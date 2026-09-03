"""The one file that touches ROS.

Everything robot-specific enters here as a parameter and leaves as a plain
dict, so the service, the model and the collision check never import rclpy.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from urllib.parse import quote
import collections
import re
import signal
import threading
import time
import xml.etree.ElementTree as ElementTree

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from ..interfaces import (CommandSpec, EFFORT_SOURCES, SignalMap,
                          TelemetrySpec)
from ..profile import RobotProfile
from .http_server import DashboardServer
from .service import DashboardConfig, IdentificationService

MESH_TYPES = {".stl": "model/stl", ".dae": "model/vnd.collada+xml",
              ".obj": "text/plain", ".png": "image/png", ".jpg": "image/jpeg",
              ".tga": "image/x-tga"}
DEFAULT_ACTION = "/joint_trajectory_controller/follow_joint_trajectory"
# Frames kept for the live panel to collect. A browser polling ten times a
# second would otherwise see one frame in twenty on a 200 Hz arm, and a current
# spike between two polls would simply never have happened.
TELEMETRY_DEPTH = 3000
# How often a mapping that matches nothing on the topic may say so. Loud enough
# to be found, quiet enough not to drown the log at the telemetry rate.
MISMATCH_WARN_S = 5.0
DESCRIPTION_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)


class DashboardNode(Node):
    """Bridges standard ROS interfaces to the dashboard service."""

    def __init__(self) -> None:
        super().__init__("robot_parameter_identification")
        get = self._declare
        port = int(get("port", 8300))
        telemetry = TelemetrySpec(
            joint_state_topic=str(get("joint_state_topic", "/joint_states")),
            dynamic_joint_state_topic=str(
                get("dynamic_joint_state_topic", "/dynamic_joint_states")),
            extra_dynamic_joint_state_topics=tuple(
                self._declare_strings("extra_telemetry_topics")),
            stale_after_s=float(get("telemetry_stale_s", 0.5)),
            signals=SignalMap(
                position=str(get("signal.position", "position")),
                velocity=_optional(get("signal.velocity", "velocity")),
                # Map whichever the drive publishes; map both if it has both.
                current=_optional(get("signal.current", "current")),
                torque=_optional(get("signal.torque", "effort")),
                effort_source=str(get("effort_source", "current")),
                temperature=_optional(get("signal.temperature", "temperature")),
                # Mapped by default: both are unambiguous on any drive and
                # need no threshold guessed. The bus-voltage window does need
                # one, so it stays off until an operator supplies a profile.
                enabled=_optional(get("signal.enabled", "enabled")),
                fault_code=_optional(get("signal.fault_code", "fault_code")),
                voltage=_optional(get("signal.voltage", "")),
            ))
        commands = CommandSpec(
            follow_joint_trajectory_action=self._resolve_action(),
            robot_description_topic=str(
                get("robot_description_topic", "/robot_description")))
        # The launch argument is symmetric; the panel may make it not.
        workspace = self._declare_floats("workspace_limit_deg")
        config = DashboardConfig(
            profile_path=str(get("profile_path", "")),
            output_directory=str(get("output_directory",
                                     "identification_results")),
            config_file_path=str(get("config_file_path", "")),
            safety_margin_m=float(get("safety_margin_m", 0.02)),
            workspace_range_deg=tuple((-value, value) for value in workspace),
            maximum_speed_deg_s=float(get("maximum_speed_deg_s", 0.0)),
            telemetry=telemetry, commands=commands)

        profile = None
        if config.profile_path:
            try:
                profile = RobotProfile.from_yaml(Path(config.profile_path))
            except Exception as error:  # noqa: BLE001
                self.get_logger().error(f"profile not loaded: {error}")

        self.service = IdentificationService(config, bridge=self,
                                             profile=profile)
        self._lock = threading.Lock()
        self._sample: dict | None = None
        self._sample_at = 0.0
        self._observed: set = set()
        self._complained_at = 0.0
        self._history: collections.deque = collections.deque(
            maxlen=TELEMETRY_DEPTH)
        self._sequence = 0
        # Signals arriving on their own topics, by topic: (arrival, per joint).
        self._extra: dict[str, tuple[float, dict]] = {}
        # Joints this dashboard does not drive, in radians by name.
        self._elsewhere: dict[str, float] = {}
        self._visuals: list[dict] = []
        self._package_dirs: dict[str, Path] = {}

        self.create_subscription(String, commands.robot_description_topic,
                                 self._on_description, DESCRIPTION_QOS)
        self._subscribe_telemetry(telemetry)
        self._subscribe_controller_state(commands)

        self.server = DashboardServer(self.service, port=port,
                                      mesh_resolver=self._read_mesh, node=self)
        self.server.start()
        self.get_logger().info(
            f"dashboard on http://localhost:{self.server.port} "
            f"| telemetry {telemetry.topic()} ({telemetry.transport()})")
        for guard in telemetry.signals.missing_guards():
            self.get_logger().warn(f"guard unavailable: {guard}")

    def _declare(self, name: str, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _resolve_action(self) -> str:
        """Which controller this dashboard drives, named the short way.

        ``controller:=left_arm_jtc`` is the whole of it; the action and the
        controller-state topic both follow. The full action path stays
        available for a controller that does not follow the usual layout, and
        wins when it is given, because it is the more specific statement.
        """
        action = str(self._declare("follow_joint_trajectory_action",
                                   DEFAULT_ACTION))
        controller = str(self._declare("controller", "")).strip()
        if controller and action == DEFAULT_ACTION:
            return CommandSpec.for_controller(
                controller).follow_joint_trajectory_action
        return action

    def _declare_strings(self, name: str) -> list[str]:
        """A string list parameter, with blank entries meaning 'not set'.

        An empty list default is inferred as a byte array and then refuses the
        strings the operator passes, so the default carries one blank instead.
        """
        value = self._declare(name, [""]) or []
        return [str(entry).strip() for entry in value if str(entry).strip()]

    def _declare_floats(self, name: str) -> list[float]:
        """A float list parameter, with zero meaning 'not set'.

        An empty list default is inferred as a byte array and then refuses the
        doubles the operator passes, so the default carries one zero instead.
        Every value this reads is a positive cap, so zero is unambiguous.
        """
        value = self._declare(name, [0.0]) or []
        return [float(entry) for entry in value if float(entry) > 0.0]

    # -- which joints are we identifying ---------------------------------

    def _subscribe_controller_state(self, commands: CommandSpec) -> None:
        """Ask the controller which joints it drives, rather than being told.

        A dual-arm URDF has twice the joints the action moves, and identifying
        a model the controller cannot command is meaningless. The joint list is
        published on a standard topic, so this needs no per-robot config.
        """
        from control_msgs.msg import JointTrajectoryControllerState  # noqa: PLC0415

        self.create_subscription(JointTrajectoryControllerState,
                                 commands.controller_state_topic,
                                 self._on_controller_state, 5)

    def _on_controller_state(self, message) -> None:
        names = [str(entry) for entry in message.joint_names]
        if names:
            self.service.adopt_driven_joints(names)

    # -- telemetry -------------------------------------------------------

    def _subscribe_telemetry(self, spec: TelemetrySpec) -> None:
        self._spec = spec
        from control_msgs.msg import DynamicJointState  # noqa: PLC0415

        if spec.transport() == "dynamic_joint_states":
            self.create_subscription(DynamicJointState, spec.topic(),
                                     self._on_dynamic_state, 20)
        else:
            from sensor_msgs.msg import JointState  # noqa: PLC0415

            self.create_subscription(JointState, spec.topic(),
                                     self._on_joint_state, 20)
        for topic in spec.extra_dynamic_joint_state_topics:
            self.create_subscription(
                DynamicJointState, topic,
                lambda message, source=topic: self._on_extra_state(source,
                                                                   message),
                20)

    def _on_extra_state(self, topic: str, message) -> None:
        """A signal that reaches ROS on its own topic rather than the main one.

        Merged by joint name, so a republisher only has to name the joints and
        the interface it carries; it need match nothing about the layout of the
        main state topic.
        """
        with self._lock:
            self._extra[topic] = (time.monotonic(), _by_joint(message))

    def _merge_extra(self, primary: dict) -> dict:
        with self._lock:
            sources = list(self._extra.values())
        return merge_by_joint(primary, sources, time.monotonic(),
                              self._spec.stale_after_s)

    def _assemble(self, by_joint: dict, wanted: list) -> None:
        """One telemetry frame, or nothing: a partial frame is not a sample."""
        signals = self._spec.signals
        roles = signals.optional_interfaces()
        channels = signals.effort_channels()
        rows: dict[str, list[float]] = {}
        seen: set[str] = set()
        for name in wanted:
            values = by_joint.get(name)
            if values is None:
                self._complain(f"{self._spec.topic()} carries no joint {name!r}")
                return
            seen.update(values)
            if signals.position not in values:
                break
            rows.setdefault("position", []).append(
                float(values[signals.position]))
            # Every mapped effort channel is read, not just the fitted one: a
            # drive reporting both is worth showing in full, and which one gets
            # fitted is settled from what actually arrives.
            for role, interface in {**roles, **channels}.items():
                if interface in values:
                    rows.setdefault(role, []).append(float(values[interface]))
        count = len(wanted)
        if count == 0:
            return
        # A channel short on any joint is unusable for all of them, and a frame
        # with no effort at all is not a sample whichever channel is missing.
        if (len(rows.get("position", [])) != count
                or not any(len(rows.get(source, [])) == count
                           for source in EFFORT_SOURCES)):
            self._complain(
                f"no usable frame: need {signals.position!r} and one of "
                f"{sorted(channels.values())}; {self._spec.topic()} carries "
                f"{sorted(seen)}")
            return
        self._store(rows, count)

    def _on_dynamic_state(self, message) -> None:
        signals = self._spec.signals
        wanted = self._joint_names()
        by_name = _by_joint(message)
        self._note_everything_else(
            {name: values.get(signals.position)
             for name, values in by_name.items()}, wanted)
        self._assemble(self._merge_extra(by_name), wanted)

    def _on_joint_state(self, message) -> None:
        signals = self._spec.signals
        wanted = self._joint_names()
        by_name: dict[str, dict[str, float]] = {}
        for index, name in enumerate(message.name):
            values: dict[str, float] = {}
            if index < len(message.position):
                values[signals.position] = float(message.position[index])
            if signals.velocity and index < len(message.velocity):
                values[signals.velocity] = float(message.velocity[index])
            # JointState has one effort field and no name for it, so it stands
            # in for whichever channel the fit was pointed at.
            if index < len(message.effort):
                values[signals.effort] = float(message.effort[index])
            by_name[name] = values
        self._note_everything_else(
            {name: values.get(signals.position)
             for name, values in by_name.items()}, wanted)
        self._assemble(self._merge_extra(by_name), wanted)

    def _joint_names(self) -> list[str]:
        if self.service.profile is not None:
            return list(self.service.profile.joint_names)
        if self.service.arm is not None:
            return list(self.service.arm.joint_names)
        return []

    def _settle(self, arrived: list) -> SignalMap:
        """Fit against a channel the robot publishes, not one it was asked for.

        Both channels are mapped by default, so the usual case is that the
        preference holds and this changes nothing. When it does change, the
        unit of every identified parameter changes with it, so it is announced.
        """
        signals = self._spec.signals
        settled = signals.settled_among(arrived)
        if settled is signals:
            return signals
        self._spec = replace(self._spec, signals=settled)
        self.get_logger().warn(
            f"{signals.effort_source} is not published; fitting against "
            f"{settled.effort_source} ({settled.effort_unit}) instead")
        self.service.adopt_signals(settled)
        return settled

    def _complain(self, message: str) -> None:
        """Say why no frame is being made, at most every few seconds.

        A mapping that names an interface the robot does not publish stops
        telemetry dead, and the only symptom is a panel that never fills.
        """
        now = time.monotonic()
        if now - self._complained_at < MISMATCH_WARN_S:
            return
        self._complained_at = now
        self.get_logger().warn(message)

    def _note_everything_else(self, positions: dict, driven: list) -> None:
        """Where the rest of the robot is, in radians, by joint name.

        This dashboard drives one arm, but it draws the whole robot and its
        collision scene holds every link the URDF ships. Without this the other
        arm is pinned at the pose it was reduced against -- drawn at neutral
        wherever it actually is, and, more to the point, screened there too.
        """
        elsewhere = {name: float(value) for name, value in positions.items()
                     if name not in driven and value is not None
                     and np.isfinite(value)}
        if not elsewhere:
            return
        with self._lock:
            first = not self._elsewhere
            self._elsewhere = elsewhere
        # The latched URDF beats the first joint state, so the screen is
        # usually built before this is known. Once per session, rebuild it.
        if first:
            self.service.adopt_elsewhere()

    def elsewhere(self) -> dict:
        with self._lock:
            return dict(self._elsewhere)

    def _store(self, rows: dict[str, list[float]], count: int) -> None:
        if count == 0:
            return
        arrived = [source for source in EFFORT_SOURCES
                   if len(rows.get(source, [])) == count]
        if len(rows.get("position", [])) != count or not arrived:
            return
        signals = self._settle(arrived)
        scale = 1.0 if signals.position_in_degrees else 180.0 / np.pi
        sample = {
            "position_deg": [value * scale for value in rows["position"]],
            # The fitted channel, under a name that predates there being two.
            "current_a": list(rows[signals.effort_source]),
        }
        # A drive may publish both, so the panel gets each under its own name
        # rather than having to guess which quantity it is looking at.
        if "current" in arrived:
            sample["drive_current_a"] = list(rows["current"])
        if "torque" in arrived:
            sample["joint_torque_nm"] = list(rows["torque"])
        if "velocity" in rows:
            sample["speed_deg_s"] = [value * scale for value in rows["velocity"]]
        for role, key in (("temperature", "temperature_c"),
                          ("voltage", "voltage_v")):
            if role in rows:
                sample[key] = list(rows[role])
        if "enabled" in rows:
            sample["enabled"] = [value > 0.5 for value in rows["enabled"]]
        if "fault_code" in rows:
            sample["fault_code"] = [int(value) for value in rows["fault_code"]]
        if not all(np.isfinite(sample["position_deg"])):
            return
        with self._lock:
            self._sample = sample
            self._sample_at = time.monotonic()
            self._observed = {role for role, values in rows.items()
                              if len(values) == count}
            self._sequence += 1
            self._history.append((self._sequence, self._sample_at, sample))

    def history_since(self, cursor: int, limit: int = 400) -> dict:
        """Every frame after ``cursor``, newest last, capped at ``limit``.

        A cursor rather than a timestamp: the panel then knows whether it fell
        behind, instead of silently plotting a decimated signal as though it
        were the whole of it.
        """
        with self._lock:
            newest = self._sequence
            wanted = [entry for entry in self._history if entry[0] > cursor]
        dropped = max(0, len(wanted) - limit)
        wanted = wanted[-limit:]
        now = time.monotonic()
        return {
            "cursor": newest,
            "dropped": dropped,
            "frames": [{"age_s": round(now - at, 4), **frame}
                       for _seq, at, frame in wanted],
        }

    def latest_sample(self) -> dict | None:
        with self._lock:
            if self._sample is None:
                return None
            if time.monotonic() - self._sample_at > self._spec.stale_after_s:
                return None
            return dict(self._sample)

    def health(self) -> dict:
        with self._lock:
            age = (None if self._sample is None
                   else time.monotonic() - self._sample_at)
        return {
            "telemetry_ok": age is not None and age <= self._spec.stale_after_s,
            "sample_age_s": None if age is None else round(age, 3),
            "action_ok": self._action_available(),
        }

    def observed_signals(self) -> set:
        """Roles that have actually arrived, not merely been named."""
        with self._lock:
            return set(self._observed)

    def _action_available(self) -> bool:
        """Whether anyone is offering the trajectory action.

        An action is not a topic, but rclpy exposes its feedback and status
        under ``<action>/_action/``, and those are enough to answer the only
        question the operator has: is there a server there.
        """
        name = self.service.config.commands.follow_joint_trajectory_action
        prefix = f"{name}/_action/"
        for topic, _types in self.get_topic_names_and_types():
            if topic.startswith(prefix):
                return True
        return False

    def hardware_plant(self, profile, collision_scene,
                       require_neutral_start: bool = True,
                       expected_start_deg=(),
                       maximum_speed_deg_s: float | None = None):
        from ..plants.ros_control import HardwareConfig, HardwarePlant  # noqa: PLC0415

        config = HardwareConfig(
            action=self.service.config.commands.follow_joint_trajectory_action,
            state_topic=self._spec.topic(),
            signals=self._spec.signals,
            require_neutral_start=require_neutral_start,
            expected_start_deg=tuple(float(v) for v in expected_start_deg))
        if maximum_speed_deg_s is not None:
            config.maximum_speed_deg_s = float(maximum_speed_deg_s)
        # The plant must build its OWN node, context and executor. Lending it
        # this one puts the campaign thread and rclpy.spin() on the same wait
        # set, which corrupts it and takes the dashboard down mid-run.
        plant = HardwarePlant(profile, config=config,
                              collision_model=collision_scene)
        plant.open()
        return plant

    # -- robot description ----------------------------------------------

    def _on_description(self, message: String) -> None:
        if self.service.adopt_description(message.data):
            self._visuals = parse_visuals(message.data)
            self.get_logger().info(
                f"model adopted: {len(self._visuals)} mesh visuals")

    def visuals(self) -> list[dict]:
        return list(self._visuals)

    def _read_mesh(self, package: str, relative: str) -> tuple[bytes, str]:
        directory = self._package_dir(package)
        target = Path(directory) / relative
        # Lexical containment, then read: meshes under a symlinked share tree
        # resolve outside the package dir and a resolve() check would 404 them.
        import os  # noqa: PLC0415

        if not os.path.normpath(str(target)).startswith(
                os.path.normpath(str(directory))):
            raise ValueError("path escapes the package")
        if not target.is_file():
            raise KeyError(f"{relative} not found in {package}")
        return target.read_bytes(), MESH_TYPES.get(target.suffix.lower(),
                                                   "application/octet-stream")

    def _package_dir(self, package: str) -> Path:
        if package not in self._package_dirs:
            from ament_index_python.packages import (  # noqa: PLC0415
                get_package_share_directory)

            self._package_dirs[package] = Path(
                get_package_share_directory(package))
        return self._package_dirs[package]

    def destroy_node(self) -> bool:
        try:
            self.server.stop()
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def _optional(value) -> str | None:
    text = str(value or "").strip()
    return text or None


def _by_joint(message) -> dict[str, dict[str, float]]:
    """A DynamicJointState flattened to joint -> interface -> value."""
    return {name: dict(zip(entry.interface_names, entry.values))
            for name, entry in zip(message.joint_names,
                                   message.interface_values)}


def merge_by_joint(primary: dict, sources, now: float,
                   stale_after_s: float) -> dict:
    """The main state topic, then the topics the operator pointed at.

    The extra topics win. Naming one is a statement about where a signal comes
    from, and a driver that fills its own effort field with zeros is exactly
    why an operator goes looking for another source. A source that has stopped
    publishing drops out rather than freezing its last reading into every
    subsequent frame.
    """
    merged = {name: dict(values) for name, values in primary.items()}
    for arrived_at, values in sources:
        if now - arrived_at > stale_after_s:
            continue
        for name, interfaces in values.items():
            merged.setdefault(name, {}).update(interfaces)
    return merged


def parse_visuals(urdf_xml: str) -> list[dict]:
    """Mesh visuals per link, as URLs the browser can fetch through /mesh."""
    try:
        root = ElementTree.fromstring(urdf_xml)
    except ElementTree.ParseError:
        return []
    found = []
    for link in root.findall("link"):
        name = link.get("name", "")
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename", "")
            match = re.match(r"package://([^/]+)/(.+)", filename)
            if not match:
                continue
            origin = visual.find("origin")
            found.append({
                "link": name,
                "url": f"/mesh?pkg={quote(match.group(1))}"
                       f"&path={quote(match.group(2))}",
                "xyz": _triple(origin, "xyz", (0.0, 0.0, 0.0)),
                "rpy": _triple(origin, "rpy", (0.0, 0.0, 0.0)),
                "scale": _triple(mesh, "scale", (1.0, 1.0, 1.0)),
            })
    return found


def _triple(element, attribute: str, fallback) -> list[float]:
    if element is None or not element.get(attribute):
        return list(fallback)
    try:
        values = [float(part) for part in element.get(attribute).split()]
    except ValueError:
        return list(fallback)
    return values if len(values) == 3 else list(fallback)


def main(args=None) -> None:
    # rclpy's own handler shuts the context down from inside the signal, which
    # destroys the subscription handles the executor is mid-take on: every
    # Ctrl-C and every pkill then ends in "Unable to convert call argument to
    # Python object" and exit code 1. Stopping the loop first and tearing down
    # afterwards is the same shutdown in the order the objects allow.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DashboardNode()
    stopping = threading.Event()

    def stop(signum, _frame) -> None:
        stopping.set()
        # A second one still forces out, in case teardown is what is stuck.
        signal.signal(signum, signal.SIG_DFL)

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, stop)
    try:
        while rclpy.ok() and not stopping.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
