"""The one file that touches ROS.

Everything robot-specific enters here as a parameter and leaves as a plain
dict, so the service, the model and the collision check never import rclpy.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import re
import threading
import time
import xml.etree.ElementTree as ElementTree

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_msgs.msg import String

from ..interfaces import CommandSpec, SignalMap, TelemetrySpec
from ..profile import RobotProfile
from .http_server import DashboardServer
from .service import DashboardConfig, IdentificationService

MESH_TYPES = {".stl": "model/stl", ".dae": "model/vnd.collada+xml",
              ".obj": "text/plain", ".png": "image/png", ".jpg": "image/jpeg",
              ".tga": "image/x-tga"}
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
            stale_after_s=float(get("telemetry_stale_s", 0.5)),
            signals=SignalMap(
                position=str(get("signal.position", "position")),
                velocity=_optional(get("signal.velocity", "velocity")),
                effort=str(get("signal.effort", "current")),
                temperature=_optional(get("signal.temperature", "temperature")),
                voltage=_optional(get("signal.voltage", "")),
                enabled=_optional(get("signal.enabled", "")),
                fault_code=_optional(get("signal.fault_code", "")),
                effort_unit=str(get("effort_unit", "ampere")),
            ))
        commands = CommandSpec(
            follow_joint_trajectory_action=str(get(
                "follow_joint_trajectory_action",
                "/joint_trajectory_controller/follow_joint_trajectory")),
            robot_description_topic=str(
                get("robot_description_topic", "/robot_description")))
        config = DashboardConfig(
            profile_path=str(get("profile_path", "")),
            output_directory=str(get("output_directory",
                                     "identification_results")),
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

    # -- which joints are we identifying ---------------------------------

    def _subscribe_controller_state(self, commands: CommandSpec) -> None:
        """Ask the controller which joints it drives, rather than being told.

        A dual-arm URDF has twice the joints the action moves, and identifying
        a model the controller cannot command is meaningless. The joint list is
        published on a standard topic, so this needs no per-robot config.
        """
        from control_msgs.msg import JointTrajectoryControllerState  # noqa: PLC0415

        base = commands.follow_joint_trajectory_action.rsplit(
            "/follow_joint_trajectory", 1)[0]
        self.create_subscription(JointTrajectoryControllerState,
                                 f"{base}/controller_state",
                                 self._on_controller_state, 5)

    def _on_controller_state(self, message) -> None:
        names = [str(entry) for entry in message.joint_names]
        if names:
            self.service.adopt_driven_joints(names)

    # -- telemetry -------------------------------------------------------

    def _subscribe_telemetry(self, spec: TelemetrySpec) -> None:
        self._spec = spec
        if spec.transport() == "dynamic_joint_states":
            from control_msgs.msg import DynamicJointState  # noqa: PLC0415

            self.create_subscription(DynamicJointState, spec.topic(),
                                     self._on_dynamic_state, 20)
        else:
            from sensor_msgs.msg import JointState  # noqa: PLC0415

            self.create_subscription(JointState, spec.topic(),
                                     self._on_joint_state, 20)

    def _on_dynamic_state(self, message) -> None:
        signals = self._spec.signals
        wanted = self._joint_names()
        by_name = dict(zip(message.joint_names, message.interface_values))
        rows: dict[str, list[float]] = {}
        for name in wanted:
            entry = by_name.get(name)
            if entry is None:
                return
            values = dict(zip(entry.interface_names, entry.values))
            for role in ("position", "velocity", "effort", "temperature",
                         "voltage", "enabled", "fault_code"):
                interface = getattr(signals, role)
                if not interface:
                    continue
                if interface not in values:
                    if role in ("position", "effort"):
                        return
                    continue
                rows.setdefault(role, []).append(float(values[interface]))
        self._store(rows, len(wanted))

    def _on_joint_state(self, message) -> None:
        wanted = self._joint_names()
        index = {name: i for i, name in enumerate(message.name)}
        rows: dict[str, list[float]] = {}
        for name in wanted:
            position = index.get(name)
            if position is None:
                return
            rows.setdefault("position", []).append(
                float(message.position[position]))
            if message.velocity and position < len(message.velocity):
                rows.setdefault("velocity", []).append(
                    float(message.velocity[position]))
            if message.effort and position < len(message.effort):
                rows.setdefault("effort", []).append(
                    float(message.effort[position]))
        if len(rows.get("effort", [])) != len(wanted):
            return
        self._store(rows, len(wanted))

    def _joint_names(self) -> list[str]:
        if self.service.profile is not None:
            return list(self.service.profile.joint_names)
        if self.service.arm is not None:
            return list(self.service.arm.joint_names)
        return []

    def _store(self, rows: dict[str, list[float]], count: int) -> None:
        if count == 0 or len(rows.get("position", [])) != count:
            return
        scale = 1.0 if self._spec.signals.position_in_degrees else 180.0 / np.pi
        sample = {
            "position_deg": [value * scale for value in rows["position"]],
            "current_a": list(rows.get("effort", [])),
        }
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

    def hardware_plant(self, profile, collision_scene):
        from ..plants.ros_control import HardwareConfig, HardwarePlant  # noqa: PLC0415

        config = HardwareConfig(
            action=self.service.config.commands.follow_joint_trajectory_action,
            state_topic=self._spec.topic())
        plant = HardwarePlant(profile, config=config,
                              collision_model=collision_scene, node=self)
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
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
