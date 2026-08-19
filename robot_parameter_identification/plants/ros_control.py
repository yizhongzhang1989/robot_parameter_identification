"""Real-robot plant driven through a ros2_control joint trajectory controller.

Motion goes through the commissioned position path only. Current mode is never
enabled here: during identification the current is the quantity being measured,
so commanding it would beg the question.

Nothing in this file is specific to one arm - the joint names, limits and
envelope all come from the profile. ROS is imported lazily so the rest of the
package stays testable without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time

import numpy as np

from ..interfaces import SignalMap
from ..profile import RobotProfile

# A quintic zero-velocity endpoint trajectory peaks near 1.875x its average
# speed, so durations are scaled to keep the measured peak at the request.
PEAK_TO_AVERAGE = 1.875
MINIMUM_SEGMENT_S = 1.0
SETTLE_S = 0.6
NEUTRAL_TOLERANCE_DEG = 1.0
TELEMETRY_REFRESH_S = 1.0


def differentiate(times_s, values, window: int = 5) -> np.ndarray:
    """Central-difference derivative with light smoothing.

    Acceleration is not published, so phase C has to differentiate the measured
    velocity. Using the commanded value instead would hide tracking error in
    exactly the term the phase exists to measure.
    """
    times = np.asarray(times_s, dtype=float)
    series = np.asarray(values, dtype=float)
    if series.ndim == 1:
        series = series[:, None]
    count = series.shape[0]
    if count < 3:
        return np.zeros_like(series)

    span = max(1, int(window) // 2)
    derivative = np.zeros_like(series)
    for index in range(count):
        low = max(0, index - span)
        high = min(count - 1, index + span)
        if high == low:
            continue
        dt = times[high] - times[low]
        if dt <= 0.0:
            continue
        derivative[index] = (series[high] - series[low]) / dt
    return derivative


def fit_window(frames: list[dict], joint_count: int) -> dict | None:
    """Collapse a burst of consecutive frames into one high-quality sample.

    Position is fitted quadratically against the publisher's own timestamp, so
    speed comes from the slope and acceleration from the curvature. The drive's
    reported velocity is quantised and, on this arm, reads up to 7.8 deg/s while
    the joint provably has not moved; a fit over a tenth of a second of 0.001
    deg position resolves speed some three orders of magnitude finer.
    """
    if len(frames) < 3:
        return None
    stamps = np.asarray([f.get("stamp_s", 0.0) for f in frames], dtype=float)
    span = float(stamps[-1] - stamps[0])
    if not np.isfinite(span) or span <= 0.0:
        return None
    centre = 0.5 * (stamps[0] + stamps[-1])
    t = stamps - centre

    positions = np.asarray([f["position_deg"] for f in frames], dtype=float)
    if positions.shape[1] != joint_count:
        return None
    basis = np.vstack([np.ones_like(t), t, t ** 2]).T
    coefficients, *_ = np.linalg.lstsq(basis, positions, rcond=None)
    residual = positions - basis @ coefficients

    sample = {
        "position_deg": coefficients[0].tolist(),
        "speed_deg_s": coefficients[1].tolist(),
        "acceleration_deg_s2": (2.0 * coefficients[2]).tolist(),
        "stamp_s": float(centre),
        "window_frames": len(frames),
        "window_span_s": round(span, 5),
        # How straight the motion was over the window. A pass that was still
        # ramping, or a frame dropped mid-window, shows up here rather than
        # silently biasing the speed.
        "window_fit_rms_deg": float(np.sqrt(np.mean(residual ** 2))),
    }
    for key in ("current_a", "temperature_c", "voltage_v"):
        values = [f.get(key, []) for f in frames]
        if all(len(v) == joint_count for v in values):
            sample[key] = np.asarray(values, dtype=float).mean(axis=0).tolist()
        else:
            sample[key] = []
    for key, combine in (("enabled", all), ("fault_code", max)):
        values = [f.get(key, []) for f in frames]
        if all(len(v) == joint_count for v in values):
            sample[key] = [combine(v[index] for v in values)
                           for index in range(joint_count)]
        else:
            sample[key] = []
    sample["arm_status"] = None
    return sample


def pick_windows(frames: list[dict], count: int, span_s: float) -> list[list[dict]]:
    """Evenly spaced bursts from the settled middle of a motion.

    The ends are ramps, so they are excluded: what the friction phase wants is
    the part where the joint is already up to speed.
    """
    if not frames or count < 1:
        return []
    stamps = [f.get("stamp_s", 0.0) for f in frames]
    total = stamps[-1] - stamps[0]
    if total <= 0.0:
        return []
    # Keep clear of the ramps at either end.
    usable = [f for f in frames
              if stamps[0] + 0.2 * total <= f.get("stamp_s", 0.0)
              <= stamps[0] + 0.8 * total]
    if len(usable) < 3:
        usable = frames
    windows = []
    for index in range(count):
        share = (index + 0.5) / count
        centre = usable[0]["stamp_s"] + share * (
            usable[-1]["stamp_s"] - usable[0]["stamp_s"])
        burst = [f for f in usable
                 if abs(f.get("stamp_s", 0.0) - centre) <= 0.5 * span_s]
        if len(burst) >= 3:
            windows.append(burst)
    return windows


@dataclass
class HardwareConfig:
    action: str = "/joint_trajectory_controller/follow_joint_trajectory"
    state_topic: str = "/dynamic_joint_states"
    # Which published interface carries each quantity. Naming a fixed set here
    # is what stopped this plant working on any arm but the one it was written
    # for: a torque-only drive has no "current", and plenty have no bus voltage.
    signals: SignalMap = field(default_factory=SignalMap)
    maximum_speed_deg_s: float = 10.0
    settle_s: float = SETTLE_S
    settle_samples: int = 5
    stream_rate_hz: float = 50.0
    # One motion yields this many fitted samples, each from a burst of raw
    # frames. Repeated frames at one pose are repeated rows of the same
    # regressor, so collecting thousands of them buys noise averaging and
    # nothing else, while out-voting the sweeps that carry the speed content.
    #
    # One window, not several: measured on this arm, the current wanders about
    # twenty times more within a pass than the pass mean moves between
    # repeats. Three short windows are three correlated looks at the same
    # pass; one long one is a better measurement of it.
    window_span_s: float = 0.1
    window_maximum_span_s: float = 1.5
    # How far the joint may travel inside a window. Averaging across a wider
    # arc than this starts averaging across a changing gravity term.
    window_arc_deg: float = 6.0
    windows_per_move: int = 1
    # Every raw frame behind those samples, kept for the run folder.
    keep_raw_frames: bool = True
    require_neutral_start: bool = True
    goal_timeout_margin_s: float = 8.0


class TelemetryUnavailable(RuntimeError):
    """Raised when the arm is not publishing usable joint state."""


class HardwarePlant:
    """Campaign plant backed by the joint trajectory controller.

    ``collision_free`` is delegated to a simulated twin when one is supplied,
    because the hardware cannot be asked whether a pose it has not reached yet
    would collide.
    """

    def __init__(self, profile: RobotProfile,
                 config: HardwareConfig | None = None,
                 collision_model=None, node=None) -> None:
        self.profile = profile
        self.joint_names = list(profile.joint_names)
        self.joint_count = profile.joint_count
        self.config = config or HardwareConfig()
        self.collision_model = collision_model
        self._node = node
        self._owns_node = node is None
        self._owns_context = False
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_at = 0.0
        # Every frame of the motion in flight, at the publisher's full rate.
        self._buffer: list[dict] = []
        self._buffering = False
        self._passes = 0
        self.raw_frames: list[dict] = []
        self._previous_position: tuple[np.ndarray | None, float] = (None, 0.0)
        self._client = None
        self._rclpy = None
        self._context = None
        self._executor = None

    # -- lifecycle -------------------------------------------------------

    def open(self, timeout_s: float = 15.0) -> None:
        import rclpy  # noqa: PLC0415
        from control_msgs.action import FollowJointTrajectory  # noqa: PLC0415
        from control_msgs.msg import DynamicJointState  # noqa: PLC0415
        from rclpy.action import ActionClient  # noqa: PLC0415
        from rclpy.executors import SingleThreadedExecutor  # noqa: PLC0415
        from rclpy.node import Node  # noqa: PLC0415

        self._rclpy = rclpy
        self._action_type = FollowJointTrajectory
        if self._owns_node:
            # Own context, own executor. Sharing a host's context means two
            # executors mutate one wait set from two threads, which corrupts it
            # ("wait set index too big") and kills the host process.
            self._context = rclpy.context.Context()
            rclpy.init(context=self._context)
            self._owns_context = True
            self._node = Node("rm_impedance_calibration_plant",
                              context=self._context)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
        self._node.create_subscription(
            DynamicJointState, self.config.state_topic, self._on_state, 50)
        self._client = ActionClient(self._node, FollowJointTrajectory,
                                    self.config.action)
        if not self._client.wait_for_server(timeout_sec=timeout_s):
            raise TelemetryUnavailable(
                f"{self.config.action} is unavailable")
        self._spin_for(2.0)
        if self.sample() is None:
            raise TelemetryUnavailable(
                f"{self.config.state_topic} is not publishing "
                f"{self.config.signals.required_interfaces()} for every joint")
        if self.config.require_neutral_start:
            position = np.asarray(self.sample()["position_deg"], dtype=float)
            if np.max(np.abs(position)) > NEUTRAL_TOLERANCE_DEG:
                raise TelemetryUnavailable(
                    "the arm must start within "
                    f"{NEUTRAL_TOLERANCE_DEG:g} deg of neutral, it is at "
                    f"{[round(v, 2) for v in position]}")

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._node is not None and self._owns_node:
            self._node.destroy_node()
            self._node = None
        # Only tear down the context if this plant created it.  A plant that
        # merely borrowed a running context -- a dashboard node, say -- must
        # never shut it down, or it kills its own host process.
        if self._owns_context and self._rclpy is not None:
            if self._rclpy.ok(context=self._context):
                self._rclpy.shutdown(context=self._context)
            self._owns_context = False
            self._context = None

    # -- telemetry -------------------------------------------------------

    def _on_state(self, message) -> None:
        signals = self.config.signals
        by_name = dict(zip(message.joint_names, message.interface_values))
        required = signals.required_interfaces()
        optional = signals.optional_interfaces()
        columns: dict[str, list[float]] = {}
        for name in self.joint_names:
            entry = by_name.get(name)
            if entry is None:
                return
            values = dict(zip(entry.interface_names, entry.values))
            if not all(interface in values for interface in required):
                return
            columns.setdefault("position", []).append(values[signals.position])
            columns.setdefault("effort", []).append(values[signals.effort])
            for role, interface in optional.items():
                if interface in values:
                    columns.setdefault(role, []).append(values[interface])

        count = len(self.joint_names)
        # A channel that is short on any joint is unusable for all of them.
        columns = {role: values for role, values in columns.items()
                   if len(values) == count}
        if not np.isfinite(np.asarray(columns["position"], dtype=float)).all():
            return
        if not np.isfinite(np.asarray(columns["effort"], dtype=float)).all():
            return

        position = np.degrees(np.asarray(columns["position"], dtype=float))
        frame = {
            "position_deg": position.tolist(),
            "speed_deg_s": self._speed(columns, position),
            "current_a": [float(v) for v in columns["effort"]],
            "temperature_c": [float(v) for v in columns.get("temperature", [])],
            "voltage_v": [float(v) for v in columns.get("voltage", [])],
            "enabled": [v > 0.999 for v in columns.get("enabled", [])],
            "fault_code": [int(v) for v in columns.get("fault_code", [])],
            "arm_status": None,
        }
        spare = "torque" if signals.effort_source == "current" else "current"
        if spare in columns:
            frame[f"{spare}_measured"] = [float(v) for v in columns[spare]]
        # The publisher's own stamp, not the time this callback ran. Messages
        # arrive in bursts whenever the executor gets a slice, so arrival time
        # compresses a second of motion into a millisecond and any speed
        # differentiated from it is meaningless.
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        if stamp is not None:
            frame["stamp_s"] = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        else:
            frame["stamp_s"] = time.monotonic()
        with self._lock:
            self._latest = frame
            self._latest_at = time.monotonic()
            if self._buffering:
                self._buffer.append(frame)

    def _speed(self, columns: dict, position_deg: np.ndarray) -> list[float]:
        """Mapped velocity when the drive publishes one, else differentiated.

        The signal map promises velocity is optional; without this that promise
        was false, because the fit needs a speed for every frame.
        """
        if "velocity" in columns:
            return np.degrees(
                np.asarray(columns["velocity"], dtype=float)).tolist()
        now = time.monotonic()
        previous, at = self._previous_position
        self._previous_position = (position_deg, now)
        gap = now - at
        if previous is None or gap <= 1e-4 or gap > 0.5:
            return [0.0] * len(position_deg)
        return ((position_deg - previous) / gap).tolist()

    def _spin_for(self, seconds: float) -> None:
        if self._rclpy is None or self._node is None:
            return
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._spin_once(0.02)

    def _spin_once(self, timeout_s: float) -> None:
        if self._executor is not None:
            self._executor.spin_once(timeout_sec=timeout_s)
        else:
            self._rclpy.spin_once(self._node, timeout_sec=timeout_s)

    def _wait(self, future, timeout_s: float) -> None:
        if self._executor is not None:
            self._executor.spin_until_future_complete(future, timeout_sec=timeout_s)
        else:
            self._rclpy.spin_until_future_complete(
                self._node, future, timeout_sec=timeout_s)

    def sample(self, maximum_age_s: float = 0.5) -> dict | None:
        with self._lock:
            if self._latest is None:
                return None
            if time.monotonic() - self._latest_at > maximum_age_s:
                return None
            return dict(self._latest)

    def _capture(self) -> None:
        with self._lock:
            self._buffer = []
            self._buffering = True

    def _harvest(self, tag: str, phase: str = "") -> list[dict]:
        """Stop buffering and return the motion's frames, keeping a copy."""
        with self._lock:
            self._buffering = False
            frames = self._buffer
            self._buffer = []
        if self.config.keep_raw_frames:
            for index, frame in enumerate(frames):
                stored = dict(frame)
                stored["motion"] = tag
                stored["phase"] = phase
                stored["frame"] = index
                self.raw_frames.append(stored)
        return frames

    def _window_span(self, speed_deg_s: float) -> float:
        """The longest window this speed can fill without crossing much arc."""
        speed = abs(float(speed_deg_s))
        if speed <= 0.0:
            return self.config.window_maximum_span_s
        by_arc = self.config.window_arc_deg / speed
        return float(min(max(by_arc, self.config.window_span_s),
                         self.config.window_maximum_span_s))

    def _observations(self, frames: list[dict], tag: str,
                      speed_deg_s: float = 0.0) -> list[dict]:
        """The fitted samples one motion contributes to the regression."""
        windows = pick_windows(frames, self.config.windows_per_move,
                               self._window_span(speed_deg_s))
        found = []
        for index, burst in enumerate(windows):
            fitted = fit_window(burst, self.joint_count)
            if fitted is None:
                continue
            fitted["motion"] = tag
            fitted["window"] = index
            found.append(fitted)
        return found

    def _require_sample(self) -> dict:
        frame = self.sample()
        if frame is None:
            # Subscription callbacks only run while we spin, and callers may
            # compute for seconds between moves. Listen before calling the link
            # dead, or planning work looks like a telemetry failure.
            self._spin_for(TELEMETRY_REFRESH_S)
            frame = self.sample()
        if frame is None:
            raise TelemetryUnavailable("joint state is stale")
        return frame

    # -- plant interface -------------------------------------------------

    def limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        limit = np.asarray(self.profile.position_limit_deg, dtype=float)
        return -limit, limit

    def collision_free(self, pose_deg) -> bool:
        low, high = self.limits_deg()
        if np.any(pose_deg < low) or np.any(pose_deg > high):
            return False
        if self.collision_model is None:
            return True
        return bool(self.collision_model.collision_free(pose_deg))

    def _duration_for(self, target_deg, speed_deg_s: float) -> float:
        current = np.asarray(self._require_sample()["position_deg"], dtype=float)
        distance = float(np.max(np.abs(np.asarray(target_deg, dtype=float) - current)))
        speed = max(float(speed_deg_s), 0.1)
        return max(MINIMUM_SEGMENT_S, PEAK_TO_AVERAGE * distance / speed)

    def _goal(self, points):
        from trajectory_msgs.msg import JointTrajectoryPoint  # noqa: PLC0415

        goal = self._action_type.Goal()
        goal.trajectory.joint_names = self.joint_names
        for position_deg, velocity_deg_s, time_from_start in points:
            point = JointTrajectoryPoint()
            point.positions = list(np.radians(np.asarray(position_deg, dtype=float)))
            point.velocities = list(
                np.radians(np.asarray(velocity_deg_s, dtype=float)))
            seconds = int(time_from_start)
            point.time_from_start.sec = seconds
            point.time_from_start.nanosec = int(
                round((time_from_start - seconds) * 1e9))
            goal.trajectory.points.append(point)
        goal.goal_time_tolerance.sec = 4
        return goal

    def _execute(self, points, on_frame=None):
        """Send one trajectory and pump telemetry until the controller is done."""
        goal = self._goal(points)
        duration = points[-1][2]
        send_future = self._client.send_goal_async(goal)
        self._wait(send_future, 5.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("trajectory was rejected by the controller")

        result_future = handle.get_result_async()
        started = time.monotonic()
        period = 1.0 / max(self.config.stream_rate_hz, 1.0)
        next_sample = started

        def capture(now: float) -> None:
            if on_frame is None:
                return
            frame = self.sample()
            if frame is not None:
                on_frame(now - started, frame)

        # Sample the endpoints explicitly: a short or already-finished goal
        # would otherwise skip the wait loop and silently record nothing.
        capture(started)
        try:
            while not result_future.done():
                self._spin_once(0.01)
                now = time.monotonic()
                if now >= next_sample:
                    next_sample = now + period
                    capture(now)
                if now - started > duration + self.config.goal_timeout_margin_s:
                    raise RuntimeError("trajectory timed out")
        except BaseException:
            cancel = handle.cancel_goal_async()
            self._wait(cancel, 3.0)
            raise
        capture(time.monotonic())
        wrapped = result_future.result()
        if wrapped is None or (
                wrapped.result.error_code
                != self._action_type.Result.SUCCESSFUL):
            raise RuntimeError("trajectory did not complete successfully")

    def hold_pose(self, pose_deg) -> dict:
        """Move there, let the servo settle, and average a few frames at rest."""
        target = np.asarray(pose_deg, dtype=float)
        duration = self._duration_for(target, self.config.maximum_speed_deg_s)
        self._execute([(target, np.zeros(self.joint_count), duration)])
        self._spin_for(self.config.settle_s)
        return self.dwell()

    def dwell(self, pose_deg=None) -> dict:
        """Fit a burst of frames where the arm already stands.

        Fitting rather than averaging because the slope is worth having: it
        says whether the arm is truly still, which the drive's own velocity
        channel cannot, and it is the speed the regression will use.
        """
        self._capture()
        self._spin_for(max(self.config.window_span_s * 3.0, 0.15))
        frames = self._harvest("dwell", "A_gravity")
        fitted = fit_window(frames, self.joint_count) if frames else None
        if fitted is not None:
            return fitted
        # No buffered frames means the subscription is not delivering; fall
        # back to the latest single frame so the caller sees a real failure
        # from the guard rather than an empty sample here.
        return self._require_sample()

    def park(self) -> dict:
        """Return to neutral so no joint is left holding a load."""
        return self.hold_pose(np.zeros(self.joint_count))

    def traverse(self, joint: int, start_deg, distance_deg: float,
                 speed_deg_s: float):
        """Constant-speed pass so friction separates from acceleration."""
        start = np.asarray(start_deg, dtype=float)
        self.hold_pose(start)

        end = start.copy()
        end[joint] += distance_deg
        speed = max(speed_deg_s, 0.1)
        velocity = np.zeros(self.joint_count)
        velocity[joint] = math.copysign(speed, distance_deg)

        # Ramp in and out so the cruise segment really is constant speed, and
        # shorten the cruise by the time the ramps already spent covering it.
        ramp = min(0.25 * abs(distance_deg) / speed, 1.0)
        cruise = abs(distance_deg) / speed - ramp
        entry = start.copy()
        entry[joint] += math.copysign(0.5 * speed * ramp, distance_deg)
        exit_pose = end.copy()
        exit_pose[joint] -= math.copysign(0.5 * speed * ramp, distance_deg)
        points = [
            (entry, velocity, ramp),
            (exit_pose, velocity, ramp + cruise),
            (end, np.zeros(self.joint_count), ramp + cruise + ramp),
        ]

        collected: list[dict] = []
        self._capture()
        self._passes += 1
        # The repeat index is part of the tag: without it three repeats of one
        # rung share a name and cannot be told apart in the raw record.
        tag = (f"traverse:j{joint}:{speed:g}:"
               f"{'+' if distance_deg > 0 else '-'}:{self._passes}")
        try:
            self._execute(points, lambda _t, frame: collected.append(frame))
        finally:
            frames = self._harvest(tag, "B_friction")
        for observation in self._observations(frames, tag, speed):
            yield observation

    def probe_pose(self, pose_deg, delta_deg: float, speed_deg_s: float):
        """Cross the pose slowly both ways instead of holding still on it.

        Standing still leaves static friction free to take any value inside its
        band, so the current at rest is gravity plus an unknowable offset. Once
        the joint moves, friction has a determined sign, and the two passes
        bracket gravity between them.
        """
        pose = np.asarray(pose_deg, dtype=float)
        low, high = self.limits_deg()
        step = np.full(self.joint_count, abs(delta_deg))
        start = np.clip(pose - step, low, high)
        end = np.clip(pose + step, low, high)
        self.hold_pose(start)
        return self._sweep(start, end, speed_deg_s) + \
            self._sweep(end, start, speed_deg_s)

    def _sweep(self, start_deg, end_deg, speed_deg_s: float) -> list[dict]:
        """One constant-speed straight line in joint space, frames collected."""
        start = np.asarray(start_deg, dtype=float)
        end = np.asarray(end_deg, dtype=float)
        travel = end - start
        distance = float(np.max(np.abs(travel)))
        if distance < 1e-6:
            return []
        speed = max(speed_deg_s, 0.01)
        direction = travel / distance
        velocity = direction * speed
        ramp = min(0.25 * distance / speed, 1.0)
        # The ramps already cover speed*ramp of the distance, so the cruise
        # segment must be shortened by that much or it runs slower than asked.
        cruise = distance / speed - ramp
        entry = start + 0.5 * velocity * ramp
        exit_pose = end - 0.5 * velocity * ramp
        points = [
            (entry, velocity, ramp),
            (exit_pose, velocity, ramp + cruise),
            (end, np.zeros(self.joint_count), ramp + cruise + ramp),
        ]
        collected: list[dict] = []
        self._capture()
        self._passes += 1
        tag = f"sweep:{speed:g}:{self._passes}"
        try:
            self._execute(points, lambda _t, frame: collected.append(frame))
        finally:
            frames = self._harvest(tag, "A_gravity")
        return self._observations(frames, tag, speed)

    def track(self, trajectory, rate_hz: float):
        """Follow the Fourier design and fit consecutive windows of it.

        Acceleration comes out of the same quadratic fit as the speed, so the
        phase no longer differentiates the drive's quantised velocity channel,
        which amplified its noise into exactly the term this phase exists to
        measure.
        """
        start, _velocity, _acceleration = trajectory.sample(0.0)
        self.hold_pose(start)

        steps = max(2, int(trajectory.duration_s * max(rate_hz, 1.0)))
        points = []
        for step in range(1, steps + 1):
            time_s = trajectory.duration_s * step / steps
            position, velocity, _ = trajectory.sample(time_s)
            points.append((position, velocity, time_s))

        self._capture()
        try:
            self._execute(points)
        finally:
            frames = self._harvest("track", "C_inertia")
        if len(frames) < 3:
            raise RuntimeError(
                f"phase C captured {len(frames)} frames, which is too few to "
                "fit; check the joint state rate")
        span = max(self.config.window_span_s, 1e-3)
        burst: list[dict] = []
        produced = 0
        for frame in frames:
            if burst and frame["stamp_s"] - burst[0]["stamp_s"] >= span:
                fitted = fit_window(burst, self.joint_count)
                if fitted is not None:
                    fitted["motion"] = "track"
                    fitted["window"] = produced
                    produced += 1
                    yield fitted
                burst = []
            burst.append(frame)
        fitted = fit_window(burst, self.joint_count)
        if fitted is not None:
            fitted["motion"] = "track"
            fitted["window"] = produced
            yield fitted


def _average(frames: list[dict], joint_count: int) -> dict:
    """Mean of the numeric channels; the discrete ones take the worst case."""
    if len(frames) == 1:
        return frames[0]
    averaged = dict(frames[-1])
    for key in ("position_deg", "speed_deg_s", "current_a",
                "temperature_c", "voltage_v"):
        if not all(len(frame.get(key, [])) == joint_count for frame in frames):
            continue
        stacked = np.asarray([frame[key] for frame in frames], dtype=float)
        averaged[key] = stacked.mean(axis=0).tolist()
    for key, combine in (("enabled", all), ("fault_code", max)):
        # An arm that publishes no such interface reports it empty, not short.
        if not all(len(frame.get(key, [])) == joint_count for frame in frames):
            averaged[key] = []
            continue
        averaged[key] = [combine(frame[key][index] for frame in frames)
                         for index in range(joint_count)]
    return averaged
