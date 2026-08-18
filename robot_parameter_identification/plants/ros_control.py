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

from ..profile import RobotProfile

INTERFACES = (
    "position", "velocity", "current", "temperature", "voltage",
    "enabled", "fault_code",
)
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


@dataclass
class HardwareConfig:
    action: str = "/joint_trajectory_controller/follow_joint_trajectory"
    state_topic: str = "/dynamic_joint_states"
    maximum_speed_deg_s: float = 10.0
    settle_s: float = SETTLE_S
    settle_samples: int = 5
    stream_rate_hz: float = 50.0
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
                f"{self.config.state_topic} is not publishing all of {INTERFACES}")
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
        by_name = dict(zip(message.joint_names, message.interface_values))
        rows = []
        for name in self.joint_names:
            entry = by_name.get(name)
            if entry is None:
                return
            values = dict(zip(entry.interface_names, entry.values))
            if not all(interface in values for interface in INTERFACES):
                return
            rows.append([values[interface] for interface in INTERFACES])
        joints = np.asarray(rows, dtype=float)
        if not np.isfinite(joints).all():
            return
        frame = {
            "position_deg": np.degrees(joints[:, 0]).tolist(),
            "speed_deg_s": np.degrees(joints[:, 1]).tolist(),
            "current_a": joints[:, 2].tolist(),
            "temperature_c": joints[:, 3].tolist(),
            "voltage_v": joints[:, 4].tolist(),
            "enabled": [value > 0.999 for value in joints[:, 5]],
            "fault_code": [int(value) for value in joints[:, 6]],
            "arm_status": None,
        }
        with self._lock:
            self._latest = frame
            self._latest_at = time.monotonic()

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
        """Average fresh frames where the arm already stands, commanding nothing.

        Repeated readings at one pose are independent draws of sensor noise.
        They neither need nor should pay for another trajectory goal: a
        zero-distance goal still costs a full minimum segment plus settle.
        """
        frames = []
        for _ in range(max(1, self.config.settle_samples)):
            frames.append(self._require_sample())
            self._spin_for(0.05)
        return _average(frames, self.joint_count)

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
        self._execute(points, lambda _t, frame: collected.append(frame))
        for frame in collected:
            yield frame

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
        self._execute(points, lambda _t, frame: collected.append(frame))
        return collected

    def track(self, trajectory, rate_hz: float):
        """Follow the Fourier design, then differentiate what actually happened."""
        start, _velocity, _acceleration = trajectory.sample(0.0)
        self.hold_pose(start)

        steps = max(2, int(trajectory.duration_s * max(rate_hz, 1.0)))
        points = []
        for step in range(1, steps + 1):
            time_s = trajectory.duration_s * step / steps
            position, velocity, _ = trajectory.sample(time_s)
            points.append((position, velocity, time_s))

        stamps: list[float] = []
        collected: list[dict] = []

        def capture(elapsed, frame):
            stamps.append(elapsed)
            collected.append(frame)

        self._execute(points, capture)
        if len(collected) < 3:
            raise RuntimeError(
                f"phase C captured {len(collected)} frames, which is too few to "
                "differentiate; check the joint state rate")
        speeds = np.asarray([f["speed_deg_s"] for f in collected], dtype=float)
        accelerations = differentiate(stamps, speeds)
        for frame, acceleration in zip(collected, accelerations):
            frame = dict(frame)
            frame["acceleration_deg_s2"] = acceleration.tolist()
            yield frame


def _average(frames: list[dict], joint_count: int) -> dict:
    """Mean of the numeric channels; the discrete ones take the worst case."""
    if len(frames) == 1:
        return frames[0]
    averaged = dict(frames[-1])
    for key in ("position_deg", "speed_deg_s", "current_a",
                "temperature_c", "voltage_v"):
        stacked = np.asarray([frame[key] for frame in frames], dtype=float)
        averaged[key] = stacked.mean(axis=0).tolist()
    averaged["enabled"] = [
        all(frame["enabled"][index] for frame in frames)
        for index in range(joint_count)
    ]
    averaged["fault_code"] = [
        max(frame["fault_code"][index] for frame in frames)
        for index in range(joint_count)
    ]
    return averaged
