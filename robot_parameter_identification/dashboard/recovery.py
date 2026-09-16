"""Passive evidence for recovery of an exited right-arm gravity hold worker.

This monitor grants no command authority and performs no I/O. Positions and
velocities are raw ros2_control radians; freshness uses a monotonic clock.
"""

import math
import threading
import time


RIGHT_JOINTS = tuple(f"right_arm_joint{index}" for index in range(1, 8))
CURRENT_CONTROLLER = "right_arm_forward_current_controller"
MAX_AGE_S = 0.1
INVENTORY_MAX_AGE_S = 3.0
HEALTHY_S = 1.0
MAX_TRAVEL_RAD = math.radians(0.5)
MAX_SPEED_RAD_S = math.radians(1.0)
SAFE_FLAGS = {
    "direct_current_active": 0,
    "direct_current_stop_confirmed": 1,
    "direct_current_fault_latched": 0,
    "enabled": 1,
    "fault_code": 0,
}


class RecoveryMonitor:
    """Thread-safe, fail-closed evidence accumulator, independent of ROS."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._action = None
        self._statuses = None
        self._selection = None
        self._since = None
        self._at = None
        self._sequences = None
        self._advanced_at = None
        self._positions = None
        self._travel = [0.0] * 7
        self._reason = "waiting for complete hardware telemetry"

    def _reset(self, reason):
        self._since = None
        self._travel = [0.0] * 7
        self._reason = reason

    def action_status(self, action, statuses):
        """Record a received GoalStatusArray; None means unknown, [] is idle."""
        with self._lock:
            if action != self._action:
                self._reset("trajectory action changed")
            self._action = action
            self._statuses = None if statuses is None else tuple(statuses)
            if self._statuses is None or any(
                    status not in (4, 5, 6) for status in self._statuses):
                self._reset("trajectory goal status unknown or nonterminal")

    def _context_reason(self, joints, action, inventory, require_goal_status=True):
        selection = (tuple(joints), action)
        if selection != self._selection:
            self._reset("recovery selection changed")
            self._selection = selection
        if len(joints) != 7 or set(joints) != set(RIGHT_JOINTS):
            return "recovery evidence is only for the seven driven right-arm joints"
        suffix = "/follow_joint_trajectory"
        if not isinstance(action, str) or not action.endswith(suffix):
            return "selected trajectory action is invalid"
        controller = action[:-len(suffix)].strip("/")
        if not controller or controller == CURRENT_CONTROLLER:
            return "selected trajectory controller is invalid"
        if self._action not in (None, action):
            return "trajectory action changed"
        if self._statuses is None and require_goal_status:
            return "trajectory goal status unknown"
        if any(status not in (4, 5, 6) for status in (self._statuses or ())):
            return "trajectory goal is nonterminal"
        try:
            age = inventory["age_s"]
            if (inventory["available"] is not True or inventory.get("error")
                    or age is None or not math.isfinite(age)
                    or not 0 <= age < INVENTORY_MAX_AGE_S):
                return "controller inventory unavailable or stale"
            items = inventory["items"]
            current = [item for item in items
                       if item["name"] == CURRENT_CONTROLLER]
            selected = [item for item in items if item["name"] == controller]
            if len(current) != 1 or current[0]["state"] != "inactive":
                return "right-arm current controller is not confirmed inactive"
            if len(selected) != 1 or selected[0]["state"] != "active":
                return "selected trajectory controller is not active"
            if selected[0].get("type") != "joint_trajectory_controller/JointTrajectoryController":
                return "selected controller is not a joint trajectory controller"
            required = {f"{joint}/position" for joint in joints}
            if not required.issubset(set(selected[0]["claimed_interfaces"])):
                return "selected trajectory controller lacks position resources"
        except (KeyError, TypeError, ValueError):
            return "controller inventory unavailable or malformed"
        return ""

    def _fresh_reason(self, now):
        if self._at is None or not 0 <= now - self._at <= MAX_AGE_S:
            return "hardware telemetry stale or missing"
        if self._advanced_at is None or any(
                not 0 <= now - advanced <= MAX_AGE_S
                for advanced in self._advanced_at):
            return "hardware telemetry sequence frozen or stale"
        return ""

    def observe(self, by_joint, joints, action, inventory):
        """Consume every primary dynamic frame, never merged or assembled data."""
        with self._lock:
            now = self._clock()
            context_reason = self._context_reason(
                joints, action, inventory, require_goal_status=False)
            gap_reason = self._fresh_reason(now)
            try:
                rows = [by_joint[joint] for joint in RIGHT_JOINTS]
                for row in rows:
                    for flag, expected in SAFE_FLAGS.items():
                        if row[flag] != expected:
                            raise ValueError(f"unsafe hardware flag {flag}")
                positions = [float(row["position"]) for row in rows]
                speeds = [float(row["velocity"]) for row in rows]
                sequences = [float(row["telemetry_sequence"]) for row in rows]
                if not all(math.isfinite(value)
                           for value in positions + speeds + sequences):
                    raise ValueError("nonfinite hardware telemetry")
                if any(value < 0 or not value.is_integer() for value in sequences):
                    raise ValueError("invalid hardware telemetry sequence")
                if any(abs(value) > MAX_SPEED_RAD_S for value in speeds):
                    raise ValueError("right arm is moving too fast")
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                self._reset(f"incomplete or unsafe hardware telemetry: {error}")
                self._at = None
                self._sequences = None
                self._advanced_at = None
                self._positions = None
                return
            regressed = self._sequences is not None and any(
                value < previous for value, previous in zip(sequences, self._sequences))
            if regressed or self._sequences is None:
                self._advanced_at = [now] * 7
            else:
                self._advanced_at = [
                    now if value > previous else advanced
                    for value, previous, advanced in zip(
                        sequences, self._sequences, self._advanced_at)]
            self._at = now
            self._sequences = sequences
            reason = (context_reason or
                      ("hardware telemetry sequence regressed" if regressed else "")
                      or gap_reason or self._fresh_reason(now))
            if reason:
                self._reset(reason)
            if context_reason or regressed or self._fresh_reason(now):
                self._positions = positions
                return
            if self._since is None:
                self._since = now
            elif self._positions is not None:
                self._travel = [travel + abs(value - previous)
                                for travel, value, previous in zip(
                                    self._travel, positions, self._positions)]
                if any(travel > MAX_TRAVEL_RAD for travel in self._travel):
                    self._reset("right-arm total travel exceeds 0.5 degrees")
            self._positions = positions
            if self._since is not None:
                self._reason = "waiting for one second of continuously safe evidence"

    def status(self, joints, action, inventory, *, require_goal_status=True) -> dict:
        """Read current evidence, resetting the window if any prerequisite fails."""
        with self._lock:
            now = self._clock()
            reason = (self._context_reason(joints, action, inventory, require_goal_status)
                      or self._fresh_reason(now))
            if reason:
                self._reset(reason)
            if self._since is not None and self._at - self._since >= HEALTHY_S:
                return {"ready": True, "reason": "right-arm recovery evidence ready"}
            return {"ready": False, "reason": self._reason}
