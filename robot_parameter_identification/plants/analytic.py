"""A robot that exists only as equations, for rehearsing a campaign.

Its job is to prove the pipeline end to end -- experiment design, phase
sequencing, guards, regression, verdict -- before any of it is pointed at
hardware. It is deliberately the cheapest thing that can do that: rigid-body
torque from Pinocchio, a friction model bolted on, telemetry-shaped noise.

**What this gate does and does not prove.** It shares its rigid-body model with
the identifier, so recovering the injected parameters shows the *pipeline* is
sound, not that the *physics* is. It cannot catch an error that lives in the
model itself. When an independent engine is available, cross-check against it;
:func:`independent_engine_available` says whether one is installed. Calling this
a validation of the dynamics would be overclaiming, so the report it produces
labels itself ``self_consistency``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pinocchio as pin

from .. import excitation
from ..profile import RobotProfile

SETTLE_SAMPLES = 3
MINIMUM_SEGMENT_S = 0.2


def independent_engine_available() -> bool:
    """True when a second physics engine could cross-check this one."""
    try:
        import mujoco  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


def _vector(value, count: int, fallback: float) -> np.ndarray:
    if value is None:
        return np.full(count, float(fallback))
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(count, float(array))
    if array.size != count:
        raise ValueError(f"expected {count} values, got {array.size}")
    return array.astype(float)


@dataclass
class AnalyticPlant:
    """Rigid-body dynamics plus a friction model, sampled like a driver would.

    Every gain here is a quantity a real calibration would go on to *measure*;
    they are set to plausible values so the rehearsal exercises realistic
    magnitudes, and the recovered values are compared against them.
    """

    model: pin.Model
    profile: RobotProfile
    effort_per_torque: np.ndarray | None = None
    coulomb: np.ndarray | None = None
    coulomb_transition_deg_s: float = 1.8
    viscous_per_deg_s: np.ndarray | None = None
    offset: np.ndarray | None = None
    noise: float = 0.0
    temperature_c: float = 32.0
    seed: int = 0
    collision_scene: object | None = None
    # Matched to the hardware plant: a rehearsal whose data is shaped nothing
    # like a real run rehearses nothing worth knowing.
    windows_per_move: int = 1
    window_span_s: float = 0.1

    def __post_init__(self) -> None:
        count = self.profile.joint_count
        if self.model.nq != count:
            raise ValueError(
                f"model has {self.model.nq} joints, profile names {count}")
        self.data = self.model.createData()
        self.joint_names = list(self.profile.joint_names)
        self.joint_count = count
        self.effort_per_torque = _vector(self.effort_per_torque, count, 1.0)
        self.coulomb = _vector(self.coulomb, count, 0.0)
        self.viscous_per_deg_s = _vector(self.viscous_per_deg_s, count, 0.0)
        self.offset = _vector(self.offset, count, 0.0)
        self._rng = np.random.default_rng(self.seed)
        self._clock = 0.0

    # -- the Plant protocol ---------------------------------------------

    def limits_deg(self) -> tuple[np.ndarray, np.ndarray]:
        upper = np.asarray(self.profile.position_limit_deg, dtype=float)
        return -upper, upper

    def collision_free(self, pose_deg) -> bool:
        if self.collision_scene is None:
            return True
        return bool(self.collision_scene.collision_free(pose_deg))

    def hold_pose(self, pose_deg) -> dict:
        self._clock += 0.5
        return self._sample(pose_deg, np.zeros(self.joint_count),
                            np.zeros(self.joint_count), "hold")

    def dwell(self, pose_deg) -> dict:
        self._clock += 0.05
        return self._sample(pose_deg, np.zeros(self.joint_count),
                            np.zeros(self.joint_count), "dwell")

    def traverse(self, joint: int, start_deg, distance_deg: float,
                 speed_deg_s: float):
        start = np.asarray(start_deg, dtype=float).copy()
        speed = abs(float(speed_deg_s))
        span = abs(float(distance_deg))
        if speed <= 0.0 or span <= 0.0:
            return
        direction = math.copysign(1.0, float(distance_deg))
        duration = max(span / speed, MINIMUM_SEGMENT_S)
        # Sampled where the hardware plant samples: a few points from the
        # settled middle, not the ramps at either end.
        count = max(1, int(self.windows_per_move))
        for index in range(count):
            share = 0.2 + 0.6 * (index + 0.5) / count
            pose = start.copy()
            pose[joint] = start[joint] + direction * span * share
            velocity = np.zeros(self.joint_count)
            velocity[joint] = direction * speed
            self._clock += duration / count
            yield self._sample(pose, velocity, np.zeros(self.joint_count),
                               "traverse")

    def track(self, trajectory: excitation.FourierTrajectory, rate_hz: float):
        spacing = max(self.window_span_s, 1.0 / max(1.0, float(rate_hz)))
        steps = max(2, int(trajectory.duration_s / spacing))
        for index in range(steps + 1):
            moment = trajectory.duration_s * index / steps
            pose, velocity, acceleration = trajectory.sample(moment)
            self._clock += trajectory.duration_s / steps
            yield self._sample(pose, velocity, acceleration, "track")

    # -- effort model ----------------------------------------------------

    def effort(self, pose_deg, velocity_deg_s, acceleration_deg_s2) -> np.ndarray:
        """What a driver would report, in the profile's effort unit."""
        q = np.radians(np.asarray(pose_deg, dtype=float))
        v = np.radians(np.asarray(velocity_deg_s, dtype=float))
        a = np.radians(np.asarray(acceleration_deg_s2, dtype=float))
        torque = pin.rnea(self.model, self.data, q, v, a)
        effort = np.asarray(torque, dtype=float) / self.effort_per_torque
        speed = np.asarray(velocity_deg_s, dtype=float)
        width = self.coulomb_transition_deg_s
        reversal = np.tanh(speed / width) if width > 0.0 else np.sign(speed)
        effort = effort + self.coulomb * reversal
        effort = effort + self.viscous_per_deg_s * speed
        return effort + self.offset

    def _sample(self, pose_deg, velocity_deg_s, acceleration_deg_s2,
                source: str) -> dict:
        effort = self.effort(pose_deg, velocity_deg_s, acceleration_deg_s2)
        if self.noise > 0.0:
            effort = effort + self._rng.normal(0.0, self.noise, self.joint_count)
        count = self.joint_count
        return {
            "time_s": self._clock,
            "source": source,
            "position_deg": list(np.asarray(pose_deg, dtype=float)),
            "speed_deg_s": list(np.asarray(velocity_deg_s, dtype=float)),
            "acceleration_deg_s2": list(
                np.asarray(acceleration_deg_s2, dtype=float)),
            "current_a": list(effort),
            "temperature_c": [self.temperature_c] * count,
            "voltage_v": [24.0] * count,
            "enabled": [True] * count,
            "fault_code": [0] * count,
        }
