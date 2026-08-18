"""Experiment design: pick the motions that make parameters observable.

Random poses identify a model that fits the data and generalises badly. The
design criterion used throughout the identification literature is the condition
number of the stacked regressor, so candidate poses and trajectories are scored
by how much they improve it, subject to joint limits and self-collision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import identification as ident


@dataclass
class DesignLimits:
    """Bounds every planned motion must respect."""

    lower_deg: np.ndarray
    upper_deg: np.ndarray
    margin_deg: float = 5.0
    maximum_speed_deg_s: float = 10.0
    maximum_acceleration_deg_s2: float = 20.0

    def usable(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower_deg + self.margin_deg, self.upper_deg - self.margin_deg

    def clip(self, pose_deg: np.ndarray) -> np.ndarray:
        low, high = self.usable()
        return np.clip(pose_deg, low, high)


@dataclass
class StaticPlan:
    """Poses for gravity identification, plus the conditioning they achieve."""

    poses_deg: list[list[float]] = field(default_factory=list)
    condition_number: float = float("inf")
    candidates_screened: int = 0
    rejected_by_collision: int = 0
    rejected_by_path: int = 0
    travel_deg: float = 0.0

    def as_dict(self) -> dict:
        return {
            "poses_deg": [[round(v, 3) for v in pose] for pose in self.poses_deg],
            "condition_number": float(self.condition_number),
            "candidates_screened": self.candidates_screened,
            "rejected_by_collision": self.rejected_by_collision,
            "rejected_by_path": self.rejected_by_path,
            "travel_deg": round(float(self.travel_deg), 1),
        }


def _static_rows(arm: ident.ArmModel, pose_deg) -> np.ndarray:
    return arm.static_regressor(pose_deg)


def design_static_poses(
    arm: ident.ArmModel, limits: DesignLimits, count: int,
    candidates: int = 400, seed: int = 0, collision_free=None,
    start_deg=None,
) -> StaticPlan:
    """Greedily pick poses so the stacked gravity regressor is well conditioned."""
    rng = np.random.default_rng(seed)
    low, high = limits.usable()
    plan = StaticPlan()
    pool: list[np.ndarray] = []
    while len(pool) < candidates:
        pose = rng.uniform(low, high)
        plan.candidates_screened += 1
        if collision_free is not None and not collision_free(pose):
            plan.rejected_by_collision += 1
            if plan.candidates_screened > candidates * 20:
                break
            continue
        pool.append(pose)
        if plan.candidates_screened > candidates * 20:
            break
    if not pool:
        return plan

    rows = [_static_rows(arm, pose) for pose in pool]
    chosen: list[int] = [0]
    while len(chosen) < min(count, len(pool)):
        best_index, best_score = None, np.inf
        current = [rows[index] for index in chosen]
        for index in range(len(pool)):
            if index in chosen:
                continue
            score = ident.stacked_condition_number(current + [rows[index]])
            if score < best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        chosen.append(best_index)

    ordered, plan.rejected_by_path = _short_tour(
        [pool[index] for index in chosen], start_deg, collision_free)
    if not ordered:
        return plan
    plan.poses_deg = [pose.tolist() for pose in ordered]
    plan.condition_number = ident.stacked_condition_number(
        [_static_rows(arm, pose) for pose in ordered])
    plan.travel_deg = _tour_length(ordered, start_deg)
    return plan


def _path_free(start, end, collision_free, steps: int = 32) -> bool:
    """Joints interpolate between poses, so the swept path needs screening too.

    Two poses can each be clear while the straight line between them sweeps the
    arm through an obstacle.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    for fraction in np.linspace(0.0, 1.0, steps + 1)[1:-1]:
        if not collision_free(start + fraction * (end - start)):
            return False
    return True



def _tour_length(poses, start_deg=None) -> float:
    """Total worst-joint travel, which is what sets the time on hardware."""
    if not poses:
        return 0.0
    previous = np.zeros_like(poses[0]) if start_deg is None else np.asarray(
        start_deg, dtype=float)
    total = 0.0
    for pose in poses:
        total += float(np.max(np.abs(pose - previous)))
        previous = pose
    return total


def _short_tour(poses, start_deg=None, collision_free=None):
    """Visit the same poses in a nearer-neighbour order, by a clear path.

    Conditioning depends on the set of poses, not the order they are visited in,
    so shortening the tour is free. It matters because travel is most of the
    wall-clock time, and because a long unnecessary swing is alarming to whoever
    is standing next to the arm.

    A pose unreachable from here may be reachable later, so blocked transits
    reorder the tour rather than discard the pose. Only poses no ordering can
    reach are dropped.
    """
    if not poses:
        return [], 0
    remaining = list(poses)
    current = np.zeros_like(poses[0]) if start_deg is None else np.asarray(
        start_deg, dtype=float)
    ordered = []
    while remaining:
        order = sorted(range(len(remaining)),
                       key=lambda i: float(np.max(np.abs(remaining[i] - current))))
        chosen = next(
            (i for i in order
             if collision_free is None
             or _path_free(current, remaining[i], collision_free)),
            None)
        if chosen is None:
            return ordered, len(remaining)
        current = remaining.pop(chosen)
        ordered.append(current)
    return ordered, 0


@dataclass
class FrictionSweep:
    """One constant-velocity pass used to separate Coulomb from viscous terms."""

    joint: int
    start_deg: list[float]
    amplitude_deg: float
    speeds_deg_s: list[float]

    def as_dict(self) -> dict:
        return {
            "joint": self.joint,
            "start_deg": [round(v, 3) for v in self.start_deg],
            "amplitude_deg": self.amplitude_deg,
            "speeds_deg_s": list(self.speeds_deg_s),
        }


def design_friction_sweeps(
    arm: ident.ArmModel, limits: DesignLimits, amplitude_deg: float = 20.0,
    speeds_deg_s: tuple[float, ...] = (2.0, 5.0, 8.0),
    home_deg: np.ndarray | None = None,
) -> list[FrictionSweep]:
    """Sweep each joint about a pose where its gravity term barely changes."""
    low, high = limits.usable()
    base = np.zeros(arm.joint_count) if home_deg is None else np.asarray(
        home_deg, dtype=float)
    base = np.clip(base, low, high)
    speeds = [speed for speed in speeds_deg_s
              if 0.0 < speed <= limits.maximum_speed_deg_s]
    sweeps = []
    for joint in range(arm.joint_count):
        room = min(amplitude_deg, (high[joint] - low[joint]) / 2.0 - 1.0)
        if room <= 1.0:
            continue
        start = base.copy()
        start[joint] = np.clip(base[joint] - room / 2.0, low[joint], high[joint])
        sweeps.append(FrictionSweep(joint, start.tolist(), float(room), speeds))
    return sweeps


@dataclass
class FourierTrajectory:
    """Band-limited periodic excitation, the standard inertia-identification input."""

    centre_deg: list[float]
    amplitudes_deg: list[list[float]]
    phases: list[list[float]]
    base_frequency_hz: float
    duration_s: float

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centre = np.asarray(self.centre_deg, dtype=float)
        position = centre.copy()
        velocity = np.zeros_like(centre)
        acceleration = np.zeros_like(centre)
        omega = 2.0 * np.pi * self.base_frequency_hz
        for harmonic, (amplitude, phase) in enumerate(
                zip(self.amplitudes_deg, self.phases), start=1):
            rate = omega * harmonic
            amplitude = np.asarray(amplitude, dtype=float)
            phase = np.asarray(phase, dtype=float)
            position += amplitude * np.sin(rate * time_s + phase)
            velocity += amplitude * rate * np.cos(rate * time_s + phase)
            acceleration -= amplitude * rate * rate * np.sin(rate * time_s + phase)
        return position, velocity, acceleration

    def as_dict(self) -> dict:
        return {
            "centre_deg": [round(v, 3) for v in self.centre_deg],
            "amplitudes_deg": [[round(v, 4) for v in row]
                               for row in self.amplitudes_deg],
            "phases": [[round(v, 4) for v in row] for row in self.phases],
            "base_frequency_hz": self.base_frequency_hz,
            "duration_s": self.duration_s,
        }


def _trajectory_violates(
    trajectory: FourierTrajectory, limits: DesignLimits, samples: int = 120,
) -> bool:
    low, high = limits.usable()
    for step in range(samples):
        time_s = trajectory.duration_s * step / samples
        position, velocity, acceleration = trajectory.sample(time_s)
        if np.any(position < low) or np.any(position > high):
            return True
        if np.max(np.abs(velocity)) > limits.maximum_speed_deg_s:
            return True
        if np.max(np.abs(acceleration)) > limits.maximum_acceleration_deg_s2:
            return True
    return False


def design_fourier_trajectory(
    arm: ident.ArmModel, limits: DesignLimits, centre_deg: np.ndarray | None = None,
    harmonics: int = 3, base_frequency_hz: float = 0.1, duration_s: float = 20.0,
    attempts: int = 60, seed: int = 0, collision_free=None,
) -> FourierTrajectory | None:
    """Search random Fourier coefficients for the best-conditioned feasible one."""
    rng = np.random.default_rng(seed)
    low, high = limits.usable()
    centre = np.zeros(arm.joint_count) if centre_deg is None else np.clip(
        np.asarray(centre_deg, dtype=float), low, high)
    omega = 2.0 * np.pi * base_frequency_hz
    # Keep every harmonic under the speed and acceleration ceilings by construction.
    speed_budget = limits.maximum_speed_deg_s / max(1, harmonics)
    accel_budget = limits.maximum_acceleration_deg_s2 / max(1, harmonics)

    best, best_score = None, np.inf
    for _ in range(attempts):
        amplitudes, phases = [], []
        for harmonic in range(1, harmonics + 1):
            rate = omega * harmonic
            ceiling = min(speed_budget / rate, accel_budget / (rate * rate))
            amplitudes.append(
                rng.uniform(0.2 * ceiling, ceiling, arm.joint_count).tolist())
            phases.append(rng.uniform(0.0, 2.0 * np.pi, arm.joint_count).tolist())
        trajectory = FourierTrajectory(
            centre.tolist(), amplitudes, phases, base_frequency_hz, duration_s)
        if _trajectory_violates(trajectory, limits):
            continue
        rows, blocked = [], False
        for step in range(40):
            time_s = duration_s * step / 40
            position, velocity, acceleration = trajectory.sample(time_s)
            if collision_free is not None and not collision_free(position):
                blocked = True
                break
            rows.append(arm.torque_regressor(position, velocity, acceleration))
        if blocked or not rows:
            continue
        score = ident.stacked_condition_number(rows)
        if score < best_score:
            best, best_score = trajectory, score
    return best
