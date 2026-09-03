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

# Joint travel between consecutive collision checks along a path. A fixed
# sample count tunnels: measured on this workspace a 300 deg swing checked at
# 32 points left 9.4 deg between checks, which at arm's length is a 16 cm gap,
# and a transit that passed took the wrist to 0.0 mm from the other arm.
PATH_RESOLUTION_DEG = 2.0
# Ceiling so one enormous swing cannot make a design take minutes.
MAXIMUM_PATH_STEPS = 400


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
    # Where the tour leaves the arm standing, so the next one can start there.
    final_deg: list[float] = field(default_factory=list)
    condition_number: float = float("inf")
    candidates_screened: int = 0
    rejected_by_collision: int = 0
    rejected_by_crossing: int = 0
    rejected_by_path: int = 0
    travel_deg: float = 0.0

    def as_dict(self) -> dict:
        return {
            "poses_deg": [[round(v, 3) for v in pose] for pose in self.poses_deg],
            "final_deg": [round(v, 3) for v in self.final_deg],
            "condition_number": float(self.condition_number),
            "candidates_screened": self.candidates_screened,
            "rejected_by_collision": self.rejected_by_collision,
            "rejected_by_crossing": self.rejected_by_crossing,
            "rejected_by_path": self.rejected_by_path,
            "travel_deg": round(float(self.travel_deg), 1),
        }


def _static_rows(arm: ident.ArmModel, pose_deg) -> np.ndarray:
    return arm.static_regressor(pose_deg)


def design_static_poses(
    arm: ident.ArmModel, limits: DesignLimits, count: int,
    candidates: int = 400, seed: int = 0, collision_free=None,
    start_deg=None, crossing_deg: float = 0.0,
) -> StaticPlan:
    """Greedily pick poses so the stacked gravity regressor is well conditioned.

    ``crossing_deg`` is the half-width the gravity probe will drive on every
    joint at this pose. When given, a candidate has to clear that whole sweep,
    and the tour is planned between the sweep's own start rather than the
    centre -- which is where the arm actually goes.
    """
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
        if not _crossing_free(pose, crossing_deg, limits, collision_free):
            plan.rejected_by_crossing += 1
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

    # The arm travels to where the crossing starts, not to the centre, and it
    # is left there when the crossing returns; the tour has to screen that.
    approach = None
    if crossing_deg > 0.0:
        bounds = limits.usable()
        approach = (lambda pose: np.clip(np.asarray(pose, dtype=float)
                                         - crossing_deg, *bounds))
    ordered, plan.rejected_by_path = _short_tour(
        [pool[index] for index in chosen], start_deg, collision_free, approach)
    if not ordered:
        return plan
    plan.poses_deg = [pose.tolist() for pose in ordered]
    # Where the arm is left standing, which is the crossing's start and not
    # the pose's centre. The next tour's first transit is screened from here.
    reach = approach or (lambda pose: pose)
    plan.final_deg = [float(value) for value in reach(ordered[-1])]
    plan.condition_number = ident.stacked_condition_number(
        [_static_rows(arm, pose) for pose in ordered])
    plan.travel_deg = _tour_length(ordered, start_deg)
    return plan


def _path_free(start, end, collision_free, steps: int = 0) -> bool:
    """Joints interpolate between poses, so the swept path needs screening too.

    Two poses can each be clear while the straight line between them sweeps the
    arm through an obstacle. The number of checks follows the distance -- a
    caller's ``steps`` is a floor, not the answer -- because the gap between
    checks is what decides whether a thin obstacle can be stepped over.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    travel = float(np.max(np.abs(end - start)))
    needed = int(np.ceil(travel / PATH_RESOLUTION_DEG))
    count = min(max(int(steps), needed, 8), MAXIMUM_PATH_STEPS)
    for fraction in np.linspace(0.0, 1.0, count + 1)[1:-1]:
        if not collision_free(start + fraction * (end - start)):
            return False
    return True


def _crossing_free(pose, crossing_deg: float, limits: "DesignLimits",
                   collision_free) -> bool:
    """The whole +/-delta the gravity probe drives, not just its centre.

    The probe crosses the pose in both directions on every joint at once, so a
    pose that clears can still put the arm somewhere ten degrees away that does
    not. Measured here: a screened pose whose crossing reached 0.0 mm from the
    other arm.
    """
    if collision_free is None or crossing_deg <= 0.0:
        return True
    low, high = limits.usable()
    pose = np.asarray(pose, dtype=float)
    start = np.clip(pose - crossing_deg, low, high)
    end = np.clip(pose + crossing_deg, low, high)
    if not collision_free(start) or not collision_free(end):
        return False
    return _path_free(start, end, collision_free)


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


def _short_tour(poses, start_deg=None, collision_free=None, approach=None):
    """Visit the same poses in a nearer-neighbour order, by a clear path.

    Conditioning depends on the set of poses, not the order they are visited in,
    so shortening the tour is free. It matters because travel is most of the
    wall-clock time, and because a long unnecessary swing is alarming to whoever
    is standing next to the arm.

    ``approach`` maps a pose to the configuration the arm actually drives to
    and is left at. Screening the centres instead would check a path the arm
    never flies.

    A pose unreachable from here may be reachable later, so blocked transits
    reorder the tour rather than discard the pose. Only poses no ordering can
    reach are dropped.
    """
    if not poses:
        return [], 0
    reach = approach or (lambda pose: pose)
    remaining = list(poses)
    current = np.zeros_like(poses[0]) if start_deg is None else np.asarray(
        start_deg, dtype=float)
    ordered = []
    while remaining:
        order = sorted(range(len(remaining)),
                       key=lambda i: float(np.max(np.abs(reach(remaining[i])
                                                         - current))))
        chosen = next(
            (i for i in order
             if collision_free is None
             or _path_free(current, reach(remaining[i]), collision_free)),
            None)
        if chosen is None:
            return ordered, len(remaining)
        picked = remaining.pop(chosen)
        current = reach(picked)
        ordered.append(picked)
    return ordered, 0


@dataclass
class FrictionSweep:
    """One constant-velocity pass used to separate Coulomb from viscous terms."""

    joint: int
    start_deg: list[float]
    amplitude_deg: float
    speeds_deg_s: list[float]
    # What this joint carries at this posture: axial torque Nm, radial force N,
    # thrust force N, tilting moment Nm. Recorded because it is the variable the
    # posture was chosen to spread, and without it the run cannot say what load
    # its friction was measured under. Four terms rather than one because no
    # single term varies on every joint: joint one's radial force is fixed by
    # its horizontal axis, joint seven's axial torque is zero everywhere.
    load: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def gravity_nm(self) -> float:
        """The part the motor has to push against."""
        return self.load[0]

    def as_dict(self) -> dict:
        return {
            "joint": self.joint,
            "start_deg": [round(v, 3) for v in self.start_deg],
            "amplitude_deg": self.amplitude_deg,
            "speeds_deg_s": list(self.speeds_deg_s),
            "gravity_nm": round(self.gravity_nm, 4),
            "load": [round(term, 4) for term in self.load],
        }


def _sweep_free(base, joint: int, room: float, collision_free,
                steps: int = 24) -> bool:
    """The whole pass, not just where it starts.

    A posture that clears at the centre can still put the arm through the bench
    thirty degrees later, and the sweep spends its whole length there.
    """
    if collision_free is None:
        return True
    start = np.asarray(base, dtype=float).copy()
    end = start.copy()
    start[joint] -= room / 2.0
    end[joint] += room / 2.0
    if not collision_free(start) or not collision_free(end):
        return False
    return _path_free(start, end, collision_free, steps)


def design_friction_sweeps(
    arm: ident.ArmModel, limits: DesignLimits, amplitude_deg: float = 20.0,
    speeds_deg_s: tuple[float, ...] = (2.0, 5.0, 8.0),
    home_deg: np.ndarray | None = None, postures: int = 1,
    collision_free=None, candidates: int = 160, seed: int = 0,
) -> list[FrictionSweep]:
    """Sweep each joint about poses that span the load it has to carry.

    One posture measures a joint under one gravity load, and which load that is
    depends on where the other joints happen to be. On this arm joint one
    carries 0.17 Nm at home and up to 4.7 Nm with the arm extended, so friction
    fitted at home alone describes a small corner of its working range, while
    joint two barely varies and one posture would have served it.
    """
    low, high = limits.usable()
    base = np.zeros(arm.joint_count) if home_deg is None else np.asarray(
        home_deg, dtype=float)
    base = np.clip(base, low, high)
    speeds = [speed for speed in speeds_deg_s
              if 0.0 < speed <= limits.maximum_speed_deg_s]
    rng = np.random.default_rng(seed)
    sweeps = []
    for joint in range(arm.joint_count):
        room = min(amplitude_deg, (high[joint] - low[joint]) / 2.0 - 1.0)
        if room <= 1.0:
            continue
        centre = float(np.clip(base[joint], low[joint] + room / 2.0,
                               high[joint] - room / 2.0))
        for posture in _sweep_postures(arm, joint, base, centre, room, low,
                                       high, postures, collision_free,
                                       candidates, rng):
            start = posture.copy()
            start[joint] = posture[joint] - room / 2.0
            sweeps.append(FrictionSweep(
                joint, start.tolist(), float(room), speeds,
                load=tuple(arm.joint_loads(posture)[joint].tolist())))
    return sweeps


def _sweep_postures(arm, joint, base, centre, room, low, high, wanted,
                    collision_free, candidates, rng):
    """Collision-free postures for one joint, spread over the load it carries.

    Spread across all four load terms rather than the gravity torque alone. No
    single term varies on every joint -- joint one's radial force is fixed by
    its horizontal axis, joint seven carries no axial torque in any pose -- so
    choosing on one term picks arbitrarily on the joints where that term is
    constant, and the run then cannot say whether their friction moved with
    posture or not.
    """
    found = []
    home = base.copy()
    home[joint] = centre
    if _sweep_free(home, joint, room, collision_free):
        found.append(home)
    # Without a screen there is nothing that could make a drawn posture safe,
    # and a posture nobody checked must not be swept. Home is the exception:
    # it is the pose the arm is already sitting in.
    if wanted > 1 and collision_free is not None:
        for _ in range(candidates):
            pose = rng.uniform(low, high)
            # The swept joint's own angle is part of the load on it: joint one's
            # axis is horizontal, so pinning its centre at home caps it at a
            # third of the torque it carries across the range.
            pose[joint] = float(np.clip(pose[joint], low[joint] + room / 2.0,
                                        high[joint] - room / 2.0))
            if _sweep_free(pose, joint, room, collision_free):
                found.append(pose)
    if not found or wanted < 1:
        return found[:max(wanted, 0)]
    if len(found) <= wanted:
        return found
    return _spread_by_load(arm, joint, found, wanted)


def _spread_by_load(arm, joint, poses, wanted):
    """Pick the postures furthest apart in load, keeping home as the reference.

    Each load term is scaled by its own range first, so a joint whose radial
    force swings 22 N does not drown out the 4.7 Nm of torque beside it, and a
    joint where one term never moves simply contributes nothing on that axis
    instead of dominating.
    """
    loads = np.array([arm.joint_loads(pose)[joint] for pose in poses])
    return [poses[index] for index in _choose_by_load(loads, wanted)]


def _choose_by_load(loads, wanted):
    """Indices spanning the load range the joint actually carries.

    Load-dependent friction is linear in the axial torque, so where that term
    varies the rungs are spread evenly across it: the heaviest posture is the
    one measurement that cannot be replaced, and a middle rung is what tells a
    straight line from a curve. Farthest-point spread over all four terms does
    neither, because a posture extreme in a lesser term wins the distance while
    the torque the motor works against goes unswept.

    Where gravity cannot load the joint about its own axis -- the last joint
    carries no axial torque in any pose -- the terms that do move decide.
    """
    loads = np.asarray(loads, dtype=float)
    wanted = max(int(wanted), 0)
    if not wanted or not len(loads):
        return []
    axial = np.abs(loads[:, 0])
    if axial.max() - axial.min() > 1e-9:
        chosen: list[int] = []
        for target in np.linspace(axial.min(), axial.max(), wanted):
            order = np.argsort(np.abs(axial - target))
            for index in order:
                if int(index) not in chosen:
                    chosen.append(int(index))
                    break
        return chosen

    span = loads.max(axis=0) - loads.min(axis=0)
    scaled = loads / np.where(span > 1e-9, span, 1.0)
    # Home is index zero when it was admissible, and it is the posture every
    # earlier run used, so keeping it makes this run comparable with those.
    chosen = [0]
    gap = np.linalg.norm(scaled - scaled[0], axis=1)
    while len(chosen) < wanted:
        pick = int(np.argmax(gap))
        if gap[pick] <= 0.0:
            break
        chosen.append(pick)
        gap = np.minimum(gap, np.linalg.norm(scaled - scaled[pick], axis=1))
    return chosen


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


@dataclass
class RampedFourierTrajectory:
    """The same Fourier path with zero-velocity, zero-acceleration endpoints."""

    trajectory: FourierTrajectory
    ramp_s: float

    @property
    def duration_s(self) -> float:
        return float(self.trajectory.duration_s + self.ramp_s)

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wall_time = float(np.clip(time_s, 0.0, self.duration_s))
        core_duration = float(self.trajectory.duration_s)
        if self.ramp_s <= 0.0:
            return self.trajectory.sample(min(wall_time, core_duration))
        if wall_time <= 0.0:
            position, _velocity, _acceleration = self.trajectory.sample(0.0)
            return position, np.zeros_like(position), np.zeros_like(position)
        if wall_time >= self.duration_s:
            position, _velocity, _acceleration = self.trajectory.sample(
                core_duration)
            return position, np.zeros_like(position), np.zeros_like(position)

        ramp = float(self.ramp_s)
        if wall_time < ramp:
            share = wall_time / ramp
            phase = ramp * (share ** 3 - 0.5 * share ** 4)
            phase_rate = 3.0 * share ** 2 - 2.0 * share ** 3
            phase_acceleration = (6.0 * share - 6.0 * share ** 2) / ramp
        elif wall_time <= core_duration:
            phase = wall_time - 0.5 * ramp
            phase_rate = 1.0
            phase_acceleration = 0.0
        else:
            share = (wall_time - core_duration) / ramp
            phase = (core_duration - 0.5 * ramp
                     + ramp * (share - share ** 3 + 0.5 * share ** 4))
            phase_rate = 1.0 - 3.0 * share ** 2 + 2.0 * share ** 3
            phase_acceleration = (-6.0 * share + 6.0 * share ** 2) / ramp

        position, velocity, acceleration = self.trajectory.sample(phase)
        wall_velocity = velocity * phase_rate
        wall_acceleration = (
            acceleration * phase_rate ** 2 + velocity * phase_acceleration)
        return position, wall_velocity, wall_acceleration

    def as_dict(self) -> dict:
        payload = self.trajectory.as_dict()
        payload["ramp_s"] = float(self.ramp_s)
        payload["execution_duration_s"] = self.duration_s
        return payload


def ramp_fourier_trajectory(
    trajectory: FourierTrajectory, ramp_s: float,
) -> FourierTrajectory | RampedFourierTrajectory:
    """Return a time-scaled path that enters and leaves the Fourier motion at rest."""
    ramp = float(ramp_s)
    if not np.isfinite(ramp) or ramp < 0.0:
        raise ValueError("ramp_s must be finite and non-negative")
    if ramp == 0.0:
        return trajectory
    return RampedFourierTrajectory(
        trajectory, min(ramp, 0.25 * float(trajectory.duration_s)))


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
    conditioning_rows: list[np.ndarray] | None = None,
    randomize_centre: bool = False, start_deg=None,
    minimum_amplitude_fraction: float = 0.2,
    joint_amplitude_scale=None,
) -> FourierTrajectory | None:
    """Search random Fourier coefficients for the best-conditioned feasible one.

    ``conditioning_rows`` makes a sequence informative as a whole instead of
    selecting several individually good but redundant trajectories.  A
    randomized centre explores different gravity configurations, while
    ``start_deg`` screens the straight transit that ``Plant.track`` performs
    before following the periodic motion.
    """
    minimum_amplitude_fraction = float(minimum_amplitude_fraction)
    if (not np.isfinite(minimum_amplitude_fraction)
            or not 0.0 <= minimum_amplitude_fraction <= 1.0):
        raise ValueError("minimum_amplitude_fraction must be between zero and one")
    amplitude_scale = np.ones(arm.joint_count) if joint_amplitude_scale is None else np.asarray(
        joint_amplitude_scale, dtype=float)
    if (amplitude_scale.shape != (arm.joint_count,)
            or not np.isfinite(amplitude_scale).all()
            or np.any(amplitude_scale < 0.0)
            or np.any(amplitude_scale > 1.0)):
        raise ValueError("joint_amplitude_scale must have one 0..1 value per joint")
    rng = np.random.default_rng(seed)
    low, high = limits.usable()
    centre = np.zeros(arm.joint_count) if centre_deg is None else np.clip(
        np.asarray(centre_deg, dtype=float), low, high)
    prior_rows = list(conditioning_rows or [])
    omega = 2.0 * np.pi * base_frequency_hz
    # Keep every harmonic under the speed and acceleration ceilings by construction.
    speed_budget = limits.maximum_speed_deg_s / max(1, harmonics)
    accel_budget = limits.maximum_acceleration_deg_s2 / max(1, harmonics)

    best, best_score = None, np.inf
    for _ in range(attempts):
        amplitudes, phases = [], []
        for harmonic in range(1, harmonics + 1):
            rate = omega * harmonic
            ceiling = (min(speed_budget / rate,
                           accel_budget / (rate * rate)) * amplitude_scale)
            amplitudes.append(
                rng.uniform(minimum_amplitude_fraction * ceiling, ceiling,
                            arm.joint_count).tolist())
            phases.append(rng.uniform(0.0, 2.0 * np.pi, arm.joint_count).tolist())
        candidate_centre = centre
        if randomize_centre and centre_deg is None:
            excursion = np.sum(np.abs(np.asarray(amplitudes, dtype=float)), axis=0)
            centre_low, centre_high = low + excursion, high - excursion
            if np.any(centre_low >= centre_high):
                continue
            candidate_centre = rng.uniform(centre_low, centre_high)
        trajectory = FourierTrajectory(
            candidate_centre.tolist(), amplitudes, phases,
            base_frequency_hz, duration_s)
        if _trajectory_violates(trajectory, limits):
            continue
        if (collision_free is not None and start_deg is not None
                and not _path_free(start_deg, trajectory.sample(0.0)[0],
                                   collision_free)):
            continue
        blocked = False
        if collision_free is not None:
            for step in range(max(120, 30 * harmonics)):
                time_s = duration_s * step / max(120, 30 * harmonics)
                position, _velocity, _acceleration = trajectory.sample(time_s)
                if not collision_free(position):
                    blocked = True
                    break
        if blocked:
            continue
        rows = []
        for step in range(40):
            time_s = duration_s * step / 40
            position, velocity, acceleration = trajectory.sample(time_s)
            rows.append(arm.torque_regressor(position, velocity, acceleration))
        if not rows:
            continue
        score = ident.stacked_condition_number(prior_rows + rows)
        if score < best_score:
            best, best_score = trajectory, score
    return best
