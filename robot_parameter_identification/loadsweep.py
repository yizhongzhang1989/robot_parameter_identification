"""Sweep every joint at a series of gravity loads, and record what it takes.

Friction on this arm rises with the load the joint carries. On joint one it
rose 0.12 A for every newton metre held, which at the heaviest pose the arm can
reach is half again the Coulomb term at rest. Measuring that for one joint took
a script with the answer written into it -- joint two sets the load, ninety
degrees is where it holds still. Neither statement is true of joint five, so
for seven joints the levels have to be designed rather than named, because what
a joint can be made to carry depends on where every other joint is standing.

Three things decide whether a posture is usable, and all three are settled here
before the arm is asked to move.

**The load has to hold still while the sample is taken.** Sweeping a joint
moves the joint, and moving it changes the gravity torque on it. Joint one at
zero degrees runs from 2.9 Nm down to 0.2 and back up inside a single seventy
degree pass, so "one sweep at one load" is not something that exists there. It
exists at ninety, where the sine is stationary. Rather than solve for such a
point per joint, the drift is measured numerically -- and measured across the
window that is actually fitted, not across the pass, whose ramps are driven and
then thrown away. Judging the pass instead is what once held this experiment to
24 deg/s for the sake of an excursion nothing ever sampled.

**The reachable range is a property of the joint, not a setting.** Gravity can
load joint one over 4.9 Nm and joint seven over 0.001, because joint seven's
axis lies within a degree of gravity in every pose the arm can hold. Asking for
ten levels there returns ten copies of one measurement and a load slope fitted
to rounding error. The number of levels therefore follows from the span the
search actually finds, divided by the smallest gap the measurement can resolve.
A joint that cannot be loaded is swept once and said to be unloadable.

**Gravity direction, not merely posture.** The torque on a joint is the
component of the weight it carries about *its own axis*, and where that axis
points is set by the joints upstream of it. Searching only downstream would
hold the axis still and explore a fraction of the range. The search moves every
joint, and the angle between axis and gravity is recorded with each level, so
an analysis can tell a heavily laden pose from a merely well-aligned one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import time

import numpy as np
import pinocchio as pin

from . import campaign as campaign_module
from .interfaces import MotionFailed

# Bumped when a field changes meaning. New fields do not need it: readers take
# what they know and ignore the rest, which is the point of a record per line.
SCHEMA = 1

RECORDS_NAME = "records.jsonl"
MANIFEST_NAME = "manifest.json"


@dataclass
class SweepPlan:
    """What to measure. Every field here is a choice, not a constant."""

    maximum_levels: int = 10
    speeds: int = 18
    slowest_deg_s: float = 0.5
    fastest_deg_s: float = 60.0
    repeats: int = 3
    arc_ceiling_deg: float = 70.0
    # Transits between levels are large moves across the workspace, and are not
    # measurements. They are driven well below the top sweep speed because a
    # pass being fast is the point of a pass, whereas a transit being fast is
    # only a way to arrive at the next posture harder.
    transit_speed_deg_s: float = 20.0
    # How far the load may move across the fitted window, as a share of the gap
    # between neighbouring levels. Judged against the gap rather than against
    # each level's own size because the absolute swing is nearly the same at
    # every level, which makes a percentage test vacuous at the top of the
    # range and impossible at the bottom.
    drift_share: float = 0.35
    # The smallest load gap worth separating. Joint one's friction rose 0.12 A
    # per Nm against a residual of 0.023 A, so two levels closer together than
    # this are two readings of the same thing wearing different labels.
    minimum_level_gap_nm: float = 0.20
    search_samples: int = 20000
    refine_steps: int = 300
    # Spent per level looking for a steadier posture at the same load.
    settle_steps: int = 250
    seed: int = 0
    # Empty means every joint.
    joints: tuple[int, ...] = ()
    pass_attempts: int = 3
    pass_retry_s: float = 2.0
    # Consecutive failures on one joint before it is given up and the sweep
    # moves on. A joint that has stopped answering will not start answering.
    joint_failure_budget: int = 30

    def speed_ladder(self) -> list[float]:
        """Log spaced: the Stribeck dip lives in the bottom decade."""
        return [round(float(v), 3) for v in np.geomspace(
            self.slowest_deg_s, self.fastest_deg_s, max(2, self.speeds))]


@dataclass
class Level:
    index: int
    wanted_nm: float
    load_nm: float
    load_drift_nm: float
    pose_deg: list[float]
    axis_gravity_deg: float
    terms: dict


@dataclass
class JointDesign:
    joint: int
    name: str
    levels: list[Level] = field(default_factory=list)
    passes: list[dict] = field(default_factory=list)
    refused: list[dict] = field(default_factory=list)
    reachable_nm: tuple[float, float] = (0.0, 0.0)
    span_nm: float = 0.0
    # The widest pass every level has room for. Below the widest arc in the
    # ladder this says the joint ran out of travel, not that the sweep is
    # misconfigured.
    arc_room_deg: float = 0.0
    loadable: bool = True
    note: str = ""

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["reachable_nm"] = list(self.reachable_nm)
        return payload


# -- geometry -------------------------------------------------------------


def window_arc_deg(speed_deg_s: float, config) -> float:
    """How much joint angle the fitted sample actually spans.

    A pass is not a measurement. The plant fits one window from the middle of
    the motion and sizes it so the joint crosses at most ``window_arc_deg``
    while it is open; the ramps at either end are driven but never sampled.
    """
    speed = abs(float(speed_deg_s))
    if speed <= 0.0:
        return 0.0
    span = min(max(config.window_arc_deg / speed, config.window_span_s),
               config.window_maximum_span_s)
    return speed * span


def load_terms(arm, pose_deg, joint: int) -> np.ndarray:
    """Axial torque Nm, radial force N, thrust force N, tilting moment Nm."""
    return np.asarray(arm.joint_loads(pose_deg)[joint], dtype=float)


def load_at(arm, pose_deg, joint: int) -> float:
    """The gravity torque the motor has to hold, which is what friction scales with."""
    return abs(float(arm.joint_loads(pose_deg)[joint][0]))


def axis_gravity_deg(arm, pose_deg, joint: int) -> float:
    """Angle between the joint's own axis and gravity, in degrees.

    A joint carrying a heavy arm feels nothing if its axis points along
    gravity, and the load levels below are otherwise indistinguishable from
    poses that happen to be well aligned. Recorded so the two can be told apart.
    """
    q = np.radians(np.asarray(pose_deg, dtype=float))
    pin.forwardKinematics(arm.model, arm.data, q)
    axis = np.asarray(arm.data.joints[joint + 1].S).reshape(6)[3:]
    world = arm.data.oMi[joint + 1].rotation @ axis
    norm = float(np.linalg.norm(world))
    if norm <= 0.0:
        return float("nan")
    down = np.array([0.0, 0.0, -1.0])
    cosine = float(np.clip((world / norm) @ down, -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(cosine))))


def pose_with(pose_deg, joint: int, angle_deg: float) -> np.ndarray:
    pose = np.asarray(pose_deg, dtype=float).copy()
    pose[joint] = float(angle_deg)
    return pose


def load_across(arm, pose_deg, joint: int, arc_deg: float,
                points: int = 9) -> tuple[float, float]:
    """Mean load across an arc centred on the pose, and how far it moves."""
    centre = float(np.asarray(pose_deg, dtype=float)[joint])
    if arc_deg <= 0.0:
        value = load_at(arm, pose_deg, joint)
        return value, 0.0
    values = [load_at(arm, pose_with(pose_deg, joint, centre + offset), joint)
              for offset in np.linspace(-arc_deg / 2.0, arc_deg / 2.0, points)]
    return float(np.mean(values)), float(max(values) - min(values))


def clear_arc(scene, pose_deg, joint: int, arc_deg: float,
              steps: int = 25) -> bool:
    """The whole pass, not just its ends."""
    if scene is None:
        return True
    centre = float(np.asarray(pose_deg, dtype=float)[joint])
    return all(
        scene.collision_free(pose_with(pose_deg, joint, centre + offset))
        for offset in np.linspace(-arc_deg / 2.0, arc_deg / 2.0, steps))


def path_free(scene, start_deg, end_deg, steps: int = 32) -> bool:
    """Is the straight joint-space move between two poses clear?

    The levels are found by searching the whole workspace, so consecutive ones
    are unrelated postures that may be most of a revolution apart. The
    controller interpolates between them in joint space and the plant screens
    neither end against the middle, so without this the arm can be sent
    cleanly from one safe pose to another straight through itself.
    """
    if scene is None:
        return True
    start = np.asarray(start_deg, dtype=float)
    end = np.asarray(end_deg, dtype=float)
    return all(scene.collision_free(start + fraction * (end - start))
               for fraction in np.linspace(0.0, 1.0, max(2, steps)))


def proven_scene(scene, joint_count: int) -> None:
    """Refuse a collision screen that has never been able to say no.

    A screen with no geometry loaded returns True for everything, which is
    indistinguishable from a clear path right up to the moment the arm closes
    on itself. This experiment drives every joint far from home.
    """
    if scene is None:
        raise RuntimeError("no collision scene; refusing to plan hardware motion")
    report = scene.geometry_report()
    if not report.get("self_collision_checked"):
        raise RuntimeError(
            "the collision scene cannot see the arm "
            f"({report.get('robot_geometry_error') or 'no link shapes'}); "
            "source the workspace so package:// mesh paths resolve")
    if scene.collision_free(np.full(int(joint_count), 150.0)):
        raise RuntimeError("the collision screen passed a folded pose; "
                           "it is not actually checking anything")


# -- design ---------------------------------------------------------------


def _room(pose_deg, joint: int, arc_deg: float, lower, upper) -> bool:
    centre = float(np.asarray(pose_deg, dtype=float)[joint])
    return (centre - arc_deg / 2.0 >= lower[joint]
            and centre + arc_deg / 2.0 <= upper[joint])


def _extremise(arm, joint, start, lower, upper, sign, steps, rng, usable):
    """Push the load as far one way as the pose can be made to go.

    Random sampling finds the shape of the reachable set but rarely its
    corners, and the corners are the whole point: the span sets how many levels
    there is any sense in measuring.
    """
    best = np.asarray(start, dtype=float).copy()
    value = sign * load_at(arm, best, joint)
    scale = 0.2 * (upper - lower)
    for _ in range(max(0, steps)):
        trial = np.clip(best + rng.normal(0.0, 1.0, best.size) * scale,
                        lower, upper)
        if not usable(trial):
            scale *= 0.99
            continue
        score = sign * load_at(arm, trial, joint)
        if score > value:
            best, value = trial, score
        else:
            scale *= 0.99
    return best


def _settle(arm, joint, start, target, tolerance, window, lower, upper, steps,
            rng, room_ok, bounds=None):
    """Hold the load and quieten it: same level, steadier posture.

    The load on a joint goes as the sine of its own angle plus a phase the
    downstream joints set, so a level can be held at many postures and they are
    not equally good. The flat top of that sine is where sweeping the joint
    barely moves its own load, and which angle the flat top sits at is a free
    choice -- the downstream chain can put it where there is room to sweep
    rather than against the workspace cap. Picking the nearest pose to a target
    load takes whatever phase the random draw happened to offer; this looks for
    the quietest one at the same load.
    """
    best = np.asarray(start, dtype=float).copy()
    value = load_across(arm, best, joint, window, points=5)[1]
    scale = 0.15 * (upper - lower)
    for _ in range(max(0, steps)):
        trial = np.clip(best + rng.normal(0.0, 1.0, best.size) * scale,
                        lower, upper)
        if not room_ok(trial):
            scale *= 0.99
            continue
        mean, drift = load_across(arm, trial, joint, window, points=5)
        # Staying inside the reported reachable range matters as much as
        # staying near the target: a level quietly settled past the range the
        # search found makes that range a lie.
        if bounds is not None and not bounds[0] <= mean <= bounds[1]:
            scale *= 0.99
            continue
        if abs(mean - target) > tolerance or drift >= value:
            scale *= 0.99
            continue
        best, value = trial, drift
    return best


def design_joint(arm, scene, joint: int, plan: SweepPlan, config,
                 lower, upper, rng=None) -> JointDesign:
    """Find what loads this joint can be held at, and lay levels across them."""
    rng = rng or np.random.default_rng(plan.seed + joint)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    speeds = plan.speed_ladder()
    windows = {speed: window_arc_deg(speed, config) for speed in speeds}
    arcs = {speed: campaign_module.pass_amplitude_deg(speed, plan.arc_ceiling_deg)
            for speed in speeds}
    widest_window = max(windows.values())

    design = JointDesign(joint=joint, name=str(arm.joint_names[joint]))

    # The drift budget needs the gap between levels, and the gap needs the
    # span, which is what the search is for. Bootstrapped from the raw
    # reachable range, then the levels are laid out inside whatever survives.
    raw = np.array([load_at(arm, pose, joint) for pose in
                    rng.uniform(lower, upper, size=(2000, lower.size))])
    guess_span = float(raw.max() - raw.min())
    allowed = max(plan.drift_share * guess_span / max(plan.maximum_levels - 1, 1),
                  1e-4)

    def usable_with(pose, arc) -> bool:
        if not _room(pose, joint, arc, lower, upper):
            return False
        _, drift = load_across(arm, pose, joint, widest_window, points=5)
        return drift <= allowed

    # Room for the ladder is a feasibility condition, not a preference. The
    # load is steadiest where it is stationary in the joint's own angle, and on
    # this arm that point sits against the workspace cap: joint one's load goes
    # as sin(j1), so the flat top is at ninety degrees, which is exactly where
    # there is no room left to sweep. Choosing poses on drift alone therefore
    # buys a beautifully steady load that can only be measured at walking pace,
    # and the fast half of the ladder -- the half that carries the viscous term
    # -- is refused afterwards. The arc every level must have room for is
    # settled first, widest first, and the levels are then designed among the
    # poses that can actually be driven through the whole ladder.
    ladder = sorted(set(arcs.values()), reverse=True)
    keep: list = []
    required = ladder[-1] if ladder else 0.0
    poses = rng.uniform(lower, upper, size=(plan.search_samples, lower.size))
    for arc in ladder:
        found = [pose for pose in poses if usable_with(pose, arc)]
        if len(found) >= 50:
            keep, required = found, arc
            break
    if not keep:
        keep = [pose for pose in poses if usable_with(pose, ladder[-1])]
        required = ladder[-1] if ladder else 0.0
    design.arc_room_deg = round(float(required), 3)
    if ladder and required < ladder[0]:
        design.note = (
            f"levels are placed where the joint has room for a {required:.0f} "
            f"deg pass, not the full {ladder[0]:.0f}; the fastest rungs of the "
            "ladder need the most room and this joint runs out of travel first")

    def usable(pose) -> bool:
        return usable_with(pose, required)

    if not keep:
        design.note = ("no pose holds this joint's load still enough to "
                       "measure; it moves its own load wherever it stands")
        design.loadable = False
        return design

    values = np.array([load_at(arm, pose, joint) for pose in keep])
    lightest = _extremise(arm, joint, keep[int(np.argmin(values))], lower, upper,
                          -1.0, plan.refine_steps, rng, usable)
    heaviest = _extremise(arm, joint, keep[int(np.argmax(values))], lower, upper,
                          +1.0, plan.refine_steps, rng, usable)
    keep = [np.asarray(lightest), np.asarray(heaviest), *keep]
    values = np.array([load_at(arm, pose, joint) for pose in keep])
    low, high = float(values.min()), float(values.max())
    span = high - low
    design.reachable_nm = (round(low, 5), round(high, 5))
    design.span_nm = round(span, 5)

    # How many levels the span can carry without two of them being the same
    # measurement twice.
    count = int(min(plan.maximum_levels,
                    max(1, round(span / max(plan.minimum_level_gap_nm, 1e-9)) + 1)))
    if count < 2:
        design.loadable = False
        design.note = (
            f"gravity moves this joint's load by only {span:.4f} Nm across the "
            "whole workspace, which is below the resolution of the current "
            "measurement; swept at one posture, with no load axis")
    wanted = (np.linspace(low, high, count) if count > 1
              else np.array([float(np.median(values))]))

    allowed = max(plan.drift_share * (span / max(count - 1, 1)), 1e-4) \
        if count > 1 else allowed
    taken: list[int] = []
    gap = span / max(count - 1, 1) if count > 1 else max(span, 1e-6)
    for index, target in enumerate(wanted):
        # Among the poses that sit at this load, take the steadiest one. Any
        # candidate within a sixth of the gap is the same level as far as the
        # analysis is concerned, so the tie is worth spending on drift: a pose
        # whose load barely moves while the joint sweeps is a cleaner
        # measurement of that load than one merely closer to a round number.
        near = [k for k in range(len(keep))
                if k not in taken and abs(values[k] - target) <= gap / 6.0]
        ranked = sorted(near, key=lambda k: (
            -sum(1 for a in arcs.values()
                 if _room(keep[k], joint, a, lower, upper)),
            load_across(arm, keep[k], joint, widest_window, points=5)[1], k))
        ranked += sorted((k for k in range(len(keep)) if k not in taken
                          and k not in near),
                         key=lambda k: (abs(values[k] - target), k))
        clear: list[int] = []
        for candidate in ranked:
            if candidate in taken:
                continue
            pose = keep[candidate]
            # Proved clear over the widest arc this pose will actually be
            # driven through, which is the widest one its joint has room for.
            room = [a for a in arcs.values()
                    if _room(pose, joint, a, lower, upper)]
            if not room or not clear_arc(scene, pose, joint, max(room)):
                continue
            clear.append(candidate)
            if len(clear) >= 5:
                break
        if not clear:
            design.refused.append({"level": index + 1,
                                   "wanted_nm": round(float(target), 4),
                                   "why": "no collision free pose at this load"})
            continue
        taken.append(clear[0])
        # Several starting postures, not one. The quiet postures at a given
        # load are scattered, and a local search only ever finds the valley it
        # was dropped into.
        pose, quietest = None, float("inf")
        for candidate in clear:
            start = np.asarray(keep[candidate], dtype=float)
            settled = _settle(arm, joint, start, float(target), gap / 6.0,
                              widest_window, lower, upper, plan.settle_steps,
                              rng, lambda p: _room(p, joint, required,
                                                   lower, upper),
                              bounds=(low, high))
            for option in (settled, start):
                drift = load_across(arm, option, joint, widest_window,
                                    points=9)[1]
                if drift < quietest and clear_arc(scene, option, joint, required):
                    pose, quietest = option, drift
        if pose is None:
            pose = np.asarray(keep[clear[0]], dtype=float)
        mean, drift = load_across(arm, pose, joint, widest_window, points=21)
        terms = load_terms(arm, pose, joint)
        design.levels.append(Level(
            index=len(design.levels) + 1,
            wanted_nm=round(float(target), 4),
            load_nm=round(float(mean), 5),
            load_drift_nm=round(float(drift), 5),
            pose_deg=[round(float(v), 4) for v in pose],
            axis_gravity_deg=round(axis_gravity_deg(arm, pose, joint), 2),
            terms={"axial_nm": round(float(terms[0]), 5),
                   "radial_n": round(float(terms[1]), 4),
                   "thrust_n": round(float(terms[2]), 4),
                   "tilt_nm": round(float(terms[3]), 5)}))

    for level in design.levels:
        pose = np.asarray(level.pose_deg, dtype=float)
        for speed in speeds:
            arc, window = arcs[speed], windows[speed]
            if not _room(pose, joint, arc, lower, upper):
                design.refused.append({"level": level.index, "speed_deg_s": speed,
                                       "arc_deg": round(arc, 2),
                                       "why": "no room inside the joint limits"})
                continue
            mean, drift = load_across(arm, pose, joint, window, points=9)
            if drift > allowed:
                design.refused.append({"level": level.index, "speed_deg_s": speed,
                                       "window_deg": round(window, 2),
                                       "drift_nm": round(drift, 5),
                                       "why": "load moves across the fitted window"})
                continue
            if not clear_arc(scene, pose, joint, arc):
                design.refused.append({"level": level.index, "speed_deg_s": speed,
                                       "arc_deg": round(arc, 2),
                                       "why": "collision"})
                continue
            design.passes.append({
                "level": level.index, "speed_deg_s": speed,
                "arc_deg": round(arc, 3), "window_deg": round(window, 3),
                "load_nm": round(mean, 5), "load_drift_nm": round(drift, 5)})
    return design


def design_all(arm, scene, plan: SweepPlan, config, lower, upper,
               progress=None) -> list[JointDesign]:
    wanted = plan.joints or tuple(range(arm.joint_count))
    designs = []
    for joint in wanted:
        if progress is not None:
            progress("designing", {"joint": int(joint) + 1,
                                   "joint_name": str(arm.joint_names[joint])})
        designs.append(design_joint(arm, scene, int(joint), plan, config,
                                    lower, upper))
    return designs


def estimate_seconds(designs, plan: SweepPlan, per_pass_s: float = 4.7) -> float:
    """Measured on joint one: 1080 passes in 83 minutes."""
    passes = sum(len(d.passes) for d in designs)
    return passes * max(1, plan.repeats) * 2 * per_pass_s


# -- recording ------------------------------------------------------------


def record_key(joint: int, level: int, speed: float, direction: str,
               repeat: int) -> str:
    return f"{joint}:{level}:{float(speed):g}:{direction}:{repeat}"


class Recorder:
    """Append-only JSON lines.

    One self-describing record per line, rather than a table with a header.
    A later run can add a field without rewriting what is already on disk and
    without breaking a reader that has never heard of it, which is the whole
    reason this is not a CSV.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def already_done(self) -> set[str]:
        """Keys written by an earlier attempt, so a resumed run skips them."""
        done: set[str] = set()
        if not self.path.is_file():
            return done
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                key = payload.get("key")
                if key:
                    done.add(str(key))
        return done

    def write(self, record: dict) -> None:
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass
