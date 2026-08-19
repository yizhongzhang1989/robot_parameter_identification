"""Four-phase calibration campaign, run identically in simulation and on hardware.

The phases are ordered by what each one needs from the previous:

* **A - gravity.** Hold still at well-conditioned poses. With no motion there is
  no friction and no inertia term, so the measured current is gravity alone.
* **B - friction.** Sweep one joint at a time at several constant speeds. Speed
  is constant, so acceleration contributes nothing and what changes with speed
  is friction.
* **C - inertia.** Follow a band-limited Fourier trajectory. Only here does
  acceleration appear, and friction is already known from B, so the remaining
  signal is inertial.
* **D - validation.** Predict states the fit never saw. A model that only
  reproduces its own training data has not been identified, it has been
  memorised.

Every phase yields the same observation record, so the regression and the
report do not care whether the samples came from MuJoCo or from the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Iterator, Protocol
import time

import numpy as np

from . import excitation, identification as ident
from .model import ModelComponents
from .profile import RobotProfile

PHASE_GRAVITY = "A_gravity"
PHASE_FRICTION = "B_friction"
PHASE_INERTIA = "C_inertia"
PHASE_VALIDATION = "D_validation"
PHASES = (PHASE_GRAVITY, PHASE_FRICTION, PHASE_INERTIA, PHASE_VALIDATION)

MAXIMUM_CONDITION = 1.0e3
"""Directions weaker than this fraction of the strongest are dropped.

Measured on this arm: raising the cap to 1e4 lowered training error and raised
independent error, which is the signature of fitting noise."""

# A fit that predicts unseen states far worse than seen ones has memorised the
# data. Measured on this arm: an under-sized static phase (10 poses) produced a
# 1.0 A validation error against a 0.003 A residual, while 16 poses gave 0.004 A.
VERDICT_WARN_RATIO = 3.0
VERDICT_FAIL_RATIO = 10.0
# Below this fraction of the largest current the run ever saw, the ratio test is
# comparing two numbers that are both noise, so it is not applied.
VERDICT_NEGLIGIBLE_FRACTION = 0.005


def judge_joint(entry: dict, signal_a: float = 0.0) -> tuple[str, str]:
    """Say plainly whether one joint's fit can be used."""
    residual = float(entry.get("residual_rms_a") or 0.0)
    validation = entry.get("validation_rms_a")
    if validation is None:
        return "unknown", "no independent validation was collected"
    validation = float(validation)

    negligible = VERDICT_NEGLIGIBLE_FRACTION * max(float(signal_a), 0.0)
    if negligible > 0.0 and validation <= negligible:
        return "pass", (
            f"validation {validation:.4f} A is under {negligible:.4f} A, "
            f"which is {VERDICT_NEGLIGIBLE_FRACTION:.1%} of the largest current "
            f"seen ({signal_a:.2f} A)")

    floor = max(residual, 1e-6)
    ratio = validation / floor
    if ratio >= VERDICT_FAIL_RATIO:
        return "fail", (
            f"validation {validation:.4f} A is {ratio:.0f}x the training "
            f"residual {residual:.4f} A: the fit has not generalised")
    if ratio >= VERDICT_WARN_RATIO:
        return "warn", (
            f"validation {validation:.4f} A is {ratio:.1f}x the training "
            f"residual; treat the parameters as provisional")
    return "pass", f"validation {validation:.4f} A tracks the training residual"


class Plant(Protocol):
    """What a campaign needs from a robot, real or simulated."""

    def limits_deg(self) -> tuple[np.ndarray, np.ndarray]: ...

    def collision_free(self, pose_deg) -> bool: ...

    def hold_pose(self, pose_deg) -> dict:
        """Move there, let it settle, return one telemetry frame at rest."""

    def dwell(self, pose_deg) -> dict:
        """Resample at rest without commanding motion. Optional."""

    def traverse(self, joint: int, start_deg, distance_deg: float,
                 speed_deg_s: float) -> Iterator[dict]:
        """Move one joint at constant speed, yielding frames along the way."""

    def track(self, trajectory: excitation.FourierTrajectory,
              rate_hz: float) -> Iterator[dict]:
        """Follow a trajectory, yielding frames with the achieved state."""


class EnvelopeMonitor(Protocol):
    """Anything that can veto a telemetry frame."""

    def check(self, sample: dict, now: float) -> str | None: ...


@dataclass
class DriveMonitor:
    """Stops the run when the drives themselves say they are unfit to move.

    Only checks quantities that need no guessed threshold. A disabled drive and
    a non-zero fault word mean the same thing on every arm. The bus-voltage
    window does not: a derived profile carries a default, not a measurement, so
    it is checked only when an operator supplied one.

    Channels the robot does not publish arrive empty and are skipped, so a
    partially instrumented arm gets the checks it can support rather than none.
    """

    minimum_voltage_v: float | None = None
    maximum_voltage_v: float | None = None
    # Campaign frames are pulled synchronously, so a gap between them is
    # deliberate dwell rather than lost telemetry; kept for protocol parity.
    last_sample_at: float | None = None

    def check(self, sample: dict, now: float) -> str | None:
        for index, live in enumerate(sample.get("enabled") or []):
            if not live:
                return f"joint{index + 1} reports its drive disabled"
        for index, code in enumerate(sample.get("fault_code") or []):
            if code:
                return f"joint{index + 1} reports fault code {int(code)}"
        if self.minimum_voltage_v is None or self.maximum_voltage_v is None:
            return None
        for index, volts in enumerate(sample.get("voltage_v") or []):
            if not self.minimum_voltage_v <= volts <= self.maximum_voltage_v:
                return (f"joint{index + 1} bus at {volts:.1f} V, outside "
                        f"{self.minimum_voltage_v:.1f}-"
                        f"{self.maximum_voltage_v:.1f} V")
        return None

    def guards(self) -> tuple[str, ...]:
        """What this monitor is actually able to enforce."""
        active = ["drive-enabled check", "fault-code check"]
        if self.minimum_voltage_v is not None:
            active.append("bus-voltage window")
        return tuple(active)


@dataclass
class Observation:
    """One usable sample: a state and the current it required."""

    phase: str
    time_s: float
    position_deg: list[float]
    velocity_deg_s: list[float]
    acceleration_deg_s2: list[float]
    current_a: list[float]
    temperature_c: list[float] = field(default_factory=list)

    @classmethod
    def from_sample(cls, phase: str, time_s: float, sample: dict,
                    acceleration_deg_s2=None) -> "Observation":
        count = len(sample["position_deg"])
        acceleration = (np.zeros(count) if acceleration_deg_s2 is None
                        else np.asarray(acceleration_deg_s2, dtype=float))
        return cls(
            phase=phase, time_s=time_s,
            position_deg=[float(v) for v in sample["position_deg"]],
            velocity_deg_s=[float(v) for v in sample["speed_deg_s"]],
            acceleration_deg_s2=[float(v) for v in acceleration],
            current_a=[float(v) for v in sample["current_a"]],
            temperature_c=[float(v) for v in sample.get("temperature_c", [])],
        )


@dataclass
class CampaignPlan:
    """Everything the operator can choose before the arm moves."""

    static_poses: int = 24
    static_candidates: int = 200
    settle_samples: int = 3
    friction_amplitude_deg: float = 20.0
    friction_speeds_deg_s: tuple[float, ...] = (2.0, 5.0, 8.0)
    fourier_harmonics: int = 4
    fourier_base_frequency_hz: float = 0.08
    fourier_duration_s: float = 30.0
    fourier_attempts: int = 40
    sample_rate_hz: float = 20.0
    validation_poses: int = 12
    validation_speeds_deg_s: tuple[float, ...] = (3.5, 6.5)
    validation_trajectory_s: float = 12.0
    maximum_speed_deg_s: float = 10.0
    maximum_acceleration_deg_s2: float = 20.0
    position_margin_deg: float = 5.0
    temperature_ceiling_c: float = 45.0
    workspace_limit_deg: tuple[float, ...] = ()
    gravity_probe_deg: float = 5.0
    gravity_probe_speed_deg_s: float = 2.0
    # Width of the Coulomb reversal. Measured on run #5 data: sweeping this
    # from a hard sign() to 1.8 deg/s cuts worst-joint validation error 11%
    # and the gain survives the non-negativity constraint, unlike Stribeck.
    coulomb_transition_deg_s: float = 1.8
    # Superseded by coulomb_transition_deg_s, which buys the same curvature
    # with one non-negative column instead of a cancelling pair. Left available
    # because a different transmission may genuinely show static > dynamic.
    stribeck: bool = False
    stribeck_speed_deg_s: float = 2.0
    load_friction: bool = False
    seed: int = 0

    def design_limits(self, arm: ident.ArmModel,
                      plant_limits: tuple[np.ndarray, np.ndarray] | None = None,
                      ) -> excitation.DesignLimits:
        """Joint range is the tighter of what the model and the plant allow."""
        low, high = arm.limits_deg()
        if plant_limits is not None:
            low = np.maximum(low, plant_limits[0])
            high = np.minimum(high, plant_limits[1])
        if self.workspace_limit_deg:
            cap = np.abs(np.asarray(self.workspace_limit_deg, dtype=float))
            low, high = np.maximum(low, -cap), np.minimum(high, cap)
        return excitation.DesignLimits(
            lower_deg=low, upper_deg=high,
            margin_deg=self.position_margin_deg,
            maximum_speed_deg_s=self.maximum_speed_deg_s,
            maximum_acceleration_deg_s2=self.maximum_acceleration_deg_s2,
        )

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["friction_speeds_deg_s"] = list(self.friction_speeds_deg_s)
        payload["validation_speeds_deg_s"] = list(self.validation_speeds_deg_s)
        payload["workspace_limit_deg"] = list(self.workspace_limit_deg)
        return payload

    def model_components(self) -> ModelComponents:
        return ModelComponents(
            coulomb_transition_deg_s=self.coulomb_transition_deg_s,
            stribeck=self.stribeck,
            stribeck_speed_deg_s=self.stribeck_speed_deg_s,
            load_friction=self.load_friction)


@dataclass
class PhaseReport:
    """What one phase produced, and why it stopped."""

    phase: str
    observations: int = 0
    duration_s: float = 0.0
    peak_temperature_c: float = 0.0
    peak_speed_deg_s: float = 0.0
    peak_current_a: float = 0.0
    detail: dict = field(default_factory=dict)
    aborted: str | None = None

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "observations": self.observations,
            "duration_s": round(self.duration_s, 3),
            "peak_temperature_c": round(self.peak_temperature_c, 2),
            "peak_speed_deg_s": round(self.peak_speed_deg_s, 3),
            "peak_current_a": round(self.peak_current_a, 4),
            "detail": self.detail,
            "aborted": self.aborted,
        }


@dataclass
class CampaignResult:
    """The identified model plus the evidence that it generalises."""

    plan: dict = field(default_factory=dict)
    phases: list[dict] = field(default_factory=list)
    joints: list[dict] = field(default_factory=list)
    validation_rms_a: list[float] = field(default_factory=list)
    validation_samples: int = 0
    aborted: str | None = None
    # The fitted regressions themselves, kept out of as_dict because they are
    # arrays, not a document. Diagnostics need them to predict per observation.
    fits: list = field(default_factory=list, repr=False)

    @property
    def complete(self) -> bool:
        return self.aborted is None and bool(self.joints)

    def verdict(self) -> dict:
        """Whether this run may be used, and which joint decided that."""
        signal = max(
            (float(phase.get("peak_current_a") or 0.0) for phase in self.phases),
            default=0.0)
        joints = []
        for index, entry in enumerate(self.joints):
            state, reason = judge_joint(entry, signal)
            joints.append({"joint": index + 1, "state": state, "reason": reason})
        states = {item["state"] for item in joints}
        if self.aborted is not None:
            overall = "fail"
        elif not joints or "fail" in states:
            overall = "fail"
        elif "warn" in states or "unknown" in states:
            overall = "warn"
        else:
            overall = "pass"
        failed = [item["joint"] for item in joints if item["state"] == "fail"]
        return {"state": overall, "joints": joints, "failed_joints": failed}

    def as_dict(self) -> dict:
        return {
            "plan": self.plan,
            "phases": self.phases,
            "joints": self.joints,
            "validation_rms_a": [round(v, 5) for v in self.validation_rms_a],
            "validation_samples": self.validation_samples,
            "aborted": self.aborted,
            "complete": self.complete,
            "verdict": self.verdict(),
        }


class Abort(RuntimeError):
    """Raised when a guard stops the campaign; the arm is left at rest."""


# Phases A to C are position controlled, so the speed that matters is the
# profile's sustained limit rather than any current-mode figure. The
# acceleration bound is that same ceiling reached from rest in a quarter second.
PROBE_SPEED_FRACTION = 0.5
ACCELERATION_PER_SPEED = 4.0

# The sweep speeds are what actually excites friction, so they are fractions of
# the speed ceiling rather than fixed figures. Fixed figures meant raising the
# ceiling changed nothing: the sweep kept running at the old speeds and the
# viscous term stayed invisible.
FRICTION_SPEED_FRACTIONS = (0.1, 0.3, 0.6, 1.0)
VALIDATION_SPEED_FRACTIONS = (0.35, 0.65)

# The plant ramps a sweep over a quarter of its nominal duration, so a pass of
# `distance` at `speed` implies 4*speed^2/distance of acceleration. A short
# sweep at high speed is therefore a violent one, and has almost no constant
# speed left in the middle to measure. Amplitude is sized from this.
SWEEP_ACCELERATION_DEG_S2 = 360.0

_STATIC_BOUNDS = {
    "static_poses": (4, 60),
    "static_candidates": (10, 400),
    "settle_samples": (1, 20),
    "friction_amplitude_deg": (5.0, 40.0),
    "fourier_harmonics": (1, 6),
    "fourier_base_frequency_hz": (0.02, 0.3),
    "fourier_duration_s": (5.0, 120.0),
    "fourier_attempts": (5, 200),
    "sample_rate_hz": (5.0, 100.0),
    "validation_poses": (3, 40),
    "validation_trajectory_s": (4.0, 60.0),
    "position_margin_deg": (3.0, 30.0),
}


def campaign_bounds(profile: RobotProfile) -> dict:
    """Operator-settable ranges, capped by this arm's envelope."""
    speed = profile.sustained_speed_deg_s
    return {
        **_STATIC_BOUNDS,
        "maximum_speed_deg_s": (1.0, speed),
        "maximum_acceleration_deg_s2": (2.0, ACCELERATION_PER_SPEED * speed),
        "temperature_ceiling_c": (30.0, profile.temperature_c),
    }


_INTEGER_FIELDS = frozenset({
    "static_poses", "static_candidates", "settle_samples", "fourier_harmonics",
    "fourier_attempts", "validation_poses", "seed",
})


def sweep_speeds(maximum_speed_deg_s: float, fractions) -> tuple[float, ...]:
    """Speeds spread across the ceiling, so raising it reaches new ground."""
    speeds = {round(fraction * maximum_speed_deg_s, 2) for fraction in fractions}
    return tuple(sorted(speed for speed in speeds if speed > 0.0))


def sweep_amplitude_deg(maximum_speed_deg_s: float, requested: float) -> float:
    """Enough travel to reach the top speed without a violent ramp."""
    low, high = _STATIC_BOUNDS["friction_amplitude_deg"]
    needed = (ACCELERATION_PER_SPEED * maximum_speed_deg_s ** 2
              / SWEEP_ACCELERATION_DEG_S2)
    return float(min(max(requested, needed, low), high))


def default_plan(profile: RobotProfile) -> CampaignPlan:
    """A plan that already respects this arm's envelope."""
    plan = CampaignPlan()
    plan.temperature_ceiling_c = min(
        plan.temperature_ceiling_c, profile.temperature_c)
    plan.maximum_speed_deg_s = min(
        plan.maximum_speed_deg_s, profile.sustained_speed_deg_s)
    plan.maximum_acceleration_deg_s2 = min(
        plan.maximum_acceleration_deg_s2,
        ACCELERATION_PER_SPEED * profile.sustained_speed_deg_s)
    plan.position_margin_deg = max(
        plan.position_margin_deg, profile.position_margin_deg)
    plan.workspace_limit_deg = tuple(profile.workspace_limit_deg)
    _follow_speed(plan)
    return plan


def _follow_speed(plan: CampaignPlan) -> None:
    """Re-derive everything that only means something relative to the ceiling."""
    plan.friction_speeds_deg_s = sweep_speeds(
        plan.maximum_speed_deg_s, FRICTION_SPEED_FRACTIONS)
    plan.validation_speeds_deg_s = sweep_speeds(
        plan.maximum_speed_deg_s, VALIDATION_SPEED_FRACTIONS)
    plan.friction_amplitude_deg = sweep_amplitude_deg(
        plan.maximum_speed_deg_s, plan.friction_amplitude_deg)


def clamp_campaign_plan(
    request: dict | None, profile: RobotProfile,
) -> tuple[CampaignPlan, list[str]]:
    """Build a plan from a dashboard payload, reporting every clamped field."""
    payload = request or {}
    plan = default_plan(profile)
    bounds = campaign_bounds(profile)
    notes: list[str] = []

    for name, (low, high) in bounds.items():
        if name not in payload:
            continue
        try:
            value = float(payload[name])
        except (TypeError, ValueError):
            notes.append(f"{name} was not a number, kept {getattr(plan, name)}")
            continue
        if not np.isfinite(value):
            notes.append(f"{name} was not finite, kept {getattr(plan, name)}")
            continue
        bounded = min(max(value, low), high)
        if bounded != value:
            notes.append(f"{name} {value:g} clamped to {bounded:g}")
        setattr(plan, name, int(bounded) if name in _INTEGER_FIELDS else bounded)

    if "maximum_speed_deg_s" in payload:
        # Otherwise a raised ceiling leaves the sweep running at the old speeds.
        _follow_speed(plan)
        if "maximum_acceleration_deg_s2" not in payload:
            plan.maximum_acceleration_deg_s2 = min(
                ACCELERATION_PER_SPEED * plan.maximum_speed_deg_s,
                bounds["maximum_acceleration_deg_s2"][1])

    speeds = payload.get("friction_speeds_deg_s")
    if speeds is not None:
        kept = []
        for entry in speeds if isinstance(speeds, (list, tuple)) else []:
            try:
                value = float(entry)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and 0.0 < value <= plan.maximum_speed_deg_s:
                kept.append(value)
            else:
                notes.append(f"friction speed {entry} dropped")
        if kept:
            plan.friction_speeds_deg_s = tuple(sorted(set(kept)))
        else:
            notes.append("no usable friction speeds, kept the defaults")

    validation_speeds = payload.get("validation_speeds_deg_s")
    if validation_speeds is not None:
        kept = []
        for entry in (validation_speeds
                      if isinstance(validation_speeds, (list, tuple)) else []):
            try:
                value = float(entry)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and 0.0 < value <= plan.maximum_speed_deg_s:
                kept.append(value)
            else:
                notes.append(f"validation speed {entry} dropped")
        if kept:
            plan.validation_speeds_deg_s = tuple(sorted(set(kept)))
        else:
            notes.append("no usable validation speeds, kept the defaults")

    if "seed" in payload:
        try:
            plan.seed = int(payload["seed"])
        except (TypeError, ValueError):
            notes.append("seed was not an integer, kept 0")

    dropped = [speed for speed in plan.friction_speeds_deg_s
               if speed > plan.maximum_speed_deg_s]
    if dropped:
        plan.friction_speeds_deg_s = tuple(
            speed for speed in plan.friction_speeds_deg_s
            if speed <= plan.maximum_speed_deg_s
        ) or (plan.maximum_speed_deg_s,)
        notes.append(
            f"friction speeds {dropped} exceeded the plan speed and were dropped")

    stale = [speed for speed in plan.validation_speeds_deg_s
             if speed > plan.maximum_speed_deg_s]
    if stale:
        plan.validation_speeds_deg_s = tuple(
            speed for speed in plan.validation_speeds_deg_s
            if speed <= plan.maximum_speed_deg_s
        )
        notes.append(
            f"validation speeds {stale} exceeded the plan speed and were dropped")
    return plan, notes


MONITORED_FIELDS = (
    "position_deg", "speed_deg_s", "current_a", "temperature_c",
    "voltage_v", "enabled", "fault_code",
)


def _fully_instrumented(sample: dict) -> bool:
    # Empty, not just absent: a plant on an arm without a bus-voltage interface
    # reports the key with nothing in it, and that is not instrumentation.
    # Kept for callers that want to know; the monitor no longer waits for it,
    # because a guard that can run on the channels present should run.
    return all(sample.get(field) for field in MONITORED_FIELDS)


class Campaign:
    """Runs the phases against a plant and regresses the result."""

    def __init__(self, arm: ident.ArmModel, plant: Plant,
                 plan: CampaignPlan | None = None,
                 progress: Callable[[str, dict], None] | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 monitor: EnvelopeMonitor | None = None,
                 pose_admissible: Callable[[np.ndarray], bool] | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.arm = arm
        self.plant = plant
        self.plan = plan or CampaignPlan()
        self.progress = progress or (lambda phase, detail: None)
        self.should_stop = should_stop or (lambda: False)
        self.monitor = monitor
        self.pose_admissible = pose_admissible
        self.clock = clock or time.monotonic
        self.limits = self.plan.design_limits(arm, self._plant_limits())
        self.observations: list[Observation] = []
        self.reports: list[PhaseReport] = []
        self.aborted: str | None = None
        self._started = self.clock()

    def _plant_limits(self):
        getter = getattr(self.plant, "limits_deg", None)
        return None if getter is None else getter()

    def _collision_free(self):
        """Self-collision plus any caller-supplied admission rule."""
        collision_free = getattr(self.plant, "collision_free", None)
        if collision_free is None and self.pose_admissible is None:
            return None

        def admissible(pose_deg) -> bool:
            if collision_free is not None and not collision_free(pose_deg):
                return False
            return self.pose_admissible is None or self.pose_admissible(pose_deg)

        return admissible

    # -- guards ----------------------------------------------------------

    def _guard(self, sample: dict) -> None:
        if self.should_stop():
            raise Abort("operator stop")

        # Current is an outcome here, not a command: phases A to C are position
        # controlled, so the only honest current limit is the measured one.
        # The commissioned monitor owns those thresholds.
        if self.monitor is not None:
            # The stall detector belongs to the streaming current-control loop.
            # Campaign frames are pulled synchronously, so the gap between them
            # is deliberate dwell at a pose, not lost telemetry.
            self.monitor.last_sample_at = None
            trip = self.monitor.check(sample, self.clock())
            if trip:
                raise Abort(trip)

        temperatures = sample.get("temperature_c") or []
        for index, value in enumerate(temperatures):
            if value >= self.plan.temperature_ceiling_c:
                raise Abort(
                    f"joint{index + 1} reached {value:.1f} C, ceiling is "
                    f"{self.plan.temperature_ceiling_c:.1f} C")
        speeds = np.abs(np.asarray(sample.get("speed_deg_s") or [0.0]))
        if speeds.size and speeds.max() > self.limits.maximum_speed_deg_s * 2.0:
            raise Abort(f"joint speed {speeds.max():.1f} deg/s exceeded the plan")

    def _record(self, phase: str, sample: dict, report: PhaseReport,
                acceleration_deg_s2=None) -> Observation:
        self._guard(sample)
        observation = Observation.from_sample(
            phase, self.clock() - self._started, sample, acceleration_deg_s2)
        self.observations.append(observation)
        report.observations += 1
        # A run lasts tens of minutes, so something has to be able to watch it.
        self.progress(phase, {"observations": len(self.observations)})
        if observation.temperature_c:
            report.peak_temperature_c = max(
                report.peak_temperature_c, max(observation.temperature_c))
        report.peak_speed_deg_s = max(
            report.peak_speed_deg_s,
            float(np.max(np.abs(observation.velocity_deg_s))))
        report.peak_current_a = max(
            report.peak_current_a,
            float(np.max(np.abs(observation.current_a))))
        return observation

    # -- phases ----------------------------------------------------------

    def _open(self, phase: str) -> tuple[PhaseReport, float]:
        """Register the report up front so an abort still leaves its evidence."""
        report = PhaseReport(phase)
        self.reports.append(report)
        return report, self.clock()

    def _probe(self, pose_deg) -> list:
        """Gravity samples at a pose, with friction forced to a known sign.

        A joint held still balances gravity with any value inside its stiction
        band, so a standstill reading is gravity plus an unknowable offset.
        Crossing the pose in both directions costs a few seconds and makes the
        friction contribution cancel between the pair.
        """
        probe = getattr(self.plant, "probe_pose", None)
        if probe is None:
            samples = [self.plant.hold_pose(pose_deg)]
            samples += [self._dwell(pose_deg)
                        for _ in range(max(1, self.plan.settle_samples) - 1)]
            return samples
        speed = self.plan.gravity_probe_speed_deg_s
        frames = probe(pose_deg, self.plan.gravity_probe_deg, speed)
        # Ramp-in and ramp-out pass through near-zero speed, where friction has
        # no determined sign again; keeping those frames would put the very
        # contamination this probe removes straight back into the fit.
        floor = PROBE_SPEED_FRACTION * speed
        return [frame for frame in frames
                if max(abs(v) for v in frame["speed_deg_s"]) >= floor]

    def _dwell(self, pose_deg) -> dict:
        """Take another reading at rest, re-commanding only if the plant needs it."""
        dwell = getattr(self.plant, "dwell", None)
        return self.plant.hold_pose(pose_deg) if dwell is None else dwell(pose_deg)

    def run_gravity(self) -> PhaseReport:
        report, start = self._open(PHASE_GRAVITY)
        try:
            design = excitation.design_static_poses(
                self.arm, self.limits, count=self.plan.static_poses,
                candidates=self.plan.static_candidates, seed=self.plan.seed,
                collision_free=self._collision_free())
            report.detail = design.as_dict()
            report.detail.pop("poses_deg", None)
            report.detail["poses"] = len(design.poses_deg)
            for index, pose in enumerate(design.poses_deg):
                target = np.asarray(pose, dtype=float)
                for sample in self._probe(target):
                    self._record(PHASE_GRAVITY, sample, report)
                self.progress(PHASE_GRAVITY, {
                    "pose": index + 1, "poses": len(design.poses_deg)})
        finally:
            report.duration_s = self.clock() - start
        return report

    def run_friction(self) -> PhaseReport:
        report, start = self._open(PHASE_FRICTION)
        try:
            sweeps = excitation.design_friction_sweeps(
                self.arm, self.limits,
                amplitude_deg=self.plan.friction_amplitude_deg,
                speeds_deg_s=self.plan.friction_speeds_deg_s)
            report.detail = {"sweeps": [sweep.as_dict() for sweep in sweeps]}
            for sweep in sweeps:
                for speed in sweep.speeds_deg_s:
                    for distance in (sweep.amplitude_deg, -sweep.amplitude_deg):
                        origin = np.asarray(sweep.start_deg, dtype=float)
                        if distance < 0:
                            origin = origin.copy()
                            origin[sweep.joint] += sweep.amplitude_deg
                        for sample in self.plant.traverse(
                                sweep.joint, origin, distance, speed):
                            self._record(PHASE_FRICTION, sample, report)
                self.progress(PHASE_FRICTION, {
                    "joint": sweep.joint + 1, "joints": len(sweeps)})
        finally:
            report.duration_s = self.clock() - start
        return report

    def run_inertia(self) -> PhaseReport:
        report, start = self._open(PHASE_INERTIA)
        try:
            trajectory = excitation.design_fourier_trajectory(
                self.arm, self.limits, harmonics=self.plan.fourier_harmonics,
                base_frequency_hz=self.plan.fourier_base_frequency_hz,
                duration_s=self.plan.fourier_duration_s,
                attempts=self.plan.fourier_attempts, seed=self.plan.seed + 1,
                collision_free=self._collision_free())
            if trajectory is None:
                report.detail = {"trajectory": None}
                report.aborted = "no feasible trajectory within the limits"
                return report

            report.detail = {"trajectory": trajectory.as_dict()}
            emitted = 0
            for sample in self.plant.track(trajectory, self.plan.sample_rate_hz):
                self._record(
                    PHASE_INERTIA, sample, report,
                    sample.get("acceleration_deg_s2"))
                emitted += 1
                if emitted % 20 == 0:
                    self.progress(PHASE_INERTIA, {"samples": emitted})
        finally:
            report.duration_s = self.clock() - start
        return report

    def run_validation(self) -> PhaseReport:
        """States the fit never saw: new poses, new speeds, a new trajectory.

        Static poses alone would only exercise the gravity columns, leaving
        friction and inertia unvalidated. So this phase also sweeps at speeds
        absent from training and follows an independently seeded trajectory.
        """
        report, start = self._open(PHASE_VALIDATION)
        try:
            rng = np.random.default_rng(self.plan.seed + 977)
            low, high = self.limits.usable()
            admissible = self._collision_free()
            accepted = 0
            attempts = 0
            while accepted < self.plan.validation_poses and attempts < 40 * (
                    self.plan.validation_poses + 1):
                attempts += 1
                pose = rng.uniform(low, high)
                if admissible is not None and not admissible(pose):
                    continue
                sample = self.plant.hold_pose(pose)
                self._record(PHASE_VALIDATION, sample, report)
                accepted += 1
            report.detail = {"poses": accepted, "candidates": attempts}

            report.detail["speeds_deg_s"] = list(self._validation_speeds())
            # Sweep from a different home than training, so validation differs in
            # configuration as well as in speed.
            home = self._validation_home(rng, admissible)
            report.detail["sweep_home_deg"] = [round(v, 3) for v in home]
            swept = 0
            for sweep in excitation.design_friction_sweeps(
                    self.arm, self.limits,
                    amplitude_deg=self.plan.friction_amplitude_deg,
                    speeds_deg_s=self._validation_speeds(),
                    home_deg=home):
                for speed in sweep.speeds_deg_s:
                    origin = np.asarray(sweep.start_deg, dtype=float)
                    for frame in self.plant.traverse(
                            sweep.joint, origin, sweep.amplitude_deg, speed):
                        self._record(PHASE_VALIDATION, frame, report)
                        swept += 1
            report.detail["sweep_samples"] = swept

            trajectory = excitation.design_fourier_trajectory(
                self.arm, self.limits, harmonics=self.plan.fourier_harmonics,
                base_frequency_hz=self.plan.fourier_base_frequency_hz * 1.3,
                duration_s=self.plan.validation_trajectory_s,
                attempts=self.plan.fourier_attempts,
                seed=self.plan.seed + 4231,
                collision_free=admissible)
            tracked = 0
            if trajectory is not None:
                for frame in self.plant.track(
                        trajectory, self.plan.sample_rate_hz):
                    self._record(PHASE_VALIDATION, frame, report,
                                 frame.get("acceleration_deg_s2"))
                    tracked += 1
            report.detail["trajectory_samples"] = tracked
        finally:
            report.duration_s = self.clock() - start
        return report

    def _validation_home(self, rng, admissible) -> np.ndarray:
        """A reachable pose away from the training home, leaving sweep room."""
        low, high = self.limits.usable()
        room = self.plan.friction_amplitude_deg
        inner_low, inner_high = low + room, high - room
        if np.any(inner_low >= inner_high):
            return np.zeros(self.arm.joint_count)
        for _ in range(60):
            candidate = rng.uniform(inner_low, inner_high)
            if admissible is None or admissible(candidate):
                return candidate
        return np.zeros(self.arm.joint_count)

    def _validation_speeds(self) -> tuple[float, ...]:
        trained = set(self.plan.friction_speeds_deg_s)
        speeds = tuple(
            speed for speed in self.plan.validation_speeds_deg_s
            if speed not in trained and 0.0 < speed <= self.plan.maximum_speed_deg_s
        )
        if speeds:
            return speeds
        # Fall back to a speed between two trained ones so the phase still moves.
        ordered = sorted(s for s in trained if s <= self.plan.maximum_speed_deg_s)
        if len(ordered) >= 2:
            return (0.5 * (ordered[0] + ordered[1]),)
        return (min(self.plan.maximum_speed_deg_s, 1.0),)

    # -- regression ------------------------------------------------------

    def _rows(self, observations: Iterable[Observation]):
        regressors, velocities, currents = [], [], []
        for record in observations:
            regressors.append(self.arm.torque_regressor(
                record.position_deg, record.velocity_deg_s,
                record.acceleration_deg_s2))
            velocities.append(np.asarray(record.velocity_deg_s, dtype=float))
            currents.append(np.asarray(record.current_a, dtype=float))
        return regressors, velocities, currents

    def fit(self) -> CampaignResult:
        """Regress on phases A-C and score on phase D."""
        result = CampaignResult(
            plan=self.plan.as_dict(),
            phases=[report.as_dict() for report in self.reports])

        training = [record for record in self.observations
                    if record.phase != PHASE_VALIDATION]
        holdout = [record for record in self.observations
                   if record.phase == PHASE_VALIDATION]
        if not training:
            result.aborted = "no training observations"
            return result

        train_rows = self._rows(training)
        holdout_rows = self._rows(holdout) if holdout else None
        result.validation_samples = len(holdout)

        for joint in range(self.arm.joint_count):
            fit = ident.fit_joint(
                joint, train_rows[0],
                [v[joint] for v in train_rows[1]],
                [c[joint] for c in train_rows[2]],
                maximum_condition=MAXIMUM_CONDITION,
                components=self.plan.model_components(),
                seed=self.plan.seed)
            entry = fit.as_dict()
            if holdout_rows is not None:
                predicted = np.array([
                    ident.predict_joint(fit, regressor, velocity[joint])
                    for regressor, velocity in zip(
                        holdout_rows[0], holdout_rows[1])])
                truth = np.array([c[joint] for c in holdout_rows[2]])
                error = float(np.sqrt(np.mean((predicted - truth) ** 2)))
                entry["validation_rms_a"] = round(error, 6)
                result.validation_rms_a.append(error)
            result.fits.append(fit)
            result.joints.append(entry)
        return result

    def run(self) -> CampaignResult:
        """All four phases; an abort keeps whatever was already measured."""
        self.aborted = None
        for phase in (self.run_gravity, self.run_friction,
                      self.run_inertia, self.run_validation):
            try:
                report = phase()
            except Abort as stop:
                self.aborted = str(stop)
                self.reports[-1].aborted = self.aborted
                break
            if report.aborted:
                self.aborted = f"{report.phase}: {report.aborted}"
                break
        result = self.fit()
        if self.aborted:
            result.aborted = self.aborted
        return result
