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
import inspect
import math
import re
import time

import numpy as np

from . import excitation, identification as ident
from .interfaces import DriveLimitExceeded, MotionFailed, MotionPaused
from .model import ModelComponents
from .profile import RobotProfile

PHASE_GRAVITY = "A_gravity"
PHASE_FRICTION = "B_friction"
PHASE_INERTIA = "C_inertia"
PHASE_VALIDATION = "D_validation"

# Reversal widths offered to each joint's fit. Hardware puts the best value
# between 0.28 and 1.37 deg/s depending on the joint, so the grid brackets that
# generously rather than asserting one figure for the whole arm.
COULOMB_TRANSITION_SEARCH = tuple(
    round(float(w), 4) for w in np.geomspace(0.08, 6.0, 25))
STRIBECK_SPEED_SEARCH = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
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
    """Anything that can veto a telemetry frame.

    A monitor that also fills ``last_trip`` with the joint and the kind lets
    the campaign answer a trip it can answer -- too much current for one joint
    -- instead of only reporting it. Without that detail every trip stops the
    run, which is the safe reading of an unattributed veto.
    """

    last_trip: dict | None

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
    maximum_speed_deg_s: float | None = None
    maximum_temperature_c: float | None = None
    peak_current_a: tuple[float, ...] = ()
    continuous_current_a: tuple[float, ...] = ()
    sustained_current_window_s: float = 0.5
    _over_current_since: list[float | None] = field(
        default_factory=list, init=False, repr=False)
    # Campaign frames are pulled synchronously, so a gap between them is
    # deliberate dwell rather than lost telemetry; kept for protocol parity.
    last_sample_at: float | None = None
    # Which joint tripped and why, so a caller can answer it rather than only
    # report it. Set by every trip; the message alone would have to be parsed.
    last_trip: dict | None = field(default=None, init=False, repr=False)

    def _trip(self, joint: int | None, kind: str, message: str) -> str:
        self.last_trip = {"joint": joint, "kind": kind, "message": message}
        return message

    def check(self, sample: dict, now: float) -> str | None:
        for index, live in enumerate(sample.get("enabled") or []):
            if not live:
                return self._trip(
                    index, "disabled",
                    f"joint{index + 1} reports its drive disabled")
        for index, code in enumerate(sample.get("fault_code") or []):
            if code:
                return self._trip(
                    index, "fault",
                    f"joint{index + 1} reports fault code {int(code)}")
        if self.maximum_speed_deg_s is not None:
            for index, value in enumerate(
                    sample.get("safety_speed_deg_s") or []):
                if abs(float(value)) > self.maximum_speed_deg_s:
                    return self._trip(
                        index, "speed",
                        f"joint{index + 1} position-derived speed "
                        f"{abs(float(value)):.1f} deg/s exceeded "
                        f"{self.maximum_speed_deg_s:.1f} deg/s")
        if self.maximum_temperature_c is not None:
            for index, value in enumerate(sample.get("temperature_c") or []):
                if float(value) >= self.maximum_temperature_c:
                    return self._trip(
                        index, "temperature",
                        f"joint{index + 1} temperature {float(value):.1f} C "
                        f"reached {self.maximum_temperature_c:.1f} C")
        currents = sample.get("current_a") or []
        if currents and self.peak_current_a:
            if len(currents) != len(self.peak_current_a):
                return self._trip(
                    None, "envelope",
                    f"current telemetry has {len(currents)} joints, "
                    f"the envelope has {len(self.peak_current_a)}")
            for index, (value, ceiling) in enumerate(
                    zip(currents, self.peak_current_a)):
                if abs(float(value)) > ceiling:
                    return self._trip(
                        index, "peak_current",
                        f"joint{index + 1} peak current "
                        f"{abs(float(value)):.3f} A exceeded "
                        f"{ceiling:.3f} A")
        if currents and self.continuous_current_a:
            if len(currents) != len(self.continuous_current_a):
                return self._trip(
                    None, "envelope",
                    f"current telemetry has {len(currents)} joints, "
                    f"the continuous envelope has "
                    f"{len(self.continuous_current_a)}")
            if len(self._over_current_since) != len(currents):
                self._over_current_since = [None] * len(currents)
            for index, (value, ceiling) in enumerate(
                    zip(currents, self.continuous_current_a)):
                if abs(float(value)) <= ceiling:
                    self._over_current_since[index] = None
                    continue
                since = self._over_current_since[index]
                if since is None or now < since:
                    self._over_current_since[index] = now
                    continue
                if now - since >= self.sustained_current_window_s:
                    return self._trip(
                        index, "continuous_current",
                        f"joint{index + 1} continuous current "
                        f"{abs(float(value)):.3f} A exceeded "
                        f"{ceiling:.3f} A for "
                        f"{self.sustained_current_window_s:.3f} s")
        if self.minimum_voltage_v is None or self.maximum_voltage_v is None:
            return None
        for index, volts in enumerate(sample.get("voltage_v") or []):
            if not self.minimum_voltage_v <= volts <= self.maximum_voltage_v:
                return self._trip(
                    index, "voltage",
                    f"joint{index + 1} bus at {volts:.1f} V, outside "
                    f"{self.minimum_voltage_v:.1f}-"
                    f"{self.maximum_voltage_v:.1f} V")
        return None

    def guards(self) -> tuple[str, ...]:
        """What this monitor is actually able to enforce."""
        active = ["drive-enabled check", "fault-code check"]
        if self.maximum_speed_deg_s is not None:
            active.append("position-rate ceiling")
        if self.maximum_temperature_c is not None:
            active.append("temperature ceiling")
        if self.peak_current_a:
            active.append("peak-current ceiling")
        if self.continuous_current_a:
            active.append("sustained-current ceiling")
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
    # Which motion this came out of and how well the window fitted, so a
    # suspect point can be traced back to the pass that produced it.
    motion: str = ""
    window_frames: int = 0
    window_fit_rms_deg: float = 0.0

    @classmethod
    def from_sample(cls, phase: str, time_s: float, sample: dict,
                    acceleration_deg_s2=None) -> "Observation":
        count = len(sample["position_deg"])
        if acceleration_deg_s2 is None:
            acceleration_deg_s2 = sample.get("acceleration_deg_s2")
        acceleration = (np.zeros(count) if acceleration_deg_s2 is None
                        else np.asarray(acceleration_deg_s2, dtype=float))
        return cls(
            phase=phase, time_s=time_s,
            position_deg=[float(v) for v in sample["position_deg"]],
            velocity_deg_s=[float(v) for v in sample["speed_deg_s"]],
            acceleration_deg_s2=[float(v) for v in acceleration],
            current_a=[float(v) for v in sample["current_a"]],
            temperature_c=[float(v) for v in sample.get("temperature_c", [])],
            motion=str(sample.get("motion", "")),
            window_frames=int(sample.get("window_frames", 0)),
            window_fit_rms_deg=float(sample.get("window_fit_rms_deg", 0.0)),
        )


@dataclass
class CampaignPlan:
    """Everything the operator can choose before the arm moves."""

    static_poses: int = 24
    static_candidates: int = 200
    settle_samples: int = 3
    friction_amplitude_deg: float = 20.0
    friction_speeds_deg_s: tuple[float, ...] = (2.0, 5.0, 8.0)
    # Passes per speed and direction. One pass gives no way to notice that a
    # pass went wrong; three disagree visibly when one does.
    friction_repeats: int = 3
    # Postures each joint is swept at. Its gravity load depends on where the
    # other joints are, so one posture measures friction under one load: joint
    # one of this arm sees 0.17 Nm at home and 4.5 Nm with the arm extended.
    friction_postures: int = 3
    fourier_harmonics: int = 4
    fourier_base_frequency_hz: float = 0.08
    fourier_duration_s: float = 30.0
    fourier_ramp_s: float = 4.0
    fourier_attempts: int = 40
    # The dedicated optimal-excitation campaign replaces the single inertia
    # trajectory with a sequence selected against the cumulative regressor.
    # Validation uses separately seeded trajectories and never enters the fit.
    optimal_training_trajectories: int = 12
    optimal_validation_trajectories: int = 3
    optimal_fourier_amplitude_fraction: float = 0.20
    # Fourier reversals contain low-speed points, but they are accelerating
    # transients rather than the steady windows a Stribeck curve assumes.
    optimal_friction_speeds_deg_s: tuple[float, ...] = (
        0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    optimal_friction_repeats: int = 2
    optimal_friction_postures: int = 3
    sample_rate_hz: float = 20.0
    validation_poses: int = 12
    validation_speeds_deg_s: tuple[float, ...] = (3.5, 6.5)
    validation_trajectory_s: float = 12.0
    maximum_speed_deg_s: float = 10.0
    maximum_acceleration_deg_s2: float = 20.0
    position_margin_deg: float = 5.0
    temperature_ceiling_c: float = 45.0
    workspace_limit_deg: tuple[float, ...] = ()
    # Where each joint may go, low and high. A cell is not symmetric: an arm
    # mounted at an angle meets the bench swinging one way and nothing the
    # other, so one +/- number cannot describe it. Wins over the symmetric
    # cap above when both are set.
    workspace_range_deg: tuple[tuple[float, float], ...] = ()
    gravity_probe_deg: float = 5.0
    gravity_probe_speed_deg_s: float = 2.0
    # Two speeds rather than one. The difference between the pair-averaged
    # currents at two speeds IS the viscous term, so a gravity run measures it
    # instead of assuming the probe was slow enough to ignore it. Empty falls
    # back to the single speed above, which is what the four-phase campaign
    # has always used.
    gravity_probe_speeds_deg_s: tuple[float, ...] = ()
    # Poses held back from a gravity run, crossed exactly the way training was.
    gravity_validation_poses: int = 8
    # Width of the Coulomb reversal. Measured on run #5 data: sweeping this
    # from a hard sign() to 1.8 deg/s cuts worst-joint validation error 11%
    # and the gain survives the non-negativity constraint, unlike Stribeck.
    coulomb_transition_deg_s: float = 1.8
    # Offered to each joint's fit instead of asserting one width for the arm.
    coulomb_transition_search: tuple[float, ...] = COULOMB_TRANSITION_SEARCH
    # Superseded by coulomb_transition_deg_s, which buys the same curvature
    # with one non-negative column instead of a cancelling pair. Left available
    # because a different transmission may genuinely show static > dynamic.
    stribeck: bool = False
    # Offered to every joint; each one's data decides. On this arm three of
    # seven take it and the rest are better without.
    stribeck_search: bool = True
    stribeck_speed_deg_s: float = 1.6
    stribeck_speed_search: tuple[float, ...] = ()
    load_friction: bool = False
    # Offered to every joint and decided per joint. Measured on three postures
    # per joint it is worth twenty-nine per cent of the validation error on the
    # heaviest and nothing on the wrist, which carries no load in any pose.
    load_friction_search: bool = True
    load_stribeck: bool = False
    load_stribeck_search: bool = False
    # Motions the arm may refuse before the run is called off. A few gaps in a
    # ladder of thousands cost almost nothing; an arm refusing everything is
    # not producing a dataset and should not be left running for hours.
    skip_budget: int = 40
    # Where the arm is standing when the run begins. The first transit of a
    # tour is screened from here and it is the longest one, so a design that
    # assumes neutral and an arm that is not there is an unscreened swing.
    # Empty means neutral, which is what every campaign but gravity designs
    # from.
    start_deg: tuple = ()
    seed: int = 0

    def design_limits(self, arm: ident.ArmModel,
                      plant_limits: tuple[np.ndarray, np.ndarray] | None = None,
                      ) -> excitation.DesignLimits:
        """Joint range is the tighter of what the model and the plant allow."""
        low, high = arm.limits_deg()
        if plant_limits is not None:
            low = np.maximum(low, plant_limits[0])
            high = np.minimum(high, plant_limits[1])
        if self.workspace_range_deg:
            bounds = np.asarray(self.workspace_range_deg, dtype=float)
            low, high = (np.maximum(low, bounds[:, 0]),
                         np.minimum(high, bounds[:, 1]))
        elif self.workspace_limit_deg:
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
        payload["optimal_friction_speeds_deg_s"] = list(
            self.optimal_friction_speeds_deg_s)
        payload["gravity_probe_speeds_deg_s"] = list(
            self.gravity_probe_speeds_deg_s)
        payload["start_deg"] = list(self.start_deg)
        payload["workspace_limit_deg"] = list(self.workspace_limit_deg)
        payload["workspace_range_deg"] = [list(pair)
                                          for pair in self.workspace_range_deg]
        payload["coulomb_transition_search"] = list(
            self.coulomb_transition_search)
        payload["stribeck_speed_search"] = list(self.stribeck_speed_search)
        return payload

    def model_components(self) -> ModelComponents:
        return ModelComponents(
            coulomb_transition_deg_s=self.coulomb_transition_deg_s,
            coulomb_transition_search=tuple(self.coulomb_transition_search),
            stribeck=self.stribeck,
            stribeck_search=self.stribeck_search,
            stribeck_speed_deg_s=self.stribeck_speed_deg_s,
            stribeck_speed_search=tuple(self.stribeck_speed_search),
            load_friction=self.load_friction,
            load_friction_search=self.load_friction_search,
            load_stribeck=self.load_stribeck,
            load_stribeck_search=self.load_stribeck_search)


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
    comparison: dict = field(default_factory=dict)
    data_quality: dict = field(default_factory=dict)
    steady_friction_audit: dict = field(default_factory=dict)
    gravity_compensation: dict = field(default_factory=dict)
    # Motions the arm refused. A run with gaps in it is still a run, but the
    # report must not present it as one that measured everything it planned to.
    skipped: list = field(default_factory=list)
    # The fitted regressions themselves, kept out of as_dict because they are
    # arrays, not a document. Diagnostics need them to predict per observation.
    fits: list = field(default_factory=list, repr=False)

    @property
    def complete(self) -> bool:
        return self.aborted is None and bool(self.joints)

    def verdict(self) -> dict:
        """Whether this run may be used, and which joint decided that."""
        gravity_verdict = self.gravity_compensation.get("verdict")
        if self.gravity_compensation.get("available") and gravity_verdict:
            return dict(gravity_verdict)
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
            "comparison": dict(self.comparison),
            "data_quality": dict(self.data_quality),
            "steady_friction_audit": dict(self.steady_friction_audit),
            "gravity_compensation": dict(self.gravity_compensation),
            "skipped": self.skipped,
            "complete": self.complete,
            "verdict": self.verdict(),
        }


class Abort(RuntimeError):
    """Raised when a guard stops the campaign; the arm is left at rest."""


# Amplitude can answer these; it cannot answer a fault word, a disabled drive,
# a bus outside its window, or heat already in the joint.
RECOVERABLE_TRIPS = frozenset({"peak_current", "continuous_current", "speed"})


class DriveTrip(Abort):
    """One motion asked for more than the drive would give."""

    def __init__(self, message: str, joint: int | None = None,
                 kind: str = "") -> None:
        super().__init__(message)
        self.joint = joint
        self.kind = kind

    @property
    def recoverable(self) -> bool:
        return self.joint is not None and self.kind in RECOVERABLE_TRIPS


# Phases A to C are position controlled, so the speed that matters is the
# profile's sustained limit rather than any current-mode figure. The
# acceleration bound is that same ceiling reached from rest in a quarter second.
PROBE_SPEED_FRACTION = 0.5
ACCELERATION_PER_SPEED = 4.0

# How much of its swing a joint gives up after a trip, and how many times one
# motion may be re-planned before the run moves on without it.
TRIP_BACKOFF = 0.7
TRIP_RETRIES = 3

# The sweep speeds are what actually excites friction, so they are derived from
# the speed ceiling rather than fixed figures. Fixed figures meant raising the
# ceiling changed nothing: the sweep kept running at the old speeds and the
# viscous term stayed invisible.
#
# Spaced logarithmically. Friction changes fastest near zero -- the Coulomb
# reversal is a couple of degrees per second wide -- so an evenly spaced ladder
# spends most of its rungs where the curve is already a straight line and none
# where it bends.
FRICTION_SPEED_STEPS = 20
FRICTION_MINIMUM_SPEED_DEG_S = 0.5
VALIDATION_SPEED_FRACTIONS = (0.35, 0.65)

# Constant-speed time each pass must hold, which is what sizes its travel. A
# fixed arc would make a slow pass take a minute and a fast one a fraction of a
# second, for the same three fitted samples.
#
# Held longer when the joint is crawling: the current wanders far more within a
# pass than the pass mean moves between repeats, so the averaging window is
# what limits the measurement, and at half a degree per second four seconds of
# it costs two degrees of travel.
FRICTION_CRUISE_S = 1.0
FRICTION_MAXIMUM_CRUISE_S = 4.0
FRICTION_CRUISE_ARC_DEG = 6.0
# A speed is fitted out of distance travelled, so a crawl needs arc, not time:
# four seconds at 0.02 deg/s is 0.08 deg, which no position fit can turn into a
# speed. Bounded so a rung slower than the ladder cannot stall the run.
FRICTION_MINIMUM_ARC_DEG = 0.4
FRICTION_CRAWL_CRUISE_S = 24.0

# The plant ramps a sweep over a quarter of its nominal duration, so a pass of
# `distance` at `speed` implies 4*speed^2/distance of acceleration. A short
# sweep at high speed is therefore a violent one, and has almost no constant
# speed left in the middle to measure. Amplitude is sized from this.
SWEEP_ACCELERATION_DEG_S2 = 360.0

_STATIC_BOUNDS = {
    "static_poses": (4, 60),
    "static_candidates": (10, 400),
    "settle_samples": (1, 20),
    "friction_amplitude_deg": (5.0, 80.0),
    "fourier_harmonics": (1, 6),
    "fourier_base_frequency_hz": (0.02, 0.3),
    "fourier_duration_s": (5.0, 120.0),
    "fourier_ramp_s": (0.0, 10.0),
    "fourier_attempts": (5, 200),
    "optimal_training_trajectories": (2, 32),
    "optimal_validation_trajectories": (1, 8),
    "optimal_friction_repeats": (1, 5),
    "optimal_friction_postures": (1, 5),
    "sample_rate_hz": (5.0, 100.0),
    "validation_poses": (3, 40),
    "validation_trajectory_s": (4.0, 60.0),
    "position_margin_deg": (3.0, 30.0),
    "gravity_probe_deg": (1.0, 20.0),
    "gravity_validation_poses": (2, 30),
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
    "fourier_attempts", "optimal_training_trajectories",
    "optimal_validation_trajectories", "optimal_friction_repeats",
    "optimal_friction_postures", "validation_poses",
    "gravity_validation_poses", "seed",
})


def coerce_plan_value(name: str, value: float):
    """A plan field's own type, so a panel's number lands as one."""
    return int(round(value)) if name in _INTEGER_FIELDS else float(value)


def _takes_keyword(probe, name: str) -> bool:
    try:
        parameters = inspect.signature(probe).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        entry.kind is inspect.Parameter.VAR_KEYWORD
        for entry in parameters.values())


def _takes_tag(probe) -> bool:
    return _takes_keyword(probe, "tag")


def sweep_speeds(maximum_speed_deg_s: float, fractions) -> tuple[float, ...]:
    """Speeds spread across the ceiling, so raising it reaches new ground."""
    speeds = {round(fraction * maximum_speed_deg_s, 2) for fraction in fractions}
    return tuple(sorted(speed for speed in speeds if speed > 0.0))


def friction_speed_ladder(maximum_speed_deg_s: float,
                          steps: int = FRICTION_SPEED_STEPS,
                          minimum_deg_s: float = FRICTION_MINIMUM_SPEED_DEG_S,
                          ) -> tuple[float, ...]:
    """Logarithmic rungs from a crawl up to the ceiling.

    The bottom is absolute rather than a fraction of the ceiling: what makes a
    low speed worth measuring is the width of the Coulomb reversal, which is a
    property of the joint and does not move when the operator raises the top
    speed.
    """
    top = float(maximum_speed_deg_s)
    if top <= 0.0:
        return ()
    low = min(float(minimum_deg_s), top)
    if steps < 2 or low >= top:
        return (round(top, 3),)
    ratio = (top / low) ** (1.0 / (steps - 1))
    speeds = {round(low * ratio ** step, 3) for step in range(steps - 1)}
    speeds.add(round(top, 3))
    return tuple(sorted(speed for speed in speeds if speed > 0.0))


def friction_cruise_s(speed_deg_s: float) -> float:
    """How long one pass holds its speed. Longer when that is cheap."""
    speed = abs(float(speed_deg_s))
    if speed <= 0.0:
        return FRICTION_MAXIMUM_CRUISE_S
    wanted = FRICTION_CRUISE_ARC_DEG / speed
    held = min(FRICTION_MAXIMUM_CRUISE_S, max(FRICTION_CRUISE_S, wanted))
    return float(min(max(held, FRICTION_MINIMUM_ARC_DEG / speed),
                     FRICTION_CRAWL_CRUISE_S))


def pass_amplitude_deg(speed_deg_s: float, ceiling_deg: float) -> float:
    """Travel for one pass at one speed: both ramps plus the cruise.

    Sized per speed so that every pass costs about the same time and yields the
    same few fitted samples, instead of a slow pass dragging a fixed arc out
    for a minute while a fast one has no constant-speed middle at all.
    """
    speed = max(float(speed_deg_s), 0.0)
    ramps = speed * speed / SWEEP_ACCELERATION_DEG_S2
    return float(min(float(ceiling_deg),
                     ramps + speed * friction_cruise_s(speed)))


def sweep_amplitude_deg(maximum_speed_deg_s: float, requested: float) -> float:
    """Room the fastest pass needs; slower passes take less of it."""
    low, high = _STATIC_BOUNDS["friction_amplitude_deg"]
    needed = pass_amplitude_deg(maximum_speed_deg_s, high)
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
    plan.friction_speeds_deg_s = friction_speed_ladder(plan.maximum_speed_deg_s)
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


def _swept_rows(observations, joint: int) -> list[bool]:
    """Rows where this joint was the one being swept, one pass at one pose."""
    rows = []
    for record in observations:
        swept = False
        if getattr(record, "phase", "") == PHASE_FRICTION:
            for part in (getattr(record, "motion", "") or "").split(":"):
                if part.startswith("j") and part[1:].isdigit():
                    swept = int(part[1:]) == joint
                    break
        rows.append(swept)
    return rows


def _steady_curve_summary(points, low_speed_ceiling: float = 0.5,
                          minimum_peak_a: float = 0.02) -> dict:
    """Classify a controlled same-load curve, not a mixed scatter cloud."""
    grouped = {}
    for speed, friction in points:
        grouped.setdefault(float(speed), []).append(float(friction))
    curve = [(speed, float(np.median(values)))
             for speed, values in sorted(grouped.items())]
    if not curve:
        return {"classical_low_speed_peak": False,
                "low_speed_peak_a": 0.0, "interior_peak_a": 0.0,
                "curve": []}
    terminal = curve[-1][1]
    low = [friction for speed, friction in curve
           if speed <= low_speed_ceiling]
    low_peak = max(low, default=curve[0][1]) - terminal
    interior_peak = max(friction for _speed, friction in curve) - terminal
    return {
        "classical_low_speed_peak": bool(low_peak > minimum_peak_a),
        "low_speed_peak_a": float(low_peak),
        "interior_peak_a": float(interior_peak),
        "curve": [{"speed_deg_s": speed, "friction_a": friction}
                  for speed, friction in curve],
    }


class Campaign:
    """Runs the phases against a plant and regresses the result."""

    def __init__(self, arm: ident.ArmModel, plant: Plant,
                 plan: CampaignPlan | None = None,
                 progress: Callable[[str, dict], None] | None = None,
                 should_stop: Callable[[], bool] | None = None,
                 monitor: EnvelopeMonitor | None = None,
                 pose_admissible: Callable[[np.ndarray], bool] | None = None,
                 clock: Callable[[], float] | None = None,
                 wait_if_paused: Callable[[str, int, int], bool] | None = None) -> None:
        self.arm = arm
        self.plant = plant
        self.plan = plan or CampaignPlan()
        self.progress = progress or (lambda phase, detail: None)
        self.should_stop = should_stop or (lambda: False)
        self.wait_if_paused = wait_if_paused or (
            lambda _phase, _pose, _poses: True)
        self.monitor = monitor
        self.pose_admissible = pose_admissible
        self.clock = clock or time.monotonic
        self.limits = self.plan.design_limits(arm, self._plant_limits())
        self.observations: list[Observation] = []
        self.reports: list[PhaseReport] = []
        self.skipped: list[dict] = []
        # Where the last tour left the arm, so the next one screens its first
        # transit from there rather than from wherever the run began.
        self._left_at: list[float] | None = None
        # Shrunk for a joint that drew more than its drive would give, so the
        # next design asks that joint for less instead of tripping again.
        self.joint_amplitude_scale = np.ones(arm.joint_count)
        self.aborted: str | None = None
        self._started = self.clock()

    def _back_off(self, trip: DriveTrip) -> None:
        joint = int(trip.joint)
        self.joint_amplitude_scale[joint] *= TRIP_BACKOFF
        setter = getattr(self.plant, "set_monitor", None)
        if setter is not None:
            setter(self.monitor)
        self.progress(self.reports[-1].phase if self.reports else "", {
            "backed_off_joint": joint + 1,
            "amplitude_scale": round(
                float(self.joint_amplitude_scale[joint]), 4),
            "reason": str(trip),
        })

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

    def _self_collision_checked(self) -> bool:
        """Whether the scene can actually see the arm, not merely be asked.

        When the link meshes fail to resolve -- a changed package path is
        enough -- the scene keeps answering, and it answers clear to
        everything, because with no link shapes there are no pairs to test.
        Sweeping at home survives that: the pose is neutral and known good.
        Sweeping at postures drawn from the whole range does not, because the
        only thing that made them safe was the screen.
        """
        model = getattr(self.plant, "collision_model", None)
        report = getattr(model, "geometry_report", None)
        return bool(report and report().get("self_collision_checked"))

    def _friction_postures(self, requested: int | None = None) -> tuple[int, str | None]:
        """How many postures may be swept, given what can be verified."""
        wanted = max(1, int(
            self.plan.friction_postures if requested is None else requested))
        if wanted == 1 or self._self_collision_checked():
            return wanted, None
        return 1, ("collision geometry unavailable, so postures away from home "
                   f"could not be screened: swept home only, not {wanted}")

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
                detail = getattr(self.monitor, "last_trip", None) or {}
                raise DriveTrip(trip, joint=detail.get("joint"),
                                kind=detail.get("kind", ""))

        temperatures = sample.get("temperature_c") or []
        for index, value in enumerate(temperatures):
            if value >= self.plan.temperature_ceiling_c:
                raise Abort(
                    f"joint{index + 1} reached {value:.1f} C, ceiling is "
                    f"{self.plan.temperature_ceiling_c:.1f} C")
        speeds = np.abs(np.asarray(sample.get("speed_deg_s") or [0.0]))
        if speeds.size and speeds.max() > self.limits.maximum_speed_deg_s * 2.0:
            raise Abort(f"joint speed {speeds.max():.1f} deg/s exceeded the plan")

    def _stamp_origin(self, stamp: float) -> float:
        """First publisher stamp seen, so recorded times start near zero."""
        if getattr(self, "_first_stamp", None) is None:
            self._first_stamp = stamp
        return self._first_stamp

    def _record(self, phase: str, sample: dict, report: PhaseReport,
                acceleration_deg_s2=None) -> Observation:
        self._guard(sample)
        # The publisher's stamp when the plant supplies one: the moment a frame
        # was appended says nothing useful, because frames arrive in bursts.
        stamp = sample.get("stamp_s")
        moment = (self.clock() - self._started if stamp is None
                  else float(stamp) - self._stamp_origin(float(stamp)))
        observation = Observation.from_sample(
            phase, moment, sample, acceleration_deg_s2)
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

    def _attempt(self, report: PhaseReport, what: str, motion) -> bool:
        """Run one motion, and let the campaign outlive it failing.

        A pass that will not run is one row of several thousand. Ending the run
        over it throws away every hour already spent and asks the operator to
        start again, which is a worse outcome than a gap in the ladder. Failures
        are counted, and enough of them still stops the run: an arm refusing
        everything is not producing a dataset, it is producing a log.

        A drive that asked for more current than it may draw is the same kind
        of problem when amplitude can answer it: the joint gives up some swing
        and the run carries on. A fault word or a disabled drive cannot be
        answered that way and still stops everything.
        """
        try:
            motion()
            return True
        except DriveTrip as trip:
            if not trip.recoverable:
                raise
            self._skipped(report, what, f"drive trip: {trip}")
            self._back_off(trip)
            return False
        except DriveLimitExceeded as failure:
            trip = DriveTrip(str(failure), joint=failure.joint,
                             kind=failure.kind)
            if not trip.recoverable:
                raise
            self._skipped(report, what, f"drive trip: {trip}")
            self._back_off(trip)
            return False
        except MotionFailed as failure:
            self._skipped(report, what, str(failure))
            return False

    def _skipped(self, report: PhaseReport, what: str, reason: str) -> None:
        self.skipped.append({"phase": report.phase, "motion": what,
                             "reason": reason})
        report.detail.setdefault("skipped", []).append(
            {"motion": what, "reason": reason})
        self.progress(report.phase, {"skipped": what, "reason": reason,
                                     "skipped_total": len(self.skipped)})
        if len(self.skipped) > self.plan.skip_budget:
            raise Abort(
                f"{len(self.skipped)} motions failed, over the budget of "
                f"{self.plan.skip_budget}: {reason}")

    def _crossing_deg(self) -> float:
        """How far the gravity probe swings each joint either side of a pose.

        Zero when the plant cannot probe, because then the arm only ever stands
        on the pose and there is no sweep to screen.
        """
        if getattr(self.plant, "probe_pose", None) is None:
            return 0.0
        return float(self.plan.gravity_probe_deg)

    def _standing(self):
        """Where the arm is before the next tour, or None for neutral."""
        if self._left_at is not None:
            return np.asarray(self._left_at, dtype=float)
        return (np.asarray(self.plan.start_deg, dtype=float)
                if self.plan.start_deg else None)

    def _probe(self, pose_deg, tag: str = "",
               phase: str = PHASE_GRAVITY) -> list:
        """Gravity samples at a pose, with friction forced to a known sign.

        A joint held still balances gravity with any value inside its stiction
        band, so a standstill reading is gravity plus an unknowable offset.
        Crossing the pose in both directions costs a few seconds and makes the
        friction contribution cancel between the pair.

        Crossed at every configured speed, because the pair mean still carries
        the viscous term and only a second speed can say how much.
        """
        probe = getattr(self.plant, "probe_pose", None)
        if probe is None:
            samples = [self.plant.hold_pose(pose_deg)]
            samples += [self._dwell(pose_deg)
                        for _ in range(max(1, self.plan.settle_samples) - 1)]
            return samples
        speeds = (tuple(self.plan.gravity_probe_speeds_deg_s)
                  or (self.plan.gravity_probe_speed_deg_s,))
        # ``probe_pose`` is an optional capability found by name, so a plant
        # written before the pair tag existed must still be callable.
        extra = {"tag": tag} if _takes_tag(probe) else {}
        if _takes_keyword(probe, "phase"):
            extra["phase"] = phase
        collected: list = []
        for speed in speeds:
            frames = probe(pose_deg, self.plan.gravity_probe_deg, speed,
                           **extra)
            # Ramp-in and ramp-out pass through near-zero speed, where friction
            # has no determined sign again; keeping those frames would put the
            # very contamination this probe removes straight back into the fit.
            floor = PROBE_SPEED_FRACTION * speed
            collected.extend(
                frame for frame in frames
                if max(abs(v) for v in frame["speed_deg_s"]) >= floor)
        return collected

    def _dwell(self, pose_deg) -> dict:
        """Take another reading at rest, re-commanding only if the plant needs it."""
        dwell = getattr(self.plant, "dwell", None)
        return self.plant.hold_pose(pose_deg) if dwell is None else dwell(pose_deg)

    def _raw_frame_checkpoint(self):
        checkpoint = getattr(self.plant, "raw_frame_checkpoint", None)
        return checkpoint() if checkpoint is not None else None

    def _rollback_pose(self, report: PhaseReport, observations: int,
                       report_state: tuple, raw_checkpoint) -> None:
        """Remove every fitted and raw sample from an unfinished pose."""
        del self.observations[observations:]
        (report.observations, report.peak_temperature_c,
         report.peak_speed_deg_s, report.peak_current_a) = report_state
        rollback = getattr(self.plant, "rollback_raw_frames", None)
        if rollback is not None and raw_checkpoint is not None:
            rollback(raw_checkpoint)

    def _measure_gravity_poses(self, report: PhaseReport, phase: str,
                               poses, tag_prefix: str, label: str) -> None:
        """Measure complete poses transactionally, retrying one after pause."""
        total = len(poses)
        completed: list[int] = []
        index = 0
        while index < total:
            target = np.asarray(poses[index], dtype=float)
            pose_deg = [round(float(value), 3) for value in target]
            target_detail = {
                "target_pose": index + 1,
                "poses": total,
                "pose_deg": pose_deg,
                "completed_pose_indices": list(completed),
            }
            self.progress(phase, target_detail)
            if not self.wait_if_paused(phase, index + 1, total):
                raise Abort("operator stop")

            observation_count = len(self.observations)
            report_state = (report.observations, report.peak_temperature_c,
                            report.peak_speed_deg_s, report.peak_current_a)
            raw_checkpoint = self._raw_frame_checkpoint()
            try:
                accepted = self._attempt(
                    report, f"{label} {index + 1}",
                    lambda target=target, index=index: [
                        self._record(phase, sample, report)
                        for sample in self._probe(
                            target, f"{tag_prefix}:p{index + 1}", phase)])
            except MotionPaused:
                self._rollback_pose(report, observation_count, report_state,
                                    raw_checkpoint)
                self.progress(phase, {
                    **target_detail,
                    "interrupted_pose": index + 1,
                    "observations": len(self.observations),
                })
                if not self.wait_if_paused(phase, index + 1, total):
                    raise Abort("operator stop")
                continue
            except BaseException:
                self._rollback_pose(report, observation_count, report_state,
                                    raw_checkpoint)
                raise

            if not accepted:
                self._rollback_pose(report, observation_count, report_state,
                                    raw_checkpoint)
                self.progress(phase, {
                    **target_detail,
                    "skipped_pose": index + 1,
                    "observations": len(self.observations),
                })
                index += 1
                continue

            completed.append(index + 1)
            self.progress(phase, {
                "pose": index + 1,
                "completed_pose": index + 1,
                "completed_pose_indices": list(completed),
                "target_pose": index + 1,
                "poses": total,
                "pose_deg": pose_deg,
            })
            index += 1

    def _execution_trajectory(self, trajectory):
        return excitation.ramp_fourier_trajectory(
            trajectory, self.plan.fourier_ramp_s)

    def run_gravity(self) -> PhaseReport:
        report, start = self._open(PHASE_GRAVITY)
        try:
            design = excitation.design_static_poses(
                self.arm, self.limits, count=self.plan.static_poses,
                candidates=self.plan.static_candidates, seed=self.plan.seed,
                collision_free=self._collision_free(),
                start_deg=self._standing(),
                crossing_deg=self._crossing_deg())
            report.detail = design.as_dict()
            report.detail["poses"] = len(design.poses_deg)
            self._left_at = design.final_deg or self._left_at
            # Published before the first move so a viewer has the whole tour
            # rather than one pose at a time.
            self.progress(PHASE_GRAVITY, {"designed": design.poses_deg})
            self._measure_gravity_poses(
                report, PHASE_GRAVITY, design.poses_deg, "gravity", "pose")
        finally:
            report.duration_s = self.clock() - start
        return report

    def run_friction(self) -> PhaseReport:
        report, start = self._open(PHASE_FRICTION)
        try:
            postures, downgraded = self._friction_postures()
            sweeps = excitation.design_friction_sweeps(
                self.arm, self.limits,
                amplitude_deg=self.plan.friction_amplitude_deg,
                speeds_deg_s=self.plan.friction_speeds_deg_s,
                postures=postures,
                collision_free=self._collision_free(),
                seed=self.plan.seed)
            report.detail = {"sweeps": [sweep.as_dict() for sweep in sweeps],
                             "postures": postures}
            if downgraded:
                report.detail["downgraded"] = downgraded
            for index, sweep in enumerate(sweeps):
                # The design hands back the room available; each speed takes
                # only the part of it that speed needs, centred on the same
                # pose so every pass measures the same gravity term.
                centre = (np.asarray(sweep.start_deg, dtype=float)[sweep.joint]
                          + sweep.amplitude_deg / 2.0)
                for speed in sweep.speeds_deg_s:
                    amplitude = pass_amplitude_deg(speed, sweep.amplitude_deg)
                    for _repeat in range(max(1, self.plan.friction_repeats)):
                        for distance in (amplitude, -amplitude):
                            origin = np.asarray(sweep.start_deg, dtype=float)
                            origin = origin.copy()
                            origin[sweep.joint] = centre - distance / 2.0
                            self._attempt(
                                report,
                                f"j{sweep.joint + 1} {speed:g} deg/s "
                                f"{'+' if distance > 0 else '-'}",
                                lambda joint=sweep.joint, origin=origin,
                                distance=distance, speed=speed: [
                                    self._record(PHASE_FRICTION, sample, report)
                                    for sample in self.plant.traverse(
                                        joint, origin, distance, speed)])
                self.progress(PHASE_FRICTION, {
                    "joint": sweep.joint + 1,
                    "sweep": index + 1, "sweeps": len(sweeps)})
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

            trajectory = self._execution_trajectory(trajectory)
            report.detail = {"trajectory": trajectory.as_dict()}
            emitted = 0

            def follow():
                nonlocal emitted
                for sample in self.plant.track(
                        trajectory, self.plan.sample_rate_hz):
                    self._record(
                        PHASE_INERTIA, sample, report,
                        sample.get("acceleration_deg_s2"))
                    emitted += 1
                    if emitted % 20 == 0:
                        self.progress(PHASE_INERTIA, {"samples": emitted})

            self._attempt(report, "fourier trajectory", follow)
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
                if self._attempt(
                        report, f"validation pose {accepted + 1}",
                        lambda pose=pose: self._record(
                            PHASE_VALIDATION, self.plant.hold_pose(pose),
                            report)):
                    accepted += 1
            report.detail = {"poses": accepted, "candidates": attempts}

            report.detail["speeds_deg_s"] = list(self._validation_speeds())
            # Sweep from a different home than training, so validation differs in
            # configuration as well as in speed.
            home = self._validation_home(rng, admissible)
            report.detail["sweep_home_deg"] = [round(v, 3) for v in home]
            swept = 0

            def sweep_once(joint, origin, arc, speed):
                nonlocal swept
                for frame in self.plant.traverse(joint, origin, arc, speed):
                    self._record(PHASE_VALIDATION, frame, report)
                    swept += 1

            for sweep in excitation.design_friction_sweeps(
                    self.arm, self.limits,
                    amplitude_deg=self.plan.friction_amplitude_deg,
                    speeds_deg_s=self._validation_speeds(),
                    home_deg=home):
                for speed in sweep.speeds_deg_s:
                    origin = np.asarray(sweep.start_deg, dtype=float)
                    self._attempt(
                        report,
                        f"validation j{sweep.joint + 1} {speed:g} deg/s",
                        lambda joint=sweep.joint, origin=origin,
                        arc=sweep.amplitude_deg, speed=speed: sweep_once(
                            joint, origin, arc, speed))
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
                trajectory = self._execution_trajectory(trajectory)
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

    def _main_training_observations(self, usable):
        """Rows that determine the predictor; subclasses may own extra phases."""
        return [record for record in usable
                if record.phase != PHASE_VALIDATION]

    def fit(self) -> CampaignResult:
        """Regress on phases A-C and score on phase D."""
        acceleration_ceiling = 2.0 * self.limits.maximum_acceleration_deg_s2
        excluded = [
            record for record in self.observations
            if record.acceleration_deg_s2
            and max(abs(value) for value in record.acceleration_deg_s2)
            > acceleration_ceiling]
        usable = [record for record in self.observations
                  if record not in excluded]
        result = CampaignResult(
            plan=self.plan.as_dict(),
            phases=[report.as_dict() for report in self.reports],
            data_quality={
                "observations_recorded": len(self.observations),
                "observations_used": len(usable),
                "excluded_acceleration_outliers": len(excluded),
                "acceleration_exclusion_deg_s2": acceleration_ceiling,
            })

        training = self._main_training_observations(usable)
        training_ids = {id(record) for record in training}
        auxiliary = [record for record in usable
                 if record.phase != PHASE_VALIDATION
                 and id(record) not in training_ids]
        holdout = [record for record in usable
                   if record.phase == PHASE_VALIDATION]
        if not training:
            result.aborted = "no training observations"
            return result

        train_rows = self._rows(training)
        holdout_rows = self._rows(holdout) if holdout else None
        result.validation_samples = len(holdout)
        result.data_quality["main_training_observations"] = len(training)
        result.data_quality["auxiliary_friction_observations"] = len(auxiliary)
        selection_groups = [
            f"{record.phase}:{record.motion or 'untagged'}"
            for record in training]
        result.data_quality["optional_selection_groups"] = len(
            set(selection_groups))

        for joint in range(self.arm.joint_count):
            fit = ident.fit_joint(
                joint, train_rows[0],
                [v[joint] for v in train_rows[1]],
                [c[joint] for c in train_rows[2]],
                maximum_condition=MAXIMUM_CONDITION,
                components=self.plan.model_components(),
                transition_rows=_swept_rows(training, joint),
                selection_groups=selection_groups,
                seed=self.plan.seed)
            entry = fit.as_dict()
            entry["peak_measured_effort"] = max(
                abs(float(record.current_a[joint])) for record in usable)
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

    def _run_phases(self, phases) -> CampaignResult:
        """Run each phase in turn and fit whatever was measured.

        Hours of measurement are not worth less because the last minute of it
        failed, so a phase that raises stops the run but never discards the
        data already taken.
        """
        self.aborted = None
        for phase in phases:
            try:
                report = phase()
            except Abort as stop:
                self.aborted = str(stop)
                self.reports[-1].aborted = self.aborted
                break
            except Exception as error:  # noqa: BLE001 - see the docstring
                self.aborted = f"{self.reports[-1].phase}: {error}"
                self.reports[-1].aborted = str(error)
                break
            if report.aborted:
                self.aborted = f"{report.phase}: {report.aborted}"
                break
        result = self.fit()
        if self.aborted:
            result.aborted = self.aborted
        result.skipped = list(self.skipped)
        return result

    def run(self) -> CampaignResult:
        """All four phases; anything that stops one keeps what it measured."""
        return self._run_phases((self.run_gravity, self.run_friction,
                                 self.run_inertia, self.run_validation))


class GravityCampaign(Campaign):
    """Gravity terms only, from bidirectional passes through chosen poses.

    Standing still on a pose leaves static friction free to take any value
    inside its band, so every pose is crossed both ways, at every configured
    speed. The mean of a pair is gravity plus the drive's own offset, the half
    difference is Coulomb friction at that pose, and the change between two
    speeds is the viscous term.

    These passes carry no acceleration, so the inertia tensor columns are
    identically zero and the base-parameter reduction drops them. What this
    yields holds the arm up; it does not move it fast.
    """

    def _main_training_observations(self, usable):
        return [record for record in usable if record.phase == PHASE_GRAVITY]

    def _pair_averaged_gravity_rows(self, phase: str):
        """One static regressor/current row per complete bidirectional pose."""
        prefix = "gravity" if phase == PHASE_GRAVITY else "gravity_check"
        pattern = re.compile(
            rf"^{re.escape(prefix)}:p(?P<pose>\d+):"
            r"(?P<speed>[^:]+):(?P<direction>[+-])$")
        grouped: dict[tuple[int, float, str], list[Observation]] = {}
        malformed = []
        for record in self.observations:
            if record.phase != phase:
                continue
            match = pattern.match(record.motion or "")
            if match is None:
                malformed.append({"motion": record.motion,
                                  "reason": "tag does not match the gravity schema"})
                continue
            try:
                speed = float(match.group("speed"))
            except (TypeError, ValueError):
                malformed.append({"motion": record.motion,
                                  "reason": "probe speed is not numeric"})
                continue
            if not math.isfinite(speed) or speed <= 0.0:
                malformed.append({"motion": record.motion,
                                  "reason": "probe speed is not finite and positive"})
                continue
            key = (int(match.group("pose")), round(speed, 6),
                   match.group("direction"))
            grouped.setdefault(key, []).append(record)

        configured = tuple(round(float(speed), 6) for speed in (
            self.plan.gravity_probe_speeds_deg_s
            or (self.plan.gravity_probe_speed_deg_s,)))
        expected = tuple(dict.fromkeys(configured))
        configuration_errors = ([] if len(expected) == len(configured) else [{
            "reason": "configured probe speeds contain duplicates",
            "configured": list(configured),
        }])
        report = next((entry for entry in self.reports if entry.phase == phase),
                      None)
        designed = int((report.detail if report else {}).get("poses") or 0)
        if not designed:
            designed = (self.plan.static_poses if phase == PHASE_GRAVITY
                        else self.plan.gravity_validation_poses)
        poses = list(range(1, designed + 1))
        unexpected = [{
            "pose": pose, "speed_deg_s": speed, "direction": direction,
            "observations": len(records),
        } for (pose, speed, direction), records in sorted(grouped.items())
            if speed not in expected or pose not in poses]
        regressors, currents, accepted, incomplete = [], [], [], []
        consistency = [[] for _ in range(self.arm.joint_count)]
        for pose in poses:
            pair_regressors, pair_currents = [], []
            errors = []
            for speed in expected:
                positive = grouped.get((pose, speed, "+"), [])
                negative = grouped.get((pose, speed, "-"), [])
                if len(positive) != 1 or len(negative) != 1:
                    errors.append({"speed_deg_s": speed,
                                   "positive": len(positive),
                                   "negative": len(negative)})
                    continue
                pair = (positive[0], negative[0])
                pair_regressors.append(np.mean([
                    self.arm.static_regressor(record.position_deg)
                    for record in pair], axis=0))
                pair_currents.append(np.mean([
                    record.current_a for record in pair], axis=0))
            if errors or len(pair_currents) != len(expected):
                incomplete.append({"pose": pose, "missing_or_duplicate": errors})
                continue
            pair_currents_array = np.asarray(pair_currents, dtype=float)
            if pair_currents_array.shape[0] > 1:
                spread = np.ptp(pair_currents_array, axis=0)
                for joint, value in enumerate(spread):
                    consistency[joint].append(float(value))
            regressors.append(np.mean(pair_regressors, axis=0))
            currents.append(np.mean(pair_currents_array, axis=0))
            accepted.append(pose)
        return {
            "regressors": regressors,
            "currents": np.asarray(currents, dtype=float),
            "poses": accepted,
            "incomplete": incomplete,
            "malformed": malformed,
            "unexpected": unexpected,
            "configuration_errors": configuration_errors,
            "speed_consistency": consistency,
            "expected_speeds": list(expected),
        }

    def _fit_gravity_compensation(self) -> dict:
        """Fit gravity+offset after cancelling odd friction by direction."""
        training = self._pair_averaged_gravity_rows(PHASE_GRAVITY)
        validation = self._pair_averaged_gravity_rows(PHASE_VALIDATION)
        audit_errors = [
            *training["incomplete"], *validation["incomplete"],
            *training["malformed"], *validation["malformed"],
            *training["unexpected"], *validation["unexpected"],
            *training["configuration_errors"],
            *validation["configuration_errors"],
        ]
        if audit_errors:
            return {
                "available": False,
                "reason": "gravity direction/speed pairing audit failed",
                "pairing_audit": {
                    "incomplete_training_poses": training["incomplete"],
                    "incomplete_validation_poses": validation["incomplete"],
                    "malformed_training_tags": training["malformed"],
                    "malformed_validation_tags": validation["malformed"],
                    "unexpected_training_observations": training["unexpected"],
                    "unexpected_validation_observations": validation["unexpected"],
                    "configuration_errors": [
                        *training["configuration_errors"],
                        *validation["configuration_errors"],
                    ],
                },
            }
        if not training["regressors"]:
            return {"available": False,
                    "reason": "no complete bidirectional training poses"}
        if not validation["regressors"]:
            return {"available": False,
                    "reason": "no complete bidirectional validation poses"}

        components = ModelComponents(friction=False, offset=True)
        joints, validation_rms = [], []
        peak_effort = float(max(
            np.max(np.abs(training["currents"])),
            np.max(np.abs(validation["currents"]))))
        for joint in range(self.arm.joint_count):
            try:
                fit = ident.fit_joint(
                    joint, training["regressors"],
                    [0.0] * len(training["regressors"]),
                    training["currents"][:, joint].tolist(),
                    maximum_condition=MAXIMUM_CONDITION,
                    components=components, seed=self.plan.seed)
            except ValueError as error:
                return {"available": False,
                        "reason": f"joint {joint + 1}: {error}"}
            predicted = np.asarray([
                ident.predict_joint(fit, regressor, 0.0)
                for regressor in validation["regressors"]])
            truth = validation["currents"][:, joint]
            error = float(np.sqrt(np.mean((predicted - truth) ** 2)))
            entry = fit.as_dict()
            entry["external_validation_rms"] = error
            entry["validation_rms_a"] = error
            entry["peak_measured_effort"] = float(max(
                np.max(np.abs(training["currents"][:, joint])),
                np.max(np.abs(validation["currents"][:, joint]))))
            spread = training["speed_consistency"][joint]
            entry["speed_pair_consistency_rms"] = (
                float(np.sqrt(np.mean(np.square(spread)))) if spread else 0.0)
            joints.append(entry)
            validation_rms.append(error)
        verdict_joints = []
        for joint, entry in enumerate(joints):
            state, reason = judge_joint(entry, peak_effort)
            verdict_joints.append({"joint": joint + 1,
                                   "state": state, "reason": reason})
        states = {entry["state"] for entry in verdict_joints}
        overall = ("fail" if "fail" in states else
                   "warn" if states & {"warn", "unknown"} else "pass")
        return {
            "available": True,
            "method": "bidirectional_pair_mean_static_regressor_plus_offset",
            "friction_cancellation": (
                "average positive and negative crossings at each speed, "
                "then average speed pairs per pose"),
            "training_phase": PHASE_GRAVITY,
            "validation_phase": PHASE_VALIDATION,
            "probe_speeds_deg_s": training["expected_speeds"],
            "training_pose_ids": training["poses"],
            "validation_pose_ids": validation["poses"],
            "training_poses": len(training["poses"]),
            "validation_poses": len(validation["poses"]),
            "incomplete_training_poses": training["incomplete"],
            "incomplete_validation_poses": validation["incomplete"],
            "malformed_training_tags": training["malformed"],
            "malformed_validation_tags": validation["malformed"],
            "unexpected_training_observations": training["unexpected"],
            "unexpected_validation_observations": validation["unexpected"],
            "configuration_errors": [
                *training["configuration_errors"],
                *validation["configuration_errors"],
            ],
            "validation_rms": validation_rms,
            "joints": joints,
            "verdict": {
                "state": overall,
                "joints": verdict_joints,
                "failed_joints": [entry["joint"] for entry in verdict_joints
                                   if entry["state"] == "fail"],
            },
        }

    def fit(self) -> CampaignResult:
        result = super().fit()
        result.gravity_compensation = self._fit_gravity_compensation()
        return result

    def run_gravity_validation(self) -> PhaseReport:
        """Poses the fit never saw, crossed exactly the way training was.

        Scored the same way it was trained, because a gravity model that is
        only checked where it was fitted has not been checked.
        """
        report, start = self._open(PHASE_VALIDATION)
        try:
            design = excitation.design_static_poses(
                self.arm, self.limits,
                count=self.plan.gravity_validation_poses,
                candidates=self.plan.static_candidates,
                seed=self.plan.seed + 7717,
                collision_free=self._collision_free(),
                start_deg=self._standing(),
                crossing_deg=self._crossing_deg())
            report.detail = design.as_dict()
            report.detail["poses"] = len(design.poses_deg)
            self._left_at = design.final_deg or self._left_at
            if not design.poses_deg:
                report.aborted = "no feasible validation pose"
                return report
            self.progress(PHASE_VALIDATION, {"designed": design.poses_deg})
            self._measure_gravity_poses(
                report, PHASE_VALIDATION, design.poses_deg,
                "gravity_check", "validation pose")
        finally:
            report.duration_s = self.clock() - start
        return report

    def run(self) -> CampaignResult:
        return self._run_phases((self.run_gravity, self.run_gravity_validation))


class OptimalExcitationCampaign(Campaign):
    """Identify from one designed set of complementary excitation trajectories.

    Every training trajectory is selected against the regressors of the ones
    already accepted, so the added rows carry new parameter directions rather
    than merely repeating one attractive motion. Short load-conditioned,
    constant-speed subtrajectories supply the steady low-speed rows a friction
    law needs; Fourier trajectories supply the inertial rows. Validation uses
    separate Fourier seeds and frequencies and is never included in the fit.
    """

    _TRAINING_FREQUENCY_SCALES = (0.75, 0.9, 1.0, 1.1, 1.25)
    _VALIDATION_FREQUENCY_SCALES = (1.3, 1.45, 1.6)
    _DESIGN_SAMPLES = 40
    _FRICTION_MOTION = re.compile(
        r"optimal_friction:j(\d+):([0-9.]+):([+-]):s(\d+):r(\d+)")

    def _main_training_observations(self, usable):
        """Steady B data diagnoses friction; C alone owns the dynamic model."""
        return [record for record in usable
                if record.phase == PHASE_INERTIA]

    def _steady_friction_audit(self, fits) -> dict:
        """Controlled same-load curves, evaluated against the frozen rigid fit."""
        swept = [record for record in self.observations
                 if record.phase == PHASE_FRICTION]
        joints = []
        for joint, fit in enumerate(fits):
            levels = {}
            for record in swept:
                match = self._FRICTION_MOTION.fullmatch(record.motion or "")
                if match is None or int(match.group(1)) != joint:
                    continue
                speed = float(match.group(2))
                direction = 1.0 if match.group(3) == "+" else -1.0
                level = int(match.group(4))
                regressor = self.arm.torque_regressor(
                    record.position_deg, record.velocity_deg_s,
                    record.acceleration_deg_s2)
                acceleration = float(record.acceleration_deg_s2[joint])
                velocity = float(record.velocity_deg_s[joint])
                rigid = ident.predict_joint(
                    fit, regressor, velocity, include_friction=False,
                    acceleration=acceleration)
                levels.setdefault(level, []).append({
                    "nominal_speed_deg_s": speed,
                    "actual_speed_deg_s": abs(velocity),
                    "acceleration_deg_s2": abs(acceleration),
                    "signed_friction_a": direction * (
                        float(record.current_a[joint]) - rigid),
                    "load_a": abs(rigid),
                })
            summaries = []
            for level, rows in sorted(levels.items()):
                summary = _steady_curve_summary([
                    (row["nominal_speed_deg_s"], row["signed_friction_a"])
                    for row in rows])
                summary.update({
                    "level": level,
                    "observations": len(rows),
                    "load_median_a": float(np.median(
                        [row["load_a"] for row in rows])),
                    "actual_speed_error_median_deg_s": float(np.median([
                        abs(row["actual_speed_deg_s"]
                            - row["nominal_speed_deg_s"])
                        for row in rows])),
                    "acceleration_median_deg_s2": float(np.median([
                        row["acceleration_deg_s2"] for row in rows])),
                })
                summaries.append(summary)
            joints.append({
                "joint": joint,
                "observations": sum(level["observations"]
                                    for level in summaries),
                "load_levels": summaries,
                "classical_low_speed_peak": any(
                    level["classical_low_speed_peak"] for level in summaries),
                "maximum_low_speed_peak_a": max(
                    (level["low_speed_peak_a"] for level in summaries),
                    default=0.0),
                "maximum_interior_peak_a": max(
                    (level["interior_peak_a"] for level in summaries),
                    default=0.0),
            })
        return {
            "available": bool(swept and fits),
            "method": "same_load_direction_paired_pass_medians",
            "used_by_dynamic_fit": False,
            "low_speed_ceiling_deg_s": 0.5,
            "minimum_peak_a": 0.02,
            "observations": len(swept),
            "joints": joints,
        }

    def fit(self) -> CampaignResult:
        result = super().fit()
        if result.fits:
            result.steady_friction_audit = self._steady_friction_audit(
                result.fits)
        return result

    def reuse_low_speed_friction(self, observations, source: str,
                                 phase: dict) -> None:
        """Seed a recovery run from a completed, compatible low-speed phase."""
        if self.observations or self.reports:
            raise ValueError("low-speed data can only be reused before the run")
        records = list(observations)
        if not records or any(record.phase != PHASE_FRICTION
                              for record in records):
            raise ValueError("reused data must contain friction observations")
        detail = dict(phase.get("detail") or {})
        detail.update({
            "reused": True,
            "reused_from": str(source),
            "source_observations": len(records),
        })
        self.observations.extend(records)
        self.reports.append(PhaseReport(
            phase=PHASE_FRICTION,
            observations=len(records),
            duration_s=float(phase.get("duration_s") or 0.0),
            peak_temperature_c=float(phase.get("peak_temperature_c") or 0.0),
            peak_speed_deg_s=float(phase.get("peak_speed_deg_s") or 0.0),
            peak_current_a=float(phase.get("peak_current_a") or 0.0),
            detail=detail))

    def _planned_rows(self, trajectory) -> list[np.ndarray]:
        rows = []
        for step in range(self._DESIGN_SAMPLES):
            time_s = trajectory.duration_s * step / self._DESIGN_SAMPLES
            position, velocity, acceleration = trajectory.sample(time_s)
            rows.append(self.arm.torque_regressor(
                position, velocity, acceleration))
        return rows

    def _current_pose(self, fallback) -> np.ndarray:
        sample = getattr(self.plant, "sample", None)
        if sample is not None:
            try:
                return np.asarray(sample()["position_deg"], dtype=float)
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        return np.asarray(fallback, dtype=float)

    def run_low_speed_friction(self) -> PhaseReport:
        """Collect steady low-speed rows at collision-screened load postures."""
        report, start = self._open(PHASE_FRICTION)
        speeds = tuple(sorted({
            float(speed) for speed in self.plan.optimal_friction_speeds_deg_s
            if 0.0 < float(speed) <= self.plan.maximum_speed_deg_s
        }))
        postures, downgraded = self._friction_postures(
            self.plan.optimal_friction_postures)
        repeats = max(1, int(self.plan.optimal_friction_repeats))
        room = max(
            (pass_amplitude_deg(speed, self.plan.friction_amplitude_deg)
             for speed in speeds), default=0.0)
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=room,
            speeds_deg_s=speeds, postures=postures,
            collision_free=self._collision_free(), seed=self.plan.seed + 7001)
        report.detail = {
            "purpose": "steady_low_speed_load_conditioned",
            "speeds_deg_s": list(speeds),
            "repeats_per_direction": repeats,
            "postures": postures,
            "planned_passes": len(sweeps) * len(speeds) * repeats * 2,
            "completed_passes": 0,
            "sweeps": [sweep.as_dict() for sweep in sweeps],
        }
        if downgraded:
            report.detail["downgraded"] = downgraded
        try:
            for sweep_index, sweep in enumerate(sweeps):
                centre = (np.asarray(sweep.start_deg, dtype=float)[sweep.joint]
                          + sweep.amplitude_deg / 2.0)
                for speed in speeds:
                    amplitude = pass_amplitude_deg(speed, sweep.amplitude_deg)
                    for repeat in range(repeats):
                        for direction in (1.0, -1.0):
                            distance = direction * amplitude
                            origin = np.asarray(sweep.start_deg, dtype=float).copy()
                            origin[sweep.joint] = centre - distance / 2.0
                            tag = (
                                f"optimal_friction:j{sweep.joint}:{speed:g}:"
                                f"{'+' if direction > 0.0 else '-'}:"
                                f"s{sweep_index + 1}:r{repeat + 1}")

                            def follow(joint=sweep.joint, origin=origin,
                                       distance=distance, speed=speed,
                                       tag=tag):
                                for frame in self.plant.traverse(
                                        joint, origin, distance, speed):
                                    sample = dict(frame)
                                    sample["motion"] = tag
                                    self._record(PHASE_FRICTION, sample, report)

                            if self._attempt(report, tag, follow):
                                report.detail["completed_passes"] += 1
                            self.progress(PHASE_FRICTION, {
                                "pass": report.detail["completed_passes"],
                                "passes": report.detail["planned_passes"],
                                "joint": sweep.joint + 1,
                                "speed_deg_s": speed,
                            })
            if not report.detail["completed_passes"]:
                report.aborted = "no feasible low-speed friction trajectory"
        finally:
            report.duration_s = self.clock() - start
        return report

    def _run_trajectory_set(self, phase: str, count: int, scales,
                            seed_base: int) -> PhaseReport:
        report, start = self._open(phase)
        selected_rows: list[np.ndarray] = []
        current = self._current_pose(np.zeros(self.arm.joint_count))
        report.detail = {
            "requested_trajectories": int(count),
            "selection": "cumulative_regressor_condition",
            "trajectories": [],
        }
        try:
            for index in range(max(0, int(count))):
                scale = scales[index % len(scales)]
                frequency = min(
                    0.3, self.plan.fourier_base_frequency_hz * scale)
                # Re-planned rather than abandoned: a trip narrows the joint
                # that drew too much, so the next design asks it for less.
                for retry in range(TRIP_RETRIES + 1):
                    trajectory = excitation.design_fourier_trajectory(
                        self.arm, self.limits,
                        harmonics=self.plan.fourier_harmonics,
                        base_frequency_hz=frequency,
                        duration_s=self.plan.fourier_duration_s,
                        attempts=self.plan.fourier_attempts,
                        seed=seed_base + 997 * index + 31 * retry,
                        collision_free=self._collision_free(),
                        conditioning_rows=selected_rows,
                        randomize_centre=True,
                        start_deg=current,
                        minimum_amplitude_fraction=(
                            self.plan.optimal_fourier_amplitude_fraction),
                        joint_amplitude_scale=self.joint_amplitude_scale)
                    if trajectory is None:
                        self.skipped.append({
                            "phase": phase,
                            "motion": f"optimal trajectory {index + 1}",
                            "reason": "no feasible trajectory within the limits",
                        })
                        break

                    execution = self._execution_trajectory(trajectory)

                    emitted = 0

                    def follow():
                        nonlocal emitted
                        for frame in self.plant.track(
                                execution, self.plan.sample_rate_hz):
                            sample = dict(frame)
                            sample["motion"] = (
                                f"optimal:{'validation' if phase == PHASE_VALIDATION else 'training'}:"
                                f"{index + 1}")
                            self._record(
                                phase, sample, report,
                                sample.get("acceleration_deg_s2"))
                            emitted += 1
                            if emitted % 20 == 0:
                                self.progress(phase, {
                                    "trajectory": index + 1,
                                    "trajectories": count,
                                    "trajectory_samples": emitted,
                                })

                    completed = self._attempt(
                        report, f"optimal trajectory {index + 1}", follow)
                    current = self._current_pose(
                        execution.sample(execution.duration_s)[0]
                        if completed else current)
                    if not completed:
                        continue
                    selected_rows.extend(self._planned_rows(execution))
                    detail = execution.as_dict()
                    detail.update({
                        "index": index + 1,
                        "observations": emitted,
                        "retries": retry,
                        "cumulative_condition": round(
                            ident.stacked_condition_number(selected_rows), 6),
                    })
                    report.detail["trajectories"].append(detail)
                    self.progress(phase, {
                        "trajectory": index + 1,
                        "trajectories": count,
                        "trajectory_samples": emitted,
                    })
                    break
            if not report.detail["trajectories"]:
                report.aborted = "no feasible optimal excitation trajectory"
        finally:
            report.detail["completed_trajectories"] = len(
                report.detail["trajectories"])
            report.duration_s = self.clock() - start
        return report

    def run_training(self) -> PhaseReport:
        return self._run_trajectory_set(
            PHASE_INERTIA, self.plan.optimal_training_trajectories,
            self._TRAINING_FREQUENCY_SCALES, self.plan.seed + 10001)

    def run_optimal_validation(self) -> PhaseReport:
        return self._run_trajectory_set(
            PHASE_VALIDATION, self.plan.optimal_validation_trajectories,
            self._VALIDATION_FREQUENCY_SCALES, self.plan.seed + 50021)

    def run(self) -> CampaignResult:
        """Low-speed and Fourier training, then independent Fourier validation."""
        phases = ([self.run_low_speed_friction]
                  if not self.reports else [])
        phases.extend((self.run_training, self.run_optimal_validation))
        return self._run_phases(phases)
