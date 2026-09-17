"""Dashboard state: what is connected, what is planned, what was measured.

Holds no ROS. The node supplies a bridge object; everything else here is plain
Python so the whole surface can be exercised in tests without a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
from pathlib import Path
import csv
import hashlib
from importlib import metadata
import json
import math
import os
import platform
import re
import secrets
import signal
import subprocess
import tempfile
import threading
import time
import traceback
import xml.etree.ElementTree as ElementTree

import numpy as np

from .. import autoprofile
from .. import campaign as campaign_module
from .. import excitation, identification as ident
from .. import loadsweep as loadsweep_module
from .. import loadsweep_report
from .. import model as model_module
from .. import report as report_module
from ..arm_identity import ArmBinding, ArmIdentity
from ..interfaces import CommandSpec, TelemetrySpec
from ..loadsweep_run import LoadSweepRun
from ..model import ModelComponents
from .. import obstacles as obstacles_module
from ..obstacles import Obstacle, ObstacleScene
from ..profile import RobotProfile
from ..system_config import (
    SystemConfig, checked_value, configured_range, default_system_config,
    plan_defaults, resolved_controls, system_default, write_system_config_snapshot,
)
from . import hold_plan as hold_plan_module

IDLE, RUNNING, PAUSED, JOGGING = "idle", "running", "paused", "jogging"
GRAVITY_HOLD_TEST = "gravity_hold_test"
GRAVITY_DRAG_TEST = "gravity_drag_test"
GRAVITY_TEST_ACKNOWLEDGEMENT = "I_AM_HOLDING_ARM_AND_ESTOP_READY"
# An envelope typed into the panel is an operator's envelope, so it is labelled
# and guarded exactly as a hand-written file is.
EDITED_SOURCE = "<edited in the dashboard>"
PROFILE_FILE_NAME = re.compile(r"[A-Za-z0-9_.-]+\.yaml")
CONFIG_FILE_NAME = re.compile(r"[A-Za-z0-9_.-]+\.json")
DEFAULT_CONFIG_FILE = system_default("storage", "cell_config_filename")
# How far a joint this dashboard does not drive may move away from where the
# screen was built around it before every pose it cleared is suspect. Small,
# because it is a distance at the wrist that matters and leverage is long.
SCREEN_DRIFT_DEG = system_default("runtime", "screen_drift_deg")
# The starting pose is rounded to this before it enters a design or an arming
# signature, so a servo holding still reads the same number twice.
STANDING_QUANTUM_DEG = system_default("runtime", "standing_quantum_deg")
# Sample points the canvas may ask for along one transit. A ceiling because
# the request comes off a web surface listening on every interface.
MAXIMUM_ANIMATION_STEPS = system_default("runtime", "maximum_animation_steps")
# A ceiling nobody supplied is infinite, and JSON has no way to say so.
UNBOUNDED_LIMITS = ("continuous_current_a", "peak_current_a")
PINOCCHIO_INERTIAL_TERMS = (
    "mass", "first_moment_x", "first_moment_y", "first_moment_z",
    "inertia_xx", "inertia_xy", "inertia_yy", "inertia_xz",
    "inertia_yz", "inertia_zz",
)


def _without_infinities(value):
    if isinstance(value, dict):
        return {key: _without_infinities(entry)
                for key, entry in value.items()}
    if isinstance(value, list):
        return [_without_infinities(entry) for entry in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _with_infinities(payload: dict) -> dict:
    """A blank ceiling comes back from the panel as null; restore what it means."""
    document = dict(payload or {})
    limits = dict(document.get("limits") or {})
    for key in UNBOUNDED_LIMITS:
        values = limits.get(key)
        if isinstance(values, (list, tuple)):
            limits[key] = [math.inf if entry is None or entry == "" else entry
                           for entry in values]
    document["limits"] = limits
    return document


def parse_gravity_terms(urdf_text: str) -> list[dict]:
    """Each link's mass and where that mass sits, from the URDF's <inertial>.

    Those two are the whole of gravity. A static balance uses the mass and the
    lever and nothing else in the block, so the rotation tensor is left out
    rather than shown as though it carried a gravity term, and ``origin rpy``
    goes with it: it turns the tensor, not the centre of mass. A link with no
    mass has no term at all and is dropped rather than drawn at zero.

    Each entry names the link frame, which is a frame the viewer already has a
    world pose for, so a marker lands wherever forward kinematics puts it.
    """
    try:
        root = ElementTree.fromstring(urdf_text or "")
    except ElementTree.ParseError:
        return []
    terms = []
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = inertial.find("mass")
        try:
            kilograms = float(mass.get("value")) if mass is not None else 0.0
        except (TypeError, ValueError):
            continue
        if not math.isfinite(kilograms) or kilograms <= 0.0:
            continue
        terms.append({"link": link.get("name", ""),
                      "mass_kg": kilograms,
                      "com_m": _origin_xyz(inertial.find("origin"))})
    return terms


def _origin_xyz(origin) -> list[float]:
    if origin is None or not origin.get("xyz"):
        return [0.0, 0.0, 0.0]
    try:
        values = [float(part) for part in origin.get("xyz").split()]
    except ValueError:
        return [0.0, 0.0, 0.0]
    return values if len(values) == 3 else [0.0, 0.0, 0.0]


def _swept_here(record, joint: int) -> bool:
    """Was *this* joint the one being swept when the sample was taken?

    The friction phase moves one joint at a time, so six sevenths of a
    seven-joint arm's sweep-phase samples are a record of the joint standing
    still. Marking those as sweeps put a wall of them on the zero line of every
    chart and invited exactly the comparison the two colours exist to prevent.
    """
    if getattr(record, "phase", "") != campaign_module.PHASE_FRICTION:
        return False
    motion = getattr(record, "motion", "") or ""
    for part in motion.split(":"):
        if part.startswith("j") and part[1:].isdigit():
            return int(part[1:]) == joint
    # Older runs carry no motion tag; fall back to whether it actually moved.
    try:
        return abs(float(record.velocity_deg_s[joint])) > 0.5
    except (AttributeError, IndexError, TypeError):
        return False

def _parked_here(record, joint: int) -> bool:
    """A friction-phase row for a joint that pass never drove.

    Measured on this arm, a parked joint's fitted window speed runs to 0.037
    deg/s at the ninety-ninth percentile and 0.145 at worst, which is the same
    range the slowest rungs are commanded in. So the pass tag decides, not the
    speed: a threshold high enough to catch these would delete the crawl.
    """
    if getattr(record, "phase", "") != campaign_module.PHASE_FRICTION:
        return False
    for part in (getattr(record, "motion", "") or "").split(":"):
        if part.startswith("j") and part[1:].isdigit():
            return int(part[1:]) != joint
    return False


# A campaign yields tens of thousands of samples; a scatter plot stops being
# readable long before a browser stops being able to draw them.
MAX_PLOT_POINTS = system_default("runtime", "max_plot_points")
# Below this a joint has not established a direction of travel, so measured
# minus rigid is wherever the position servo happened to settle inside the
# stiction band, not friction at a speed. It sits well below the slowest
# commanded rung; the parked rows of a sweep are excluded by their tag instead.
STILL_SPEED_DEG_S = system_default("runtime", "still_speed_deg_s")
# The rehearsal plants known friction and must find it again; a run that merely
# completes proves the code executes, not that it computes.
REHEARSAL_NOISE = system_default("rehearsal", "noise")
REHEARSAL_TOLERANCE = system_default("rehearsal", "recovery_tolerance")
# What a gravity dry run's held-out error may be before its answer is refused.
# Recovering the planted friction is not enough on its own: the bidirectional
# pairing separates friction from gravity, so a joint whose gravity columns are
# underdetermined still returns its Coulomb term exactly. Measured on this arm,
# fourteen poses gave joint 1 a held-out error of 0.667 A while its Coulomb
# error was 0.003 -- the friction gate passed a model that could not hold the
# arm up. Ten times the planted noise; twenty-four poses land at 0.002.
GRAVITY_HOLDOUT_TOLERANCE = system_default("rehearsal", "gravity_holdout_tolerance")
# Homing is a recovery move from an unknown pose, so it goes slowly whatever
# speed the campaign was configured for.
HOMING_SPEED_DEG_S = system_default("dashboard", "home")["transit_speed_deg_s"]
# Jogging is hand-driven, so it is capped well below anything the campaign uses:
# the operator is watching the arm, not a plot, and has no undo.
JOG_SPEED_DEG_S = system_default("dashboard", "jog")["transit_speed_deg_s"]
# How long the jog worker naps when no new pose has been asked for.
JOG_POLL_S = system_default("runtime", "jog_poll_s")
OPTIMAL_MODE = "optimal_excitation"
# Gravity alone, and its dry run. Split out because the answer it produces is
# usable on its own -- it is what holds the arm up -- and because it takes
# minutes rather than the hour a full campaign does.
GRAVITY_MODE = "gravity"
GRAVITY_REHEARSAL = "gravity_rehearsal"
# Modes that move the real arm, and therefore need a passing dry run first.
HARDWARE_MODES = ("hardware", OPTIMAL_MODE, GRAVITY_MODE)
# Slow enough that the viscous term is small, fast enough to clear the 0.145
# deg/s a parked joint reaches from coupling alone. Two of them, because their
# difference is the only measurement of the viscous term a gravity run makes.
DEFAULT_GRAVITY_PROBE_SPEEDS = tuple(
    system_default("dashboard", "gravity")["gravity_probe_speeds_deg_s"])
# Plan fields the gravity card may set.
GRAVITY_OPTIONS = ("static_poses", "gravity_validation_poses",
                   "gravity_probe_deg")


def _comparison(optimal_errors, sweep: dict, joint_names) -> dict:
    """Put both models on one untouched validation set and judge the target."""
    if not sweep.get("available"):
        return dict(sweep)
    optimal = [float(value) for value in optimal_errors]
    baseline = [float(value) for value in sweep.get("validation_rms_a") or []]
    names = list(joint_names)
    if not optimal or len(optimal) != len(baseline):
        return {"available": False,
                "reason": "the two models did not score the same joints"}
    joints = []
    for index, (new, old) in enumerate(zip(optimal, baseline)):
        gain = (100.0 * (old - new) / old) if old > 0.0 else None
        joints.append({
            "joint": index,
            "name": names[index] if index < len(names) else f"joint{index + 1}",
            "optimal_validation_rms_a": new,
            "sweep_validation_rms_a": old,
            "improvement_percent": gain,
            "optimal_better": new < old,
        })
    optimal_mean, sweep_mean = float(np.mean(optimal)), float(np.mean(baseline))
    optimal_worst, sweep_worst = max(optimal), max(baseline)
    target_met = optimal_mean < sweep_mean and optimal_worst < sweep_worst
    return {
        "available": True,
        "basis": "same_unseen_optimal_validation_trajectories",
        "target_met": target_met,
        "source": sweep.get("source"),
        "sweep_method": sweep.get("method"),
        "validation_samples": sweep.get("validation_samples"),
        "optimal_mean_validation_rms_a": optimal_mean,
        "sweep_mean_validation_rms_a": sweep_mean,
        "optimal_worst_validation_rms_a": optimal_worst,
        "sweep_worst_validation_rms_a": sweep_worst,
        "mean_improvement_percent": (
            100.0 * (sweep_mean - optimal_mean) / sweep_mean
            if sweep_mean > 0.0 else None),
        "worst_improvement_percent": (
            100.0 * (sweep_worst - optimal_worst) / sweep_worst
            if sweep_worst > 0.0 else None),
        "joints": joints,
    }


@dataclass
class DashboardConfig:
    profile_path: str = system_default("ros", "profile_path")
    output_directory: str = system_default("ros", "output_directory")
    gravity_test_source: str = system_default("ros", "gravity_test_source")
    telemetry: TelemetrySpec = field(default_factory=TelemetrySpec)
    commands: CommandSpec = field(default_factory=CommandSpec)
    # Frame the 3D view renders in. Empty means the model root.
    display_frame: str = ""
    # How far the campaign may swing each joint. The URDF describes the arm,
    # not the stand it is bolted to, so this is often tighter than the URDF.
    workspace_limit_deg: tuple[float, ...] = ()
    # Per-joint lower and upper bound, which is what the panel edits. Set, it
    # replaces the symmetric cap above entirely.
    workspace_range_deg: tuple[tuple[float, float], ...] = ()
    # Top sweep speed. Zero keeps the conservative derived default, which is
    # too slow to see viscous friction on a full-size arm.
    maximum_speed_deg_s: float = system_default("ros", "maximum_speed_deg_s")
    # Where everything this panel edits lives between sessions: the obstacle
    # scene, the planner envelope and the gravity card's numbers. All three
    # describe the cell rather than the robot, so none of them can come from
    # the URDF and all of them are lost on restart without this. Empty
    # disables the file entirely.
    config_file_path: str = system_default("ros", "config_file_path")
    # How close the arm may come to anything before a pose is refused.
    safety_margin_m: float = system_default("ros", "safety_margin_m")
    system_config: SystemConfig = field(default_factory=default_system_config)

    @classmethod
    def from_system_config(cls, settings: SystemConfig, **overrides):
        ros = settings.values["ros"]
        configured = {name: ros[name] for name in (
            "profile_path", "output_directory", "gravity_test_source",
            "config_file_path", "maximum_speed_deg_s", "safety_margin_m")}
        configured["workspace_limit_deg"] = tuple(
            float(value) for value in ros["workspace_limit_deg"] if value > 0)
        return cls(system_config=settings, **(configured | overrides))


class IdentificationService:
    """One campaign at a time, plus the scene it runs in."""

    def __init__(self, config: DashboardConfig, bridge=None,
                 profile: RobotProfile | None = None,
                 process_launcher=None) -> None:
        self.config = config
        self.system = config.system_config.values
        for path in (config.config_file_path, config.profile_path):
            if path:
                self._writable_settings_path(Path(path).expanduser())
        self.bridge = bridge
        self.profile = profile
        self.arm: ident.ArmModel | None = None
        self.whole: ident.ArmModel | None = None
        self.scene: ObstacleScene | None = None
        self.urdf_text = ""
        self.gravity_terms: list[dict] = []
        self.driven_joints: list[str] = []
        # Where the joints this dashboard cannot drive were when the collision
        # model was reduced around them. The screen is true while they are
        # still there and false the moment they are not.
        self.screen_reference: dict = {}
        self.configured_profile = profile
        # What the launch supplied, kept so an edit can be undone.
        self.launch_profile = profile
        self.profile_source = "configured" if profile is not None else "none"
        self.components = ModelComponents()
        self.plan = None
        self.result: dict | None = None
        self.progress: dict = {"phase": "idle"}
        self.notes: list[str] = []
        self.events: list[dict] = []
        self._event_sequence = 0
        self._reports: dict[str, dict] = {}
        self._reports_scanned = False
        self.rehearsal_passed = False
        # What a passing gravity dry run actually validated. Re-tuning the card
        # changes the experiment, so the arming does not carry over.
        self.gravity_armed = ""
        # The gravity card's last committed numbers, kept across restarts.
        self.gravity_options: dict = {}
        self._gravity_seed: int | None = None
        # What the config file held, so the parts that need a model can wait
        # for one and the parts that do not are in force before the first plan.
        self._stored: dict = {}
        self.preview: dict = {"available": False}
        self.preview_token = 0
        self._hold_plan: dict = {}
        self._hold_recovery_required = False
        self._hold_recovery_selection = None
        self._hold_current_started = False
        self._default_gravity_source = None
        # Poses the phases of a running campaign have designed so far.
        self._designed: dict = {}
        self._completed_poses: dict[str, list[int]] = {}
        # Designing takes about as long as a short move and moves nothing, so
        # nothing else reports it; without this the panel looks dead.
        self.planning = False
        self._running_signature = ""
        self._state = IDLE
        self._activity = ""
        self._worker: threading.Thread | None = None
        self._abort = threading.Event()
        self._run_gate = threading.Event()
        self._run_gate.set()
        self._lock = threading.RLock()
        self._started_at = 0.0
        self._samples: list[dict] = []
        self._options: dict = {}
        # The pose a slider last asked for, or None once it has been driven.
        self._jog_target: list[float] | None = None
        self._process_launcher = process_launcher or subprocess.Popen
        self._external_process = None
        self._external_status_file: Path | None = None
        self._external_summary_file: Path | None = None
        self._restore_settings()

    def system_config_payload(self) -> dict:
        return {
            "path": str(self.config.system_config.path or ""),
            "values": deepcopy(self.system),
            "controls": resolved_controls(self.system),
            "control_ranges": self.control_ranges(),
            "constraints": {
                "profile_source": self.profile_source,
                "profile_sustained_speed_deg_s": (
                    self.profile.sustained_speed_deg_s if self.profile else None),
                "profile_temperature_c": self.profile.temperature_c if self.profile else None,
                "hardware_limits": "enforced independently by the robot controller and hardware plugin",
            },
        }

    def control_ranges(self) -> dict:
        campaign = (campaign_module.campaign_bounds(self.profile, system_config=self.system)
                    if self.profile is not None else {})
        result = {}
        for name, descriptor in self.system["controls"].items():
            reference = descriptor.get("range")
            if not reference:
                continue
            low, high = configured_range(self.system, reference)
            if self.profile is not None and reference == "motion.transit_speed_deg_s":
                high = min(high, self.profile.sustained_speed_deg_s)
            elif reference == "gravity.probe_speed_deg_s" and self.plan is not None:
                high = min(high, self.plan.maximum_speed_deg_s)
            elif reference.startswith("campaign.") and campaign:
                low, high = campaign[reference.split(".", 1)[1]]
            result[name] = {"min": low, "max": high if math.isfinite(high) else None,
                            "source": f"ranges.{reference}"}
        return result

    def _writable_settings_path(self, path: Path) -> Path:
        source = self.config.system_config.path
        if source is not None and path.resolve() == source.resolve():
            raise ValueError("system_config must be separate from cell settings and robot profiles")
        return path

    # -- model -----------------------------------------------------------

    def publish_event(self, message: str, level: str = "info",
                      source: str = "") -> dict:
        """Publish one operator-facing event through the dashboard-wide API."""
        text = str(message)
        severity = str(level or "info").lower()
        if severity not in ("info", "warning", "error"):
            severity = "info"
        clock = time.strftime("%H:%M:%S")
        with self._lock:
            self._event_sequence += 1
            event = {
                "sequence": self._event_sequence,
                "time": clock,
                "stamp_s": time.time(),
                "level": severity,
                "source": str(source or self._activity or "system"),
                "message": text,
            }
            self.events.append(event)
            del self.events[:-200]
            self.notes.append(f"{clock} {text}")
            del self.notes[:-200]
            return dict(event)

    def note(self, message: str) -> None:
        """Compatibility name for the one project-wide event publisher."""
        self.publish_event(message)

    def activity_payload(self) -> dict:
        """Current progress and durable recent events for every dashboard mode."""
        with self._lock:
            return {
                "sequence": self._event_sequence,
                "state": self._state,
                "activity": self._activity,
                "progress": dict(self.progress),
                "events": [dict(event) for event in self.events[-80:]],
            }

    def adopt_description(self, urdf_text: str) -> bool:
        """Build the model from a freshly received /robot_description."""
        if not urdf_text or urdf_text == self.urdf_text:
            return False
        with self._lock:
            self.urdf_text = urdf_text
            self._hold_plan = {}
        return self._rebuild()

    def adopt_driven_joints(self, names) -> bool:
        """Restrict identification to the joints the controller actually moves.

        A dual-arm URDF carries twice the joints the action can command, and
        identifying a model the controller cannot move is meaningless.
        """
        names = [str(entry) for entry in names]
        if names == self.driven_joints:
            return False
        with self._lock:
            self.driven_joints = names
            self._hold_plan = {}
            self.preview = {"available": False}
            self.preview_token += 1
            self._designed = {}
            self._completed_poses = {}
            self.gravity_armed = ""
            self.rehearsal_passed = False
        self.note(f"controller drives {len(names)} joints")
        return self._rebuild()

    def adopt_elsewhere(self) -> bool:
        """Rebuild the screen the first time the rest of the robot is visible.

        The URDF is latched and arrives before the first joint state, so the
        screen is often built before anything is known about the other arm and
        holds it at zero. This is not a change of policy, it is the same policy
        applied to a reading that had not arrived yet.
        """
        if self.arm is None or self.screen_reference:
            return False
        return bool(self._elsewhere_rad()) and self._rebuild()

    def adopt_signals(self, signals) -> None:
        """Take the effort channel the robot turned out to publish.

        Recorded rather than silent: the unit of every identified parameter
        follows from which channel is fitted.
        """
        if signals == self.config.telemetry.signals:
            return
        self.config.telemetry = replace(self.config.telemetry, signals=signals)
        self.note(f"effort read from {signals.effort_source} "
                  f"in {signals.effort_unit}")

    def _rebuild(self) -> bool:
        """Model, profile and obstacle scene, from whatever is known so far."""
        if not self.urdf_text:
            return False
        profile, source = self.configured_profile, "configured"
        if profile is None and self.driven_joints:
            try:
                cap = self._symmetric_cap() or None
                profile = autoprofile.derive_profile(
                    self.urdf_text, self.driven_joints,
                    workspace_limit_deg=cap,
                    speed_limit_deg_s=self._requested_speed(),
                    policy=self.system["profile_derivation"])
                source = "derived"
            except Exception as error:  # noqa: BLE001
                self.note(f"profile could not be derived: {error}")
        # Whatever the rest of the robot is doing right now is what the screen
        # is built against. Neutral would be a guess, and a wrong one is a
        # cleared path through an arm that is standing in it.
        elsewhere = self._elsewhere_rad()
        try:
            if profile is not None:
                arm = ident.ArmModel.from_profile(self.urdf_text, profile,
                                                  elsewhere)
            else:
                arm = ident.ArmModel.from_urdf_text(self.urdf_text, "")
                source = "none"
        except Exception as error:  # noqa: BLE001 - surfaced, never fatal
            self.note(f"robot_description rejected: {error}")
            return False
        with self._lock:
            self.arm = arm
            self.profile = profile
            self.profile_source = source
            self.screen_reference = elsewhere
            # The picture and the collision scene cover the whole robot, not
            # just the arm this dashboard drives, so the drawing needs a model
            # that still has the other arm's joints in it.
            try:
                self.whole = ident.ArmModel.from_urdf_text(self.urdf_text, "")
            except Exception as error:  # noqa: BLE001 - drawing is not the job
                self.whole = None
                self.note(f"whole-robot view unavailable: {error}")
            self.gravity_terms = parse_gravity_terms(self.urdf_text)
            previous = self.scene.as_list() if self.scene else []
            self.scene = ObstacleScene(
                arm.model, urdf_text=self.urdf_text,
                safety_margin_m=self.config.safety_margin_m)
            if previous:
                try:
                    self.scene.replace_all(previous)
                except KeyError as error:
                    self.note(f"obstacles dropped, frames changed: {error}")
            self.plan = (self._build_plan(profile)
                         if profile is not None else None)
        if not previous:
            # First model of the session: bring back whatever was drawn last.
            self._restore_obstacles()
        self.note(f"model ready: {arm.joint_count} joints, "
                  f"{arm.parameter_count} parameters, profile {source}")
        if elsewhere:
            self.note(f"collision screen built against "
                      f"{len(elsewhere)} joints this dashboard does not "
                      f"drive, where they are now")
        return True

    def _elsewhere_rad(self) -> dict:
        """The joints this dashboard cannot drive, in radians, as read.

        An optional bridge capability, and a bridge without it leaves the
        screen on the neutral assumption it always had.
        """
        reader = getattr(self.bridge, "elsewhere", None)
        if reader is None:
            return {}
        driven = set(self.driven_joints)
        return {name: float(value) for name, value in (reader() or {}).items()
                if name not in driven and math.isfinite(float(value))}

    def _requested_speed(self) -> float | None:
        speed = float(self.config.maximum_speed_deg_s or 0.0)
        return speed if speed > 0.0 else None

    def _build_plan(self, profile: RobotProfile):
        """The plan follows the operator's speed, clamped to the envelope."""
        speed = self._requested_speed()
        if speed is None:
            plan = campaign_module.default_plan(
                profile, defaults=self.system["campaign"], system_config=self.system)
        else:
            plan, notes = campaign_module.clamp_campaign_plan(
                {"maximum_speed_deg_s": speed}, profile,
                defaults=self.system["campaign"], system_config=self.system)
            for entry in notes:
                self.note(entry)
            self.note(f"sweep speeds {list(plan.friction_speeds_deg_s)} deg/s, "
                      f"amplitude {plan.friction_amplitude_deg:g} deg")
        # The panel's envelope wins over whatever the profile carries: it is a
        # statement about the cell the arm stands in, and a profile written for
        # a bare bench knows nothing about that. A stored one from another arm
        # is dropped rather than stretched: a bound list of the wrong length is
        # not a tighter cell, it is a shape mismatch that would otherwise
        # surface as a broadcast error several steps into a design.
        joints = self.arm.joint_count if self.arm is not None else 0
        if self.config.workspace_range_deg and joints and (
                len(self.config.workspace_range_deg) != joints):
            self.note(f"stored planner envelope covers "
                      f"{len(self.config.workspace_range_deg)} joints and this "
                      f"arm has {joints}; envelope ignored")
            self.config.workspace_range_deg = ()
        if self.config.workspace_range_deg:
            plan.workspace_range_deg = tuple(self.config.workspace_range_deg)
        elif self.config.workspace_limit_deg:
            plan.workspace_limit_deg = tuple(self.config.workspace_limit_deg)
        return plan

    def _symmetric_cap(self) -> list[float]:
        """The envelope as one magnitude per joint, for the derived profile.

        A profile's position limit is a single number, so an asymmetric range
        has to be widened to the larger side to be expressed at all. The plan
        keeps the true bounds; this is only what the envelope's own paperwork
        can hold.
        """
        if self.config.workspace_range_deg:
            return [max(abs(low), abs(high))
                    for low, high in self.config.workspace_range_deg]
        return list(self.config.workspace_limit_deg)

    def urdf_limit_deg(self) -> list[float]:
        """The arm's own travel, the ceiling any envelope sits under."""
        if self.arm is None:
            return []
        _low, high = self.arm.limits_deg()
        return [round(float(value), 1) for value in high]

    def urdf_range_deg(self) -> list[list[float]]:
        """The arm's own lower and upper bound per joint."""
        if self.arm is None:
            return []
        low, high = self.arm.limits_deg()
        return [[round(float(a), 1), round(float(b), 1)]
                for a, b in zip(low, high)]

    def workspace_payload(self) -> dict:
        return {
            "joint_names": list(self.arm.joint_names) if self.arm else [],
            "urdf_limit_deg": self.urdf_limit_deg(),
            "urdf_range_deg": self.urdf_range_deg(),
            "limit_deg": list(self.config.workspace_limit_deg),
            "range_deg": [list(pair)
                          for pair in self.config.workspace_range_deg],
            "effective_deg": self.jog_limits_deg(),
            "effective_range_deg": self.jog_range_deg(),
            "editable": self._state == IDLE,
        }

    def set_workspace_limit(self, values) -> dict:
        """The symmetric shorthand: one magnitude per joint, meaning +/- it."""
        if not values:
            return self.set_workspace_range([])
        try:
            given = [float(value) for value in values]
        except (TypeError, ValueError):
            return {"ok": False, "message": f"not a list of degrees: {values!r}"}
        if any(value <= 0.0 for value in given):
            return {"ok": False, "message": "every limit must be positive"}
        return self.set_workspace_range([[-value, value] for value in given])

    def set_workspace_range(self, ranges) -> dict:
        """Where each joint may go, low and high, set from the panel.

        Two bounds rather than one magnitude because a cell is not symmetric:
        this arm is mounted at an angle, so swinging one way meets the bench
        and the other way meets nothing. Unset, the planner uses the arm's own
        range, which describes the arm and not the cell it stands in: here that
        is +/-178 deg, which puts planned poses within reach of the other arm.
        Empty resets to that deliberately, because a bare bench is a real case.
        """
        self._require_idle("the envelope cannot change while a run is going")
        ceiling = self.urdf_range_deg()
        if not ceiling:
            return {"ok": False, "message": "no model yet"}
        cleaned: list[tuple[float, float]] = []
        if ranges:
            rows = list(ranges)
            if len(rows) == 1:
                rows = rows * len(ceiling)
            if len(rows) != len(ceiling):
                return {"ok": False,
                        "message": f"expected 1 or {len(ceiling)} ranges, "
                                   f"got {len(rows)}"}
            trimmed = []
            for index, (row, bound) in enumerate(zip(rows, ceiling)):
                try:
                    low, high = (float(row[0]), float(row[1]))
                except (TypeError, ValueError, IndexError, KeyError):
                    return {"ok": False,
                            "message": f"joint {index + 1} needs a low and a "
                                       f"high, got {row!r}"}
                if not (math.isfinite(low) and math.isfinite(high)):
                    return {"ok": False,
                            "message": f"joint {index + 1} bounds must be finite"}
                if low >= high:
                    return {"ok": False,
                            "message": f"joint {index + 1} low {low:g} is not "
                                       f"below high {high:g}"}
                capped = (max(low, bound[0]), min(high, bound[1]))
                if capped != (low, high):
                    trimmed.append(index + 1)
                cleaned.append((round(capped[0], 3), round(capped[1], 3)))
            if trimmed:
                self.note(f"envelope capped by the URDF on joints {trimmed}")
        with self._lock:
            self.config.workspace_range_deg = tuple(cleaned)
            self.config.workspace_limit_deg = ()
            # The poses a dry run validated are not the poses this envelope
            # will design, so the arming goes with it.
            self.gravity_armed = ""
            self.rehearsal_passed = False
        self._rebuild()
        self._persist_config()
        self.note("planner envelope "
                  + (f"set to {[list(pair) for pair in cleaned]} deg"
                     if cleaned else "reset to the arm's own range"))
        return {"ok": True, "workspace": self.workspace_payload()}

    def plan_preview(self, mode: str, options: dict | None = None) -> dict:
        """Design the poses and publish them, without moving anything.

        The point is that a human looks at them before the arm does. Designing
        is the one expensive part of a run that has no consequences, so it is
        worth being able to do on its own.
        """
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        if self.profile is None:
            return {"ok": False, "message": "no robot profile loaded"}
        self._require_idle("a run is going; wait for it to finish")
        if mode not in (GRAVITY_MODE, GRAVITY_HOLD_TEST):
            return {"ok": False, "message": f"cannot preview {mode!r}"}
        with self._lock:
            if self.planning or self._state != IDLE:
                return {"ok": False, "message": "already planning"}
            previous_progress = dict(self.progress)
            self.planning = True
            self.progress = {"mode": "planning", "phase": "designing",
                             "target": mode}
        self.publish_event("designing and screening poses", source="planner")
        try:
            answer = (self._plan_hold(dict(options or {}))
                      if mode == GRAVITY_HOLD_TEST else
                      self._plan_gravity(dict(options or {})))
            with self._lock:
                self.progress = previous_progress
            return answer
        except Exception as error:
            with self._lock:
                self.progress = {"mode": "planning", "phase": "failed",
                                 "target": mode, "error": str(error)}
            self.publish_event(f"planning failed: {error}", level="error",
                               source="planner")
            raise
        finally:
            with self._lock:
                self.planning = False

    def _plan_gravity(self, options: dict) -> dict:
        with self._lock:
            previous_seed = (self._gravity_seed if self._gravity_seed is not None
                             else self.plan.seed)
            self._gravity_seed = (previous_seed + 1 + secrets.randbelow(2**31 - 1)) % 2**31
            self.gravity_armed = ""
            self.rehearsal_passed = False
        plan = self._gravity_plan(options)
        self._remember_gravity(plan)
        limits = plan.design_limits(self.arm)
        screen = self.scene.collision_free if self.scene is not None else None
        phases = []
        standing = (list(plan.start_deg) if plan.start_deg else None)
        for phase, count, seed in (
                (campaign_module.PHASE_GRAVITY, plan.static_poses, plan.seed),
                (campaign_module.PHASE_VALIDATION,
                 plan.gravity_validation_poses, plan.seed + 7717)):
            design = excitation.design_static_poses(
                self.arm, limits, count=count,
                candidates=plan.static_candidates, seed=seed,
                collision_free=screen, start_deg=standing,
                crossing_deg=float(plan.gravity_probe_deg))
            standing = design.final_deg or standing
            phases.append({"phase": phase, "detail": design.as_dict()})
        preview = self._build_preview("plan", {"phases": phases})
        with self._lock:
            if preview.get("available"):
                self.preview = preview
                self.preview_token += 1
        drawn = sum(len(entry["detail"].get("poses_deg") or [])
                    for entry in phases)
        astray = self.screen_drift()
        if astray:
            where = ", ".join(f"{item['joint']} moved {item['moved_deg']:+g} deg"
                              for item in astray[:4])
            self.publish_event(
                f"these {drawn} poses were screened against a placement the "
                f"rest of the robot has since left: {where}",
                level="warning", source="planner")
        self.publish_event(f"planned {drawn} poses for review; nothing has moved",
                           source="planner")
        return {"ok": True, "preview": self.preview_payload(),
                "astray": astray}

    def have_model(self) -> bool:
        return self.arm is not None

    def _hold_context(self) -> str:
        try:
            source = self._gravity_source(validate=False)
        except (OSError, ValueError, TypeError, OverflowError, ImportError, LookupError) as error:
            source = {"configured": self.config.gravity_test_source, "error": str(error)}
        payload = {
            "urdf": self.urdf_text, "range": self.jog_range_deg(),
            "obstacles": self.scene.as_list() if self.scene else [],
            "margin": self.config.safety_margin_m,
            "reference": self.screen_reference,
            "joints": list(self.arm.joint_names) if self.arm else [],
            "driven_joints": list(self.driven_joints),
            "source": source,
            "commands": self.config.commands.follow_joint_trajectory_action,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _hold_position(self) -> list:
        sample = self.latest_sample()
        health = self.connection()
        if not sample or not health.get("telemetry_ok"):
            raise ValueError("fresh joint telemetry is required for a hold plan")
        pose = np.asarray(sample.get("position_deg"), dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("hold position requires seven finite joint angles")
        return pose.tolist()

    def _hold_path_clear(self, start, target) -> bool:
        limits = np.asarray(self.jog_range_deg(), dtype=float)
        if self.scene is None or limits.shape != (7, 2):
            return False
        for fraction in np.linspace(0.0, 1.0, self.system["dashboard"]["hold_test"]["path_samples"]):
            pose = np.asarray(start) + (np.asarray(target) - start) * fraction
            if (np.any(pose < limits[:, 0]) or np.any(pose > limits[:, 1])
                    or not self.scene.collision_free(pose)):
                return False
        return True

    def hold_plan_payload(self) -> dict:
        """Describe the plan permission without exposing mutable execution targets."""
        plan = self._hold_plan
        available = bool(plan and plan["context"] == self._hold_context()
                         and not self.screen_drift())
        return {"available": available, "id": plan.get("id", ""),
                "poses": plan.get("count", 0)}

    def _plan_hold(self, options: dict) -> dict:
        capability = self.gravity_test_capability()
        if not capability["available"]:
            raise ValueError(capability["reason"])
        source = self._gravity_source(ArmIdentity(capability["arm"]))
        if self.scene is None or self.screen_drift():
            raise ValueError("refresh the collision scene before planning holds")
        count = checked_value(options.get("poses", self.system["dashboard"]["hold_test"]["poses"]),
                      self.system, "hold_test.poses", integer=True)
        context = self._hold_context()
        start = self._hold_position()
        limits = np.asarray(self.jog_range_deg())

        def clear(pose):
            return bool(np.all(pose >= limits[:, 0]) and
                        np.all(pose <= limits[:, 1]) and
                        self.scene.collision_free(pose))

        with self._lock:
            previous = self._hold_plan
            self._hold_plan = {}
        for attempt in range(self.system["dashboard"]["hold_test"]["replan_attempts"]):
            plan = hold_plan_module.build_hold_plan(
                source, list(self.arm.joint_names), start, count, clear,
                system_config=self.system, urdf_text=self.urdf_text,
                exclude_poses_deg=options.get("exclude_poses_deg"))
            if plan["poses_deg"] != previous.get("poses_deg"):
                break
        else:
            raise ValueError("could not find a different hold pose set; reduce the pose count "
                             "or use a model with more recorded poses")
        if context != self._hold_context() or self.screen_drift():
            raise ValueError("scene changed during planning; plan again")
        if np.max(np.abs(np.asarray(self._hold_position()) - start)) > self.system["dashboard"]["hold_test"]["planning_drift_deg"]:
            raise ValueError("arm moved during planning; plan again")
        plan["context"] = context
        plan["id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        preview = self._build_preview("plan", {"phases": [
            {"phase": "hold_set", "detail": {"poses_deg": plan["poses_deg"]}}]})
        if not preview.get("available"):
            raise ValueError("hold preview is unavailable")
        with self._lock:
            self._hold_plan = plan
            self.preview = preview
            self.preview_token += 1
            self._completed_poses = {}
        return {"ok": True, "preview": self.preview_payload(),
                "hold_plan": self.hold_plan_payload()}

    def kinematics(self, payload: dict) -> dict:
        """Link poses along the straight joint-space line between two poses.

        The canvas cannot do forward kinematics and must not learn how: the
        picture agreeing with the model that screened the motion is the whole
        reason it is drawn from the model rather than from TF. This is that
        model, sampled along the path the arm is actually commanded to fly, so
        an animation of a plan cannot show a path the plan does not contain.

        Only the driven arm's own frames are returned. Everything the model
        holds still -- the other arm, the stand -- hangs off the universe
        joint after the reduction and is already drawn where it is.
        """
        if self.arm is None:
            return {"ok": False, "message": "no model yet"}
        count = self.arm.joint_count
        try:
            end = np.asarray(payload["to_deg"], dtype=float)
            start = (np.asarray(payload["from_deg"], dtype=float)
                     if payload.get("from_deg") else end)
        except (KeyError, TypeError, ValueError):
            return {"ok": False, "message": "to_deg is a list of degrees"}
        if len(end) != count or len(start) != count:
            return {"ok": False,
                    "message": f"this arm has {count} joints"}
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            return {"ok": False, "message": "every angle must be finite"}
        try:
            steps = int(payload.get("steps", 1))
        except (TypeError, ValueError):
            steps = 1
        steps = min(max(steps, 1), self.system["runtime"]["maximum_animation_steps"])
        driven = {frame.name for frame in self.arm.model.frames
                  if frame.parentJoint > 0}
        frames = []
        for index in range(steps + 1):
            pose = start + (end - start) * (index / steps)
            placed = self.arm.link_transforms(pose)
            frames.append({name: flat for name, flat in placed.items()
                           if name in driven})
        return {"ok": True, "frames": frames}

    # -- robot profile ---------------------------------------------------

    def profile_payload(self) -> dict:
        """The envelope in force, in the shape a profile file has.

        There is always one to edit, including the first time an arm is ever
        run: with no file the module derives a profile from the URDF and the
        controller's joint list, and that derivation is what the panel edits.
        A file is somewhere to save the answer, not a prerequisite for having
        one.
        """
        profile = self.profile
        hardware_limits = None
        try:
            identity = ArmIdentity.from_joint_names(self.driven_joints)
            hardware_limits = ArmBinding.from_description(
                identity, self.urdf_text).current_limits
        except (ValueError, TypeError, ElementTree.ParseError):
            pass
        return {
            "have_profile": profile is not None,
            "source": self.profile_source,
            "origin": profile.source if profile is not None else "",
            "edited": profile is not None and profile.source == EDITED_SOURCE,
            "save_target": str(self._save_target()),
            "editable": self._state == IDLE,
            "current_guard": (profile is not None
                              and autoprofile.current_guard_active(profile)),
            "dark_guards": list(self._dark_guards()),
            "hardware_current_limits": hardware_limits,
            "profile": (_without_infinities(profile.as_dict())
                        if profile is not None else None),
        }

    def apply_profile(self, payload: dict) -> dict:
        """Install an edited envelope, on the same terms as a written one.

        A number shown in a form and applied by an operator is that operator's
        number, which is the standard a hand-written file is held to as well.
        The one ceiling nobody can guess still gates on its own value: leave a
        current limit at infinity and the current guard stays off regardless.
        """
        self._require_idle("the envelope cannot change while a run is going")
        document = _with_infinities(payload)
        notes = dict(document.get("notes") or {})
        # The derivation's note says the current ceilings are unset, which an
        # edit may have just made untrue.
        notes.pop("derived", None)
        notes["edited"] = ("Applied from the dashboard. Every value here was "
                           "entered by an operator.")
        document["notes"] = notes
        profile = RobotProfile.from_dict(document, source=EDITED_SOURCE)
        with self._lock:
            self.configured_profile = profile
        if not self._rebuild():
            self.note(f"envelope edited: {profile.name}, waiting for a model")
            return {"ok": True, "profile": self.profile_payload()}
        self.note(f"envelope edited: {profile.name}")
        return {"ok": True, "profile": self.profile_payload()}

    def reset_profile(self) -> dict:
        """Back to the launch's file, or to a fresh derivation from the URDF."""
        self._require_idle("the envelope cannot change while a run is going")
        with self._lock:
            self.configured_profile = self.launch_profile
        self._rebuild()
        self.note("envelope reset to what the launch supplied")
        return {"ok": True, "profile": self.profile_payload()}

    def save_profile(self, name: str = "") -> dict:
        if self.profile is None:
            return {"ok": False, "message": "no profile to save yet"}
        path = self.profile.to_yaml(self._save_target(name))
        self.note(f"profile written to {path}")
        return {"ok": True, "path": str(path)}

    def _save_target(self, name: str = "") -> Path:
        """Where a save may land, which is never wherever the caller says.

        The web surface listens on every interface, so honouring a path from a
        request would be an arbitrary file write. A name is only a name, and it
        is written beside the results this dashboard was started with.
        """
        root = Path(self.config.output_directory)
        cleaned = str(name or "").strip()
        if not cleaned:
            return self._writable_settings_path(
                Path(self.config.profile_path).expanduser() if self.config.profile_path
                else root / self.system["storage"]["profile_filename"])
        if not PROFILE_FILE_NAME.fullmatch(cleaned):
            raise ValueError(
                "a profile file name may use letters, digits, dot, dash and "
                f"underscore, and must end in .yaml: {cleaned!r}")
        return self._writable_settings_path(root / cleaned)

    # -- obstacles -------------------------------------------------------

    def obstacles(self) -> list[dict]:
        return self.scene.as_list() if self.scene else []

    def frame_names(self) -> list[str]:
        return self.scene.frame_names() if self.scene else []

    def add_obstacle(self, payload: dict) -> dict:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        defaults = self.system["ui"]["obstacle"]
        box = self.scene.add(Obstacle.from_dict({
            "size_m": defaults["size_m"], "xyz_m": defaults["xyz_m"],
            "rpy_deg": defaults["rpy_deg"], **payload}))
        self.note(f"obstacle {box.name} bolted to {box.parent_frame}")
        self._persist_config()
        return box.as_dict()

    def update_obstacle(self, obstacle_id: str, changes: dict) -> dict:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        box = self.scene.update(obstacle_id, **changes).as_dict()
        self._persist_config()
        return box

    def remove_obstacle(self, obstacle_id: str) -> None:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        self.scene.remove(obstacle_id)
        self._persist_config()

    def replace_obstacles(self, payloads: list[dict]) -> list[dict]:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        self.scene.replace_all(payloads)
        self._persist_config()
        return self.scene.as_list()

    # -- configuration file ----------------------------------------------

    def config_file(self) -> Path | None:
        raw = (self.config.config_file_path or "").strip()
        return self._writable_settings_path(Path(raw).expanduser()) if raw else None

    def save_config(self, name: str = "") -> dict:
        """Write the configuration where the operator says.

        The launch path is where edits are kept automatically; this is how a
        cell's settings get a name worth carrying to another robot.
        """
        try:
            target = self._config_save_target(name)
        except ValueError as error:
            return {"ok": False, "message": str(error)}
        try:
            obstacles_module.write_document(target, self._config_document())
        except OSError as error:
            return {"ok": False, "message": f"not saved: {error}"}
        self.note(f"configuration written to {target}")
        return {"ok": True, "path": str(target)}

    def _config_save_target(self, name: str = "") -> Path:
        """Where a save may land, which is never wherever the caller says.

        Same rule as the profile: the web surface listens on every interface,
        so honouring a path from a request would be an arbitrary file write. A
        name is only a name.
        """
        cleaned = str(name or "").strip()
        if not cleaned:
            launched = self.config_file()
            return self._writable_settings_path(
                launched if launched is not None
                else Path(self.config.output_directory)
                / self.system["storage"]["cell_config_filename"])
        if not CONFIG_FILE_NAME.fullmatch(cleaned):
            raise ValueError(
                "a config file name may use letters, digits, dot, dash and "
                f"underscore, and must end in .json: {cleaned!r}")
        return self._writable_settings_path(Path(self.config.output_directory) / cleaned)

    def _config_document(self) -> dict:
        """Everything this panel edits, in the form written to disk.

        Built on top of whatever was read, so a section this build has no
        model for yet -- obstacles, before a URDF arrives -- is carried
        forward rather than blanked by the first unrelated edit.
        """
        document = dict(self._stored)
        document["schema_version"] = obstacles_module.SCHEMA_VERSION
        if self.scene is not None:
            document["obstacles"] = self.scene.as_list()
        document["workspace_range_deg"] = [
            list(pair) for pair in self.config.workspace_range_deg]
        if self.gravity_options:
            document["gravity"] = dict(self.gravity_options)
        return document

    def _persist_config(self) -> None:
        """Save after every edit; a setting lost on restart is one retyped."""
        path = self.config_file()
        if path is None:
            return
        document = self._config_document()
        try:
            obstacles_module.write_document(path, document)
        except OSError as error:
            self.note(f"configuration not saved to {path}: {error}")
            return
        self._stored = document

    def _restore_settings(self) -> None:
        """Read the file and take everything that needs no model.

        The envelope and the gravity card have to be in force before the first
        plan is built, and neither of them mentions a frame, so neither has to
        wait for a URDF. Obstacles do, and are placed later.
        """
        path = self.config_file()
        # An empty configuration has three causes and they are not
        # interchangeable: nothing was asked for, what was asked for is not
        # there, or the file was read. Saying nothing makes all three look
        # like the same bug.
        if path is None:
            self.note("config_file_path was not set: the obstacle scene, the "
                      "planner envelope and the gravity settings start empty "
                      "and every edit is lost on restart")
            return
        if not path.exists():
            self.note(f"no config file at {path.resolve()} yet: settings "
                      "start empty and the first edit creates it there")
            return
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            self.note(f"config file {path} not read: {error}")
            return
        if not isinstance(document, dict):
            self.note(f"config file {path} is not an object")
            return
        version = int(document.get("schema_version", 0) or 0)
        if version > obstacles_module.SCHEMA_VERSION:
            self.note(f"config file {path} is schema version {version}; this "
                      f"build understands up to "
                      f"{obstacles_module.SCHEMA_VERSION}")
            return
        self._stored = document
        self._restore_envelope(document.get("workspace_range_deg"))
        self._restore_gravity(document.get("gravity"))

    def _restore_envelope(self, ranges) -> None:
        """The stored envelope, unchecked against a URDF nobody has sent yet.

        Every design intersects it with the arm's own travel, so a stored
        range wider than the robot cannot widen the robot. A malformed one is
        dropped whole rather than half-applied: half an envelope is a cell
        the arm may leave on the joints that went missing.
        """
        if not ranges:
            return
        try:
            bounds = [(float(low), float(high)) for low, high in ranges]
        except (TypeError, ValueError):
            self.note(f"stored planner envelope ignored: {ranges!r}")
            return
        if any(not (math.isfinite(low) and math.isfinite(high))
               or low >= high for low, high in bounds):
            self.note(f"stored planner envelope ignored: {ranges!r}")
            return
        self.config.workspace_range_deg = tuple(bounds)
        self.config.workspace_limit_deg = ()
        self.note(f"planner envelope restored: "
                  f"{[list(pair) for pair in bounds]} deg")

    def _restore_gravity(self, options) -> None:
        if not isinstance(options, dict) or not options:
            return
        self.gravity_options = dict(options)
        self.note(f"gravity settings restored: {options}")

    def _restore_obstacles(self) -> None:
        """Boxes name frames, so they are placed once a model exists."""
        if self.scene is None or not self._stored.get("obstacles"):
            return
        try:
            skipped = self.scene.load_document(self._stored)
        except ValueError as error:
            self.note(f"obstacles not loaded: {error}")
            return
        self.note(f"obstacles restored from {self.config_file()}: "
                  f"{len(self.scene.as_list())} kept"
                  + (f", {len(skipped)} dropped" if skipped else ""))
        for reason in skipped:
            self.note(f"obstacle dropped: {reason}")

    def collision_report(self, pose_deg=None) -> dict:
        """Whether the current pose is clear, and what it touches if not."""
        if self.scene is None:
            return {"available": False, "reason": "no model yet"}
        pose = pose_deg
        if pose is None:
            sample = self.latest_sample()
            pose = sample.get("position_deg") if sample else None
        if pose is None:
            return {"available": False, "reason": "no joint state yet"}
        report = dict(self.scene.geometry_report())
        report["available"] = True
        report["clear"] = self.scene.collision_free(pose)
        report["contacts"] = [] if report["clear"] else self.scene.contacts(pose)
        return report

    # -- telemetry -------------------------------------------------------

    def latest_sample(self) -> dict | None:
        if self.bridge is None:
            return None
        return self.bridge.latest_sample()

    def telemetry_since(self, cursor: int) -> dict:
        """Telemetry the panel has not seen yet, at the rate it arrived.

        Polling for the newest frame alone would alias: a 200 Hz current read
        ten times a second is not a slower current read, it is a different
        signal with the peaks removed.
        """
        names = list(self.arm.joint_names) if self.arm is not None else []
        history = getattr(self.bridge, "history_since", None)
        if history is None:
            sample = self.latest_sample()
            return {"cursor": 0, "dropped": 0, "joint_names": names,
                    "frames": [sample] if sample else []}
        payload = history(max(0, int(cursor or 0)))
        payload["joint_names"] = names
        return payload

    def connection(self) -> dict:
        described = self.config.telemetry.describe()
        described["action"] = self.config.commands.follow_joint_trajectory_action
        described["controllers"] = {
            "manager": self.config.commands.controller_manager,
            "available": False, "age_s": None, "error": "unavailable", "items": [],
        }
        # A guard is live only if the signal is mapped, the data actually
        # arrives, and something enforces it. Naming an interface is not
        # protection, and the panel used to imply it was.
        described["missing_guards"] = list(self._dark_guards())
        if self.bridge is None:
            described.update(
                {"telemetry_ok": False, "action_ok": False, "sample_age_s": None,
                 "description_ok": bool(self.urdf_text)})
            return described
        described.update(self.bridge.health())
        described["description_ok"] = bool(self.urdf_text)
        return described

    # -- 3D view ---------------------------------------------------------

    def whole_pose_deg(self) -> dict:
        """Every joint this dashboard can see, in degrees, by name.

        The driven joints come from the fitted sample; the rest come straight
        off the state topic. Both arms are on one robot and one bus, so a
        dashboard that only knows its own seven draws the other arm wherever it
        happened to be reduced against -- which is neutral, and is a picture of
        a robot that does not exist.
        """
        pose = {name: float(np.degrees(value)) for name, value in
                (self.bridge.elsewhere() if self.bridge is not None else {}).items()}
        sample = self.latest_sample()
        if sample and self.arm is not None:
            pose.update(dict(zip(self.arm.joint_names,
                                 (float(v) for v in sample["position_deg"]))))
        return pose

    def gravity_payload(self) -> dict:
        """The URDF's gravity terms, for the canvas to draw them where they are.

        Only the file's own numbers travel. Where each mass ends up follows
        from ``link_tf``, which the canvas already has, so the picture cannot
        offer a second opinion on the kinematics.
        """
        return {
            "links": list(self.gravity_terms),
            "total_mass_kg": sum(item["mass_kg"]
                                 for item in self.gravity_terms),
        }

    def viewer_state(self) -> dict:
        """Everything the 3D canvas needs for one frame."""
        if self.arm is None:
            return {"have_model": False}
        sample = self.latest_sample()
        pose = np.asarray(sample["position_deg"], dtype=float) if sample \
            else np.zeros(self.arm.joint_count)
        transforms = None
        if self.whole is not None:
            everywhere = self.whole_pose_deg()
            if everywhere:
                transforms = self.whole.link_transforms(
                    [everywhere.get(name, 0.0)
                     for name in self.whole.joint_names])
        if transforms is None:
            transforms = (self.arm.link_transforms(pose)
                          if hasattr(self.arm, "link_transforms") else {})
        payload = {
            "have_model": True,
            "joint_names": list(self.arm.joint_names),
            "joint_values_deg": pose.tolist(),
            "link_tf": transforms,
            "obstacles": self.scene.placements(pose) if self.scene else [],
            "frames": self.frame_names(),
            "gravity": self.gravity_payload(),
            "scene_activity": self.scene_activity_payload(),
            # Bumped when a run designs new poses, so the canvas fetches the
            # skeletons once instead of on every poll.
            "preview_token": self.preview_token,
        }
        focus = payload["scene_activity"]["focus"]
        if focus is not None:
            payload["moving"] = {
                "pose_deg": list(focus["pose_deg"]),
                "phase": focus["phase"], "index": focus["index"],
            }
        if self.plan is not None and getattr(self.plan, "workspace_limit_deg", None):
            payload["workspace_limit_deg"] = list(self.plan.workspace_limit_deg)
        return payload

    def scene_activity_payload(self) -> dict:
        """One mode-neutral contract for activity drawn in the 3D canvas."""
        with self._lock:
            progress = dict(self.progress)
            mode = str(progress.get("mode") or self._activity or "")
            phase = str(progress.get("phase") or "")
            active = self._state in (RUNNING, PAUSED, JOGGING) or self.planning
            focus = None
            target_index = int(progress.get("target_pose")
                               or progress.get("pose") or 0)
            target_complete = (progress.get("target_pose") is not None
                               and progress.get("completed_pose")
                               == progress.get("target_pose"))
            if progress.get("pose_deg") is not None and not target_complete:
                focus = {
                    "kind": "joint_pose",
                    "pose_deg": list(progress["pose_deg"]),
                    "phase": phase,
                    "index": target_index,
                }
            return {
                "id": (f"{mode}:{self._started_at:.6f}"
                       if active and mode else ""),
                "state": self._state,
                "mode": mode,
                "phase": phase,
                "progress": progress,
                "focus": focus,
                "completed": {
                    name: list(indices)
                    for name, indices in self._completed_poses.items()
                },
                "tour": {
                    "kind": "joint_pose_tour",
                    "token": self.preview_token,
                    "available": bool(self.preview.get("available")),
                    "autoplay": active and "rehearsal" in mode,
                },
            }

    # -- campaign --------------------------------------------------------

    def screen_drift(self, tolerance_deg: float | None = None) -> list[dict]:
        """Joints this dashboard does not drive that have moved since the
        screen was built against them.

        The collision scene carries every link the URDF ships -- sixteen on
        this robot, both arms -- but the arm this dashboard does not drive is
        *locked* inside it, held at whatever configuration the model was
        reduced against. That configuration is where the other arm was when
        the screen was built, so the screen is true exactly as long as the
        other arm has not moved since. Let it move and the screen will
        cheerfully clear a path straight through it. Nothing else catches
        this: the geometry is loaded, the pair count is right, and the screen
        refuses folded poses exactly as it should.

        Measured on this cell, an arm 70 deg from where the screen holds it
        puts its wrist 621 mm from the screen's idea of it.
        """
        if tolerance_deg is None:
            tolerance_deg = self.system["runtime"]["screen_drift_deg"]
        drifted = []
        for name, radians in self._elsewhere_rad().items():
            # A joint the screen never had a reading for is held at zero
            # inside it, so zero is what it has drifted from.
            was = self.screen_reference.get(name, 0.0)
            degrees = float(np.degrees(radians - was))
            if abs(degrees) > tolerance_deg:
                drifted.append({"joint": name,
                                "at_deg": round(float(np.degrees(radians)), 2),
                                "moved_deg": round(degrees, 2)})
        return sorted(drifted, key=lambda item: -abs(item["moved_deg"]))

    def rescreen(self) -> dict:
        """Rebuild the screen around where the rest of the robot is now.

        Every pose already designed was screened against the old placement, so
        the arming goes with it rather than carrying over onto a scene that no
        longer matches the poses it cleared.
        """
        self._require_idle("the screen cannot be rebuilt while a run is going")
        drifted = self.screen_drift()
        with self._lock:
            self.gravity_armed = ""
            self.rehearsal_passed = False
        if not self._rebuild():
            return {"ok": False, "message": "no model to rebuild"}
        self.note(f"collision screen rebuilt; {len(drifted)} joints had moved. "
                  "Plan and rehearse again: the poses that were cleared were "
                  "cleared against the old placement.")
        return {"ok": True, "drift": self.screen_drift()}

    def start(self, mode: str, options: dict | None = None) -> dict:
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        if self.profile is None:
            return {"ok": False, "message": "no robot profile loaded"}
        options = dict(options or {})
        with self._lock:
            if self._state != IDLE or self.planning:
                return {"ok": False, "message": f"{self._activity} is running"}
            if mode == GRAVITY_MODE:
                wanted = self._gravity_signature(self._gravity_plan(options))
                if not self.gravity_armed:
                    return {"ok": False,
                            "message": "rehearse the gravity run first: a dry "
                                       "run must recover what it planted "
                                       "before the arm is allowed to move"}
                if self.gravity_armed != wanted:
                    # Re-tuning is expected and allowed; running numbers that
                    # were never rehearsed is not.
                    return {"ok": False,
                            "message": "the gravity settings changed since the "
                                       "dry run that passed. Rehearse again "
                                       "with these numbers, then run."}
            elif mode in HARDWARE_MODES and not self.rehearsal_passed:
                # Not ceremony: the rehearsal plants known friction and must
                # find it again, and it is what caught the fit returning zero.
                return {"ok": False,
                        "message": "rehearse first: a dry run must pass "
                                   "before the arm is allowed to move"}
            if mode in HARDWARE_MODES + ("load_sweep",):
                astray = self.screen_drift()
                if astray:
                    where = ", ".join(
                        f"{item['joint']} moved {item['moved_deg']:+g} deg"
                        for item in astray[:4])
                    return {"ok": False,
                            "message": "the collision screen holds every joint "
                                       "this dashboard does not drive where it "
                                       "was when the screen was built, and "
                                       f"these have moved since: {where}. "
                                       "Rebuild the screen and plan again, or "
                                       "put them back."}
            self._state = RUNNING
            self._activity = (
                mode if mode in ("load_sweep", OPTIMAL_MODE, GRAVITY_MODE,
                                 GRAVITY_REHEARSAL)
                else f"campaign_{mode}")
            self._abort.clear()
            self._run_gate.set()
            self._started_at = time.monotonic()
            self._samples = []
            self._options = options
            self._designed = {}
            self._completed_poses = {}
            # The previous run's verdict is not this run's; leaving it up reads
            # as though the campaign now moving has already passed.
            self.result = None
            self.progress = {"mode": self._activity, "phase": "starting"}
        self._worker = threading.Thread(
            target=self._run, args=(mode,), daemon=True,
            name=f"identification-{mode}")
        self._worker.start()
        return {"ok": True, "message": f"{self._activity} started"}

    def stop(self) -> dict:
        self._abort.set()
        self._run_gate.set()
        with self._lock:
            process = self._external_process
            activity = self._activity
        if process is not None and process.poll() is None:
            try:
                if activity in (GRAVITY_DRAG_TEST, GRAVITY_HOLD_TEST):
                    os.killpg(process.pid, signal.SIGINT)
                else:
                    process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        return {"ok": True, "message": "stop requested"}

    def shutdown(self, timeout_s: float = 15.0) -> bool:
        """Request a safe stop and wait a bounded time for the owner to exit."""
        self.stop()
        worker = self._worker
        if worker is None or worker is threading.current_thread():
            return True
        worker.join(max(0.0, float(timeout_s)))
        return not worker.is_alive()

    def _gravity_source(self, identity: ArmIdentity | None = None, *, validate: bool = True) -> str:
        """Resolve and validate only the configured artifact for the selected arm."""
        selected = ArmIdentity.from_joint_names(
            self.driven_joints or (self.arm.joint_names if self.arm else []))
        if identity is not None and identity != selected:
            raise ValueError("selected arm changed while resolving the gravity source")
        source = str(self.config.gravity_test_source or "").strip()
        if source:
            source = source.replace("{arm}", selected.name)
        else:
            if selected.name != "right":
                raise ValueError(
                    f"an explicit gravity_test_source is required for arm {selected.name}")
            if self._default_gravity_source is None:
                self._default_gravity_source = hold_plan_module._load_backend().current.DEFAULT_SOURCE
            source = str(self._default_gravity_source)
        path = Path(source).expanduser().resolve()
        if validate:
            payload = json.loads(hold_plan_module._source_path(path).read_text("utf-8"))
            hold_plan_module._validate_source(payload, list(selected.joint_names))
        return str(path)

    def gravity_test_capability(self) -> dict:
        """Require a selected model, matching calibration and acknowledged live binding."""
        names = (list(self.driven_joints) if self.driven_joints else
                 list(self.arm.joint_names) if self.arm is not None else [])
        try:
            identity = ArmIdentity.from_joint_names(names)
        except (TypeError, ValueError):
            return {"available": False, "reason_code": "unsupported_arm",
                    "reason": "gravity validation requires one complete ordered RealMan arm"}
        try:
            source = self._gravity_source(identity)
        except (OSError, ValueError, TypeError, OverflowError, ImportError, LookupError) as error:
            return {"available": False, "arm": identity.name,
                    "reason_code": "invalid_source", "reason": str(error)}
        if self.arm is None or tuple(self.arm.joint_names) != identity.joint_names:
            return {"available": False, "arm": identity.name,
                    "reason_code": "model_mismatch",
                    "reason": "live model does not match the selected arm joints"}
        try:
            ArmBinding.from_description(identity, self.urdf_text)
        except (ValueError, TypeError, ElementTree.ParseError) as error:
            return {"available": False, "arm": identity.name,
                    "reason_code": "unsafe_binding", "reason": str(error)}
        return {"available": True, "arm": identity.name, "source": source}

    def _motion_speed_limit(self) -> float:
        maximum = configured_range(self.system, "motion.transit_speed_deg_s")[1]
        return min(maximum, float(self.profile.sustained_speed_deg_s)) if self.profile else 0.0

    def _motion_speed(self, options: dict, default: float) -> float:
        return checked_value(options.get("transit_speed_deg_s", default), self.system,
                             "motion.transit_speed_deg_s", ceiling=self._motion_speed_limit())

    def start_gravity_test(self, mode: str, options: dict | None = None) -> dict:
        """Start one standalone, guard-owned gravity validation activity."""
        options = dict(options or {})
        if mode not in (GRAVITY_HOLD_TEST, GRAVITY_DRAG_TEST):
            return {"ok": False, "message": "unknown gravity test mode"}
        if options.pop("acknowledgement", "") != GRAVITY_TEST_ACKNOWLEDGEMENT:
            return {"ok": False,
                    "message": "confirm that the arm is supported and E-stop is ready"}
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        capability = self.gravity_test_capability()
        if not capability["available"]:
            return {"ok": False, "message": capability["reason"]}
        arm = capability["arm"]
        try:
            source = self._gravity_source(ArmIdentity(arm))
        except (OSError, ValueError, TypeError, OverflowError, ImportError, LookupError) as error:
            return {"ok": False, "message": str(error)}

        if mode == GRAVITY_HOLD_TEST:
            defaults = self.system["dashboard"]["hold_test"]
            poses = checked_value(options.get("poses", defaults["poses"]), self.system,
                                  "hold_test.poses", integer=True)
            seconds = checked_value(options.get("seconds", defaults["seconds"]), self.system,
                                    "hold_test.seconds")
            speed = self._motion_speed(options, defaults["transit_speed_deg_s"])
            return self._start_planned_hold(options.get("plan_id"), poses, seconds, speed)
        speed = checked_value(options.get("maximum_speed_deg_s",
                              self.system["dashboard"]["drag_test"]["maximum_speed_deg_s"]),
                              self.system, "drag_test.maximum_speed_deg_s")
        normalized = {"arm": arm, "maximum_speed_deg_s": speed, "source": source}
        context = self._hold_context()

        with self._lock:
            if self._state != IDLE or self.planning:
                return {"ok": False, "message": f"{self._activity} is running"}
            # Reserve the one activity slot before mkdir/Popen. The HTTP
            # server is threaded, so checking and claiming in separate lock
            # sections would allow two hardware owners to launch together.
            self._state = RUNNING
            self._activity = mode
            self._abort.clear()
            self._started_at = time.monotonic()
            self._options = normalized
            self.result = None
            self.progress = {"mode": mode, "phase": "starting"}

        folder = None
        try:
            root = Path(self.config.output_directory)
            root.mkdir(parents=True, exist_ok=True)
            base = root / f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}"
            for collision in range(1000):
                candidate = (base if collision == 0
                             else base.with_name(f"{base.name}-{collision}"))
                try:
                    candidate.mkdir()
                    folder = candidate
                    break
                except FileExistsError:
                    continue
            if folder is None:
                raise OSError("could not allocate a gravity-test output directory")
            status_file = folder / "status.json"
            summary_file = folder / "gravity_test_summary.json"
            output = folder / "drag.json"
            snapshot = write_system_config_snapshot(folder, self.system)
            command = [
                "ros2", "run", "rm_control", "manual_drag",
                "--arm", arm,
                "--maximum-speed-deg-s", str(speed),
                "--output", str(output), "--status-file", str(status_file),
                "--ack", GRAVITY_TEST_ACKNOWLEDGEMENT,
                "--system-config", str(snapshot),
                "--source", source,
            ]
            with self._lock:
                if context != self._hold_context() or source != self._gravity_source(ArmIdentity(arm)):
                    raise ValueError("gravity source or selected arm changed before launch")
                ArmBinding.from_description(ArmIdentity(arm), self.urdf_text)
            process = self._process_launcher(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
        except Exception as error:  # noqa: BLE001
            if folder is not None:
                try:
                    folder.rmdir()
                except OSError:
                    pass
            with self._lock:
                self._state = IDLE
                self._activity = ""
                self.progress = {"mode": mode, "phase": "failed",
                                 "error": str(error)}
            self.publish_event(
                f"could not start {mode}: {error}", level="error", source=mode)
            return {"ok": False, "message": f"could not start gravity test: {error}"}

        with self._lock:
            self._external_process = process
            self._external_status_file = status_file
            self._external_summary_file = summary_file
            self.progress = {"mode": mode, "phase": "starting",
                             "output": str(folder)}
            stop_requested = self._abort.is_set()
        self.publish_event(f"{mode} started", source=mode)
        if stop_requested and process.poll() is None:
            try:
                if mode == GRAVITY_DRAG_TEST:
                    os.killpg(process.pid, signal.SIGINT)
                else:
                    process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        self._worker = threading.Thread(
            target=self._run_gravity_test, args=(mode, process, summary_file),
            daemon=True, name=f"identification-{mode}")
        self._worker.start()
        return {"ok": True, "message": f"{mode} started",
                "output": str(folder)}

    def _start_planned_hold(self, plan_id, poses, seconds, speed=None) -> dict:
        if speed is None:
            speed = self.system["dashboard"]["hold_test"]["transit_speed_deg_s"]
        with self._lock:
            if self._state != IDLE or self.planning:
                return {"ok": False, "message": "another activity is running"}
            permission = self.hold_plan_payload()
            if (not permission["available"] or not plan_id
                    or permission["id"] != plan_id or permission["poses"] != poses
                    or not hold_plan_module.source_is_current(self._hold_plan)):
                return {"ok": False, "message": "plan the hold poses again before execution"}
            plan = json.loads(json.dumps(self._hold_plan))
            try:
                position = self._hold_position()
                if np.max(np.abs(np.asarray(position) - plan["start_deg"])) > self.system["dashboard"]["hold_test"]["planning_drift_deg"]:
                    raise ValueError("arm moved since planning; plan again")
            except ValueError as error:
                return {"ok": False, "message": str(error)}
            self._state, self._activity = RUNNING, GRAVITY_HOLD_TEST
            self._hold_current_started = False
            self._hold_recovery_selection = (
                tuple(plan["joint_names"]),
                self.config.commands.follow_joint_trajectory_action)
            self._started_at = time.monotonic()
            self._abort.clear()
            self._completed_poses = {}
            self._options = {"poses": poses, "seconds": seconds, "plan_id": plan_id,
                             "transit_speed_deg_s": speed}
            self.result = None
            self.progress = {"mode": GRAVITY_HOLD_TEST, "phase": "starting"}
            self._worker = threading.Thread(
                target=self._run_planned_hold, args=(plan, seconds), daemon=True,
                name="identification-planned-hold")
            self._worker.start()
        return {"ok": True, "message": "planned hold started"}

    def _run_hold_child(self, command, timeout_s):
        with self._lock:
            if self._abort.is_set():
                raise RuntimeError("hold stopped before child launch")
            if (not self._hold_plan or self._hold_plan["context"] != self._hold_context()
                    or not hold_plan_module.source_is_current(self._hold_plan)):
                raise ValueError("hold plan or source changed before child launch")
            capability = self.gravity_test_capability()
            if not capability["available"]:
                raise ValueError(capability["reason"])
            try:
                arm = command[command.index("--arm") + 1]
                source = command[command.index("--source") + 1]
            except (ValueError, IndexError) as error:
                raise ValueError("hold child requires an explicit arm and source") from error
            if (arm != capability["arm"] or hold_plan_module._source_path(source)
                    != hold_plan_module._source_path(capability["source"])):
                raise ValueError("hold child arm or source differs from the selected calibration")
            endpoint = getattr(self.bridge, "controller_state_url", None)
            endpoint = endpoint() if callable(endpoint) else None
            if isinstance(endpoint, str) and endpoint:
                command = [*command, "--controller-state-url", endpoint]
            process = self._process_launcher(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True)
            self._external_process = process
            self._hold_current_started = True
        deadline = time.monotonic() + timeout_s
        interrupted = False
        while True:
            if not interrupted and (self._abort.is_set() or self.screen_drift()
                                    or time.monotonic() >= deadline):
                self.stop()
                interrupted = True
            try:
                stdout, stderr = process.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
        with self._lock:
            self._external_process = None
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    def _move_hold_target(self, pose) -> dict:
        from ..plants.ros_control import MotionStopUnverified

        if self._abort.is_set():
            raise RuntimeError("hold stopped before motion")
        context = self._hold_context()
        selection = (tuple(self.driven_joints or self.arm.joint_names),
                     self.config.commands.follow_joint_trajectory_action)
        plant = self.bridge.hardware_plant(
            self.profile, self.scene, require_neutral_start=False,
            maximum_speed_deg_s=self._motion_speed(
                self._options, self.system["dashboard"]["hold_test"]["transit_speed_deg_s"]))
        try:
            plant.set_monitor(self._monitor())
            plant.set_stop_requested(
                lambda: self._abort.is_set() or bool(self.screen_drift())
                or self._hold_context() != context
                or not self.connection().get("telemetry_ok"))
            if not self._hold_path_clear(self._hold_position(), pose):
                raise ValueError("hold transit changed before motion")
            plant.move_to(pose)
            tolerance = self.system["dashboard"]["hold_test"]["target_tolerance_deg"]
            sample = plant.wait_for_position(
                pose, tolerance_deg=tolerance,
                timeout_s=self.system["dashboard"]["hold_test"]["target_timeout_s"])
            actual = np.asarray((sample or {}).get("position_deg"), dtype=float)
            if actual.shape != (7,) or not np.isfinite(actual).all():
                raise ValueError("move returned no fresh position telemetry")
            error = float(np.max(np.abs(actual - pose)))
            return {"ok": error <= tolerance, "error_deg": error,
                    "ros_deg": actual.tolist(),
                    "reason": "" if error <= tolerance else "hold target was not reached"}
        except MotionStopUnverified:
            self._hold_recovery_required = True
            self._hold_recovery_selection = selection
            raise
        finally:
            self._release(plant)

    def _run_planned_hold(self, plan, seconds) -> None:
        folder = None
        try:
            root = Path(self.config.output_directory)
            root.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix="gravity_hold_test-", dir=root))
            (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
            preview = self._build_preview("running", {"phases": [
                {"phase": "hold_set", "detail": {"poses_deg": plan["poses_deg"]}}]})
            with self._lock:
                self.preview = preview
                self.preview_token += 1

            def progress(phase, detail):
                if plan["context"] != self._hold_context() or self.screen_drift():
                    raise ValueError("collision scene changed; hold stopped")
                if detail.get("stage") == "moving":
                    if not self._hold_path_clear(self._hold_position(), detail["pose_deg"]):
                        raise ValueError("hold transit is no longer clear")
                self._on_progress(phase, detail)

            result = hold_plan_module.execute_hold_plan(
                plan, seconds, folder, self._abort, progress, self._run_hold_child,
                move=self._move_hold_target, system_config=self.system)
            if self._hold_current_started and not result.get("stop_verified"):
                self._hold_recovery_required = True
            result.update(mode=GRAVITY_HOLD_TEST, plan_id=plan["id"],
                          transit_speed_deg_s=self._options["transit_speed_deg_s"])
            summary = folder / "gravity_test_summary.json"
            summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["evidence"] = "/runs/" + summary.relative_to(root).as_posix()
            with self._lock:
                self.result = result
                self.progress = {"mode": GRAVITY_HOLD_TEST,
                                 "phase": "complete" if result["result"] == "PASS" else "failed",
                                 "result": result["result"], "output": str(folder),
                                 "error": result.get("reason", "")}
        except Exception as error:
            with self._lock:
                if self._hold_current_started:
                    self._hold_recovery_required = True
                self.result = {"mode": GRAVITY_HOLD_TEST, "result": "FAIL", "reason": str(error)}
                self.progress = {"mode": GRAVITY_HOLD_TEST, "phase": "failed", "error": str(error)}
        finally:
            with self._lock:
                if self._hold_recovery_required:
                    self._state = PAUSED
                    self.progress["error"] = (
                        "stop or recovery was not confirmed; operator recovery required")
                else:
                    self._state, self._activity = IDLE, ""
                self._worker = None
                self._hold_plan = {}

    def _run_gravity_test(self, mode: str, process, summary_file: Path) -> None:
        try:
            output = getattr(process, "stdout", None)
            if output is not None:
                for line in output:
                    text = line.strip()
                    if not text:
                        continue
                    if text.startswith("GRAVITY_TEST "):
                        try:
                            progress = json.loads(text[len("GRAVITY_TEST "):])
                        except json.JSONDecodeError:
                            progress = {"phase": "running", "message": text}
                        self._on_progress(str(progress.get("phase", "running")), progress)
                    else:
                        self.publish_event(text[-500:], source=mode)
            exit_code = process.wait()
            if summary_file.is_file():
                result = json.loads(summary_file.read_text(encoding="utf-8"))
            else:
                result = {"mode": mode, "result": "FAIL",
                          "reason": "gravity test wrote no summary",
                          "child_exit_code": exit_code}
            result["mode"] = mode
            result["child_exit_code"] = exit_code
            if exit_code != 0 and result.get("result") == "PASS":
                result["reported_result"] = "PASS"
                result["result"] = "FAIL"
                result["reason"] = (
                    f"gravity test exited {exit_code} despite a PASS summary")
            result["evidence"] = (
                "/runs/"
                + summary_file.relative_to(
                    Path(self.config.output_directory)).as_posix())
            verdict = str(result.get("result", "FAIL"))
            phase = ("complete" if verdict == "PASS" else
                     "stopped" if verdict == "STOPPED" else "failed")
            with self._lock:
                self.result = result
                self.progress = {
                    "mode": mode,
                    "phase": phase,
                    "result": verdict,
                    "output": str(summary_file.parent),
                    "error": "" if verdict == "PASS" else
                    str(result.get("reason") or result.get("child_result", {}).get("reason", "")),
                }
            level = ("info" if verdict == "PASS" else
                     "warning" if verdict == "STOPPED" else "error")
            self.publish_event(
                f"{mode} {verdict.lower()}",
                level=level, source=mode)
        except Exception as error:  # noqa: BLE001
            with self._lock:
                self.result = {"mode": mode, "result": "FAIL", "reason": str(error)}
                self.progress = {"mode": mode, "phase": "failed",
                                 "error": str(error)}
            self.publish_event(f"{mode} failed: {error}", level="error", source=mode)
        finally:
            with self._lock:
                self._external_process = None
                self._external_status_file = None
                self._external_summary_file = None
                self._state = IDLE
                self._activity = ""
                if self._worker is threading.current_thread():
                    self._worker = None

    def pause(self) -> dict:
        """Request a gravity hardware pause at the next safe JTC boundary."""
        with self._lock:
            if self._state != RUNNING or self._activity != GRAVITY_MODE:
                return {"ok": False,
                        "message": "only a running gravity hardware campaign "
                                   "can be paused"}
            if not self._run_gate.is_set():
                return {"ok": True, "message": "pause already requested"}
            self._run_gate.clear()
            carried = dict(self.progress)
            carried.update({"pause_pending": True,
                            "updated_fields": ["pause_pending"]})
            self.progress = carried
        self.publish_event(
            "pause requested; waiting for the current trajectory to finish",
            level="warning", source=GRAVITY_MODE)
        return {"ok": True, "message": "pause requested"}

    def resume(self) -> dict:
        """Release a pending or settled gravity pause."""
        with self._lock:
            if (self._activity != GRAVITY_MODE
                    or self._state not in (RUNNING, PAUSED)
                    or self._run_gate.is_set()):
                return {"ok": False,
                        "message": "no paused gravity campaign to resume"}
        drifted = self.screen_drift()
        if drifted:
            where = ", ".join(
                f"{item['joint']} moved {item['moved_deg']:+g} deg"
                for item in drifted[:4])
            return {"ok": False,
                    "message": "resume refused: the collision screen no "
                               f"longer matches the other arm; {where}"}
        with self._lock:
            if (self._activity != GRAVITY_MODE
                    or self._state not in (RUNNING, PAUSED)
                    or self._run_gate.is_set()):
                return {"ok": False,
                        "message": "no paused gravity campaign to resume"}
            self._run_gate.set()
        self.publish_event("resume requested", source=GRAVITY_MODE)
        return {"ok": True, "message": "resume requested"}

    def _wait_if_paused(self, phase: str, pose: int, poses: int) -> bool:
        """Block the worker between goals, leaving stop able to wake it."""
        if self._run_gate.is_set():
            return not self._abort.is_set()
        with self._lock:
            if self._run_gate.is_set():
                return not self._abort.is_set()
            carried = dict(self.progress)
            carried.update({
                "mode": self._activity,
                "phase": phase,
                "paused": True,
                "pause_pending": False,
                "target_pose": pose,
                "poses": poses,
                "updated_fields": ["paused", "target_pose", "poses"],
            })
            self.progress = carried
            self._state = PAUSED
        self.publish_event(
            f"paused before pose {pose}/{poses}; that pose will restart on resume",
            level="warning", source=GRAVITY_MODE)
        while not self._abort.is_set():
            if not self._run_gate.wait(timeout=0.2):
                continue
            drifted = self.screen_drift()
            if not drifted:
                break
            self._run_gate.clear()
            where = ", ".join(
                f"{item['joint']} moved {item['moved_deg']:+g} deg"
                for item in drifted[:4])
            self.publish_event(
                "resume blocked: the collision screen no longer matches the "
                f"other arm; {where}", level="error", source=GRAVITY_MODE)
        if self._abort.is_set():
            return False
        with self._lock:
            carried = dict(self.progress)
            carried.update({
                "paused": False,
                "pause_pending": False,
                "resumed": True,
                "updated_fields": ["resumed", "target_pose", "poses"],
            })
            self.progress = carried
            self._state = RUNNING
        self.publish_event(
            f"resumed; restarting pose {pose}/{poses}", source=GRAVITY_MODE)
        return True

    def home(self, options: dict | None = None) -> dict:
        """Drive every joint back to neutral.

        A campaign leaves the arm wherever validation ended, and the hardware
        plant refuses to start more than a degree from neutral, so without this
        the next run cannot be armed without hand-driving the arm.
        """
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        if self.profile is None:
            return {"ok": False, "message": "no robot profile loaded"}
        speed = self._motion_speed(
            options or {}, self.system["dashboard"]["home"]["transit_speed_deg_s"])
        with self._lock:
            if self._state != IDLE or self.planning:
                return {"ok": False, "message": f"{self._activity} is running"}
            self._state = RUNNING
            self._activity = "homing"
            self._options = {"transit_speed_deg_s": speed}
            self._abort.clear()
            self._started_at = time.monotonic()
            self.progress = {"mode": "homing", "phase": "starting"}
        self._worker = threading.Thread(
            target=self._run_home, daemon=True, name="identification-homing")
        self._worker.start()
        return {"ok": True, "message": "homing started"}

    def _run_home(self) -> None:
        plant = None
        try:
            if self.bridge is None:
                raise RuntimeError("no ROS bridge; cannot drive hardware")
            # The neutral check exists to refuse campaigns that start off-home.
            # Homing is the one job that must run precisely then.
            plant = self.bridge.hardware_plant(
                self.profile, self.scene,
                require_neutral_start=False,
                maximum_speed_deg_s=self._motion_speed(
                    self._options, self.system["dashboard"]["home"]["transit_speed_deg_s"]))
            setter = getattr(plant, "set_monitor", None)
            if setter is not None:
                setter(self._monitor())
            before = [float(v) for v in plant.sample()["position_deg"]]
            self._on_progress("moving",
                              {"from_deg": [round(v, 2) for v in before]})
            if self._abort.is_set():
                raise RuntimeError("stopped before the arm moved")
            plant.park()
            after = [float(v) for v in plant.sample()["position_deg"]]
            worst = max(abs(v) for v in after)
            self.note(f"homed: worst joint {worst:.3f} deg from neutral")
            with self._lock:
                self.progress = {
                    "mode": "homing", "phase": "homed",
                    "elapsed_s": time.monotonic() - self._started_at,
                    "from_deg": [round(v, 2) for v in before],
                    "to_deg": [round(v, 3) for v in after],
                    "worst_deg": round(worst, 3),
                }
        except Exception as error:  # noqa: BLE001 - a crash must not be silent
            self.note(f"homing failed: {error}")
            self.progress = {"mode": "homing", "phase": "failed",
                             "error": str(error),
                             "traceback": traceback.format_exc()[-2000:]}
        finally:
            self._release(plant)
            with self._lock:
                self._state = IDLE
                self._activity = ""

    def running(self) -> bool:
        return self._state == RUNNING

    # -- jogging ---------------------------------------------------------

    def jog(self, body: dict) -> dict:
        """One endpoint for the three things a slider does."""
        action = str((body or {}).get("action") or "").strip()
        if action == "start":
            return self.jog_start(body)
        if action == "stop":
            return self.jog_stop()
        if action == "move":
            return self.jog_to((body or {}).get("position_deg"))
        return {"ok": False, "message": f"unknown jog action {action!r}"}

    def jog_start(self, options: dict | None = None) -> dict:
        """Hold a plant open for the session rather than one per slider release.

        Opening one costs an action handshake and a spin for state -- seconds,
        which is nothing once and unusable per drag.
        """
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        if self.profile is None:
            return {"ok": False, "message": "no robot profile loaded"}
        if self.bridge is None:
            return {"ok": False, "message": "no ROS bridge; cannot drive hardware"}
        speed = self._motion_speed(
            options or {}, self.system["dashboard"]["jog"]["transit_speed_deg_s"])
        with self._lock:
            if self._state == JOGGING:
                return {"ok": True, "message": "already jogging"}
            if self._state != IDLE or self.planning:
                return {"ok": False, "message": f"{self._activity} is running"}
            self._state = JOGGING
            self._activity = "jogging"
            self._options = {"transit_speed_deg_s": speed}
            self._abort.clear()
            self._jog_target = None
            self._started_at = time.monotonic()
            self.progress = {"mode": "jogging", "phase": "starting"}
        self._worker = threading.Thread(target=self._run_jog, daemon=True,
                                        name="identification-jog")
        self._worker.start()
        return {"ok": True, "message": "jogging enabled"}

    def jog_to(self, position_deg) -> dict:
        """Ask for a pose. The newest ask wins; a drag is not a queue."""
        with self._lock:
            if self._state != JOGGING:
                return {"ok": False, "message": "jogging is not enabled"}
        try:
            target = self._screened_pose(position_deg)
        except ValueError as error:
            return {"ok": False, "message": str(error)}
        with self._lock:
            self._jog_target = target
        return {"ok": True, "target_deg": [round(value, 2) for value in target]}

    def jog_stop(self) -> dict:
        with self._lock:
            if self._state != JOGGING:
                return {"ok": True, "message": "not jogging"}
        self._abort.set()
        return {"ok": True, "message": "jogging stopped"}

    def jog_range_deg(self) -> list[list[float]]:
        """Where each joint may be jogged: the campaign's own envelope.

        Deliberately not the URDF's: the URDF describes the arm, not the bench
        it is bolted to, and a slider is the easiest way there is to drive an
        arm into its surroundings. Both bounds, because the envelope may be
        asymmetric and clamping to the upper one as a magnitude would let a
        joint go somewhere the planner is forbidden from designing.
        """
        if self.plan is None or self.arm is None:
            return []
        limits = self.plan.design_limits(self.arm)
        return [[round(float(low), 1), round(float(high), 1)]
                for low, high in zip(limits.lower_deg, limits.upper_deg)]

    def jog_limits_deg(self) -> list[float]:
        """The envelope as one magnitude per joint, for anything symmetric."""
        return [round(max(abs(low), abs(high)), 1)
                for low, high in self.jog_range_deg()]

    def _screened_pose(self, position_deg) -> list[float]:
        """Clamp to the envelope, then refuse anything that would hit something.

        Both checks belong here rather than in the browser: the request does
        not have to have come from this dashboard, and the arm has no idea what
        is bolted around it.
        """
        limits = self.jog_range_deg()
        values = [float(value) for value in (position_deg or [])]
        if not limits:
            raise ValueError("no motion plan yet")
        if len(values) != len(limits):
            raise ValueError(f"expected {len(limits)} joint angles, "
                             f"got {len(values)}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint angles must be finite")
        clamped = [max(bound[0], min(bound[1], value))
                   for value, bound in zip(values, limits)]
        report = self.collision_report(clamped)
        if report.get("available") and not report.get("clear", True):
            touching = ", ".join(
                str(contact) for contact in (report.get("contacts") or [])[:3])
            raise ValueError("that pose is in collision"
                             + (f": {touching}" if touching else ""))
        return clamped

    def _run_jog(self) -> None:
        plant = None
        try:
            speed = self._motion_speed(
                self._options, self.system["dashboard"]["jog"]["transit_speed_deg_s"])
            plant = self.bridge.hardware_plant(
                self.profile, self.scene, require_neutral_start=False,
                maximum_speed_deg_s=speed)
            setter = getattr(plant, "set_monitor", None)
            if setter is not None:
                setter(self._monitor())
            self.note(f"jogging enabled at {speed:g} deg/s")
            self._on_progress("jogging", {})
            while not self._abort.is_set():
                with self._lock:
                    target, self._jog_target = self._jog_target, None
                if target is None:
                    time.sleep(self.system["runtime"]["jog_poll_s"])
                    continue
                plant.move_to(target)
                self._on_progress(
                    "jogging", {"to_deg": [round(value, 1) for value in target]})
        except Exception as error:  # noqa: BLE001 - a crash must not be silent
            self.note(f"jogging stopped: {error}")
            self.progress = {"mode": "jogging", "phase": "failed",
                             "error": str(error),
                             "traceback": traceback.format_exc()[-2000:]}
        else:
            self.note("jogging disabled")
            self.progress = {"phase": "idle"}
        finally:
            self._release(plant)
            with self._lock:
                self._state = IDLE
                self._activity = ""
                self._jog_target = None

    def _run(self, mode: str) -> None:
        if mode == "load_sweep":
            self._run_load_sweep()
            return
        plant = None
        run = None
        previous_report = self._reports.get(mode)
        failure = None
        gravity = mode in (GRAVITY_MODE, GRAVITY_REHEARSAL)
        try:
            monitor = self._monitor() if mode in HARDWARE_MODES else None
            if gravity:
                plan = self._gravity_plan(self._options)
                self._remember_gravity(plan)
            elif mode == OPTIMAL_MODE:
                plan = self._optimal_plan()
            else:
                plan = self.plan
            reused = None
            if mode == OPTIMAL_MODE and self._options.get(
                    "reuse_friction", self.system["dashboard"]["optimal"]["reuse_friction"]):
                folder = self._latest_optimal_friction(plan)
                if folder is None:
                    raise RuntimeError(
                        "no compatible completed optimal low-speed phase found")
                reused = self._read_optimal_friction(folder)
            plant = self._build_plant(mode, plan)
            pause_setter = getattr(plant, "set_pause_requested", None)
            if pause_setter is not None and mode == GRAVITY_MODE:
                pause_setter(lambda: not self._run_gate.is_set())
            setter = getattr(plant, "set_monitor", None)
            if setter is not None and monitor is not None:
                setter(monitor)
            # What this run is about to validate, read back when it finishes.
            self._running_signature = (self._gravity_signature(plan)
                                       if gravity else "")
            if gravity:
                run_type = campaign_module.GravityCampaign
            elif mode == OPTIMAL_MODE:
                run_type = campaign_module.OptimalExcitationCampaign
            else:
                run_type = campaign_module.Campaign
            run = run_type(
                self.arm, plant, plan,
                progress=self._on_progress,
                should_stop=self._abort.is_set,
                wait_if_paused=self._wait_if_paused,
                monitor=monitor)
            if reused is not None:
                records, folder, phase = reused
                run.reuse_low_speed_friction(records, str(folder), phase)
                self.note(f"reusing {len(records)} low-speed observations "
                          f"from {folder}")
            self.progress = {"mode": mode, "phase": "starting"}
            result = run.run()
            if mode == OPTIMAL_MODE:
                result.comparison = self._compare_with_load_sweep(
                    result, run.observations)
            self._finish(mode, result, run.observations,
                         getattr(plant, "raw_frames", None))
        except Exception as error:  # noqa: BLE001 - a crash must not be silent
            failure = str(error)
            self.note(f"{mode} run failed: {error}")
            self.progress = {"mode": mode, "phase": "failed",
                             "error": str(error),
                             "traceback": traceback.format_exc()[-2000:]}
            self._salvage(mode, run, plant, error)
        finally:
            try:
                if mode in HARDWARE_MODES:
                    evidence = getattr(plant, "failure_evidence", lambda: {})()
                    entry = self._reports.get(mode)
                    folder = (Path(self.config.output_directory) / entry["name"]
                              if entry is not None and entry is not previous_report else None)
                    payload = (json.loads((folder / report_module.RESULT_NAME).read_text(
                        encoding="utf-8")) if folder is not None else {
                            "mode": mode, "complete": False, "aborted": failure})
                    if evidence or failure or payload.get("aborted"):
                        payload["failure_evidence"] = evidence
                        observations = getattr(run, "observations", None) or []
                        raw_frames = getattr(plant, "raw_frames", None) or []
                        if folder is None:
                            self._write(payload, mode, observations, raw_frames)
                            entry = self._reports.get(mode)
                            if entry is not None and entry is not previous_report:
                                folder = Path(self.config.output_directory) / entry["name"]
                        with self._lock:
                            self.result = payload
                        if folder is not None:
                            (folder / "failure_evidence.json").write_text(
                                json.dumps(evidence, indent=2), encoding="utf-8")
                            (folder / report_module.RESULT_NAME).write_text(
                                json.dumps(payload, indent=2, ensure_ascii=False),
                                encoding="utf-8")
                            (folder / report_module.REPORT_NAME).write_text(
                                report_module.render_report(
                                    payload, observation_rows=len(observations),
                                    raw_rows=len(raw_frames),
                                    stamp=folder.name.removeprefix(f"{mode}-")),
                                encoding="utf-8")
            except Exception as evidence_error:
                self.note(f"could not write failure evidence: {evidence_error}")
            self._release(plant)
            self._run_gate.set()
            with self._lock:
                self._state = IDLE
                self._activity = ""

    def _apply_options(self, plan, names, options: dict) -> None:
        """Operator numbers onto a plan, bounded by this arm's envelope."""
        if self.profile is None:
            raise ValueError("a robot profile is required for campaign ranges")
        bounds = campaign_module.campaign_bounds(self.profile, system_config=self.system)
        for name in names:
            if name not in options:
                continue
            low, high = bounds[name]
            try:
                value = float(options[name])
            except (TypeError, ValueError):
                self.note(f"ignoring option {name}={options[name]!r}")
                continue
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            bounded = min(max(value, low), high)
            if bounded != value:
                self.note(f"{name} {value:g} clamped to {bounded:g}")
            setattr(plan, name,
                    campaign_module.coerce_plan_value(name, bounded))

    def _gravity_plan(self, options: dict):
        """The gravity card's numbers, bounded, on a copy of the plan."""
        options = self.gravity_options | options
        plan = replace(self.plan)
        if self._gravity_seed is not None:
            plan.seed = self._gravity_seed
        plan.transit_speed_deg_s = self._motion_speed(options, plan.transit_speed_deg_s)
        plan.gravity_probe_speeds_deg_s = tuple(
            self.system["dashboard"]["gravity"]["gravity_probe_speeds_deg_s"])
        plan.start_deg = self._standing_deg()
        self._apply_options(plan, GRAVITY_OPTIONS, options)
        speeds = options.get("gravity_probe_speeds_deg_s", plan.gravity_probe_speeds_deg_s)
        if speeds:
            minimum, maximum = configured_range(self.system, "gravity.probe_speed_deg_s")
            ceiling = min(maximum, float(plan.maximum_speed_deg_s))
            if minimum > ceiling:
                raise ValueError("gravity probe speed range does not overlap the robot/plan speed limit")
            try:
                cleaned = sorted({round(min(max(float(value), minimum), ceiling), 3)
                                  for value in speeds if math.isfinite(float(value))
                                  and float(value) > 0.0})
            except (TypeError, ValueError):
                cleaned = []
                self.note(f"ignoring probe speeds {speeds!r}")
            if cleaned:
                plan.gravity_probe_speeds_deg_s = tuple(cleaned)
        least = self._minimum_gravity_poses()
        if least and plan.static_poses < least:
            self.note(f"{plan.static_poses} poses cannot identify joint 1: it "
                      f"carries {least} gravity terms and a pose gives one row")
        return plan

    def _gravity_signature(self, plan) -> str:
        """What a gravity dry run validated, as one comparable string.

        Every field here changes the experiment, so a rehearsal of one set of
        them says nothing about another. The panel stays editable; the arming
        is what expires.
        """
        return json.dumps({
            "static_poses": plan.static_poses,
            "transit_speed_deg_s": plan.transit_speed_deg_s,
            "static_candidates": plan.static_candidates,
            "gravity_validation_poses": plan.gravity_validation_poses,
            "gravity_probe_deg": plan.gravity_probe_deg,
            "gravity_probe_speeds_deg_s": list(plan.gravity_probe_speeds_deg_s),
            "workspace_limit_deg": list(plan.workspace_limit_deg),
            "workspace_range_deg": [list(pair)
                                    for pair in plan.workspace_range_deg],
            # The tour is designed from here and its first transit is the
            # longest, so a dry run from one starting pose says nothing about
            # a real run from another.
            "start_deg": list(plan.start_deg),
            "seed": plan.seed,
            "joints": list(self.driven_joints),
        }, sort_keys=True)

    def _standing_deg(self) -> tuple:
        """Where the arm is now, quantised so a held pose reads the same twice.

        The quantum is well inside the tolerance the plant checks the start
        against, so rounding here cannot let the arm begin somewhere the
        design did not screen; it only stops encoder noise from expiring an
        arming every poll.
        """
        sample = self.latest_sample()
        pose = (sample or {}).get("position_deg")
        if not pose:
            return ()
        quantum = self.system["runtime"]["standing_quantum_deg"]
        return tuple(round(float(value) / quantum) * quantum for value in pose)

    def _optimal_plan(self):
        """Apply the small set of options exposed by the optimal-run card."""
        plan = replace(self.plan)
        self._apply_options(plan, ("optimal_training_trajectories",
                                   "optimal_validation_trajectories",
                                   "optimal_friction_repeats",
                                   "optimal_friction_postures",
                                   "fourier_base_frequency_hz",
                                   "fourier_duration_s"), self._options)
        return plan

    def _latest_load_sweep(self) -> Path | None:
        root = Path(self.config.output_directory) / "load_sweep"
        if not root.is_dir():
            return None
        folders = sorted(
            (path for path in root.glob("sweep-*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime, reverse=True)
        expected = list(self.driven_joints) or list(self.arm.joint_names)
        for folder in folders:
            records = folder / loadsweep_module.RECORDS_NAME
            try:
                if not records.is_file() or records.stat().st_size <= 0:
                    continue
            except OSError:
                continue
            try:
                _rows, manifest = loadsweep_report.read(folder)
            except (OSError, ValueError, KeyError):
                continue
            recorded = list(manifest.get("joint_names") or [])
            if not recorded or recorded == expected:
                return folder
        return None

    def _latest_optimal_friction(self, plan) -> Path | None:
        """Newest saved B phase whose design exactly matches this request."""
        root = Path(self.config.output_directory)
        if not root.is_dir():
            return None
        expected_names = list(self.driven_joints) or list(self.profile.joint_names)
        folders = sorted(
            (path for path in root.glob("optimal_excitation-*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime, reverse=True)
        for folder in folders:
            result_path = folder / report_module.RESULT_NAME
            observations_path = folder / report_module.OBSERVATIONS_NAME
            if not result_path.is_file() or not observations_path.is_file():
                continue
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            saved_plan = payload.get("plan") or {}
            if list(payload.get("joint_names") or []) != expected_names:
                continue
            if payload.get("action") != self.config.commands.follow_joint_trajectory_action:
                continue
            if payload.get("effort_source") != self.config.telemetry.signals.effort_source:
                continue
            if tuple(saved_plan.get("optimal_friction_speeds_deg_s") or ()) != tuple(
                    plan.optimal_friction_speeds_deg_s):
                continue
            if int(saved_plan.get("optimal_friction_repeats") or 0) != int(
                    plan.optimal_friction_repeats):
                continue
            if int(saved_plan.get("optimal_friction_postures") or 0) != int(
                    plan.optimal_friction_postures):
                continue
            phase = next((entry for entry in payload.get("phases") or []
                          if entry.get("phase") == campaign_module.PHASE_FRICTION),
                         None)
            detail = (phase or {}).get("detail") or {}
            if (phase is None or phase.get("aborted")
                    or int(detail.get("planned_passes") or 0) <= 0
                    or int(detail.get("completed_passes") or 0)
                    != int(detail.get("planned_passes") or 0)):
                continue
            try:
                self._read_optimal_friction(folder)
            except (OSError, ValueError, KeyError, StopIteration):
                continue
            return folder
        return None

    def _read_optimal_friction(self, folder: Path):
        """Load fitted B-phase windows; raw source frames remain in that folder."""
        payload = json.loads(
            (folder / report_module.RESULT_NAME).read_text(encoding="utf-8"))
        names = list(payload.get("joint_names") or [])
        expected = list(self.driven_joints) or list(self.profile.joint_names)
        if names != expected:
            raise ValueError("saved low-speed phase names another arm")
        phase = next(entry for entry in payload.get("phases") or []
                     if entry.get("phase") == campaign_module.PHASE_FRICTION)

        def values(row, suffix, required=True):
            raw = [row.get(f"{name}.{suffix}", "") for name in names]
            if any(value == "" for value in raw):
                if required:
                    raise ValueError(f"saved low-speed data lacks {suffix}")
                return []
            return [float(value) for value in raw]

        records = []
        with (folder / report_module.OBSERVATIONS_NAME).open(
                newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("phase") != campaign_module.PHASE_FRICTION:
                    continue
                records.append(campaign_module.Observation(
                    phase=campaign_module.PHASE_FRICTION,
                    time_s=float(row.get("time_s") or 0.0),
                    position_deg=values(row, "position_deg"),
                    velocity_deg_s=values(row, "velocity_deg_s"),
                    acceleration_deg_s2=values(row, "acceleration_deg_s2"),
                    current_a=values(row, "effort"),
                    temperature_c=values(row, "temperature_c", required=False),
                    motion=str(row.get("motion") or ""),
                    window_frames=int(row.get("window_frames") or 0),
                    window_fit_rms_deg=float(
                        row.get("window_fit_rms_deg") or 0.0)))
        if not records:
            raise ValueError("saved low-speed phase contains no fitted observations")
        return records, folder, phase

    def _compare_with_load_sweep(self, result, observations) -> dict:
        if self.config.telemetry.signals.effort_source != "current":
            return {"available": False,
                    "reason": "load sweep baseline is in current, but this "
                              "campaign identified another effort quantity"}
        folder = self._latest_load_sweep()
        if folder is None:
            return {"available": False,
                    "reason": "no compatible completed load sweep was found"}
        validation = [record for record in observations
                      if record.phase == campaign_module.PHASE_VALIDATION]
        try:
            sweep = loadsweep_report.score_saved_sweep(
                folder, self.arm, validation,
                expected_joint_names=(list(self.driven_joints)
                                      or list(self.arm.joint_names)))
        except (OSError, ValueError, KeyError) as error:
            return {"available": False,
                    "reason": f"load sweep could not be scored: {error}"}
        comparison = _comparison(
            result.validation_rms_a, sweep,
            list(self.driven_joints) or list(self.arm.joint_names))
        if comparison.get("available"):
            self.note(
                "optimal excitation vs load sweep: "
                f"mean {comparison['mean_improvement_percent']:+.1f}%, "
                f"worst {comparison['worst_improvement_percent']:+.1f}%, "
                f"target {'met' if comparison['target_met'] else 'not met'}")
        else:
            self.note("optimal excitation comparison unavailable: "
                      + comparison.get("reason", "unknown reason"))
        return comparison

    def _sweep_plan(self) -> loadsweep_module.SweepPlan:
        """The requested sweep, with anything unspecified left at its default."""
        plan = loadsweep_module.SweepPlan(**plan_defaults(self.system["load_sweep"]))
        for key, value in (getattr(self, "_options", None) or {}).items():
            if key == "resume" or not hasattr(plan, key):
                continue
            current = getattr(plan, key)
            try:
                if key == "joints":
                    setattr(plan, key, tuple(int(v) for v in value))
                elif isinstance(current, bool):
                    if type(value) is not bool:
                        raise ValueError(f"{key} must be boolean")
                    setattr(plan, key, value)
                elif isinstance(current, (int, float)):
                    setattr(plan, key, checked_value(value, self.system, f"load_sweep.{key}",
                                                     integer=isinstance(current, int)))
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid load sweep option {key}: {error}") from error
        for name in self.system["ranges"]["load_sweep"]:
            value = getattr(plan, name)
            ceiling = (float(self.profile.sustained_speed_deg_s)
                       if self.profile is not None and name in
                       ("slowest_deg_s", "fastest_deg_s", "transit_speed_deg_s") else None)
            checked_value(value, self.system, f"load_sweep.{name}",
                          integer=isinstance(value, int) and not isinstance(value, bool))
            if ceiling is not None and value > ceiling:
                checked_value(ceiling, self.system, f"load_sweep.{name}")
                setattr(plan, name, ceiling)
                self.note(f"load sweep {name} {value:g} limited to robot profile {ceiling:g}")
        if plan.slowest_deg_s <= 0 or plan.fastest_deg_s < plan.slowest_deg_s:
            raise ValueError("load sweep speeds must be positive and slowest <= fastest")
        if plan.transit_speed_deg_s <= 0 or plan.arc_ceiling_deg <= 0 or plan.minimum_level_gap_nm <= 0:
            raise ValueError("load sweep transit speed, arc and load spacing must be positive")
        if self.arm is not None and any(index < 0 or index >= self.arm.joint_count for index in plan.joints):
            raise ValueError("load sweep joint index is outside the current robot model")
        return plan

    def _sweep_folder(self) -> Path:
        """Where this sweep writes, continuing an earlier one when asked."""
        root = Path(self.config.output_directory) / "load_sweep"
        resume = (getattr(self, "_options", None) or {}).get("resume")
        if resume:
            if isinstance(resume, str) and resume not in ("", "latest", "true"):
                return root / resume
            existing = sorted((p for p in root.glob("sweep-*") if p.is_dir()),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if existing:
                return existing[0]
            self.note("no earlier sweep to resume; starting a new one")
        return root / time.strftime("sweep-%Y%m%d-%H%M%S")

    def _run_load_sweep(self) -> None:
        """Sweep every joint at every load gravity can be made to put on it.

        Separate from the campaign because it answers a different question. The
        campaign fits one friction number per joint from whatever postures the
        excitation happened to visit; this drives a speed ladder at a designed
        series of loads, so the load dependence is measured rather than
        inferred from three points that were chosen for conditioning.
        """
        plant = None
        run = None
        try:
            if self.bridge is None:
                raise RuntimeError("no ROS bridge; cannot drive hardware")
            # Before anything is planned, prove the screen can refuse. A scene
            # with no geometry passes every pose, and this sweep sends joints
            # to postures found by searching the whole workspace.
            loadsweep_module.proven_scene(self.scene, self.arm.joint_count)
            plan = self._sweep_plan()
            limits = self.plan.design_limits(self.arm)
            from ..plants.ros_control import HardwareConfig  # noqa: PLC0415

            self._on_progress("designing", {"joint": 0})
            designs = loadsweep_module.design_all(
                self.arm, self.scene, plan, HardwareConfig(),
                limits.lower_deg, limits.upper_deg,
                progress=self._on_progress)
            passes = sum(len(d.passes) for d in designs)
            hours = loadsweep_module.estimate_seconds(designs, plan) / 3600.0
            for design in designs:
                self.note(f"joint {design.joint + 1}: "
                          f"{len(design.levels)} levels over "
                          f"{design.span_nm:.3f} Nm, {len(design.passes)} passes"
                          + (f" -- {design.note}" if design.note else ""))
            self.note(f"{passes} passes designed, about {hours:.1f} hours")
            if not passes:
                raise RuntimeError("nothing to drive: no pose survived the "
                                   "load, limit and collision screens")

            folder = self._sweep_folder()
            plant = self.bridge.hardware_plant(
                self.profile, self.scene,
                maximum_speed_deg_s=plan.transit_speed_deg_s)
            setter = getattr(plant, "set_monitor", None)
            if setter is not None:
                setter(self._monitor())
            run = LoadSweepRun(self.arm, plant, plan, designs, folder,
                               progress=self._on_progress,
                               should_stop=self._abort.is_set,
                               note=self.note, scene=self.scene)
            outcome = run.run()
            self.note(f"load sweep finished: {outcome['driven']} passes driven, "
                      f"{len(outcome['skipped'])} skipped, written to {folder}")
            with self._lock:
                self.progress = {
                    "mode": "load_sweep", "phase": "complete",
                    "elapsed_s": time.monotonic() - self._started_at,
                    "driven": outcome["driven"],
                    "skipped": len(outcome["skipped"]),
                    "folder": str(folder)}
        except Exception as error:  # noqa: BLE001 - a crash must not be silent
            self.note(f"load sweep failed: {error}")
            self.progress = {"mode": "load_sweep", "phase": "failed",
                             "error": str(error),
                             "driven": getattr(run, "driven", 0),
                             "traceback": traceback.format_exc()[-2000:]}
        finally:
            self._release(plant)
            with self._lock:
                self._state = IDLE
                self._activity = ""

    def _salvage(self, mode: str, run, plant, error: Exception) -> None:
        """Write what was measured before the failure.

        A run that dies in its third hour still holds three hours of readings.
        Discarding them because the exception arrived on the way out means the
        operator is asked to spend those hours again, which is the one outcome
        the data was collected to avoid.
        """
        observations = list(getattr(run, "observations", None) or [])
        if not observations:
            return
        try:
            result = run.fit()
            result.aborted = str(error)
            result.skipped = list(getattr(run, "skipped", []))
            self._finish(mode, result, observations,
                         getattr(plant, "raw_frames", None))
            self.note(f"{mode} run salvaged: {len(observations)} observations "
                      "kept and fitted")
        except Exception as second:  # noqa: BLE001 - the raw data still matters
            # The fit needs phases the run never reached. The measurements do
            # not, and they are the expensive part.
            self._write({"mode": mode, "aborted": str(error),
                         "fit_failed": str(second)},
                        mode, observations, getattr(plant, "raw_frames", None))
            self.note(f"{mode} run salvaged unfitted: {len(observations)} "
                      f"observations kept, fit failed: {second}")

    def gravity_defaults(self) -> dict:
        """What the gravity card starts at, from the plan actually in force.

        Whatever was last committed wins over the plan's own numbers, so the
        card comes back showing the experiment this cell was tuned for rather
        than the module's defaults. Those values were bounded when they were
        committed, so they need no second clamp here.
        """
        plan = self.plan
        if plan is None:
            return {}
        defaults = {
            "static_poses": plan.static_poses,
            "transit_speed_deg_s": plan.transit_speed_deg_s,
            "gravity_validation_poses": plan.gravity_validation_poses,
            "gravity_probe_deg": plan.gravity_probe_deg,
            "gravity_probe_speeds_deg_s": list(
                self.system["dashboard"]["gravity"]["gravity_probe_speeds_deg_s"]),
        }
        defaults.update(self.gravity_options)
        defaults["minimum_poses"] = self._minimum_gravity_poses()
        return defaults

    def _remember_gravity(self, plan) -> None:
        """Keep the card's numbers, so a restart is not a retype.

        Stored after the bounding, so what comes back is what the run would
        have used and not what was typed at it.
        """
        wanted = {
            "static_poses": plan.static_poses,
            "transit_speed_deg_s": plan.transit_speed_deg_s,
            "gravity_validation_poses": plan.gravity_validation_poses,
            "gravity_probe_deg": plan.gravity_probe_deg,
            "gravity_probe_speeds_deg_s": list(plan.gravity_probe_speeds_deg_s),
        }
        if wanted == self.gravity_options:
            return
        self.gravity_options = wanted
        self._persist_config()

    def _minimum_gravity_poses(self) -> int:
        """Poses below which the first joint cannot be identified at all.

        Static passes give one row per pose per joint, and the joint nearest
        the base carries every link downstream of it: two gravity parameters
        each, plus its own Coulomb, viscous and offset terms. Fewer poses than
        that is not a noisier fit, it is an underdetermined one -- measured on
        this arm, ten poses left joint one with a held-out error of 0.35 A
        while every other joint sat at the 0.002 A noise floor.
        """
        count = self.arm.joint_count if self.arm is not None else 0
        return 2 * count + 3 if count else 0

    def _build_preview(self, mode: str, payload: dict) -> dict:
        """Where the run went, as skeletons the 3D canvas can draw.

        Read back out of the phases the run designed rather than re-derived
        from the plan, so what is drawn is what the arm was actually asked to
        visit -- including the poses a collision screen refused to place.
        """
        if self.arm is None:
            return {"available": False}
        if self.driven_joints and set(self.arm.joint_names) != set(self.driven_joints):
            return {"available": False}
        groups = []
        for phase in payload.get("phases") or []:
            name = str(phase.get("phase") or "")
            detail = phase.get("detail") or {}
            poses = list(detail.get("poses_deg") or [])
            if not poses:
                poses = [sweep.get("start_deg") for sweep
                         in detail.get("sweeps") or []
                         if sweep.get("start_deg")]
            entries = []
            for index, pose in enumerate(poses):
                try:
                    paths = self.arm.skeleton_paths(pose)
                except (ValueError, TypeError):
                    continue
                paths = [[[round(float(value), 5) for value in point]
                          for point in path] for path in paths]
                entry = {
                    "index": index + 1,
                    "pose_deg": [round(float(value), 3) for value in pose],
                    "points": paths[0] if len(paths) == 1 else [],
                    "paths": paths,
                }
                # How much room this pose has, so the operator reviewing them
                # can go straight to the tightest one instead of all of them.
                if self.scene is not None:
                    try:
                        entry["clearance"] = self.scene.clearance_rank(pose)
                    except (ValueError, RuntimeError):
                        pass
                entries.append(entry)
            if entries:
                groups.append({"phase": name, "poses": entries})
        return {
            "available": bool(groups),
            "mode": mode,
            "joint_names": list(self.arm.joint_names),
            "probes_m": list(obstacles_module.CLEARANCE_PROBES_M),
            "groups": groups,
        }

    def preview_payload(self) -> dict:
        with self._lock:
            payload = dict(self.preview)
        payload["token"] = self.preview_token
        return payload

    def _monitor(self):
        """The drive guards, with the voltage window only when it was supplied.

        A derived profile's window is a default rather than a measurement, and
        aborting a good run against a guessed threshold is worse than not
        checking; the panel says which guards are live either way.
        """
        written = self.profile is not None and self.profile_source == "configured"
        current = (written
                   and self.config.telemetry.signals.effort_source == "current"
                   and autoprofile.current_guard_active(self.profile))
        return campaign_module.DriveMonitor(
            minimum_voltage_v=self.profile.minimum_voltage_v if written else None,
            maximum_voltage_v=self.profile.maximum_voltage_v if written else None,
            maximum_speed_deg_s=(self.profile.peak_speed_deg_s
                                 if self.profile is not None else None),
            maximum_temperature_c=(self.profile.temperature_c
                                   if self.profile is not None else None),
            peak_current_a=(tuple(self.profile.peak_current_a)
                            if current else ()),
            continuous_current_a=(tuple(self.profile.continuous_current_a)
                                  if current else ()),
            sustained_current_window_s=(
                self.profile.sustained_current_window_s
                if current else 0.5))

    def _dark_guards(self) -> tuple[str, ...]:
        """Guards that will not fire, and why is not the operator's problem."""
        signals = self.config.telemetry.signals
        observed = set()
        if self.bridge is not None:
            observed = set(getattr(self.bridge, "observed_signals", set)() or ())
        live = set(self._monitor().guards())
        dark = []
        for guard, role in (("temperature ceiling", "temperature"),
                            ("drive-enabled check", "enabled"),
                            ("fault-code check", "fault_code"),
                            ("bus-voltage window", "voltage")):
            mapped = bool(getattr(signals, role, None))
            enforced = guard == "temperature ceiling" or guard in live
            if not mapped or not enforced or (observed and role not in observed):
                dark.append(guard)
        return tuple(dark)

    def _release(self, plant) -> None:
        """A hardware plant owns a ROS context; leaving it open leaks it."""
        closer = getattr(plant, "close", None)
        if closer is None:
            return
        try:
            closer()
        except Exception as error:  # noqa: BLE001
            self.note(f"plant did not close cleanly: {error}")

    def _on_progress(self, phase: str, detail: dict) -> None:
        # Within a phase, updates report different things -- pose index here,
        # sample count there -- so they merge. Across a phase boundary they
        # do not, or the old phase's pose index would haunt the new one.
        update = dict(detail or {})
        designed = update.pop("designed", None)
        updated_fields = list(update)
        if designed is not None:
            updated_fields.insert(0, "designed")
            update["designed_poses"] = len(designed)
        completed = update.get("completed_pose_indices")
        if completed is not None:
            with self._lock:
                self._completed_poses[phase] = [int(value) for value in completed]
        with self._lock:
            carried = (dict(self.progress)
                       if self.progress.get("phase") == phase else {})
            carried.update({"mode": self._activity, "phase": phase,
                            "elapsed_s": time.monotonic() - self._started_at})
            carried.update(update)
            carried["updated_fields"] = updated_fields
            self.progress = carried
        if designed is not None:
            self._publish_designed(phase, designed)
        # Between motions is the only place a run can notice that the robot it
        # was screened against has stopped being the robot standing there.
        if self._activity in HARDWARE_MODES and not self._abort.is_set():
            drifted = self.screen_drift()
            if drifted:
                where = ", ".join(f"{item['joint']} moved {item['moved_deg']:+g} deg"
                                  for item in drifted[:4])
                self.note(f"stopping: the screen no longer matches the robot; "
                          f"{where}")
                self._abort.set()

    def _publish_designed(self, phase: str, poses) -> None:
        """The poses a running phase just designed, drawn before it moves.

        A run designs its own poses, so without this the canvas has nothing to
        show until the run is over -- which is exactly when watching it stops
        being useful.
        """
        with self._lock:
            self._designed[phase] = [list(pose) for pose in poses]
            phases = [{"phase": name, "detail": {"poses_deg": entries}}
                      for name, entries in self._designed.items()]
        preview = self._build_preview("running", {"phases": phases})
        if not preview.get("available"):
            return
        with self._lock:
            self.preview = preview
            self.preview_token += 1

    def _finish(self, mode: str, result, observations, raw_frames=None) -> None:
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        payload["mode"] = mode
        # The per-joint fields are named _a for historical reasons but hold
        # whichever effort was regressed, so the file has to say which.
        signals = self.config.telemetry.signals
        payload["effort_source"] = signals.effort_source
        payload["effort_unit"] = signals.effort_unit
        # Without these a file cannot say which arm it describes, and two arms
        # writing into one directory become indistinguishable a week later.
        payload["joint_names"] = list(self.driven_joints) or (
            list(self.profile.joint_names) if self.profile else [])
        payload["action"] = self.config.commands.follow_joint_trajectory_action
        payload["provenance"] = self._provenance_payload(
            mode, observations, raw_frames)
        if mode in (GRAVITY_MODE, GRAVITY_REHEARSAL):
            payload["gravity_model"] = self._gravity_model_payload(payload)
        payload.update(self._plot_data(payload, observations,
                                       getattr(result, "fits", None)))
        aborted = payload.get("aborted")
        recovery = ({} if mode in HARDWARE_MODES else self._check_recovery(
            payload,
            self.system["rehearsal"]["gravity_holdout_tolerance"]
            if mode == GRAVITY_REHEARSAL else 0.0))
        if recovery:
            payload["rehearsal_check"] = recovery
        preview = self._build_preview(mode, payload)
        with self._lock:
            self.result = payload
            # A stopped run that still says "finished" is how a half-measured
            # model gets mistaken for a complete one.
            self.progress = {"mode": mode,
                             "phase": "stopped" if aborted else "finished"}
            if aborted:
                self.progress["error"] = str(aborted)
            passed = (bool(payload.get("complete"))
                      and bool(recovery.get("passed")))
            if mode == GRAVITY_REHEARSAL:
                self.gravity_armed = (
                    getattr(self, "_running_signature", "") if passed else "")
            elif mode not in HARDWARE_MODES:
                self.rehearsal_passed = passed
            if preview.get("available"):
                self.preview = preview
                self.preview_token += 1
        self._write(payload, mode, observations, raw_frames)
        if aborted:
            self.note(f"{mode} run stopped: {aborted}")
        elif recovery.get("available") and not recovery.get("passed"):
            self.note(f"{mode} completed but did not recover the planted "
                      f"friction: worst error "
                      f"{recovery['worst_coulomb_error']} > "
                      f"{recovery['tolerance']}")
        else:
            self.note(f"{mode} run complete")

    def _provenance_payload(self, mode: str, observations,
                            raw_frames) -> dict:
        """Evidence that distinguishes hardware data from an analytic run."""
        source = ("real_hardware" if mode in HARDWARE_MODES
                  else "analytic_rehearsal" if "rehearsal" in mode
                  else "offline_or_simulated")
        raw_count = len(raw_frames) if raw_frames is not None else 0
        observation_count = len(observations) if observations is not None else 0
        stamps = []
        motion_stamps: dict[tuple[str, str], list[float]] = {}
        raw_phase_counts: dict[str, int] = {}
        raw_phase_tag_mismatches = 0
        for frame in raw_frames or ():
            phase = str(frame.get("phase") or "")
            motion = str(frame.get("motion") or "")
            raw_phase_counts[phase] = raw_phase_counts.get(phase, 0) + 1
            expected_phase = (
                campaign_module.PHASE_VALIDATION
                if motion.startswith("gravity_check:") else
                campaign_module.PHASE_GRAVITY
                if motion.startswith("gravity:") else None)
            if expected_phase is not None and phase != expected_phase:
                raw_phase_tag_mismatches += 1
            try:
                stamp = float(frame.get("stamp_s"))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(stamp):
                stamps.append(stamp)
                key = (phase, motion)
                motion_stamps.setdefault(key, []).append(stamp)
        first = min(stamps) if stamps else None
        last = max(stamps) if stamps else None
        duration = (last - first if first is not None and last is not None
                    else None)
        motion_rates = []
        for values in motion_stamps.values():
            if len(values) < 2:
                continue
            span = max(values) - min(values)
            if span > 0.0:
                motion_rates.append((len(values) - 1) / span)
        implementation = {}
        for name, module in (("campaign", campaign_module),
                             ("identification", ident),
                             ("model", model_module)):
            try:
                implementation[name] = hashlib.sha256(
                    Path(module.__file__).read_bytes()).hexdigest()
            except (AttributeError, OSError, TypeError):
                implementation[name] = None
        try:
            package_version = metadata.version("robot_parameter_identification")
        except metadata.PackageNotFoundError:
            package_version = "source-tree"
        return {
            "schema_version": 1,
            "source": source,
            "hardware_evidence": source == "real_hardware" and raw_count > 0,
            "raw_frame_count": raw_count,
            "fitted_observation_count": observation_count,
            "publisher_stamp_start_s": first,
            "publisher_stamp_end_s": last,
            "publisher_duration_s": duration,
            "approximate_raw_rate_hz": (
                float(np.median(motion_rates)) if motion_rates else None),
            "raw_motion_groups": len(motion_stamps),
            "raw_phase_counts": raw_phase_counts,
            "raw_phase_tag_mismatches": raw_phase_tag_mismatches,
            "telemetry_transport": self.config.telemetry.transport(),
            "telemetry_topic": self.config.telemetry.topic(),
            "trajectory_action": self.config.commands.follow_joint_trajectory_action,
            "effort_source": self.config.telemetry.signals.effort_source,
            "effort_unit": self.config.telemetry.signals.effort_unit,
            "profile_source": self.profile_source,
            "software": {
                "package_version": package_version,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pinocchio_version": str(
                    getattr(getattr(ident, "pin", None), "__version__", "unknown")),
                "implementation_sha256": implementation,
            },
            "configuration": {
                "system_config": self.system_config_payload(),
                "profile": (_without_infinities(self.profile.as_dict())
                            if self.profile is not None else None),
                "collision_safety_margin_m": self.config.safety_margin_m,
                "obstacles": self.obstacles(),
                "locked_joint_positions_rad": {
                    str(name): float(value)
                    for name, value in self.screen_reference.items()
                },
            },
        }

    def _gravity_model_payload(self, payload: dict) -> dict:
        """A self-describing empirical predictor, not physical link inertias."""
        if self.arm is None:
            return {"available": False, "reason": "robot model unavailable"}
        direct = dict(payload.get("gravity_compensation") or {})
        if not direct.get("available"):
            return {
                "schema_version": 1,
                "available": False,
                "reason": direct.get(
                    "reason", "pair-averaged gravity model unavailable"),
                "pairing_audit": dict(direct.get("pairing_audit") or {}),
            }
        fit_entries = direct.get("joints") or []
        rigid_width = self.arm.parameter_count
        body_joints = [str(name) for name in self.arm.model.names[1:]]
        output_names = list(payload.get("joint_names") or [])
        joints = []
        for output, entry in enumerate(fit_entries):
            components = ModelComponents.from_dict(entry.get("components") or {})
            extra_names = components.column_names()
            retained = []
            for column, coefficient in zip(
                    entry.get("columns") or (), entry.get("parameters") or ()):
                index = int(column)
                if 0 <= index < rigid_width:
                    body = index // len(PINOCCHIO_INERTIAL_TERMS)
                    slot = index % len(PINOCCHIO_INERTIAL_TERMS)
                    body_name = (body_joints[body]
                                 if body < len(body_joints) else f"body_{body + 1}")
                    retained.append({
                        "index": index,
                        "kind": "rigid_body",
                        "body_joint": body_name,
                        "term": PINOCCHIO_INERTIAL_TERMS[slot],
                        "feature": f"{body_name}.{PINOCCHIO_INERTIAL_TERMS[slot]}",
                        "coefficient": float(coefficient),
                    })
                else:
                    extra = index - rigid_width
                    name = (extra_names[extra]
                            if 0 <= extra < len(extra_names)
                            else f"unknown_extra_{extra}")
                    retained.append({
                        "index": index,
                        "kind": "empirical_extra",
                        "term": name,
                        "feature": name,
                        "coefficient": float(coefficient),
                    })
            joints.append({
                "output_joint": (output_names[output]
                                 if output < len(output_names)
                                 else f"joint_{output + 1}"),
                "output_index": output,
                "retained_columns": retained,
                "components": components.as_dict(),
                "training_rms": entry.get("residual_rms_a"),
                "internal_holdout_rms": entry.get("holdout_rms_a"),
                "external_validation_rms": entry.get(
                    "external_validation_rms", entry.get("validation_rms_a")),
                "speed_pair_consistency_rms": entry.get(
                    "speed_pair_consistency_rms"),
            })
        gravity = getattr(getattr(self.arm.model, "gravity", None), "linear", ())
        return {
            "schema_version": 1,
            "available": bool(joints),
            "model_type": "pair_averaged_empirical_gravity_effort_regressor",
            "predictor_equation": (
                "gravity_effort_j = Y_j(q, 0, 0)[rigid_columns] * "
                "coefficients + offset_j"),
            "measurement_reduction": direct.get("friction_cancellation"),
            "fit_scope": "independent_per_output_joint",
            "effort_source": payload.get("effort_source"),
            "effort_unit": payload.get("effort_unit"),
            "position_unit": "degree",
            "velocity_unit": "degree_per_second",
            "training_phase": campaign_module.PHASE_GRAVITY,
            "external_validation_phase": campaign_module.PHASE_VALIDATION,
            "training_poses": direct.get("training_poses", 0),
            "external_validation_poses": direct.get("validation_poses", 0),
            "external_validation_rms": list(direct.get("validation_rms") or []),
            "validation_verdict": dict(direct.get("verdict") or {}),
            "pairing_audit": {
                "probe_speeds_deg_s": direct.get("probe_speeds_deg_s", []),
                "incomplete_training_poses": direct.get(
                    "incomplete_training_poses", []),
                "incomplete_validation_poses": direct.get(
                    "incomplete_validation_poses", []),
                "malformed_training_tags": direct.get(
                    "malformed_training_tags", []),
                "malformed_validation_tags": direct.get(
                    "malformed_validation_tags", []),
                "unexpected_training_observations": direct.get(
                    "unexpected_training_observations", []),
                "unexpected_validation_observations": direct.get(
                    "unexpected_validation_observations", []),
                "configuration_errors": direct.get("configuration_errors", []),
            },
            "urdf_sha256": hashlib.sha256(
                self.urdf_text.encode("utf-8")).hexdigest(),
            "urdf_artifact": report_module.MODEL_URDF_NAME,
            "driven_joint_names": output_names,
            "locked_joint_positions_rad": {
                str(name): float(value)
                for name, value in self.screen_reference.items()
            },
            "model_joint_names": body_joints,
            "gravity_vector_m_s2": [float(value) for value in gravity],
            "rigid_parameter_order": list(PINOCCHIO_INERTIAL_TERMS),
            "rigid_parameter_count": rigid_width,
            "joints": joints,
            "separate_friction_model": {
                "available": bool(payload.get("joints")),
                "source": "directional observations before pair averaging",
                "warning": (
                    "friction is an empirical nuisance model; two probe speed "
                    "magnitudes do not establish a physical Stribeck/load law"),
            },
            "physical_link_parameters": {
                "available": False,
                "mass": False,
                "center_of_mass": False,
                "rotational_inertia": False,
                "reason": (
                    "gravity-only current-domain fits are independent per "
                    "output joint and cannot recover one shared SI link model"),
            },
            "runtime": {
                "integrated_controller_loader": False,
                "reference_evaluator": "tools/identified_zero_force_drag.py",
                "requires_matching_urdf_sha256": True,
            },
        }

    def _plot_data(self, payload: dict, observations, fits=None) -> dict:
        """Scatter data for the charts, thinned to something a browser can draw.

        The friction plot carries measured current minus the rigid-body
        prediction, not raw current. Raw current is mostly gravity, which
        varies by pose over the campaign, so a friction curve laid over it was
        being compared against a cloud it never claimed to explain.

        Sweep samples are flagged. They are the only ones where a single joint
        moves fast around one nominal pose, so they are the only ones in which
        speed varies without pose varying with it. The rest of the cloud comes
        from many poses at low speed, and reading a speed trend across the two
        groups measures the pose difference as much as the speed difference.
        The flag is omitted when false to keep the polled payload small.

        Rows where the joint is standing still are left out. Friction has no
        determined sign at rest, so those rows say only where the servo
        settled inside the stiction band, and drawn against speed they pile
        into a vertical band at zero that looks like a low-speed peak.
        """
        joints = payload.get("joints") or []
        if not joints or not observations or self.arm is None:
            return {}
        acceleration_ceiling = (payload.get("data_quality") or {}).get(
            "acceleration_exclusion_deg_s2")
        moving = [record for record in observations
                  if getattr(record, "phase", "") != "D_validation"
                  and (acceleration_ceiling is None
                       or not record.acceleration_deg_s2
                       or max(abs(value) for value
                              in record.acceleration_deg_s2)
                       <= acceleration_ceiling)]
        if not moving:
            return {}
        stride = max(1, len(moving) // self.system["runtime"]["max_plot_points"])
        thinned = moving[::stride]
        fits = list(fits or [])
        if len(fits) != len(joints):
            # Without the regressions there is no prediction to subtract, and
            # an empty panel is better than a misleading one.
            return {}

        regressors = [
            self.arm.torque_regressor(record.position_deg,
                                      record.velocity_deg_s,
                                      record.acceleration_deg_s2)
            for record in thinned]

        friction, residual = [], []
        standstill = []
        for index, fit in enumerate(fits):
            curve, errors, still = [], [], 0
            for record, regressor in zip(thinned, regressors):
                try:
                    speed = float(record.velocity_deg_s[index])
                    measured = float(record.current_a[index])
                except (AttributeError, IndexError, TypeError):
                    continue
                if abs(speed) < self.system["runtime"]["still_speed_deg_s"] or _parked_here(record, index):
                    still += 1
                    continue
                acceleration = 0.0
                if record.acceleration_deg_s2 is not None and index < len(
                        record.acceleration_deg_s2):
                    acceleration = float(record.acceleration_deg_s2[index])
                rigid = ident.predict_joint(
                    fit, regressor, speed, include_friction=False,
                    acceleration=acceleration)
                whole = ident.predict_joint(
                    fit, regressor, speed, acceleration=acceleration)
                point = {"speed": round(speed, 6),
                         "effort": round(measured - rigid, 6),
                         "load": round(rigid, 6)}
                error = {"speed": round(speed, 6),
                         "residual": round(measured - whole, 6)}
                if _swept_here(record, index):
                    point["sweep"] = True
                    error["sweep"] = True
                curve.append(point)
                errors.append(error)
            friction.append(curve)
            residual.append(errors)
            standstill.append(still)
        return {"friction_samples": friction, "residual_samples": residual,
                "friction_standstill_excluded": standstill,
                "friction_standstill_speed_deg_s": self.system["runtime"]["still_speed_deg_s"]}

    def _build_plant(self, mode: str, plan=None):
        if mode in HARDWARE_MODES:
            if self.bridge is None:
                raise RuntimeError("no ROS bridge; cannot drive hardware")
            options = {"expected_start_deg": tuple(getattr(plan, "start_deg", ()) or ())}
            if mode == GRAVITY_MODE and plan is not None:
                options["maximum_speed_deg_s"] = self._motion_speed(
                    {"transit_speed_deg_s": plan.transit_speed_deg_s},
                    self.system["campaign"]["transit_speed_deg_s"])
            return self.bridge.hardware_plant(
                self.profile, self.scene, **options)
        from ..plants.analytic import AnalyticPlant  # noqa: PLC0415

        injected = self._rehearsal_friction()
        self._injected = injected
        return AnalyticPlant(
            self.arm.model, self.profile, collision_scene=self.scene,
            coulomb=injected["coulomb"],
            viscous_per_deg_s=injected["viscous"],
            coulomb_transition_deg_s=injected["transition"],
            noise=self.system["rehearsal"]["noise"])

    def _rehearsal_friction(self) -> dict:
        """Friction to plant in the rehearsal so the fit has to find something.

        A rehearsal against a frictionless robot recovers zero and reports a
        tiny residual, which looks like success and proves nothing. Values are
        arbitrary but plausible, and spread across joints so a fit that mixes
        joints up cannot pass.
        """
        count = self.arm.joint_count
        defaults = self.system["rehearsal"]
        return {
            "coulomb": [round(defaults["coulomb_base"] + defaults["coulomb_per_joint"] * index, 4)
                        for index in range(count)],
            "viscous": [round(defaults["viscous_base"] + defaults["viscous_per_joint"] * index, 5)
                        for index in range(count)],
            "transition": defaults["transition_deg_s"],
        }

    def _check_recovery(self, payload: dict,
                        holdout_tolerance: float = 0.0) -> dict:
        """Did the rehearsal get back what was planted in it?"""
        injected = getattr(self, "_injected", None)
        joints = payload.get("joints") or []
        if not injected or not joints:
            return {"available": False}
        errors = []
        for index, entry in enumerate(joints):
            friction = entry.get("friction") or {}
            expected = injected["coulomb"][index]
            actual = float(friction.get("coulomb") or 0.0)
            errors.append({
                "joint": index + 1,
                "expected": expected,
                "recovered": round(actual, 4),
                "error": round(abs(actual - expected), 4),
            })
        worst = max((item["error"] for item in errors), default=0.0)
        report = {
            "available": True,
            "worst_coulomb_error": round(worst, 4),
            "tolerance": self.system["rehearsal"]["recovery_tolerance"],
            "passed": worst <= self.system["rehearsal"]["recovery_tolerance"],
            "joints": errors,
        }
        if holdout_tolerance > 0.0:
            held = [float(value)
                    for value in payload.get("validation_rms_a") or []]
            worst_held = max(held) if held else float("inf")
            report["worst_holdout_a"] = (None if not held
                                         else round(worst_held, 5))
            report["holdout_tolerance"] = holdout_tolerance
            report["holdout_passed"] = worst_held <= holdout_tolerance
            report["passed"] = report["passed"] and report["holdout_passed"]
        return report

    def _write(self, payload: dict, mode: str, observations=None,
               raw_frames=None) -> None:
        try:
            folder = report_module.write_run(
                self.config.output_directory, payload, observations,
                raw_frames=raw_frames,
                model_urdf=(self.urdf_text if payload.get("gravity_model") else ""))
            self._remember_report(mode, folder)
            self.note(f"written to {folder}")
        except OSError as error:
            self.note(f"could not write result: {error}")

    def _remember_report(self, mode: str, folder: Path) -> dict:
        directory = Path(self.config.output_directory)
        entry = {
            "mode": str(mode),
            "name": str(Path(folder).relative_to(directory)),
            "report": (f"/runs/{Path(folder).relative_to(directory)}/"
                       f"{report_module.REPORT_NAME}"),
            "modified": Path(folder).stat().st_mtime,
        }
        with self._lock:
            current = self._reports.get(str(mode))
            if current is None or entry["modified"] >= current["modified"]:
                self._reports[str(mode)] = entry
        return entry

    @staticmethod
    def _run_mode(name: str) -> str:
        match = re.match(r"^(.+)-\d{8}-\d{6}(?:-.+)?$", Path(name).name)
        return match.group(1) if match else ""

    def reports_payload(self) -> dict:
        """Latest safely served report per mode, including after a restart."""
        if not self._reports_scanned:
            self.runs(limit=None)
        with self._lock:
            return {mode: dict(entry) for mode, entry in self._reports.items()}

    def runs(self, limit: int | None = 12) -> list[dict]:
        """Result folders that hold a readable report, newest first.

        One level deep as well as at the top, because a load sweep files
        itself under ``load_sweep/`` and would otherwise write a report the
        dashboard never offers a link to.
        """
        directory = Path(self.config.output_directory)
        if not directory.is_dir():
            return []
        found = []
        try:
            entries = list(directory.iterdir())
        except OSError:
            return []
        for entry in entries:
            if not entry.is_dir():
                continue
            if (entry / report_module.REPORT_NAME).is_file():
                nested = [entry]
            else:
                try:
                    nested = [child for child in entry.iterdir()
                              if child.is_dir()
                              and (child / report_module.REPORT_NAME).is_file()]
                except OSError:
                    continue
            for folder in nested:
                name = str(folder.relative_to(directory))
                mode = self._run_mode(name)
                found.append({
                    "mode": mode,
                    "name": name,
                    "report": f"/runs/{name}/{report_module.REPORT_NAME}",
                    "modified": folder.stat().st_mtime,
                })
        found.sort(key=lambda item: item["modified"], reverse=True)
        with self._lock:
            for entry in reversed(found):
                mode = entry.get("mode")
                if mode:
                    self._reports[mode] = dict(entry)
            self._reports_scanned = True
        return found if limit is None else found[:limit]

    # -- snapshot --------------------------------------------------------

    def _reconcile_hold_recovery(self) -> None:
        with self._lock:
            if (not self._hold_recovery_required or self._state != PAUSED
                    or self._activity != GRAVITY_HOLD_TEST or self._worker is not None
                    or self.planning):
                return
            if (self._external_process is not None
                    and self._external_process.poll() is None):
                return
            selection = (tuple(self.driven_joints or (
                self.arm.joint_names if self.arm else [])),
                self.config.commands.follow_joint_trajectory_action)
            if self._hold_recovery_selection != selection:
                return
            reader = getattr(self.bridge, "recovery_status", None)
            if not callable(reader):
                return
            records = (self.result or {}).get("records") or []
            motion_completed = bool(records) and all(
                isinstance(record, dict) and isinstance(record.get("move"), dict)
                and record["move"].get("ok") is True for record in records)
            try:
                evidence = reader(require_goal_status=not motion_completed)
            except Exception:
                return
            if not isinstance(evidence, dict) or evidence.get("ready") is not True:
                return
            self._hold_recovery_required = False
            self._hold_recovery_selection = None
            self._hold_current_started = False
            self._external_process = None
            self._hold_plan = {}
            self.gravity_armed = ""
            self.rehearsal_passed = False
            self._state, self._activity = IDLE, ""
            self.progress = {**self.progress, "recovery_verified": True,
                             "error": (self.result or {}).get("reason", "")}
            self.publish_event(
                "robot recovery verified; previous run remains failed; ready for a new task",
                source=GRAVITY_HOLD_TEST)

    def snapshot(self) -> dict:
        self._reconcile_hold_recovery()
        with self._lock:
            return {
                "state": self._state,
                "activity": self._activity,
                "connection": self.connection(),
                "have_model": self.have_model(),
                "joint_names": list(self.arm.joint_names) if self.arm else [],
                "parameter_count": self.arm.parameter_count if self.arm else 0,
                "effort_unit": self.config.telemetry.signals.effort_unit,
                "profile_source": self.profile_source,
                "profile_edited": (self.profile is not None
                                   and self.profile.source == EDITED_SOURCE),
                "current_guard": (
                    self.profile is not None
                    and self.config.telemetry.signals.effort_source == "current"
                    and autoprofile.current_guard_active(self.profile)),
                "driven_joints": list(self.driven_joints),
                "reach_deg": self.jog_limits_deg(),
                "reach_range_deg": self.jog_range_deg(),
                # Joints this dashboard cannot drive that have moved since the
                # screen was reduced around them. Everything screened while
                # this is non-empty was screened against a robot that is not
                # the one standing there.
                "astray": self.screen_drift(),
                "jogging": self._state == JOGGING,
                "rehearsal_passed": self.rehearsal_passed,
                "gravity_armed": bool(self.gravity_armed),
                "gravity_defaults": self.gravity_defaults(),
                "control_ranges": self.control_ranges(),
                "motion_speed_max_deg_s": self._motion_speed_limit(),
                "gravity_test": self.gravity_test_capability(),
                "hold_plan": self.hold_plan_payload(),
                "workspace": self.workspace_payload(),
                "preview_token": self.preview_token,
                "planning": self.planning,
                "obstacles": self.obstacles(),
                "config_file": str(self._config_save_target()),
                # Where edits are written by themselves, which is nowhere
                # unless the launch named a file.
                "config_autosave": str(self.config_file() or ""),
                "frames": self.frame_names(),
                "collision": self.collision_report(),
                "progress": dict(self.progress),
                "activity_feed": self.activity_payload(),
                "reports": self.reports_payload(),
                "result": self.result,
                "notes": list(self.notes[-40:]),
                "sample": self.latest_sample(),
            }

    # -- guards ----------------------------------------------------------

    def _require_scene(self) -> None:
        if self.scene is None:
            raise RuntimeError("no model yet; waiting for /robot_description")

    def _require_idle(self, message: str) -> None:
        if self._state != IDLE or self.planning:
            raise RuntimeError(message)
