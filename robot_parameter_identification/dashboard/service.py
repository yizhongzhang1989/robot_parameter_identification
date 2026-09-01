"""Dashboard state: what is connected, what is planned, what was measured.

Holds no ROS. The node supplies a bridge object; everything else here is plain
Python so the whole surface can be exercised in tests without a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import csv
import json
import math
import re
import threading
import time
import traceback

import numpy as np

from .. import autoprofile
from .. import campaign as campaign_module
from .. import excitation, identification as ident
from .. import loadsweep as loadsweep_module
from .. import loadsweep_report
from .. import report as report_module
from ..interfaces import CommandSpec, TelemetrySpec
from ..loadsweep_run import LoadSweepRun
from ..model import ModelComponents
from ..obstacles import Obstacle, ObstacleScene
from ..profile import RobotProfile

IDLE, RUNNING, JOGGING = "idle", "running", "jogging"
# An envelope typed into the panel is an operator's envelope, so it is labelled
# and guarded exactly as a hand-written file is.
EDITED_SOURCE = "<edited in the dashboard>"
PROFILE_FILE_NAME = re.compile(r"[A-Za-z0-9_.-]+\.yaml")
OBSTACLE_FILE_NAME = re.compile(r"[A-Za-z0-9_.-]+\.json")
# A ceiling nobody supplied is infinite, and JSON has no way to say so.
UNBOUNDED_LIMITS = ("continuous_current_a", "peak_current_a")


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
MAX_PLOT_POINTS = 1500
# Below this a joint has not established a direction of travel, so measured
# minus rigid is wherever the position servo happened to settle inside the
# stiction band, not friction at a speed. It sits well below the slowest
# commanded rung; the parked rows of a sweep are excluded by their tag instead.
STILL_SPEED_DEG_S = 0.01
# The rehearsal plants known friction and must find it again; a run that merely
# completes proves the code executes, not that it computes.
REHEARSAL_NOISE = 0.002
REHEARSAL_TOLERANCE = 0.02
# Homing is a recovery move from an unknown pose, so it goes slowly whatever
# speed the campaign was configured for.
HOMING_SPEED_DEG_S = 10.0
# Jogging is hand-driven, so it is capped well below anything the campaign uses:
# the operator is watching the arm, not a plot, and has no undo.
JOG_SPEED_DEG_S = 10.0
# How long the jog worker naps when no new pose has been asked for.
JOG_POLL_S = 0.05
OPTIMAL_MODE = "optimal_excitation"


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
    profile_path: str = ""
    output_directory: str = "identification_results"
    telemetry: TelemetrySpec = field(default_factory=TelemetrySpec)
    commands: CommandSpec = field(default_factory=CommandSpec)
    # Frame the 3D view renders in. Empty means the model root.
    display_frame: str = ""
    # How far the campaign may swing each joint. The URDF describes the arm,
    # not the stand it is bolted to, so this is often tighter than the URDF.
    workspace_limit_deg: tuple[float, ...] = ()
    # Top sweep speed. Zero keeps the conservative derived default, which is
    # too slow to see viscous friction on a full-size arm.
    maximum_speed_deg_s: float = 0.0
    # Where the drawn obstacle scene lives between sessions. Empty disables it.
    obstacle_path: str = ""


class IdentificationService:
    """One campaign at a time, plus the scene it runs in."""

    def __init__(self, config: DashboardConfig, bridge=None,
                 profile: RobotProfile | None = None) -> None:
        self.config = config
        self.bridge = bridge
        self.profile = profile
        self.arm: ident.ArmModel | None = None
        self.whole: ident.ArmModel | None = None
        self.scene: ObstacleScene | None = None
        self.urdf_text = ""
        self.driven_joints: list[str] = []
        self.configured_profile = profile
        # What the launch supplied, kept so an edit can be undone.
        self.launch_profile = profile
        self.profile_source = "configured" if profile is not None else "none"
        self.components = ModelComponents()
        self.plan = None
        self.result: dict | None = None
        self.progress: dict = {"phase": "idle"}
        self.notes: list[str] = []
        self.rehearsal_passed = False
        self._state = IDLE
        self._activity = ""
        self._worker: threading.Thread | None = None
        self._abort = threading.Event()
        self._lock = threading.RLock()
        self._started_at = 0.0
        self._samples: list[dict] = []
        self._options: dict = {}
        # The pose a slider last asked for, or None once it has been driven.
        self._jog_target: list[float] | None = None

    # -- model -----------------------------------------------------------

    def note(self, message: str) -> None:
        with self._lock:
            self.notes.append(f"{time.strftime('%H:%M:%S')} {message}")
            del self.notes[:-200]

    def adopt_description(self, urdf_text: str) -> bool:
        """Build the model from a freshly received /robot_description."""
        if not urdf_text or urdf_text == self.urdf_text:
            return False
        self.urdf_text = urdf_text
        return self._rebuild()

    def adopt_driven_joints(self, names) -> bool:
        """Restrict identification to the joints the controller actually moves.

        A dual-arm URDF carries twice the joints the action can command, and
        identifying a model the controller cannot move is meaningless.
        """
        names = [str(entry) for entry in names]
        if names == self.driven_joints:
            return False
        self.driven_joints = names
        self.note(f"controller drives {len(names)} joints")
        return self._rebuild()

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
                cap = list(self.config.workspace_limit_deg) or None
                profile = autoprofile.derive_profile(
                    self.urdf_text, self.driven_joints,
                    workspace_limit_deg=cap,
                    speed_limit_deg_s=self._requested_speed())
                source = "derived"
            except Exception as error:  # noqa: BLE001
                self.note(f"profile could not be derived: {error}")
        try:
            if profile is not None:
                arm = ident.ArmModel.from_profile(self.urdf_text, profile)
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
            # The picture and the collision scene cover the whole robot, not
            # just the arm this dashboard drives, so the drawing needs a model
            # that still has the other arm's joints in it.
            try:
                self.whole = ident.ArmModel.from_urdf_text(self.urdf_text, "")
            except Exception as error:  # noqa: BLE001 - drawing is not the job
                self.whole = None
                self.note(f"whole-robot view unavailable: {error}")
            previous = self.scene.as_list() if self.scene else []
            self.scene = ObstacleScene(arm.model, urdf_text=self.urdf_text)
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
        return True

    def _requested_speed(self) -> float | None:
        speed = float(self.config.maximum_speed_deg_s or 0.0)
        return speed if speed > 0.0 else None

    def _build_plan(self, profile: RobotProfile):
        """The plan follows the operator's speed, clamped to the envelope."""
        speed = self._requested_speed()
        if speed is None:
            return campaign_module.default_plan(profile)
        plan, notes = campaign_module.clamp_campaign_plan(
            {"maximum_speed_deg_s": speed}, profile)
        for entry in notes:
            self.note(entry)
        self.note(f"sweep speeds {list(plan.friction_speeds_deg_s)} deg/s, "
                  f"amplitude {plan.friction_amplitude_deg:g} deg")
        return plan

    def have_model(self) -> bool:
        return self.arm is not None

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
            return (Path(self.config.profile_path) if self.config.profile_path
                    else root / "profile.yaml")
        if not PROFILE_FILE_NAME.fullmatch(cleaned):
            raise ValueError(
                "a profile file name may use letters, digits, dot, dash and "
                f"underscore, and must end in .yaml: {cleaned!r}")
        return root / cleaned

    # -- obstacles -------------------------------------------------------

    def obstacles(self) -> list[dict]:
        return self.scene.as_list() if self.scene else []

    def frame_names(self) -> list[str]:
        return self.scene.frame_names() if self.scene else []

    def add_obstacle(self, payload: dict) -> dict:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        box = self.scene.add(Obstacle.from_dict(payload))
        self.note(f"obstacle {box.name} bolted to {box.parent_frame}")
        self._persist_obstacles()
        return box.as_dict()

    def update_obstacle(self, obstacle_id: str, changes: dict) -> dict:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        box = self.scene.update(obstacle_id, **changes).as_dict()
        self._persist_obstacles()
        return box

    def remove_obstacle(self, obstacle_id: str) -> None:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        self.scene.remove(obstacle_id)
        self._persist_obstacles()

    def replace_obstacles(self, payloads: list[dict]) -> list[dict]:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        self.scene.replace_all(payloads)
        self._persist_obstacles()
        return self.scene.as_list()

    def obstacle_path(self) -> Path | None:
        raw = (self.config.obstacle_path or "").strip()
        return Path(raw).expanduser() if raw else None

    def save_obstacles(self, name: str = "") -> dict:
        """Write the scene where the operator says, so each arm keeps its own.

        The launch path is where edits are kept automatically; this is how a
        scene gets a name worth carrying to another robot.
        """
        if self.scene is None:
            return {"ok": False, "message": "no model yet"}
        try:
            target = self._obstacle_save_target(name)
        except ValueError as error:
            return {"ok": False, "message": str(error)}
        try:
            self.scene.save(target)
        except OSError as error:
            return {"ok": False, "message": f"not saved: {error}"}
        self.note(f"obstacles written to {target}")
        return {"ok": True, "path": str(target)}

    def _obstacle_save_target(self, name: str = "") -> Path:
        """Where a save may land, which is never wherever the caller says.

        Same rule as the profile: the web surface listens on every interface,
        so honouring a path from a request would be an arbitrary file write. A
        name is only a name.
        """
        cleaned = str(name or "").strip()
        if not cleaned:
            launched = self.obstacle_path()
            return (launched if launched is not None
                    else Path(self.config.output_directory) / "obstacles.json")
        if not OBSTACLE_FILE_NAME.fullmatch(cleaned):
            raise ValueError(
                "an obstacle file name may use letters, digits, dot, dash and "
                f"underscore, and must end in .json: {cleaned!r}")
        return Path(self.config.output_directory) / cleaned

    def _persist_obstacles(self) -> None:
        """Save after every edit; a scene lost on restart is a scene retyped."""
        path = self.obstacle_path()
        if path is None or self.scene is None:
            return
        try:
            self.scene.save(path)
        except OSError as error:
            self.note(f"obstacles not saved to {path}: {error}")

    def _restore_obstacles(self) -> bool:
        path = self.obstacle_path()
        if path is None or self.scene is None or not path.exists():
            return False
        try:
            skipped = self.scene.load(path)
        except (OSError, ValueError) as error:
            self.note(f"obstacle file {path} not loaded: {error}")
            return False
        self.note(f"obstacles restored from {path}: "
                  f"{len(self.scene.as_list())} kept"
                  + (f", {len(skipped)} dropped" if skipped else ""))
        for reason in skipped:
            self.note(f"obstacle dropped: {reason}")
        return True

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
        }
        if self.plan is not None and getattr(self.plan, "workspace_limit_deg", None):
            payload["workspace_limit_deg"] = list(self.plan.workspace_limit_deg)
        return payload

    # -- campaign --------------------------------------------------------

    def elsewhere_off_neutral(self, tolerance_deg: float = 5.0) -> list[dict]:
        """Joints this dashboard does not drive that are not where the screen
        thinks they are.

        The collision scene carries every link the URDF ships -- sixteen on
        this robot, both arms -- but it is built on a model reduced against the
        neutral configuration, so the arm this dashboard does not drive is
        pinned at zero inside the screen. Park that arm somewhere else and the
        screen will cheerfully clear a path straight through it. Nothing else
        catches this: the geometry is loaded, the pair count is right, and the
        screen refuses folded poses exactly as it should.
        """
        driven = set(self.arm.joint_names) if self.arm is not None else set()
        astray = []
        for name, radians in (self.bridge.elsewhere()
                              if self.bridge is not None else {}).items():
            if name in driven:
                continue
            degrees = float(np.degrees(radians))
            if abs(degrees) > tolerance_deg:
                astray.append({"joint": name, "at_deg": round(degrees, 2)})
        return sorted(astray, key=lambda item: -abs(item["at_deg"]))

    def start(self, mode: str, options: dict | None = None) -> dict:
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        if self.profile is None:
            return {"ok": False, "message": "no robot profile loaded"}
        with self._lock:
            if self._state != IDLE:
                return {"ok": False, "message": f"{self._activity} is running"}
            if mode in ("hardware", OPTIMAL_MODE) and not self.rehearsal_passed:
                # Not ceremony: the rehearsal plants known friction and must
                # find it again, and it is what caught the fit returning zero.
                return {"ok": False,
                        "message": "rehearse first: a dry run must pass "
                                   "before the arm is allowed to move"}
            if mode in ("hardware", "load_sweep", OPTIMAL_MODE):
                astray = self.elsewhere_off_neutral()
                if astray:
                    where = ", ".join(f"{item['joint']} at {item['at_deg']:g} deg"
                                      for item in astray[:4])
                    return {"ok": False,
                            "message": "the collision screen places every joint "
                                       "this dashboard does not drive at neutral, "
                                       f"and these are not: {where}. Home them "
                                       "before driving, or the screen will clear "
                                       "a path through them."}
            self._state = RUNNING
            self._activity = (mode if mode in ("load_sweep", OPTIMAL_MODE)
                              else f"campaign_{mode}")
            self._abort.clear()
            self._started_at = time.monotonic()
            self._samples = []
            self._options = dict(options or {})
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
        return {"ok": True, "message": "stop requested"}

    def home(self) -> dict:
        """Drive every joint back to neutral.

        A campaign leaves the arm wherever validation ended, and the hardware
        plant refuses to start more than a degree from neutral, so without this
        the next run cannot be armed without hand-driving the arm.
        """
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        if self.profile is None:
            return {"ok": False, "message": "no robot profile loaded"}
        with self._lock:
            if self._state != IDLE:
                return {"ok": False, "message": f"{self._activity} is running"}
            self._state = RUNNING
            self._activity = "homing"
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
                maximum_speed_deg_s=HOMING_SPEED_DEG_S)
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
            return self.jog_start()
        if action == "stop":
            return self.jog_stop()
        if action == "move":
            return self.jog_to((body or {}).get("position_deg"))
        return {"ok": False, "message": f"unknown jog action {action!r}"}

    def jog_start(self) -> dict:
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
        with self._lock:
            if self._state == JOGGING:
                return {"ok": True, "message": "already jogging"}
            if self._state != IDLE:
                return {"ok": False, "message": f"{self._activity} is running"}
            self._state = JOGGING
            self._activity = "jogging"
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

    def jog_limits_deg(self) -> list[float]:
        """How far each joint may be jogged: the campaign's own envelope.

        Deliberately not the URDF's: the URDF describes the arm, not the bench
        it is bolted to, and a slider is the easiest way there is to drive an
        arm into its surroundings.
        """
        if self.plan is None or self.arm is None:
            return []
        return [round(float(value), 1)
                for value in self.plan.design_limits(self.arm).upper_deg]

    def _screened_pose(self, position_deg) -> list[float]:
        """Clamp to the envelope, then refuse anything that would hit something.

        Both checks belong here rather than in the browser: the request does
        not have to have come from this dashboard, and the arm has no idea what
        is bolted around it.
        """
        limits = self.jog_limits_deg()
        values = [float(value) for value in (position_deg or [])]
        if not limits:
            raise ValueError("no motion plan yet")
        if len(values) != len(limits):
            raise ValueError(f"expected {len(limits)} joint angles, "
                             f"got {len(values)}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint angles must be finite")
        clamped = [max(-limit, min(limit, value))
                   for value, limit in zip(values, limits)]
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
            plant = self.bridge.hardware_plant(
                self.profile, self.scene, require_neutral_start=False,
                maximum_speed_deg_s=JOG_SPEED_DEG_S)
            setter = getattr(plant, "set_monitor", None)
            if setter is not None:
                setter(self._monitor())
            self.note(f"jogging enabled at {JOG_SPEED_DEG_S:g} deg/s")
            self._on_progress("jogging", {})
            while not self._abort.is_set():
                with self._lock:
                    target, self._jog_target = self._jog_target, None
                if target is None:
                    time.sleep(JOG_POLL_S)
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
        try:
            monitor = (self._monitor()
                       if mode in ("hardware", OPTIMAL_MODE) else None)
            plan = self._optimal_plan() if mode == OPTIMAL_MODE else self.plan
            reused = None
            if mode == OPTIMAL_MODE and self._options.get("reuse_friction"):
                folder = self._latest_optimal_friction(plan)
                if folder is None:
                    raise RuntimeError(
                        "no compatible completed optimal low-speed phase found")
                reused = self._read_optimal_friction(folder)
            plant = self._build_plant(mode)
            setter = getattr(plant, "set_monitor", None)
            if setter is not None and monitor is not None:
                setter(monitor)
            run_type = (campaign_module.OptimalExcitationCampaign
                        if mode == OPTIMAL_MODE else campaign_module.Campaign)
            run = run_type(
                self.arm, plant, plan,
                progress=self._on_progress,
                should_stop=self._abort.is_set,
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
            self.note(f"{mode} run failed: {error}")
            self.progress = {"mode": mode, "phase": "failed",
                             "error": str(error),
                             "traceback": traceback.format_exc()[-2000:]}
            self._salvage(mode, run, plant, error)
        finally:
            self._release(plant)
            with self._lock:
                self._state = IDLE
                self._activity = ""

    def _optimal_plan(self):
        """Apply the small set of options exposed by the optimal-run card."""
        plan = replace(self.plan)
        bounds = campaign_module.campaign_bounds(self.profile)
        for name in ("optimal_training_trajectories",
                     "optimal_validation_trajectories",
                     "optimal_friction_repeats",
                     "optimal_friction_postures",
                     "fourier_base_frequency_hz",
                     "fourier_duration_s"):
            if name not in self._options:
                continue
            low, high = bounds[name]
            try:
                value = float(self._options[name])
            except (TypeError, ValueError):
                self.note(f"ignoring optimal excitation option "
                          f"{name}={self._options[name]!r}")
                continue
            bounded = min(max(value, low), high)
            if bounded != value:
                self.note(f"{name} {value:g} clamped to {bounded:g}")
            setattr(plan, name, int(bounded) if name.startswith("optimal_")
                    else float(bounded))
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
        plan = loadsweep_module.SweepPlan()
        for key, value in (getattr(self, "_options", None) or {}).items():
            if key == "resume" or not hasattr(plan, key):
                continue
            current = getattr(plan, key)
            try:
                if key == "joints":
                    setattr(plan, key, tuple(int(v) for v in value))
                elif isinstance(current, bool):
                    setattr(plan, key, bool(value))
                elif isinstance(current, int):
                    setattr(plan, key, int(value))
                elif isinstance(current, float):
                    setattr(plan, key, float(value))
            except (TypeError, ValueError):
                self.note(f"ignoring load sweep option {key}={value!r}")
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
        with self._lock:
            carried = (dict(self.progress)
                       if self.progress.get("phase") == phase else {})
            carried.update({"mode": self._activity, "phase": phase,
                            "elapsed_s": time.monotonic() - self._started_at})
            carried.update(detail or {})
            self.progress = carried

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
        payload.update(self._plot_data(payload, observations,
                                       getattr(result, "fits", None)))
        aborted = payload.get("aborted")
        recovery = ({} if mode in ("hardware", OPTIMAL_MODE)
                else self._check_recovery(payload))
        if recovery:
            payload["rehearsal_check"] = recovery
        with self._lock:
            self.result = payload
            # A stopped run that still says "finished" is how a half-measured
            # model gets mistaken for a complete one.
            self.progress = {"mode": mode,
                             "phase": "stopped" if aborted else "finished"}
            if aborted:
                self.progress["error"] = str(aborted)
            if mode not in ("hardware", OPTIMAL_MODE):
                self.rehearsal_passed = (bool(payload.get("complete"))
                                         and bool(recovery.get("passed")))
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
        stride = max(1, len(moving) // MAX_PLOT_POINTS)
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
                if abs(speed) < STILL_SPEED_DEG_S or _parked_here(record, index):
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
                "friction_standstill_speed_deg_s": STILL_SPEED_DEG_S}

    def _build_plant(self, mode: str):
        if mode in ("hardware", OPTIMAL_MODE):
            if self.bridge is None:
                raise RuntimeError("no ROS bridge; cannot drive hardware")
            return self.bridge.hardware_plant(self.profile, self.scene)
        from ..plants.analytic import AnalyticPlant  # noqa: PLC0415

        injected = self._rehearsal_friction()
        self._injected = injected
        return AnalyticPlant(
            self.arm.model, self.profile, collision_scene=self.scene,
            coulomb=injected["coulomb"],
            viscous_per_deg_s=injected["viscous"],
            coulomb_transition_deg_s=injected["transition"],
            noise=REHEARSAL_NOISE)

    def _rehearsal_friction(self) -> dict:
        """Friction to plant in the rehearsal so the fit has to find something.

        A rehearsal against a frictionless robot recovers zero and reports a
        tiny residual, which looks like success and proves nothing. Values are
        arbitrary but plausible, and spread across joints so a fit that mixes
        joints up cannot pass.
        """
        count = self.arm.joint_count
        return {
            "coulomb": [round(0.12 + 0.04 * index, 4) for index in range(count)],
            "viscous": [round(0.006 + 0.002 * index, 5) for index in range(count)],
            "transition": 1.8,
        }

    def _check_recovery(self, payload: dict) -> dict:
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
        return {
            "available": True,
            "worst_coulomb_error": round(worst, 4),
            "tolerance": REHEARSAL_TOLERANCE,
            "passed": worst <= REHEARSAL_TOLERANCE,
            "joints": errors,
        }

    def _write(self, payload: dict, mode: str, observations=None,
               raw_frames=None) -> None:
        try:
            folder = report_module.write_run(
                self.config.output_directory, payload, observations,
                raw_frames=raw_frames)
            self.note(f"written to {folder}")
        except OSError as error:
            self.note(f"could not write result: {error}")

    def runs(self, limit: int = 12) -> list[dict]:
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
                found.append({
                    "name": name,
                    "report": f"/runs/{name}/{report_module.REPORT_NAME}",
                    "modified": folder.stat().st_mtime,
                })
        found.sort(key=lambda item: item["modified"], reverse=True)
        return found[:limit]

    # -- snapshot --------------------------------------------------------

    def snapshot(self) -> dict:
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
                "jogging": self._state == JOGGING,
                "rehearsal_passed": self.rehearsal_passed,
                "obstacles": self.obstacles(),
                "obstacle_file": str(self._obstacle_save_target()),
                "frames": self.frame_names(),
                "collision": self.collision_report(),
                "progress": dict(self.progress),
                "result": self.result,
                "notes": list(self.notes[-40:]),
                "sample": self.latest_sample(),
            }

    # -- guards ----------------------------------------------------------

    def _require_scene(self) -> None:
        if self.scene is None:
            raise RuntimeError("no model yet; waiting for /robot_description")

    def _require_idle(self, message: str) -> None:
        if self._state != IDLE:
            raise RuntimeError(message)
