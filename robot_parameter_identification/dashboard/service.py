"""Dashboard state: what is connected, what is planned, what was measured.

Holds no ROS. The node supplies a bridge object; everything else here is plain
Python so the whole surface can be exercised in tests without a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
import time
import traceback

import numpy as np

from .. import autoprofile
from .. import campaign as campaign_module
from .. import excitation, identification as ident
from ..interfaces import CommandSpec, TelemetrySpec
from ..model import ModelComponents
from ..obstacles import Obstacle, ObstacleScene
from ..profile import RobotProfile

IDLE, RUNNING = "idle", "running"
# A campaign yields tens of thousands of samples; a scatter plot stops being
# readable long before a browser stops being able to draw them.
MAX_PLOT_POINTS = 1500
# The rehearsal plants known friction and must find it again; a run that merely
# completes proves the code executes, not that it computes.
REHEARSAL_NOISE = 0.002
REHEARSAL_TOLERANCE = 0.02
# Homing is a recovery move from an unknown pose, so it goes slowly whatever
# speed the campaign was configured for.
HOMING_SPEED_DEG_S = 10.0


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
        self.scene: ObstacleScene | None = None
        self.urdf_text = ""
        self.driven_joints: list[str] = []
        self.configured_profile = profile
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

    def connection(self) -> dict:
        described = self.config.telemetry.describe()
        described["action"] = self.config.commands.follow_joint_trajectory_action
        if self.bridge is None:
            described.update(
                {"telemetry_ok": False, "action_ok": False, "sample_age_s": None,
                 "description_ok": bool(self.urdf_text)})
            return described
        described.update(self.bridge.health())
        described["description_ok"] = bool(self.urdf_text)
        return described

    # -- 3D view ---------------------------------------------------------

    def viewer_state(self) -> dict:
        """Everything the 3D canvas needs for one frame."""
        if self.arm is None:
            return {"have_model": False}
        sample = self.latest_sample()
        pose = np.asarray(sample["position_deg"], dtype=float) if sample \
            else np.zeros(self.arm.joint_count)
        payload = {
            "have_model": True,
            "joint_names": list(self.arm.joint_names),
            "joint_values_deg": pose.tolist(),
            "link_tf": self.arm.link_transforms(pose)
            if hasattr(self.arm, "link_transforms") else {},
            "obstacles": self.scene.placements(pose) if self.scene else [],
            "frames": self.frame_names(),
        }
        if self.plan is not None and getattr(self.plan, "workspace_limit_deg", None):
            payload["workspace_limit_deg"] = list(self.plan.workspace_limit_deg)
        return payload

    # -- campaign --------------------------------------------------------

    def start(self, mode: str) -> dict:
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        if self.profile is None:
            return {"ok": False, "message": "no robot profile loaded"}
        with self._lock:
            if self._state != IDLE:
                return {"ok": False, "message": f"{self._activity} is running"}
            if mode == "hardware" and not self.rehearsal_passed:
                # Not ceremony: the rehearsal plants known friction and must
                # find it again, and it is what caught the fit returning zero.
                return {"ok": False,
                        "message": "rehearse first: a dry run must pass "
                                   "before the arm is allowed to move"}
            self._state = RUNNING
            self._activity = f"campaign_{mode}"
            self._abort.clear()
            self._started_at = time.monotonic()
            self._samples = []
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

    def _run(self, mode: str) -> None:
        plant = None
        try:
            plant = self._build_plant(mode)
            run = campaign_module.Campaign(
                self.arm, plant, self.plan,
                progress=self._on_progress,
                should_stop=self._abort.is_set)
            self.progress = {"mode": mode, "phase": "starting"}
            result = run.run()
            self._finish(mode, result, run.observations)
        except Exception as error:  # noqa: BLE001 - a crash must not be silent
            self.note(f"{mode} run failed: {error}")
            self.progress = {"mode": mode, "phase": "failed",
                             "error": str(error),
                             "traceback": traceback.format_exc()[-2000:]}
        finally:
            self._release(plant)
            with self._lock:
                self._state = IDLE
                self._activity = ""

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

    def _finish(self, mode: str, result, observations) -> None:
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        payload["mode"] = mode
        payload.update(self._plot_data(payload, observations))
        aborted = payload.get("aborted")
        recovery = {} if mode == "hardware" else self._check_recovery(payload)
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
            if mode != "hardware":
                self.rehearsal_passed = (bool(payload.get("complete"))
                                         and bool(recovery.get("passed")))
        self._write(payload, mode)
        if aborted:
            self.note(f"{mode} run stopped: {aborted}")
        elif recovery.get("available") and not recovery.get("passed"):
            self.note(f"{mode} completed but did not recover the planted "
                      f"friction: worst error "
                      f"{recovery['worst_coulomb_error']} > "
                      f"{recovery['tolerance']}")
        else:
            self.note(f"{mode} run complete")

    def _plot_data(self, payload: dict, observations) -> dict:
        """Scatter data for the charts, thinned to something a browser can draw.

        The friction plot is the reason this exists: a fitted curve on its own
        looks fine no matter how wrong it is, and only laying it over the cloud
        it came from shows a reversal model that misses at low speed.
        """
        joints = payload.get("joints") or []
        if not joints or not observations:
            return {}
        moving = [record for record in observations
                  if getattr(record, "phase", "") != "D_validation"]
        stride = max(1, len(moving) // MAX_PLOT_POINTS)
        thinned = moving[::stride]
        friction, residual = [], []
        for index, entry in enumerate(joints):
            speeds, efforts, errors = [], [], []
            for record in thinned:
                try:
                    speed = float(record.velocity_deg_s[index])
                    effort = float(record.current_a[index])
                except (AttributeError, IndexError, TypeError):
                    continue
                speeds.append(round(speed, 3))
                efforts.append(round(effort, 4))
            friction.append([{"speed": s, "effort": e}
                             for s, e in zip(speeds, efforts)])
            predicted = entry.get("predicted_a") or []
            measured = entry.get("measured_a") or []
            for step in range(0, min(len(predicted), len(measured)), stride):
                errors.append({"speed": 0.0,
                               "residual": round(
                                   measured[step] - predicted[step], 4)})
            residual.append(errors)
        return {"friction_samples": friction, "residual_samples": residual}

    def _build_plant(self, mode: str):
        if mode == "hardware":
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

    def _write(self, payload: dict, mode: str) -> None:
        directory = Path(self.config.output_directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = directory / f"{mode}-{stamp}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.note(f"written to {path}")
        except OSError as error:
            self.note(f"could not write result: {error}")

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
                "current_guard": (
                    self.profile is not None
                    and autoprofile.current_guard_active(self.profile)),
                "driven_joints": list(self.driven_joints),
                "reach_deg": ([round(v, 1) for v in self.plan.design_limits(
                    self.arm).upper_deg]
                    if self.plan is not None and self.arm is not None else []),
                "rehearsal_passed": self.rehearsal_passed,
                "obstacles": self.obstacles(),
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
