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

from .. import campaign as campaign_module
from .. import excitation, identification as ident
from ..interfaces import CommandSpec, TelemetrySpec
from ..model import ModelComponents
from ..obstacles import Obstacle, ObstacleScene
from ..profile import RobotProfile

# The operator must type this before anything moves. It is deliberately a
# sentence about the room, not a checkbox: the person clicking it is asserting
# they are next to the arm.
ACKNOWLEDGEMENT = "I_AM_AT_THE_ROBOT_AND_ESTOP_READY"
IDLE, RUNNING = "idle", "running"
# A campaign yields tens of thousands of samples; a scatter plot stops being
# readable long before a browser stops being able to draw them.
MAX_PLOT_POINTS = 1500


@dataclass
class DashboardConfig:
    profile_path: str = ""
    output_directory: str = "identification_results"
    telemetry: TelemetrySpec = field(default_factory=TelemetrySpec)
    commands: CommandSpec = field(default_factory=CommandSpec)
    # Frame the 3D view renders in. Empty means the model root.
    display_frame: str = ""


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
        try:
            if self.profile is not None:
                arm = ident.ArmModel.from_profile(urdf_text, self.profile)
            else:
                arm = ident.ArmModel.from_urdf_text(urdf_text, "")
        except Exception as error:  # noqa: BLE001 - surfaced, never fatal
            self.note(f"robot_description rejected: {error}")
            return False
        with self._lock:
            self.urdf_text = urdf_text
            self.arm = arm
            previous = self.scene.as_list() if self.scene else []
            self.scene = ObstacleScene(arm.model, urdf_text=urdf_text)
            if previous:
                try:
                    self.scene.replace_all(previous)
                except KeyError as error:
                    self.note(f"obstacles dropped, frames changed: {error}")
            self.plan = campaign_module.default_plan(self.profile) \
                if self.profile is not None else None
        self.note(f"model ready: {arm.joint_count} joints, "
                  f"{arm.parameter_count} parameters")
        return True

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
        return box.as_dict()

    def update_obstacle(self, obstacle_id: str, changes: dict) -> dict:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        return self.scene.update(obstacle_id, **changes).as_dict()

    def remove_obstacle(self, obstacle_id: str) -> None:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        self.scene.remove(obstacle_id)

    def replace_obstacles(self, payloads: list[dict]) -> list[dict]:
        self._require_scene()
        self._require_idle("obstacles cannot be edited while a run is active")
        self.scene.replace_all(payloads)
        return self.scene.as_list()

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

    def start(self, mode: str, acknowledgement: str = "") -> dict:
        if not self.have_model():
            return {"ok": False, "message": "no /robot_description yet"}
        if self.profile is None:
            return {"ok": False, "message": "no robot profile loaded"}
        with self._lock:
            if self._state != IDLE:
                return {"ok": False, "message": f"{self._activity} is running"}
            if mode == "hardware":
                if not self.rehearsal_passed:
                    return {"ok": False,
                            "message": "rehearse first: a dry run must pass "
                                       "before the arm is allowed to move"}
                if acknowledgement != ACKNOWLEDGEMENT:
                    return {"ok": False,
                            "message": "hardware runs need the operator "
                                       "acknowledgement"}
            self._state = RUNNING
            self._activity = f"campaign_{mode}"
            self._abort.clear()
            self._started_at = time.monotonic()
            self._samples = []
        self._worker = threading.Thread(
            target=self._run, args=(mode,), daemon=True,
            name=f"identification-{mode}")
        self._worker.start()
        return {"ok": True, "message": f"{self._activity} started"}

    def stop(self) -> dict:
        self._abort.set()
        return {"ok": True, "message": "stop requested"}

    def running(self) -> bool:
        return self._state == RUNNING

    def _run(self, mode: str) -> None:
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
            with self._lock:
                self._state = IDLE
                self._activity = ""

    def _on_progress(self, phase: str, detail: dict) -> None:
        with self._lock:
            self.progress = {"mode": self._activity, "phase": phase,
                             "elapsed_s": time.monotonic() - self._started_at}
            self.progress.update(detail or {})

    def _finish(self, mode: str, result, observations) -> None:
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        payload["mode"] = mode
        payload.update(self._plot_data(payload, observations))
        with self._lock:
            self.result = payload
            self.progress = {"mode": mode, "phase": "finished"}
            if mode != "hardware":
                self.rehearsal_passed = bool(payload.get("complete"))
        self._write(payload, mode)
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

        return AnalyticPlant(self.arm.model, self.profile,
                             collision_scene=self.scene, noise=0.002)

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
                "acknowledgement": ACKNOWLEDGEMENT,
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
