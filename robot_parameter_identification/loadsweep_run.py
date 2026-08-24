"""Drive the load sweep on real hardware, and keep what it measures.

Split from the planning in ``loadsweep`` because the plan is arithmetic and can
be tested without a robot, while this half is nothing but the robot: goals that
are refused, controllers that stop answering, and an arm that has to be put
back somewhere safe before anything is tried again.

A full seven-joint sweep is hours of motion. Two things follow from that, and
both are the reason this is a class rather than a loop. Every record is on disk
before the next pass starts, so an interruption costs one pass and not the
afternoon. And every record carries a key, so a resumed run reads what is
already there and drives only what is missing.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np

from .interfaces import MotionFailed
from .loadsweep import (MANIFEST_NAME, RECORDS_NAME, SCHEMA, Recorder,
                        SweepPlan, axis_gravity_deg, load_terms, path_free,
                        pose_with, record_key)


class TransitBlocked(RuntimeError):
    """No collision free way to get the arm to the next posture."""


class LoadSweepRun:
    """One sweep of one arm, written down as it happens."""

    def __init__(self, arm, plant, plan: SweepPlan, designs, folder,
                 progress=None, should_stop=None, note=None, scene=None) -> None:
        self.arm = arm
        self.plant = plant
        self.plan = plan
        self.designs = list(designs)
        self.folder = Path(folder)
        self.scene = scene
        self._progress = progress
        self._should_stop = should_stop or (lambda: False)
        self._note = note or (lambda _message: None)
        self.driven = 0
        self.skipped: list[dict] = []
        self.started_at = 0.0
        # Which level's posture the arm is standing in. Cleared whenever
        # anything moves it that is not a transit, so the path is screened
        # again before the next pass rather than assumed.
        self._standing = None

    # -- bookkeeping -----------------------------------------------------

    def manifest(self) -> dict:
        return {
            "schema": SCHEMA,
            "kind": "load_sweep",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "joint_names": [str(name) for name in self.arm.joint_names],
            "plan": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in vars(self.plan).items()},
            "speeds_deg_s": self.plan.speed_ladder(),
            "joints": [design.as_dict() for design in self.designs],
        }

    def _write_manifest(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / MANIFEST_NAME).write_text(
            json.dumps(self.manifest(), indent=1), encoding="utf-8")

    def _say(self, phase: str, payload: dict) -> None:
        if self._progress is not None:
            self._progress(phase, payload)

    # -- driving ---------------------------------------------------------

    def _recover(self) -> bool:
        """Put the arm somewhere known before trying again.

        A refused goal usually means the controller is unhappy about where the
        arm is, not about where it was asked to go, so the useful response is
        to go back to neutral rather than to repeat the request.
        """
        self._standing = None
        try:
            self.plant.park()
            return True
        except Exception as error:  # noqa: BLE001 - recovery must not raise
            self._note(f"recovery failed: {error}")
            return False

    def _go_to(self, target) -> None:
        """Move to a posture without driving the arm through itself.

        Levels are searched for across the whole workspace, so the pose that
        holds 4.8 Nm and the one that holds 4.3 Nm are unrelated and can be
        most of a revolution apart. The controller interpolates between them in
        joint space and the plant screens the ends but never the middle, so the
        straight move is checked here, and neutral is tried as a staging post
        when it is not clear.
        """
        target = np.asarray(target, dtype=float)
        sample = self.plant.sample()
        here = np.asarray(sample["position_deg"], dtype=float) if sample \
            else np.zeros(self.arm.joint_count)
        if path_free(self.scene, here, target):
            self.plant.hold_pose(target)
            return
        neutral = np.zeros(self.arm.joint_count)
        if path_free(self.scene, here, neutral) \
                and path_free(self.scene, neutral, target):
            self._note("direct move blocked; going by way of neutral")
            self.plant.park()
            self.plant.hold_pose(target)
            return
        raise TransitBlocked(
            "no collision free path to the next posture, directly or "
            "through neutral")

    def _moved(self, joint: int, frames, wanted: float) -> str:
        """Empty if the joint really went at the speed it was asked to.

        A drive can accept a goal, report itself enabled and unfaulted, and
        then not move: the arm holds position, the controller succeeds, and the
        pass is recorded as a measurement of a joint that never turned. Seen on
        this robot with both arms holding 23 V and two amps against gravity
        while a ten degree command produced a third of a degree. Nothing else
        in the stack notices, because every part of it did its job.

        Judged on the fitted window rather than the goal, because the window is
        what becomes a row.
        """
        if not frames:
            return "the arm returned no samples"
        speeds = [abs(float(frame["speed_deg_s"][joint])) for frame in frames]
        fastest = max(speeds)
        if fastest < 0.5 * abs(wanted):
            return (f"the joint moved at {fastest:.3f} deg/s when it was asked "
                    f"for {abs(wanted):.3f}; the arm is not following commands")
        return ""

    def _drive(self, joint: int, level, entry: dict, distance: float,
               repeat: int) -> list[dict] | None:
        """One pass, retried, or None if the arm would not do it."""
        origin = pose_with(level.pose_deg, joint,
                           float(level.pose_deg[joint]) - distance / 2.0)
        last = ""
        for attempt in range(1, max(1, self.plan.pass_attempts) + 1):
            if self._should_stop():
                return None
            try:
                frames = list(self.plant.traverse(
                    joint, origin, distance, entry["speed_deg_s"]))
                last = self._moved(joint, frames, entry["speed_deg_s"])
                if not last:
                    return frames
            except MotionFailed as failure:
                last = str(failure)
            if attempt < self.plan.pass_attempts:
                self._note(f"joint {joint + 1} level {level.index} at "
                           f"{entry['speed_deg_s']:g} deg/s: {last}; "
                           f"retrying ({attempt}/{self.plan.pass_attempts})")
                time.sleep(self.plan.pass_retry_s)
                self._recover()
        self.skipped.append({
            "joint": joint + 1, "level": level.index,
            "speed_deg_s": entry["speed_deg_s"], "repeat": repeat,
            "direction": "+" if distance > 0 else "-", "why": last})
        return None

    def _record(self, joint: int, design, level, entry: dict, frames,
                direction: str, repeat: int, key: str) -> None:
        for index, frame in enumerate(frames):
            here = pose_with(level.pose_deg, joint,
                             float(frame["position_deg"][joint]))
            terms = load_terms(self.arm, here, joint)
            # Signed as well as absolute. joint_loads reports magnitudes,
            # which is right for a bearing load and wrong for calibration: the
            # current spent holding gravity reverses when the torque does, so
            # regressing it against a magnitude is meaningless as soon as the
            # search picks postures either side of zero. It does.
            signed = float(self.arm.inverse_dynamics(here)[joint])
            self.recorder.write({
                "schema": SCHEMA,
                "key": key,
                "stamp": time.time(),
                "joint": joint,
                "joint_name": design.name,
                "level": level.index,
                "level_load_nm": level.load_nm,
                "level_pose_deg": list(level.pose_deg),
                "axis_gravity_deg": level.axis_gravity_deg,
                "speed_deg_s": entry["speed_deg_s"],
                "direction": direction,
                "repeat": repeat,
                "arc_deg": entry["arc_deg"],
                "window": index,
                "position_deg": round(float(frame["position_deg"][joint]), 4),
                "velocity_deg_s": round(float(frame["speed_deg_s"][joint]), 4),
                "acceleration_deg_s2": round(float(
                    (frame.get("acceleration_deg_s2")
                     or [0.0] * self.arm.joint_count)[joint]), 4),
                "current_a": round(float(frame["current_a"][joint]), 5),
                "temperature_c": round(float(frame["temperature_c"][joint]), 2),
                # Every load term, not only the one the current model uses, so
                # a later question about radial force does not need the arm
                # back.
                "load": {"axial_nm": round(float(terms[0]), 5),
                         "axial_signed_nm": round(signed, 5),
                         "radial_n": round(float(terms[1]), 4),
                         "thrust_n": round(float(terms[2]), 4),
                         "tilt_nm": round(float(terms[3]), 5)},
                "load_drift_nm": entry["load_drift_nm"],
                "window_frames": frame.get("window_frames", 0),
                "window_fit_rms_deg": round(
                    float(frame.get("window_fit_rms_deg") or 0.0), 5),
                # The whole pose, so anything not thought of today can still be
                # recomputed from what was written down.
                "pose_deg": [round(float(v), 4) for v in here],
            })

    def run(self) -> dict:
        self._write_manifest()
        self.recorder = Recorder(self.folder / RECORDS_NAME)
        done = self.recorder.already_done()
        if done:
            self._note(f"resuming: {len(done)} passes already recorded")
        self.started_at = time.monotonic()
        total = sum(len(d.passes) for d in self.designs) \
            * max(1, self.plan.repeats) * 2
        seen = 0
        try:
            self.plant.park()
            self._standing = None
            for design in self.designs:
                if self._should_stop():
                    break
                failures = 0
                levels = {level.index: level for level in design.levels}
                for entry in design.passes:
                    if self._should_stop() or failures >= self.plan.joint_failure_budget:
                        break
                    level = levels.get(entry["level"])
                    if level is None:
                        continue
                    wanted = [
                        record_key(design.joint, level.index,
                                   entry["speed_deg_s"], direction, repeat)
                        for repeat in range(1, max(1, self.plan.repeats) + 1)
                        for direction in ("+", "-")]
                    if all(key in done for key in wanted):
                        # A resumed run must not drive the arm across the
                        # workspace to re-measure what is already on disk.
                        seen += len(wanted)
                        continue
                    for repeat in range(1, max(1, self.plan.repeats) + 1):
                        for distance in (entry["arc_deg"], -entry["arc_deg"]):
                            direction = "+" if distance > 0 else "-"
                            key = record_key(design.joint, level.index,
                                             entry["speed_deg_s"], direction,
                                             repeat)
                            seen += 1
                            if key in done:
                                continue
                            if self._should_stop():
                                break
                            if self._standing != level.index:
                                try:
                                    self._go_to(level.pose_deg)
                                except (TransitBlocked, MotionFailed) as blocked:
                                    self._note(
                                        f"joint {design.joint + 1} level "
                                        f"{level.index} not reached: {blocked}")
                                    self.skipped.append(
                                        {"joint": design.joint + 1,
                                         "level": level.index,
                                         "why": str(blocked)})
                                    failures += 1
                                    break
                                self._standing = level.index
                            frames = self._drive(design.joint, level, entry,
                                                 distance, repeat)
                            if frames is None:
                                failures += 1
                                continue
                            failures = 0
                            self.driven += 1
                            self._record(design.joint, design, level, entry,
                                         frames, direction, repeat, key)
                    self._say("sweeping", {
                        "joint": design.joint + 1,
                        "joint_name": design.name,
                        "level": level.index,
                        "levels": len(design.levels),
                        "speed_deg_s": entry["speed_deg_s"],
                        "driven": self.driven,
                        "passes": seen,
                        "total": total,
                        "skipped": len(self.skipped),
                        "elapsed_s": time.monotonic() - self.started_at})
                if failures >= self.plan.joint_failure_budget:
                    self._note(f"joint {design.joint + 1} gave up after "
                               f"{failures} refusals in a row")
        finally:
            self.recorder.close()
            try:
                self.plant.park()
            except Exception as error:  # noqa: BLE001 - already unwinding
                self._note(f"could not park at the end: {error}")
            # Written even when the run was cut short, because a sweep that
            # stopped in its fifth hour still measured four.
            try:
                from .loadsweep_report import write_report  # noqa: PLC0415

                self._note(f"report written to {write_report(self.folder)}")
            except Exception as error:  # noqa: BLE001 - the data still matters
                self._note(f"could not write the report: {error}")
        return {"driven": self.driven, "skipped": self.skipped,
                "folder": str(self.folder),
                "elapsed_s": time.monotonic() - self.started_at}
