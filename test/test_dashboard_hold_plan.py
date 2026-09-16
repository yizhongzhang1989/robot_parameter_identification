"""Planned dashboard holds, using synthetic geometry and no hardware."""

import copy
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from test_dashboard import (
    DashboardConfig, IdentificationService, GRAVITY_HOLD_TEST, GRAVITY_DRAG_TEST,
    GRAVITY_TEST_ACKNOWLEDGEMENT, RUNNING, PAUSED, build_routes, dashboard_service,
    synthetic_urdf, test_profile,
)
from robot_parameter_identification.plants.ros_control import MotionStopUnverified


class HoldPlanServiceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.launch = mock.Mock(side_effect=AssertionError("unexpected process"))
        self.bridge = mock.Mock(spec=[
            "latest_sample", "health", "elsewhere", "observed_signals",
            "hardware_plant",
        ])
        self.bridge.latest_sample.return_value = {"position_deg": [0.0] * 7}
        self.bridge.health.return_value = {
            "telemetry_ok": True, "action_ok": True, "sample_age_s": 0.01,
        }
        self.bridge.elsewhere.return_value = {}
        self.bridge.observed_signals.return_value = set()
        self.bridge.hardware_plant.side_effect = AssertionError("unexpected hardware")
        self.service = IdentificationService(
            DashboardConfig(output_directory=directory.name,
                            gravity_test_source=str(self.output / "model")),
            bridge=self.bridge,
            profile=test_profile(), process_launcher=self.launch)
        self.service.adopt_description(synthetic_urdf())
        patches = ExitStack()
        self.addCleanup(patches.close)
        self.plan = {
            "source": self.service.config.gravity_test_source,
            "source_digest": "synthetic-source-digest",
            "joint_names": list(self.service.arm.joint_names),
            "start_deg": [0.0] * 7,
            "poses_deg": [[1.0] * 7, [2.0] * 7],
            "predicted_current_a": [[0.1] * 7, [0.2] * 7],
            "count": 2,
        }
        self.builder = patches.enter_context(mock.patch.object(
            dashboard_service.hold_plan_module, "build_hold_plan",
            side_effect=lambda *_args: copy.deepcopy(self.plan)))
        self.current_source = patches.enter_context(mock.patch.object(
            dashboard_service.hold_plan_module, "source_is_current",
            return_value=True))
        self.real_executor = dashboard_service.hold_plan_module.execute_hold_plan
        self.executor = patches.enter_context(mock.patch.object(
            dashboard_service.hold_plan_module, "execute_hold_plan",
            side_effect=lambda *_args, **_kwargs: self.success()))
        self.clear = patches.enter_context(mock.patch.object(
            self.service.scene, "collision_free", return_value=True))
        patches.enter_context(mock.patch.object(
            self.service.scene, "clearance_rank", return_value={}))
        self.addCleanup(self.service.shutdown, 2.0)

    @staticmethod
    def success():
        return {"result": "PASS", "reason": "verified", "stop_verified": True,
                "restore_errors": [], "records": []}

    def preview(self):
        answer = self.service.plan_preview(GRAVITY_HOLD_TEST, {"poses": 2})
        self.assertTrue(answer["ok"])
        return answer

    def options(self, **changes):
        options = {
            "acknowledgement": GRAVITY_TEST_ACKNOWLEDGEMENT,
            "plan_id": self.service.hold_plan_payload()["id"],
            "poses": 2, "seconds": 2,
        }
        options.update(changes)
        return options

    def wait(self, state="idle"):
        worker = self.service._worker
        if worker is not None:
            worker.join(2.0)
            self.assertFalse(worker.is_alive(), "hold worker did not finish")
        self.assertEqual(self.service.snapshot()["state"], state)

    def fake_plant(self, target):
        plant = mock.Mock(spec=[
            "set_monitor", "set_stop_requested", "move_to", "sample", "wait_for_position", "close",
        ])
        plant.wait_for_position.return_value = {"position_deg": list(target)}
        self.bridge.hardware_plant.side_effect = None
        self.bridge.hardware_plant.return_value = plant
        return plant

    def assert_refused(self, options=None):
        answer = self.service.start_gravity_test(
            GRAVITY_HOLD_TEST, self.options() if options is None else options)
        self.assertFalse(answer["ok"])
        self.executor.assert_not_called()
        self.launch.assert_not_called()
        self.assertEqual(self.service.snapshot()["state"], "idle")
        return answer

    def test_unknown_plan_id_is_refused_without_a_process(self):
        answer = self.service.start_gravity_test(GRAVITY_HOLD_TEST, {
            "acknowledgement": GRAVITY_TEST_ACKNOWLEDGEMENT,
            "plan_id": "never-planned", "poses": 2, "seconds": 2,
        })
        self.assertFalse(answer["ok"])
        self.assertIn("plan", answer["message"])
        self.assertEqual(self.service.snapshot()["state"], "idle")
        self.launch.assert_not_called()

    def test_planning_publishes_hold_set_without_starting_any_process(self):
        token = self.service.preview_token
        self.service._completed_poses = {"hold_set": [1]}
        answer = self.preview()
        self.builder.assert_called_once()
        source, names, start, count, clear = self.builder.call_args.args
        self.assertEqual(source, self.plan["source"])
        self.assertEqual(names, self.plan["joint_names"])
        self.assertEqual(start, self.plan["start_deg"])
        self.assertEqual(count, 2)
        self.assertTrue(clear(np.zeros(7)))
        self.clear.return_value = False
        self.assertFalse(clear(np.zeros(7)))
        self.clear.return_value = True
        self.assertFalse(clear(np.full(7, 1000.0)))
        groups = answer["preview"]["groups"]
        self.assertEqual([group["phase"] for group in groups], ["hold_set"])
        self.assertEqual([pose["pose_deg"] for pose in groups[0]["poses"]],
                         self.plan["poses_deg"])
        self.assertEqual([pose["index"] for pose in groups[0]["poses"]], [1, 2])
        self.assertTrue(all(pose["points"] for pose in groups[0]["poses"]))
        self.assertEqual(answer["preview"]["token"], token + 1)
        self.assertEqual(self.service._completed_poses, {})
        self.assertFalse(self.service.planning)
        self.assertEqual(self.service.snapshot()["state"], "idle")
        self.assertEqual(list(self.output.iterdir()), [])
        self.executor.assert_not_called()
        self.launch.assert_not_called()
        self.bridge.hardware_plant.assert_not_called()

    def test_replanning_retries_an_identical_pose_set(self):
        self.preview()
        changed = copy.deepcopy(self.plan)
        changed["poses_deg"][0][0] += 0.1
        self.builder.side_effect = [copy.deepcopy(self.plan), changed]
        second = self.preview()
        self.assertEqual(self.builder.call_count, 3)
        self.assertEqual(second["preview"]["groups"][0]["poses"][0]["pose_deg"],
                         changed["poses_deg"][0])

    def test_replanning_refuses_when_no_different_pose_set_is_found(self):
        self.preview()
        with self.assertRaisesRegex(ValueError, "different hold pose set"):
            self.preview()
        self.assertFalse(self.service.hold_plan_payload()["available"])
        self.launch.assert_not_called()

    def test_snapshot_exposes_plan_count_and_content_id_but_not_mutable_targets(self):
        answer = self.preview()
        frozen = copy.deepcopy(self.service._hold_plan)
        plan_id = frozen.pop("id")
        self.assertEqual(plan_id, hashlib.sha256(
            json.dumps(frozen, sort_keys=True).encode()).hexdigest())
        permission = {"available": True, "id": plan_id, "poses": 2}
        self.assertEqual(answer["hold_plan"], permission)
        self.assertEqual(self.service.snapshot()["hold_plan"], permission)
        answer["hold_plan"]["poses"] = 20
        answer["preview"]["groups"][0]["poses"][0]["pose_deg"][0] = 99.0
        self.assertEqual(self.service.hold_plan_payload(), permission)
        self.assertEqual(self.service._hold_plan["poses_deg"], self.plan["poses_deg"])

    def test_exact_frozen_plan_reaches_worker_executor_and_plan_json(self):
        self.preview()
        frozen = copy.deepcopy(self.service._hold_plan)
        options = self.options()
        original_options = dict(options)
        with mock.patch.object(dashboard_service.threading, "Thread") as thread:
            answer = self.service.start_gravity_test(GRAVITY_HOLD_TEST, options)
            self.assertTrue(answer["ok"])
            thread.return_value.start.assert_called_once_with()
            worker_plan, seconds = thread.call_args.kwargs["args"]
            self.assertEqual(worker_plan, frozen)
            self.assertIsNot(worker_plan, self.service._hold_plan)
            self.assertIsNot(worker_plan["poses_deg"], self.service._hold_plan["poses_deg"])
            self.service._hold_plan["poses_deg"][0][0] = 99.0
            self.assertEqual(worker_plan, frozen)
            self.assertEqual(self.service._options, {
                "poses": 2, "seconds": 2.0, "plan_id": frozen["id"],
                "transit_speed_deg_s": 5.0,
            })
        thread.call_args.kwargs["target"](worker_plan, seconds)
        self.executor.assert_called_once()
        executed, duration, folder, abort, progress, child = self.executor.call_args.args
        self.assertIs(executed, worker_plan)
        self.assertEqual(executed, frozen)
        self.assertEqual(duration, 2.0)
        self.assertIs(abort, self.service._abort)
        self.assertTrue(callable(progress))
        self.assertEqual(child, self.service._run_hold_child)
        self.assertEqual(self.executor.call_args.kwargs,
                         {"move": self.service._move_hold_target})
        self.assertEqual(json.loads((folder / "plan.json").read_text()), frozen)
        result = self.service.snapshot()["result"]
        self.assertEqual(result["plan_id"], frozen["id"])
        self.assertEqual(result["result"], "PASS")
        self.assertTrue(result["stop_verified"])
        self.assertEqual(result["restore_errors"], [])
        self.assertEqual(json.loads(
            (folder / "gravity_test_summary.json").read_text())["plan_id"], frozen["id"])
        self.assertEqual(result["evidence"],
                         f"/runs/{folder.name}/gravity_test_summary.json")
        self.assertFalse(self.service.hold_plan_payload()["available"])
        self.assertEqual(options, original_options)
        self.builder.assert_called_once()
        self.launch.assert_not_called()
        self.wait()

    def test_acknowledgement_is_required_even_with_a_valid_plan(self):
        self.preview()
        for acknowledgement in (None, "", "not-ready"):
            with self.subTest(acknowledgement=acknowledgement):
                answer = self.assert_refused(self.options(
                    acknowledgement=acknowledgement))
                self.assertIn("E-stop", answer["message"])

    def test_missing_wrong_id_or_changed_count_is_refused(self):
        self.preview()
        for changes in ({"plan_id": None}, {"plan_id": "stale"}, {"poses": 3}):
            with self.subTest(changes=changes):
                answer = self.assert_refused(self.options(**changes))
                self.assertIn("plan", answer["message"])

    def test_changed_context_is_refused(self):
        self.preview()
        options = self.options()
        self.service.config.safety_margin_m += 0.01
        self.assertFalse(self.service.hold_plan_payload()["available"])
        self.assert_refused(options)

    def test_changed_source_is_refused(self):
        self.preview()
        self.current_source.return_value = False
        answer = self.assert_refused()
        self.assertIn("plan", answer["message"])
        self.current_source.assert_called_once_with(self.service._hold_plan)

    def test_other_arm_motion_is_refused(self):
        self.preview()
        options = self.options()
        self.bridge.elsewhere.return_value = {"other_arm_joint1": math.radians(20)}
        self.assertTrue(self.service.screen_drift())
        self.assertFalse(self.service.hold_plan_payload()["available"])
        self.assert_refused(options)

    def test_changed_start_is_refused(self):
        self.preview()
        self.bridge.latest_sample.return_value = {"position_deg": [1.01] + [0.0] * 6}
        answer = self.assert_refused()
        self.assertIn("moved", answer["message"])

    def test_stale_start_telemetry_is_refused(self):
        self.preview()
        self.bridge.health.return_value["telemetry_ok"] = False
        answer = self.assert_refused()
        self.assertIn("fresh", answer["message"])

    def test_planning_requires_fresh_telemetry_and_a_current_scene(self):
        for invalid in ("stale", "missing_sample", "missing_scene", "other_arm"):
            with self.subTest(invalid=invalid), ExitStack() as patches:
                if invalid == "stale":
                    patches.enter_context(mock.patch.object(
                        self.bridge, "health", return_value={"telemetry_ok": False}))
                elif invalid == "missing_sample":
                    patches.enter_context(mock.patch.object(
                        self.bridge, "latest_sample", return_value=None))
                elif invalid == "missing_scene":
                    patches.enter_context(mock.patch.object(self.service, "scene", None))
                else:
                    patches.enter_context(mock.patch.object(
                        self.bridge, "elsewhere",
                        return_value={"other_arm_joint1": math.radians(20)}))
                with self.assertRaisesRegex(ValueError, "fresh|collision scene"):
                    self.preview()
                self.assertFalse(self.service.planning)
                self.assertFalse(self.service.hold_plan_payload()["available"])
        self.builder.assert_not_called()
        self.executor.assert_not_called()
        self.launch.assert_not_called()

    def test_invalid_start_vectors_are_refused(self):
        self.preview()
        for position in ([0.0] * 6, [float("nan")] * 7, [float("inf")] * 7):
            with self.subTest(position=position):
                self.bridge.latest_sample.return_value = {"position_deg": position}
                answer = self.assert_refused()
                self.assertIn("seven finite", answer["message"])

    def test_original_hold_parameter_bounds_are_retained(self):
        self.preview()
        for name, value in (("poses", 0), ("poses", 21), ("poses", "bad"),
                            ("seconds", 0.49), ("seconds", 10.01),
                            ("seconds", float("nan")), ("seconds", None)):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    self.service.start_gravity_test(
                        GRAVITY_HOLD_TEST, self.options(**{name: value}))
        self.executor.assert_not_called()
        self.launch.assert_not_called()

    def test_planning_reserves_activity_and_releases_it_on_failure(self):
        def blocked_builder(*_args):
            self.assertTrue(self.service.planning)
            self.assertFalse(self.service.start("rehearsal")["ok"])
            self.assertFalse(self.service.home()["ok"])
            self.assertFalse(self.service.jog_start()["ok"])
            self.assertFalse(self.service.jog({"action": "start"})["ok"])
            for mode in (GRAVITY_HOLD_TEST, GRAVITY_DRAG_TEST):
                self.assertFalse(self.service.start_gravity_test(mode, self.options())["ok"])
            raise ValueError("synthetic blocked path")

        self.builder.side_effect = blocked_builder
        with self.assertRaisesRegex(ValueError, "synthetic blocked path"):
            self.preview()
        self.assertFalse(self.service.planning)
        self.assertEqual(self.service.snapshot()["state"], "idle")
        self.assertFalse(self.service.hold_plan_payload()["available"])
        self.executor.assert_not_called()
        self.launch.assert_not_called()
        self.bridge.hardware_plant.assert_not_called()
        self.builder.side_effect = lambda *_args: copy.deepcopy(self.plan)
        self.preview()

    def test_move_uses_capped_bridge_plant_exact_target_and_releases_it(self):
        target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        plant = self.fake_plant(target)
        with mock.patch.object(self.service, "_monitor",
                               return_value=self.service._monitor()) as monitor:
            result = self.service._move_hold_target(target)
        self.bridge.hardware_plant.assert_called_once_with(
            self.service.profile, self.service.scene,
            require_neutral_start=False, maximum_speed_deg_s=5.0)
        monitor.assert_called_with()
        plant.set_monitor.assert_called_once_with(monitor.return_value)
        plant.set_stop_requested.assert_called_once()
        self.assertTrue(callable(plant.set_stop_requested.call_args.args[0]))
        plant.move_to.assert_called_once_with(target)
        plant.wait_for_position.assert_called_once_with(target, tolerance_deg=1.0, timeout_s=1.0)
        plant.sample.assert_not_called()
        plant.close.assert_called_once_with()
        self.assertEqual(result, {
            "ok": True, "error_deg": 0.0, "ros_deg": target, "reason": "",
        })
        self.assertFalse(self.service._hold_recovery_required)
        self.launch.assert_not_called()

    def test_move_closes_plant_on_setup_motion_and_sample_errors(self):
        target = [0.5] * 7
        for method in ("set_monitor", "set_stop_requested", "move_to", "wait_for_position"):
            with self.subTest(method=method):
                plant = self.fake_plant(target)
                getattr(plant, method).side_effect = RuntimeError("synthetic plant failure")
                with self.assertRaisesRegex(RuntimeError, "synthetic plant failure"):
                    self.service._move_hold_target(target)
                plant.close.assert_called_once_with()
                self.assertFalse(self.service._hold_recovery_required)
        self.launch.assert_not_called()

    def test_move_uses_requested_transit_speed(self):
        target = [0.5] * 7
        self.fake_plant(target)
        self.service._options = {"transit_speed_deg_s": 15.0}
        self.assertTrue(self.service._move_hold_target(target)["ok"])
        self.assertEqual(
            self.bridge.hardware_plant.call_args.kwargs["maximum_speed_deg_s"], 15.0)

    def test_hold_rejects_invalid_transit_speed_before_execution(self):
        for speed in (None, "bad", float("nan"), float("inf"), 0, -1, 61):
            with self.subTest(speed=speed), self.assertRaisesRegex(
                    ValueError, "transit_speed_deg_s"):
                self.service.start_gravity_test(
                    GRAVITY_HOLD_TEST, self.options(transit_speed_deg_s=speed))
        self.bridge.hardware_plant.assert_not_called()
        self.executor.assert_not_called()

    def test_move_rejects_invalid_or_off_target_samples_and_closes_plant(self):
        target = [0.5] * 7
        for sample in (None, {}, {"position_deg": [0.5] * 6},
                       {"position_deg": [float("nan")] * 7},
                       {"position_deg": [float("inf")] * 7}):
            with self.subTest(sample=sample):
                plant = self.fake_plant(target)
                plant.wait_for_position.return_value = sample
                with self.assertRaisesRegex(ValueError, "position telemetry"):
                    self.service._move_hold_target(target)
                plant.close.assert_called_once_with()
        plant = self.fake_plant([2.0] + target[1:])
        result = self.service._move_hold_target(target)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_deg"], 1.5)
        self.assertIn("not reached", result["reason"])
        plant.close.assert_called_once_with()
        self.assertFalse(self.service._hold_recovery_required)
        self.launch.assert_not_called()

    def test_move_closes_plant_when_transit_is_blocked(self):
        plant = self.fake_plant([0.5] * 7)
        self.clear.return_value = False
        with self.assertRaisesRegex(ValueError, "transit"):
            self.service._move_hold_target([0.5] * 7)
        plant.move_to.assert_not_called()
        plant.close.assert_called_once_with()

    def test_stop_callback_short_circuits_abort_and_stale_scene_before_telemetry(self):
        plant = self.fake_plant([0.5] * 7)
        self.service._move_hold_target([0.5] * 7)
        stop_requested = plant.set_stop_requested.call_args.args[0]
        self.assertFalse(stop_requested())
        for trigger in ("abort", "scene_drift", "context"):
            with self.subTest(trigger=trigger), ExitStack() as patches:
                if trigger == "abort":
                    self.service._abort.set()
                elif trigger == "scene_drift":
                    patches.enter_context(mock.patch.object(
                        self.service, "screen_drift", return_value=[{"joint": "other"}]))
                else:
                    patches.enter_context(mock.patch.object(
                        self.service, "_hold_context", return_value="changed-context"))
                telemetry = patches.enter_context(mock.patch.object(
                    self.service, "connection",
                    side_effect=AssertionError("stop must not wait for telemetry")))
                try:
                    self.assertTrue(stop_requested())
                    telemetry.assert_not_called()
                finally:
                    self.service._abort.clear()
        with mock.patch.object(self.service, "connection",
                               return_value={"telemetry_ok": False}) as telemetry:
            self.assertTrue(stop_requested())
            telemetry.assert_called_once_with()

    def test_abort_before_move_does_not_open_plant_or_latch_recovery(self):
        self.service._abort.set()
        with self.assertRaisesRegex(RuntimeError, "stopped before motion"):
            self.service._move_hold_target([0.5] * 7)
        self.bridge.hardware_plant.assert_not_called()
        self.assertFalse(self.service._hold_recovery_required)

    def run_move_failure(self, error):
        self.preview()
        plant = self.fake_plant(self.plan["poses_deg"][0])
        plant.move_to.side_effect = error
        backend = SimpleNamespace(current=SimpleNamespace(
            DEFAULT_CORRIDOR_DEG=2.0, MAXIMUM_TEMPERATURE_C=40.0,
            ACKNOWLEDGEMENT="synthetic-ack"))

        def execute(*args, move):
            return self.real_executor(*args, move=move, backend=backend)

        self.executor.side_effect = execute
        with mock.patch.object(self.service, "_run_hold_child") as child:
            self.assertTrue(self.service.start_gravity_test(
                GRAVITY_HOLD_TEST, self.options())["ok"])
            self.wait(PAUSED if isinstance(error, MotionStopUnverified) else "idle")
            child.assert_not_called()
        self.assertEqual(self.service.result["result"], "FAIL")
        self.assertIn(str(error), self.service.result["reason"])
        self.assertFalse(self.service.result["stop_verified"])
        self.assertEqual(len(self.service.result["records"]), 1)
        plant.move_to.assert_called_once_with(self.plan["poses_deg"][0])
        plant.sample.assert_not_called()
        plant.close.assert_called_once_with()
        self.launch.assert_not_called()

    def test_unverified_motion_stop_latches_paused_and_blocks_other_tasks(self):
        self.run_move_failure(MotionStopUnverified("synthetic stop unverified"))
        self.assertTrue(self.service._hold_recovery_required)
        self.assertIsNone(self.service._worker)
        self.assertFalse(self.service.hold_plan_payload()["available"])
        self.assertIn("operator recovery required", self.service.progress["error"])
        with mock.patch.object(dashboard_service.threading, "Thread") as thread:
            self.assertFalse(self.service.start("rehearsal")["ok"])
            self.assertFalse(self.service.home()["ok"])
            self.assertFalse(self.service.jog_start()["ok"])
            for mode in (GRAVITY_HOLD_TEST, GRAVITY_DRAG_TEST):
                self.assertFalse(self.service.start_gravity_test(mode, self.options())["ok"])
            self.service.stop()
            self.assertEqual(self.service.snapshot()["state"], PAUSED)
            self.assertFalse(self.service.start("rehearsal")["ok"])
            thread.assert_not_called()

    def test_unverified_current_stop_latches_paused_and_blocks_other_tasks(self):
        def execute(*_args, **_kwargs):
            self.service._hold_current_started = True
            return {"result": "FAIL", "reason": "synthetic current stop unverified",
                    "stop_verified": False, "restore_errors": ["recovery unverified"],
                    "records": []}

        self.preview()
        options = self.options()
        self.executor.side_effect = execute
        self.assertTrue(self.service.start_gravity_test(GRAVITY_HOLD_TEST, options)["ok"])
        self.wait(PAUSED)
        self.assertTrue(self.service._hold_current_started)
        self.assertTrue(self.service._hold_recovery_required)
        self.assertEqual(self.service._activity, GRAVITY_HOLD_TEST)
        self.assertIsNone(self.service._worker)
        self.assertEqual(self.service.result["result"], "FAIL")
        self.assertFalse(self.service.result["stop_verified"])
        self.assertEqual(self.service.result["restore_errors"], ["recovery unverified"])
        self.assertFalse(self.service.hold_plan_payload()["available"])
        self.assertIn("operator recovery required", self.service.progress["error"])
        with mock.patch.object(dashboard_service.threading, "Thread") as thread:
            self.assertFalse(self.service.start("rehearsal")["ok"])
            self.assertFalse(self.service.home()["ok"])
            self.assertFalse(self.service.jog_start()["ok"])
            self.assertFalse(self.service.jog({"action": "start"})["ok"])
            for mode in (GRAVITY_HOLD_TEST, GRAVITY_DRAG_TEST):
                self.assertFalse(self.service.start_gravity_test(mode, options)["ok"])
            with self.assertRaisesRegex(RuntimeError, "run is going"):
                self.service.plan_preview(GRAVITY_HOLD_TEST, {"poses": 2})
            self.assertFalse(self.service.hold_plan_payload()["available"])
            self.service.stop()
            self.assertEqual(self.service.snapshot()["state"], PAUSED)
            self.assertEqual(self.service._activity, GRAVITY_HOLD_TEST)
            self.assertFalse(self.service.start("rehearsal")["ok"])
            thread.assert_not_called()
        self.assertIsNone(self.service._worker)
        self.builder.assert_called_once()
        self.executor.assert_called_once()
        self.launch.assert_not_called()
        self.bridge.hardware_plant.assert_not_called()

    def recovery_latched_service(self):
        self.service._state = PAUSED
        self.service._activity = GRAVITY_HOLD_TEST
        self.service._hold_recovery_required = True
        self.service._hold_current_started = True
        self.service._worker = None
        self.service.result = {"result": "FAIL", "reason": "UDP state stale",
                               "stop_verified": False}
        self.service.progress = {"phase": "failed", "error": "operator recovery required"}

    def test_confirmed_live_recovery_releases_old_lock_without_resuming(self):
        self.recovery_latched_service()
        original = copy.deepcopy(self.service.result)
        self.service.gravity_armed = "old permission"
        self.service.rehearsal_passed = True
        self.bridge.attach_mock(mock.Mock(return_value={"ready": True}), "recovery_status")
        snapshot = self.service.snapshot()
        self.assertEqual(snapshot["state"], "idle")
        self.assertEqual(snapshot["activity"], "")
        self.assertTrue(snapshot["progress"]["recovery_verified"])
        self.assertEqual(snapshot["progress"]["error"], "UDP state stale")
        self.assertEqual(snapshot["result"], original)
        self.assertFalse(self.service._hold_recovery_required)
        self.assertFalse(snapshot["gravity_armed"])
        self.assertFalse(snapshot["rehearsal_passed"])
        self.assertFalse(snapshot["hold_plan"]["available"])
        notes = list(self.service.notes)
        self.service.snapshot()
        self.assertEqual(self.service.notes, notes)
        self.executor.assert_not_called()
        self.bridge.hardware_plant.assert_not_called()
        self.launch.assert_not_called()
        self.assertTrue(self.preview()["ok"])

    def test_recovery_requires_explicit_positive_evidence(self):
        self.recovery_latched_service()
        reader = mock.Mock()
        self.bridge.attach_mock(reader, "recovery_status")
        for evidence in (None, {}, {"ready": False}, {"ready": "true"}):
            with self.subTest(evidence=evidence):
                reader.return_value = evidence
                self.assertEqual(self.service.snapshot()["state"], PAUSED)
        reader.side_effect = RuntimeError("telemetry unavailable")
        self.assertEqual(self.service.snapshot()["state"], PAUSED)

    def test_only_confirmed_moves_allow_recovery_without_action_status(self):
        for records, required in (([], True), ([{}], True),
                                  ([{"move": {"ok": False}}], True),
                                  ([{"move": {"ok": True}}], False)):
            with self.subTest(records=records):
                self.recovery_latched_service()
                self.service.result["records"] = records
                reader = mock.Mock(return_value={"ready": False})
                self.bridge.attach_mock(reader, "recovery_status")
                self.service.snapshot()
                reader.assert_called_once_with(require_goal_status=required)

    def test_recovery_never_releases_an_active_owner_or_other_mode(self):
        self.recovery_latched_service()
        reader = mock.Mock(return_value={"ready": True})
        self.bridge.attach_mock(reader, "recovery_status")
        for attribute, value in (("_worker", object()), ("planning", True),
                                 ("_activity", "gravity"), ("_state", RUNNING),
                                 ("_external_process", mock.Mock(poll=lambda: None))):
            with self.subTest(attribute=attribute), mock.patch.object(
                    self.service, attribute, value):
                self.service.snapshot()
                self.assertTrue(self.service._hold_recovery_required)
        reader.assert_not_called()

    def test_verified_current_stop_after_failure_releases_activity_for_replanning(self):
        def execute(*_args, **_kwargs):
            self.service._hold_current_started = True
            return {"result": "FAIL", "reason": "synthetic hold failure with verified stop",
                    "stop_verified": True, "restore_errors": [], "records": []}

        self.preview()
        self.executor.side_effect = execute
        self.assertTrue(self.service.start_gravity_test(
            GRAVITY_HOLD_TEST, self.options())["ok"])
        self.wait()
        self.assertTrue(self.service._hold_current_started)
        self.assertFalse(self.service._hold_recovery_required)
        self.assertEqual(self.service._activity, "")
        self.assertIsNone(self.service._worker)
        self.assertEqual(self.service.result["result"], "FAIL")
        self.assertTrue(self.service.result["stop_verified"])
        self.assertEqual(self.service.result["restore_errors"], [])
        self.assertFalse(self.service.hold_plan_payload()["available"])
        self.preview()
        self.assertTrue(self.service.hold_plan_payload()["available"])
        self.executor.assert_called_once()
        self.launch.assert_not_called()
        self.bridge.hardware_plant.assert_not_called()

    def test_normal_abort_and_cancel_runtime_errors_do_not_latch_recovery(self):
        for reason in ("operator stop requested", "trajectory canceled"):
            with self.subTest(reason=reason):
                self.run_move_failure(RuntimeError(reason))
                self.assertFalse(self.service._hold_recovery_required)
                self.assertIsNone(self.service._worker)
                self.preview()
                self.executor.side_effect = lambda *_args, **_kwargs: self.success()
                self.assertTrue(self.service.start_gravity_test(
                    GRAVITY_HOLD_TEST, self.options())["ok"])
                self.wait()
                self.assertEqual(self.service.result["result"], "PASS")

    def test_planning_rejects_context_or_start_changes_during_build(self):
        for changed in ("context", "start"):
            with self.subTest(changed=changed):
                self.bridge.latest_sample.return_value = {"position_deg": [0.0] * 7}

                def change_during_build(*_args):
                    if changed == "context":
                        self.service.config.safety_margin_m += 0.01
                    else:
                        self.bridge.latest_sample.return_value = {"position_deg": [2.0] * 7}
                    return copy.deepcopy(self.plan)

                self.builder.side_effect = change_during_build
                with self.assertRaisesRegex(ValueError, "during planning"):
                    self.preview()
                self.assertFalse(self.service.planning)
                self.assertFalse(self.service.hold_plan_payload()["available"])
        self.launch.assert_not_called()

    def test_planned_hold_is_reachable_over_http(self):
        routes = build_routes(self.service, None)
        refused = routes["/api/gravity-test"][1]({
            "mode": GRAVITY_HOLD_TEST, "options": self.options(),
        })
        self.assertFalse(refused["ok"])
        planned = routes["/api/plan"][1]({
            "mode": GRAVITY_HOLD_TEST, "options": {"poses": 2},
        })
        self.assertTrue(planned["ok"])
        self.assertTrue(planned["hold_plan"]["available"])
        self.executor.assert_not_called()
        self.launch.assert_not_called()
        answer = routes["/api/gravity-test"][1]({
            "mode": GRAVITY_HOLD_TEST, "options": self.options(),
        })
        self.assertTrue(answer["ok"])
        self.wait()
        self.executor.assert_called_once()
        self.assertEqual(self.service.result["result"], "PASS")

    def test_worker_refuses_changed_scene_or_blocked_transit_before_child_execution(self):
        for changed in ("context", "other_arm", "transit", "stale"):
            with self.subTest(changed=changed), ExitStack() as patches:
                self.preview()

                def execute(plan, _seconds, _folder, _abort, progress, child, *, move):
                    if changed == "context":
                        self.service.config.safety_margin_m += 0.01
                    elif changed == "other_arm":
                        patches.enter_context(mock.patch.object(
                            self.bridge, "elsewhere",
                            return_value={"other_arm_joint1": math.radians(20)}))
                    elif changed == "transit":
                        patches.enter_context(mock.patch.object(
                            self.service.scene, "collision_free", return_value=False))
                    else:
                        patches.enter_context(mock.patch.object(
                            self.bridge, "health", return_value={"telemetry_ok": False}))
                    progress("hold_set", {
                        "stage": "moving", "target_pose": 1,
                        "pose_deg": plan["poses_deg"][0],
                    })
                    child(["must-not-launch"], 1.0)
                    return self.success()

                self.executor.side_effect = execute
                self.assertTrue(self.service.start_gravity_test(
                    GRAVITY_HOLD_TEST, self.options())["ok"])
                self.wait()
                self.assertEqual(self.service.result["result"], "FAIL")
                self.assertRegex(
                    self.service.result["reason"], "scene changed|no longer clear|fresh")
                self.assertFalse(self.service.hold_plan_payload()["available"])
                self.launch.assert_not_called()

    def test_executor_exception_releases_slot_and_requires_a_new_plan(self):
        self.preview()
        options = self.options()
        self.executor.side_effect = RuntimeError("synthetic executor failure")
        self.assertTrue(self.service.start_gravity_test(GRAVITY_HOLD_TEST, options)["ok"])
        self.wait()
        self.assertEqual(self.service.result["result"], "FAIL")
        self.assertIn("synthetic executor failure", self.service.result["reason"])
        self.assertIsNone(self.service._worker)
        self.executor.reset_mock()
        self.assert_refused(options)
        self.preview()
        self.executor.side_effect = lambda *_args, **_kwargs: self.success()
        self.assertTrue(self.service.start_gravity_test(
            GRAVITY_HOLD_TEST, self.options())["ok"])
        self.wait()
        self.assertEqual(self.service.result["result"], "PASS")
        self.launch.assert_not_called()

    def test_hold_progress_shares_next_target_index_and_completions(self):
        scenes = []
        views = []

        def execute(plan, _seconds, _folder, _abort, progress, _child, *, move):
            for index, pose in enumerate(plan["poses_deg"], 1):
                for stage in ("moving", "holding"):
                    progress("hold_set", {"target_pose": index, "pose_deg": pose,
                                          "stage": stage})
                    scenes.append(self.service.scene_activity_payload())
                    views.append(self.service.viewer_state())
                progress("hold_set", {
                    "target_pose": None, "pose_deg": None, "completed_pose": index,
                    "completed_pose_indices": list(range(1, index + 1)),
                    "stage": "completed",
                })
                scenes.append(self.service.scene_activity_payload())
                views.append(self.service.viewer_state())
            return self.success()

        self.preview()
        self.executor.side_effect = execute
        self.assertTrue(self.service.start_gravity_test(
            GRAVITY_HOLD_TEST, self.options())["ok"])
        self.wait()
        self.assertEqual(self.service.result["result"], "PASS")
        self.assertEqual(len(scenes), 6)
        for offset, index in ((0, 1), (3, 2)):
            for position in (offset, offset + 1):
                self.assertEqual(scenes[position]["mode"], GRAVITY_HOLD_TEST)
                self.assertEqual(scenes[position]["state"], RUNNING)
                self.assertEqual(scenes[position]["focus"], {
                    "kind": "joint_pose", "phase": "hold_set", "index": index,
                    "pose_deg": self.plan["poses_deg"][index - 1],
                })
                self.assertEqual(views[position]["moving"]["index"], index)
            completed = scenes[offset + 2]
            self.assertIsNone(completed["focus"])
            self.assertNotIn("moving", views[offset + 2])
            self.assertEqual(completed["completed"],
                             {"hold_set": list(range(1, index + 1))})
        self.assertEqual(scenes[3]["completed"], {"hold_set": [1]})
        self.assertEqual(self.service.scene_activity_payload()["completed"],
                         {"hold_set": [1, 2]})
        self.launch.assert_not_called()

    def test_stop_reaches_plan_executor_and_releases_activity_for_replanning(self):
        entered = threading.Event()

        def execute(_plan, _seconds, _folder, abort, _progress, _child, *, move):
            entered.set()
            if not abort.wait(2.0):
                raise AssertionError("Stop did not reach hold executor")
            return {"result": "FAIL", "reason": "operator stop requested",
                    "stop_verified": False, "restore_errors": ["recovery unverified"]}

        self.preview()
        options = self.options()
        self.executor.side_effect = execute
        self.assertTrue(self.service.start_gravity_test(GRAVITY_HOLD_TEST, options)["ok"])
        try:
            self.assertTrue(entered.wait(1.0))
            self.assertFalse(self.service.start("rehearsal")["ok"])
            for mode in (GRAVITY_HOLD_TEST, GRAVITY_DRAG_TEST):
                self.assertFalse(self.service.start_gravity_test(mode, options)["ok"])
        finally:
            build_routes(self.service, None)["/api/stop"][1]({})
            self.wait()
        self.assertEqual(self.service.result["result"], "FAIL")
        self.assertFalse(self.service.result["stop_verified"])
        self.assertEqual(self.service.result["restore_errors"], ["recovery unverified"])
        self.executor.reset_mock()
        self.assert_refused(options)
        self.preview()
        self.executor.side_effect = lambda *_args, **_kwargs: self.success()
        self.assertTrue(self.service.start_gravity_test(
            GRAVITY_HOLD_TEST, self.options())["ok"])
        self.wait()
        self.assertEqual(self.service.result["result"], "PASS")
        self.assertFalse(self.service._abort.is_set())
        self.launch.assert_not_called()


class HoldPlanCompletionProgressTest(unittest.TestCase):
    def test_real_executor_clears_shared_focus_and_commits_completed_indices(self):
        made = IdentificationService(DashboardConfig(), profile=test_profile())
        made.adopt_description(synthetic_urdf())
        made._activity = GRAVITY_HOLD_TEST
        made._state = RUNNING
        poses = [[1.0] * 7, [2.0] * 7]
        plan = {
            "source": "synthetic-source", "source_digest": "synthetic-digest",
            "joint_names": list(made.arm.joint_names), "start_deg": [0.0] * 7,
            "poses_deg": poses, "predicted_current_a": [[0.1] * 7] * 2, "count": 2,
        }
        backend = SimpleNamespace(
            current=SimpleNamespace(DEFAULT_CORRIDOR_DEG=2.0,
                                    MAXIMUM_TEMPERATURE_C=40.0,
                                    ACKNOWLEDGEMENT="synthetic-ack"),
        )
        scenes = []
        views = []
        updates = []
        moves = []
        children = []

        def progress(phase, detail):
            updates.append(copy.deepcopy(detail))
            made._on_progress(phase, detail)
            scenes.append(made.scene_activity_payload())
            views.append(made.viewer_state())

        def move(pose):
            self.assertEqual(updates[-1]["stage"], "moving")
            self.assertEqual(pose, poses[len(moves)])
            moves.append(list(pose))
            return {"ok": True, "ros_deg": list(pose), "error_deg": 0.0}

        def child(command, _timeout):
            self.assertEqual(command[3], "hold_check")
            self.assertEqual(updates[-1]["stage"], "holding")
            children.append(command)
            report = Path(command[command.index("--output") + 1])
            report.write_text(json.dumps({
                "result": "PASS", "stop_verified": True, "restore_errors": [],
            }), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                dashboard_service.hold_plan_module, "source_is_current", return_value=True), \
                mock.patch.object(dashboard_service.hold_plan_module, "_load_backend",
                                  side_effect=AssertionError("unexpected real backend")):
            result = dashboard_service.hold_plan_module.execute_hold_plan(
                plan, 2.0, directory, threading.Event(), progress, child,
                move=move, backend=backend)
        self.assertEqual(result["result"], "PASS", result["reason"])
        self.assertEqual(moves, poses)
        self.assertEqual(len(children), len(poses))
        self.assertEqual(len(updates), 6)
        for position, index in ((2, 1), (5, 2)):
            with self.subTest(index=index, field="pose_deg"):
                self.assertIn("pose_deg", updates[position])
                self.assertIsNone(updates[position]["pose_deg"])
            with self.subTest(index=index, field="completed_pose_indices"):
                self.assertEqual(updates[position].get("completed_pose_indices"),
                                 list(range(1, index + 1)))
            with self.subTest(index=index, field="shared_scene"):
                self.assertIsNone(scenes[position]["focus"])
                self.assertNotIn("moving", views[position])
                self.assertEqual(scenes[position]["completed"],
                                 {"hold_set": list(range(1, index + 1))})
        self.assertEqual(scenes[3]["focus"]["index"], 2)
        self.assertEqual(scenes[3]["completed"], {"hold_set": [1]})
