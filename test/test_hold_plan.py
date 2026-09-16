import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from robot_parameter_identification.dashboard import hold_plan


class HoldPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = hold_plan._load_backend()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.source = self.folder / "result.json"
        self.payload = {
            "complete": True, "verdict": {"state": "pass"},
            "effort_unit": "ampere", "joint_names": hold_plan.RIGHT_JOINTS.copy(),
            "joints": [{"columns": [0], "parameters": [0.1]} for _ in range(7)],
        }
        self.source.write_text(json.dumps(self.payload))
        self.candidates = np.array([[value] * 7 for value in (-10.123456789, 0, 10, 20)])
        self.model = {"joint_names": hold_plan.RIGHT_JOINTS.copy(), "columns": [[0]] * 7}
        self.backend = SimpleNamespace(
            campaign=SimpleNamespace(
                _executed_poses=Mock(side_effect=lambda *args: self.candidates.copy()),
                admissible=self.legacy.campaign.admissible,
                random_then_order=Mock(wraps=self.legacy.campaign.random_then_order),
                transit_clear=self.legacy.campaign.transit_clear),
            identified=SimpleNamespace(
                load_identification=Mock(return_value=self.model),
                gravity_current=Mock(side_effect=lambda model, arm, pose: np.abs(pose) / 100),
                CONTINUOUS_CURRENT_A=self.legacy.identified.CONTINUOUS_CURRENT_A),
            current=SimpleNamespace(
                DEFAULT_SOURCE=self.folder,
                DEFAULT_CORRIDOR_DEG=self.legacy.current.DEFAULT_CORRIDOR_DEG,
                MAXIMUM_TEMPERATURE_C=self.legacy.current.MAXIMUM_TEMPERATURE_C,
                ACKNOWLEDGEMENT=self.legacy.current.ACKNOWLEDGEMENT),
            arm_model=Mock(return_value=SimpleNamespace(parameter_count=70)))
        self.commands = []
        self.moves = []
        self.events = []
        self.stopped = False
        self.report = {"result": "PASS", "stop_verified": True,
                       "restore_errors": [], "session_failure": ""}
        self.move_ok = True
        self.hold_code = 0
        self.telemetry_offset = 0
        self.after_move = False
        self.after_hold = False
        self.write_report = True

    def build(self, **changes):
        options = dict(source=str(self.folder), joint_names=hold_plan.RIGHT_JOINTS.copy(),
                       start_deg=[0] * 7, count=3, collision_free=lambda pose: True,
                       backend=self.backend)
        options.update(changes)
        return hold_plan.build_hold_plan(**options)

    def move(self, pose):
        self.assertIsInstance(pose, list)
        self.moves.append(pose.copy())
        ros_deg = [value + self.telemetry_offset for value in pose]
        self.stopped = self.after_move
        if not self.move_ok:
            return {"ok": False, "reason": "move failed"}
        if not np.isfinite(ros_deg).all():
            return {"ok": False, "reason": "nonfinite telemetry"}
        if np.max(np.abs(np.asarray(ros_deg) - pose)) > 1.0:
            return {"ok": False, "reason": "target not reached", "ros_deg": ros_deg}
        return {"ok": True, "ros_deg": ros_deg}

    def child(self, command, timeout_s):
        self.commands.append((command, timeout_s))
        self.assertEqual(command[:4],
                         ["ros2", "run", "rm_control", "hold_check"])
        if self.write_report:
            Path(command[command.index("--output") + 1]).write_text(json.dumps(self.report))
        self.stopped = self.after_hold
        return SimpleNamespace(returncode=self.hold_code, stdout="", stderr="hold stderr")

    def execute(self, plan=None, **changes):
        options = dict(plan=self.build() if plan is None else plan, seconds=2,
                       output_directory=self.folder / "output", abort=lambda: self.stopped,
                       on_progress=lambda phase, detail: self.events.append((phase, detail)),
                       run_child=self.child, move=self.move, backend=self.backend)
        options.update(changes)
        return hold_plan.execute_hold_plan(**options)

    def test_source_hash_tracks_bytes_and_accepts_both_paths(self):
        for source in (self.source, self.folder):
            plan = {"source": str(source), "source_digest": hold_plan._digest(self.source)}
            self.assertTrue(hold_plan.source_is_current(plan))
            self.source.write_text(self.source.read_text() + "\n")
            self.assertFalse(hold_plan.source_is_current(plan))
        self.assertFalse(hold_plan.source_is_current({}))
        self.source.unlink()
        self.assertFalse(hold_plan.source_is_current(plan))

    def test_source_requires_complete_passing_ampere_right_arm_model(self):
        hold_plan._validate_source(self.payload, hold_plan.RIGHT_JOINTS)
        for key, value in (("complete", False), ("verdict", {"state": "warn"}),
                           ("effort_unit", "newton_metre"), ("joints", []),
                           ("joint_names", list(reversed(hold_plan.RIGHT_JOINTS)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                hold_plan._validate_source({**self.payload, key: value}, hold_plan.RIGHT_JOINTS)

    def test_planning_reuses_selection_and_is_json_serializable(self):
        seen = []
        with patch("subprocess.run", side_effect=AssertionError("no commands")), \
                patch("socket.socket", side_effect=AssertionError("no sockets")), \
                patch.object(hold_plan.secrets, "randbits", return_value=11):
            plan = self.build(
                source="", collision_free=lambda pose: seen.append(pose.copy()) or True)
            repeated = self.build(source=str(self.source))
        self.assertEqual(plan["source"], str(self.folder))
        self.assertEqual(plan, json.loads(json.dumps(plan, allow_nan=False)))
        self.assertEqual(plan, repeated)
        self.assertEqual(plan["seed"], 11)
        self.assertEqual(self.backend.campaign.random_then_order.call_args.kwargs, {"seed": 11})
        self.assertEqual(len(seen), 3 * 400)
        np.testing.assert_array_equal(seen[0], plan["start_deg"])
        for index, pose in enumerate(plan["poses_deg"]):
            np.testing.assert_allclose(seen[(index + 1) * 400 - 1], pose, atol=1e-12)
        np.testing.assert_allclose(plan["predicted_current_a"], np.abs(plan["poses_deg"]) / 100)

    def test_new_seeds_produce_different_pose_sets(self):
        self.candidates = np.array([[value] * 7 for value in range(20)])
        with patch.object(hold_plan.secrets, "randbits", side_effect=[11, 12]):
            first = self.build()
            second = self.build()
        self.assertNotEqual(first["seed"], second["seed"])
        self.assertNotEqual(first["poses_deg"], second["poses_deg"])

    def test_import_adapter_and_arm_model_never_run_commands_or_sockets(self):
        import sys

        before = sys.path[:]
        bindings = {name: sys.modules.get(name) for name in (
            "identified_zero_force_drag", "identified_static_hold_campaign",
            "forward_current_controller_test", "subprocess")}
        with patch("subprocess.run", side_effect=AssertionError("no commands")), \
                patch("subprocess.Popen", side_effect=AssertionError("no processes")), \
                patch("socket.socket", side_effect=AssertionError("no sockets")):
            backend = hold_plan._load_backend()
            arm = backend.arm_model()
        self.assertGreater(arm.parameter_count, 0)
        self.assertEqual(sys.path, before)
        for name, module in bindings.items():
            self.assertIs(sys.modules.get(name), module)

    def test_invalid_fits_and_nonfinite_inputs_refused(self):
        for fit in ({"columns": [], "parameters": []},
                    {"columns": [0, 1], "parameters": [1]},
                    {"columns": [0], "parameters": [float("nan")]},
                    {"columns": [0], "parameters": [[1.0]]},
                    {"columns": [0, 0], "parameters": [1, 2]}):
            with self.subTest(fit=fit), self.assertRaises(ValueError):
                hold_plan._validate_source(
                    {**self.payload, "joints": [fit] * 7}, hold_plan.RIGHT_JOINTS)
        for options in ({"start_deg": [0] * 6}, {"start_deg": [float("inf")] * 7},
                        {"count": 0}, {"count": True}, {"count": 21},
                        {"collision_free": None},
                        {"joint_names": list(reversed(hold_plan.RIGHT_JOINTS))}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.build(**options)

    def test_insufficient_finite_or_current_admissible_candidates_refused(self):
        for candidates in (np.array([[float("nan")] * 7]), np.zeros((0, 7)),
                           np.array([[900] * 7, [0] * 7])):
            self.candidates = candidates
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                self.build()
        self.candidates = np.array([[0] * 7, [float("inf")] * 7])
        self.assertEqual(self.build(count=1)["poses_deg"], [[0.0] * 7])
        self.backend.identified.gravity_current.side_effect = lambda *args: [float("nan")] * 7
        with self.assertRaises(ValueError):
            self.build(count=1)

    def test_target_and_intermediate_collisions_refused_in_degrees(self):
        self.candidates = np.array([[20] * 7])
        for guard in (lambda pose: pose[0] < 19, lambda pose: not 8 < pose[0] < 12,
                      lambda pose: pose[0] != 0):
            with self.subTest(guard=guard), self.assertRaisesRegex(ValueError, "blocked"):
                self.build(count=1, collision_free=guard)

    def test_source_change_during_planning_refused(self):
        def screen(pose):
            self.source.write_text(self.source.read_text() + " ")
            return True
        with self.assertRaisesRegex(ValueError, "changed during planning"):
            self.build(count=1, collision_free=screen)

    def test_execution_requires_keyword_only_move(self):
        arguments = (None, 2, self.folder, lambda: False, Mock(), self.child)
        with self.assertRaises(TypeError):
            hold_plan.execute_hold_plan(*arguments)
        with self.assertRaises(TypeError):
            hold_plan.execute_hold_plan(*arguments, self.move)

    def test_execution_uses_exact_targets_and_publishes_before_move_and_hold(self):
        plan = self.build()
        selected = self.backend.campaign.random_then_order.call_count

        def move(pose):
            self.assertEqual(self.events[-1][0], "hold_set")
            self.assertEqual(self.events[-1][1]["stage"], "moving")
            self.assertEqual(self.events[-1][1]["pose_deg"], pose)
            return self.move(pose)

        def child(command, timeout_s):
            self.assertEqual(self.events[-1][0], "hold_set")
            self.assertEqual(self.events[-1][1]["stage"], "holding")
            return self.child(command, timeout_s)

        result = self.execute(plan, run_child=child, move=move)
        self.assertEqual(result["result"], "PASS", result)
        self.assertTrue(result["stop_verified"])
        self.assertEqual(self.backend.campaign.random_then_order.call_count, selected)
        self.assertEqual(self.moves, plan["poses_deg"])
        self.assertEqual(len(self.commands), 3)
        for index, pose in enumerate(plan["poses_deg"]):
            hold, hold_timeout = self.commands[index]
            self.assertEqual(result["records"][index]["move"], {"ok": True, "ros_deg": pose})
            self.assertNotIn("move_returncode", result["records"][index])
            self.assertEqual(hold_timeout, 302)
            self.assertEqual(hold[hold.index("--source") + 1], plan["source"])
            self.assertEqual(hold[hold.index("--ack") + 1], "I_AM_HOLDING_ARM_AND_ESTOP_READY")
            self.assertEqual(float(hold[hold.index("--corridor-deg") + 1]), 5.0)
            self.assertLessEqual(float(hold[hold.index("--temperature-c") + 1]), 40)
            self.assertNotIn("--ignore-model-fit", hold)
        completed = [
            detail["completed_pose"] for _, detail in self.events if "completed_pose" in detail]
        self.assertEqual(completed, [1, 2, 3])
        self.assertIsNone(self.events[-1][1]["target_pose"])

    def test_execution_snapshot_resists_callback_mutation(self):
        plan = self.build()
        original = json.loads(json.dumps(plan))

        def progress(phase, detail):
            plan["poses_deg"][-1] = [999] * 7
            if detail.get("pose_deg") is not None:
                detail["pose_deg"][:] = [888] * 7

        def move(pose):
            verdict = self.move(pose)
            pose[:] = [777] * 7
            return verdict

        result = self.execute(plan, on_progress=progress, move=move)
        self.assertEqual(result["result"], "PASS", result)
        self.assertEqual(self.moves, original["poses_deg"])
        self.assertEqual(
            [record["pose_deg"] for record in result["records"]], original["poses_deg"])

    def test_failed_move_or_invalid_telemetry_never_holds(self):
        for ok, offset in ((False, 0), (True, 2), (True, float("nan")),
                           (True, float("inf"))):
            self.move_ok, self.telemetry_offset = ok, offset
            self.commands.clear()
            self.moves.clear()
            result = self.execute()
            self.assertEqual(result["result"], "FAIL", result)
            self.assertFalse(result["stop_verified"])
            self.assertFalse(result["records"][0]["move"]["ok"])
            self.assertEqual(len(self.moves), 1)
            self.assertFalse(self.commands)

    def test_every_nonpass_hold_and_recovery_error_stops_immediately(self):
        cases = [(1, self.report), (0, {**self.report, "result": "REFUSED"}),
                 (0, {**self.report, "result": "FAIL"}),
                 (130, {**self.report, "result": "STOPPED"}),
                 (0, {**self.report, "restore_errors": ["restore failed"]}),
                 (0, {**self.report, "stop_verified": False}),
                 (0, {**self.report, "session_failure": "stale telemetry"}),
                 (0, {"result": "PASS", "stop_verified": True})]
        for index, (code, report) in enumerate(cases):
            with self.subTest(report=report, code=code):
                self.report, self.hold_code = report, code
                self.commands.clear()
                self.events.clear()
                result = self.execute(output_directory=self.folder / str(index))
                self.assertEqual(result["result"], "FAIL", result)
                self.assertEqual(len(self.commands), 1)
                self.assertFalse(any("completed_pose" in detail for _, detail in self.events))
                if report.get("restore_errors"):
                    self.assertEqual(result["restore_errors"], report["restore_errors"])
                    self.assertFalse(result["stop_verified"])

    def test_session_failure_preserves_verified_stop(self):
        self.report = {**self.report, "result": "FAIL",
                       "reason": "dynamic UDP state became stale",
                       "session_failure": "dynamic UDP state became stale"}
        self.hold_code = 1
        result = self.execute()
        self.assertEqual(result["result"], "FAIL")
        self.assertTrue(result["stop_verified"])
        self.assertEqual(result["restore_errors"], [])
        self.assertEqual(len(self.commands), 1)

    def test_abort_before_move_after_move_and_after_hold(self):
        self.stopped = True
        self.assertEqual(self.execute()["result"], "FAIL")
        self.assertFalse(self.commands)
        self.assertFalse(self.moves)
        self.stopped = False
        self.after_move = True
        result = self.execute()
        self.assertEqual(result["result"], "FAIL")
        self.assertTrue(result["records"][0]["move"]["ok"])
        self.assertFalse(result["stop_verified"])
        self.assertEqual(len(self.moves), 1)
        self.assertFalse(self.commands)
        self.commands.clear()
        self.stopped = self.after_move = False
        self.after_hold = True
        self.report["restore_errors"] = ["restore failed"]
        result = self.execute()
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["stop_verified"])
        self.assertEqual(result["restore_errors"], ["restore failed"])
        self.assertEqual(len(self.commands), 1)

    def test_abort_in_progress_callback_prevents_move_or_hold(self):
        for stage in ("moving", "holding"):
            with self.subTest(stage=stage):
                self.stopped = False
                self.moves.clear()

                def progress(phase, detail):
                    self.stopped = detail["stage"] == stage

                self.assertEqual(self.execute(on_progress=progress)["result"], "FAIL")
                self.assertEqual(len(self.moves), 0 if stage == "moving" else 1)
                self.assertFalse(self.commands)

    def test_stale_source_bounds_and_stale_reports_never_launch(self):
        plan = self.build()
        for seconds in (0, 0.49, 10.01, float("inf"), float("nan")):
            self.assertEqual(self.execute(plan, seconds=seconds)["result"], "FAIL")
        self.source.write_text(self.source.read_text() + " ")
        self.assertEqual(self.execute(plan)["result"], "FAIL")
        output = self.folder / "output"
        output.mkdir(exist_ok=True)
        (output / "hold_01.json").write_text(json.dumps(self.report))
        self.assertEqual(self.execute()["result"], "FAIL")
        self.assertFalse(self.commands)
        self.assertFalse(self.moves)

    def test_missing_report_and_child_exception_fail_closed(self):
        self.write_report = False
        result = self.execute()
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["stop_verified"])
        self.assertEqual(len(self.commands), 1)
        result = self.execute(run_child=Mock(side_effect=TimeoutError("child timed out")))
        self.assertEqual(result["result"], "FAIL")
        self.assertIn("timed out", result["reason"])

    def test_source_change_after_move_prevents_hold(self):
        def move(pose):
            result = self.move(pose)
            self.source.write_text(self.source.read_text() + " ")
            return result
        result = self.execute(move=move)
        self.assertEqual(result["result"], "FAIL")
        self.assertEqual(len(self.moves), 1)
        self.assertFalse(self.commands)

    def test_malformed_or_non_true_move_verdict_never_holds(self):
        for verdict in (None, [], "no verdict", {}, {"ok": False}, {"ok": 1},
                        {"ok": "true"}, {"ok": np.bool_(True)},
                        {"ok": False, "reason": "target not reached"}):
            with self.subTest(verdict=verdict):
                move = Mock(return_value=verdict)
                result = self.execute(move=move)
                self.assertEqual(result["result"], "FAIL", result)
                self.assertFalse(result["stop_verified"])
                self.assertEqual(move.call_count, 1)
                self.assertEqual(result["records"][0]["outcome"], "FAIL")
                self.assertFalse(self.commands)

    def test_move_exception_never_holds(self):
        result = self.execute(move=Mock(side_effect=RuntimeError("move interrupted")))
        self.assertEqual(result["result"], "FAIL", result)
        self.assertIn("move interrupted", result["reason"])
        self.assertFalse(result["stop_verified"])
        self.assertFalse(self.commands)

    def test_malformed_hold_report_and_clean_stop_are_not_passes(self):
        def malformed_child(command, timeout_s):
            result = self.child(command, timeout_s)
            if command[3] == "hold_check":
                Path(command[command.index("--output") + 1]).write_text("{broken")
            return result
        result = self.execute(run_child=malformed_child)
        self.assertEqual(result["result"], "FAIL")
        self.assertFalse(result["stop_verified"])
        self.report["result"] = "STOPPED"
        self.hold_code = 130
        self.after_hold = True
        result = self.execute(output_directory=self.folder / "stopped")
        self.assertEqual(result["result"], "FAIL")
        self.assertTrue(result["stop_verified"])

    def test_importing_public_module_does_not_load_legacy_helpers(self):
        import importlib.util
        import sys

        with patch.dict(sys.modules, {"ament_index_python.packages": None}), \
                patch("subprocess.run", side_effect=AssertionError("no commands")), \
                patch("socket.socket", side_effect=AssertionError("no sockets")):
            spec = importlib.util.spec_from_file_location(
                "hold_plan_import_probe", hold_plan.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.build_hold_plan))


if __name__ == "__main__":
    unittest.main()
