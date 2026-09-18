import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np

from fixtures import synthetic_urdf
from robot_parameter_identification import gravity_current_model, hold_candidates
from robot_parameter_identification.arm_identity import ArmIdentity
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
            arm_model=Mock(return_value=SimpleNamespace(
                parameter_count=70, joint_names=hold_plan.RIGHT_JOINTS.copy())))
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

    def test_selected_instance_controls_model_and_every_hold_child(self):
        for name in ("right", "left", "station_3"):
            with self.subTest(name=name):
                identity = ArmIdentity(name)
                names = list(identity.joint_names)
                self.payload["joint_names"] = names
                self.source.write_text(json.dumps(self.payload))
                self.model["joint_names"] = names
                self.backend.arm_model.return_value.joint_names = names
                self.commands.clear()
                plan = self.build(joint_names=names)
                self.backend.identified.load_identification.assert_called_with(
                    self.folder, identity.prefix)
                self.backend.arm_model.assert_called_with(identity.prefix)
                result = self.execute(plan, output_directory=self.folder / name)
                self.assertEqual(result["result"], "PASS", result)
                for command, _timeout in self.commands:
                    self.assertEqual(command[command.index("--arm") + 1], name)

    def test_source_from_another_instance_is_refused_before_loading_model(self):
        with self.assertRaisesRegex(ValueError, "selected arm"):
            self.build(joint_names=list(ArmIdentity("left").joint_names))
        self.backend.identified.load_identification.assert_not_called()
        self.backend.arm_model.assert_not_called()

    def test_nonright_arm_requires_an_explicit_source(self):
        for name in ("left", "station_3"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "explicit"):
                self.build(source="", joint_names=list(ArmIdentity(name).joint_names))
        self.backend.identified.load_identification.assert_not_called()
        self.backend.arm_model.assert_not_called()

    def test_live_urdf_uses_the_selected_model_prefix_without_backend_geometry(self):
        for name in ("right", "left", "station_3"):
            with self.subTest(name=name):
                identity = ArmIdentity(name)
                names = list(identity.joint_names)
                self.payload["joint_names"] = names
                self.source.write_text(json.dumps(self.payload))
                self.model["joint_names"] = names
                urdf = synthetic_urdf(prefix=identity.model_prefix)
                with patch.object(hold_plan.ArmModel, "from_urdf_text",
                                  wraps=hold_plan.ArmModel.from_urdf_text) as constructor:
                    plan = self.build(joint_names=names, urdf_text=urdf)
                constructor.assert_called_once_with(urdf, identity.model_prefix)
                self.assertEqual(plan["joint_names"], names)
                arm = self.backend.identified.gravity_current.call_args.args[1]
                self.assertEqual(list(arm.joint_names), names)
                self.backend.arm_model.assert_not_called()

    def test_live_urdf_with_incomplete_selected_joints_is_refused(self):
        with self.assertRaisesRegex(ValueError, "URDF joint order"):
            self.build(urdf_text=synthetic_urdf(joints=6))
        self.backend.identified.gravity_current.assert_not_called()
        self.backend.arm_model.assert_not_called()

    def test_live_mount_prediction_matches_runtime_gravity_without_source_geometry(self):
        identity = ArmIdentity("left")
        names = list(identity.joint_names)
        columns = list(range(70))
        parameters = [0.001] * 70
        self.payload["joint_names"] = names
        self.payload["joints"] = [
            {"columns": columns, "parameters": parameters} for _index in range(7)]
        self.source.write_text(json.dumps(self.payload))
        self.model.update(joint_names=names, columns=[columns.copy() for _index in range(7)],
                          parameters=[parameters.copy() for _index in range(7)])
        self.backend.identified.gravity_current = self.legacy.identified.gravity_current
        self.candidates = np.array([[7.0] * 7])
        predictions = []
        for rotation in ("0 0 0", "0 1.2 0"):
            root = ET.fromstring(synthetic_urdf(prefix=identity.model_prefix))
            ET.SubElement(root, "link", name="world")
            mount = ET.SubElement(root, "joint", name="mount", type="fixed")
            ET.SubElement(mount, "parent", link="world")
            ET.SubElement(mount, "child", link=f"{identity.model_prefix}base_link")
            ET.SubElement(mount, "origin", xyz="0 0 0", rpy=rotation)
            urdf = ET.tostring(root, encoding="unicode")
            plan = self.build(joint_names=names, count=1, urdf_text=urdf)
            runtime_arm = hold_plan.ArmModel.from_urdf_text(urdf, identity.model_prefix)
            expected = self.legacy.identified.gravity_current(
                self.model, runtime_arm, self.candidates[0])
            np.testing.assert_allclose(plan["predicted_current_a"][0], expected, atol=1e-12)
            predictions.append(plan["predicted_current_a"][0])
        self.assertFalse(np.allclose(*predictions))
        self.backend.arm_model.assert_not_called()

    def test_execution_rechecks_source_identity_before_any_move_or_child(self):
        plan = self.build()
        plan["joint_names"] = list(ArmIdentity("left").joint_names)
        result = self.execute(plan)
        self.assertEqual(result["result"], "FAIL")
        self.assertIn("selected arm", result["reason"])
        self.assertFalse(self.moves)
        self.assertFalse(self.commands)

    def test_identity_rejects_mixed_reordered_or_invalid_joint_names(self):
        right = list(ArmIdentity("right").joint_names)
        for names in ([], right[:-1], right[::-1],
                      [*right[:-1], "left_arm_joint7"],
                      [name.replace("right", "../left") for name in right]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                ArmIdentity.from_joint_names(names)

    def test_new_seeds_produce_different_pose_sets(self):
        self.candidates = np.array([[value] * 7 for value in range(20)])
        with patch.object(hold_plan.secrets, "randbits", side_effect=[11, 12]):
            first = self.build()
            second = self.build()
        self.assertNotEqual(first["seed"], second["seed"])
        self.assertNotEqual(first["poses_deg"], second["poses_deg"])

    def test_ten_batches_select_fifty_unique_measured_poses_without_replanning(self):
        self.candidates = np.array([[value] * 7 for value in range(60)], dtype=float)
        selected = []
        for _batch in range(10):
            plan = self.build(count=5, exclude_poses_deg=selected)
            keys = {tuple(round(value, 3) for value in pose) for pose in selected}
            self.assertFalse(keys.intersection(tuple(pose) for pose in plan["poses_deg"]))
            selected.extend(plan["poses_deg"])
        self.assertEqual(len({tuple(pose) for pose in selected}), 50)
        self.assertEqual(self.backend.campaign.random_then_order.call_count, 10)

    def test_excluded_poses_use_preview_precision_and_are_frozen_in_plan(self):
        excluded = [[-10.12346] * 7]
        plan = self.build(exclude_poses_deg=excluded)
        self.assertEqual(plan["poses_deg"], [[0.0] * 7, [10.0] * 7, [20.0] * 7])
        self.assertEqual(plan["excluded_poses_deg"], [[-10.123] * 7])
        excluded[0][0] = 999
        self.assertEqual(plan["excluded_poses_deg"][0][0], -10.123)

    def test_exclusions_cannot_relax_current_or_collision_constraints(self):
        self.candidates = np.asarray([[0.0] * 7, [100.0] * 7])
        with self.assertRaisesRegex(ValueError, "insufficient"):
            self.build(count=1, exclude_poses_deg=[[0.0] * 7])
        self.candidates = np.asarray([[0.0] * 7, [10.0] * 7])
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.build(count=1, exclude_poses_deg=[[0.0] * 7], collision_free=lambda _pose: False)

    def test_instance_command_limits_control_candidate_admission_and_are_frozen(self):
        self.candidates = np.array([[10.0] * 7])
        for name in ("right", "left", "station_3"):
            identity = ArmIdentity(name)
            names = list(identity.joint_names)
            self.payload["joint_names"] = names
            self.source.write_text(json.dumps(self.payload))
            self.model["joint_names"] = names
            self.backend.arm_model.return_value.joint_names = names
            limits = [0.2] * 7
            with self.subTest(arm=name):
                plan = self.build(count=1, joint_names=names, maximum_command_a=limits)
                limits[0] = 0.05
                self.assertEqual(plan["maximum_command_a"], [0.2] * 7)
                with self.assertRaisesRegex(ValueError, "insufficient"):
                    self.build(count=1, joint_names=names, maximum_command_a=limits)

    def test_invalid_instance_limits_fail_before_model_loading(self):
        for limits in ([], [0] * 7, [float("nan")] * 7, [4] * 7, [[1]] * 7):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.build(maximum_command_a=limits)
        self.backend.identified.load_identification.assert_not_called()

    def test_execution_refuses_a_prediction_above_frozen_limits_before_moving(self):
        plan = self.build()
        plan["maximum_command_a"] = [0.05] * 7
        result = self.execute(plan)
        self.assertEqual(result["result"], "FAIL")
        self.assertIn("frozen instance", result["reason"])
        self.assertFalse(self.moves)
        self.assertFalse(self.commands)

    def test_malformed_exclusions_fail_before_backend_loading(self):
        for excluded in ("bad", {}, [[0] * 6], [[float("nan")] * 7], [[0] * 7] * 1001):
            with self.subTest(excluded=str(excluded)[:50]), self.assertRaises(ValueError):
                self.build(exclude_poses_deg=excluded)
        self.backend.identified.load_identification.assert_not_called()

    def test_configured_wider_ranges_reach_every_hold_child_snapshot(self):
        from robot_parameter_identification.system_config import load_system_config, system_defaults

        settings = system_defaults()
        settings["ranges"]["hold_test"]["poses"]["max"] = 24
        settings["ranges"]["hold_test"]["seconds"]["max"] = 15.0
        self.candidates = np.array([[value] * 7 for value in np.linspace(-10, 10, 25)])
        plan = self.build(count=21, system_config=settings)
        result = self.execute(plan, seconds=12.0, system_config=settings)
        self.assertEqual(result["result"], "PASS", result["reason"])
        self.assertEqual(len(self.commands), 21)
        snapshots = set()
        for command, timeout_s in self.commands:
            snapshots.add(command[command.index("--system-config") + 1])
            self.assertEqual(float(command[command.index("--seconds") + 1]), 12.0)
            self.assertEqual(timeout_s, 12.0 + settings["dashboard"]["hold_test"]["child_timeout_margin_s"])
        self.assertEqual(snapshots, {result["system_config"]})
        saved = load_system_config(result["system_config"]).values
        self.assertEqual(saved, settings)
        settings["ranges"]["hold_test"]["seconds"]["max"] = 30.0
        self.assertEqual(load_system_config(result["system_config"]).values, saved)

    def test_import_adapter_and_arm_model_never_run_commands_or_sockets(self):
        import sys
        import xacro

        description = self.folder / "src" / "robot_description" / "urdf" / "robot.urdf.xacro"
        description.parent.mkdir(parents=True)
        description.write_text(synthetic_urdf(), encoding="utf-8")
        before = sys.path[:]
        bindings = {name: sys.modules.get(name) for name in (
            "identified_zero_force_drag", "identified_static_hold_campaign",
            "forward_current_controller_test", "subprocess")}
        with patch("subprocess.run", side_effect=AssertionError("no commands")), \
                patch("subprocess.Popen", side_effect=AssertionError("no processes")), \
                patch("socket.socket", side_effect=AssertionError("no sockets")), \
                patch.object(hold_plan, "_workspace_root", return_value=self.folder), \
                patch.dict(sys.modules, {"common.workspace_utils": None}), \
                patch.object(xacro, "process_file", wraps=xacro.process_file) as render:
            backend = hold_plan._load_backend()
            with self.assertWarnsRegex(UserWarning, "default mounts"):
                arm = backend.arm_model()
        render.assert_called_once()
        self.assertEqual(render.call_args.args, (str(description),))
        self.assertEqual(render.call_args.kwargs["mappings"]["use_mock_hardware"], "true")
        self.assertGreater(arm.parameter_count, 0)
        self.assertEqual(sys.path, before)
        for name, module in bindings.items():
            self.assertIs(sys.modules.get(name), module)

    def test_backend_exposes_shared_functions_and_isolated_compatibility_constants(self):
        backend = hold_plan._load_backend()
        self.assertIs(backend.identified.load_identification,
                      gravity_current_model.load_identification)
        self.assertIs(backend.identified.gravity_current, gravity_current_model.gravity_current)
        self.assertIs(backend.campaign._executed_poses, hold_candidates.executed_poses)
        for name in ("admissible", "random_then_order", "spread_then_order", "transit_clear"):
            self.assertIs(getattr(backend.campaign, name), getattr(hold_candidates, name))
        self.assertEqual(backend.current.DEFAULT_SOURCE, hold_plan._workspace_root() /
                         "identification_results" /
                         "optimal_excitation_regime_separated-20260825-232541")
        self.assertEqual(backend.current.ACKNOWLEDGEMENT, "I_AM_HOLDING_ARM_AND_ESTOP_READY")
        self.assertEqual(backend.current.DEFAULT_CORRIDOR_DEG, 5.0)
        self.assertEqual(backend.current.MAXIMUM_TEMPERATURE_C, 40.0)
        np.testing.assert_array_equal(backend.identified.CONTINUOUS_CURRENT_A,
                                      hold_plan.DEFAULT_MAXIMUM_COMMAND_A)
        backend.identified.CONTINUOUS_CURRENT_A[0] = 0.0
        np.testing.assert_array_equal(hold_plan._load_backend().identified.CONTINUOUS_CURRENT_A,
                                      hold_plan.DEFAULT_MAXIMUM_COMMAND_A)

    def test_default_backend_builds_from_live_urdf_without_legacy_or_offline_geometry(self):
        import sys

        for name in ("right", "left", "station_3"):
            with self.subTest(name=name):
                identity = ArmIdentity(name)
                names = list(identity.joint_names)
                self.payload["joint_names"] = names
                self.payload["joints"] = [
                    {"columns": list(range(70)), "parameters": [0.001] * 70}
                    for _index in range(7)]
                self.source.write_text(json.dumps(self.payload))
                with (self.folder / "observations.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([f"{joint}.position_deg" for joint in names])
                    writer.writerows([[value] * 7 for value in (0, 7.04, 7.01)])
                urdf = synthetic_urdf(prefix=identity.model_prefix)
                with patch.dict(sys.modules, dict.fromkeys((
                        "identified_static_hold_campaign", "identified_zero_force_drag",
                        "forward_current_controller_test", "ament_index_python.packages",
                        "generate_mjcf", "common.workspace_utils", "xacro", "rclpy"))), \
                        patch.object(hold_plan, "_offline_arm_model",
                                     side_effect=AssertionError("live URDF only")) as offline:
                    plan = self.build(backend=None, joint_names=names, count=2,
                                      urdf_text=urdf, maximum_command_a=[0.2] * 7)
                offline.assert_not_called()
                self.assertEqual(plan["maximum_command_a"], [0.2] * 7)
                self.assertEqual({tuple(pose) for pose in plan["poses_deg"]},
                                 {(0.0,) * 7, (7.0,) * 7})
                model = gravity_current_model.load_identification(self.folder, identity.prefix)
                arm = hold_plan.ArmModel.from_urdf_text(urdf, identity.model_prefix)
                np.testing.assert_allclose(plan["predicted_current_a"], [
                    gravity_current_model.gravity_current(model, arm, pose)
                    for pose in plan["poses_deg"]], atol=1e-12)

    def test_default_backend_requires_complete_pass_ampere_before_shared_loading(self):
        for key, value in (("complete", False), ("verdict", {"state": "warn"}),
                           ("effort_unit", "newton_metre")):
            with self.subTest(key=key):
                self.source.write_text(json.dumps({**self.payload, key: value}))
                with patch.object(gravity_current_model, "load_identification") as load, \
                        patch.object(hold_plan, "_offline_arm_model") as offline:
                    with self.assertRaisesRegex(ValueError, "complete, passing ampere"):
                        self.build(backend=None, count=1, urdf_text=synthetic_urdf())
                load.assert_not_called()
                offline.assert_not_called()

    def test_offline_model_preserves_configured_mounts_and_unconfigured_fallbacks(self):
        import sys
        import xacro

        (self.folder / "robot_mounts.yaml").write_text(json.dumps({
            "right_arm": {"xyz": [1, 2, 3], "rpy": [0, 0.5, 0]},
            "left_arm": {"xyz": [-1, -2, -3]},
        }))
        workspace_utils = SimpleNamespace(get_config_dir=lambda: str(self.folder))
        with patch.dict(sys.modules, {"common.workspace_utils": workspace_utils}), \
                patch.object(hold_plan, "_workspace_root", return_value=self.folder), \
                patch.object(xacro, "process_file", return_value=SimpleNamespace(
                    toxml=synthetic_urdf)) as render:
            arm = hold_plan._load_backend().arm_model()
        self.assertEqual(list(arm.joint_names), hold_plan.RIGHT_JOINTS)
        self.assertEqual(render.call_args.kwargs["mappings"], {
            "use_mock_hardware": "true", "right_arm_xyz": "1.0 2.0 3.0",
            "right_arm_rpy": "0.0 0.5 0.0", "left_arm_xyz": "-1.0 -2.0 -3.0",
            "left_arm_rpy": "0 0 0",
        })

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
        import sys

        code = compile(Path(hold_plan.__file__).read_text(encoding="utf-8"),
                       hold_plan.__file__, "exec")
        before = sys.path[:]
        forbidden = dict.fromkeys((
            "identified_static_hold_campaign", "identified_zero_force_drag",
            "forward_current_controller_test", "identified_static_hold", "generate_mjcf",
            "ament_index_python.packages", "common.workspace_utils", "xacro", "rclpy"))
        namespace = {"__name__": "hold_plan_import_probe", "__file__": hold_plan.__file__}
        with patch.dict(sys.modules, forbidden), \
                patch("builtins.open", side_effect=AssertionError("no runtime file reads")), \
                patch.object(Path, "open", side_effect=AssertionError("no runtime file reads")), \
                patch("subprocess.run", side_effect=AssertionError("no commands")), \
                patch("subprocess.Popen", side_effect=AssertionError("no processes")), \
                patch("socket.socket", side_effect=AssertionError("no sockets")), \
                patch("socket.create_connection", side_effect=AssertionError("no network")):
            exec(code, namespace)
            backend = namespace["_load_backend"]()
            for name in forbidden:
                self.assertIsNone(sys.modules[name])
        self.assertEqual(sys.path, before)
        self.assertTrue(callable(namespace["build_hold_plan"]))
        self.assertIs(backend.campaign._executed_poses, hold_candidates.executed_poses)


if __name__ == "__main__":
    unittest.main()
