"""Pure current-model contracts independent of any hardware capability."""

import builtins
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from robot_parameter_identification import gravity_current_model as current


INSTANCES = ("right", "left", "station_3")


def payload_for(names):
    return {
        "joint_names": list(names), "effort_unit": "ampere",
        "joints": [{
            "columns": [2, -1, 3, 0, 1],
            "parameters": [0.25, 999.0, 888.0, -0.5, 0.125],
            "friction": {"coulomb": 0.2, "offset": 777.0, "viscous": None},
            "components": {"friction": True, "offset": True,
                           "coulomb_transition_deg_s": 0.0,
                           "stribeck_speed_deg_s": 2.0,
                           "stribeck_speed_search": [0.5, 1.0, 2.0]},
            "validation_rms_a": None,
        } for name in names],
    }


def arm_for(names):
    regressor = np.arange(len(names) * 3, dtype=float).reshape(len(names), 3) - 1.25
    return SimpleNamespace(joint_names=list(names), parameter_count=3,
                           torque_regressor=Mock(return_value=regressor))


class CurrentModelCoreTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.names = ["station_3_axis_pitch", "station_3_slide"]
        self.payload = payload_for(self.names)

    def write_result(self, payload):
        (self.folder / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        (self.folder / "gravity_model.json").write_text("invalid export", encoding="utf-8")

    def load(self):
        return current.load_identification(self.folder, "station_3_")

    def test_prefix_aliases_are_generic_for_all_instances_and_joint_counts(self):
        for instance in INSTANCES:
            for count in (1, 2, 6, 8):
                names = [f"{instance}_arm_joint{index}" for index in range(1, count + 1)]
                self.write_result(payload_for(names))
                for prefix in (f"{instance}_", f"{instance}_arm_"):
                    with self.subTest(instance=instance, count=count, prefix=prefix):
                        model = current.load_identification(self.folder, prefix)
                        self.assertEqual(model["joint_names"], names)
                        prediction = current.gravity_current(
                            model, arm_for(names), np.zeros(count))
                        self.assertEqual(prediction.shape, (count,))

    def test_explicit_ordered_names_need_no_prefix_or_numbered_joint_convention(self):
        names = ["slide", "pitch_axis", "tool_rotation"]
        self.write_result(payload_for(names))
        model = current.load_identification(self.folder, joint_names=tuple(names))
        self.assertEqual(model["joint_names"], names)
        with self.assertRaises(current.InvalidCurrentModel):
            current.load_identification(self.folder, joint_names=names[::-1])
        with self.assertRaises(current.InvalidCurrentModel):
            current.load_identification(self.folder, "station_3_", joint_names=names)

    def test_selection_requires_a_prefix_or_an_explicit_order(self):
        self.write_result(self.payload)
        for selection in ({}, {"prefix": ""}, {"prefix": 3}, {"joint_names": []},
                          {"joint_names": "station_3_slide"}, {"joint_names": set(self.names)},
                          {"joint_names": [self.names[0], self.names[0]]}):
            with self.subTest(selection=selection), self.assertRaises(current.InvalidCurrentModel):
                current.load_identification(self.folder, **selection)

    def test_complete_pass_is_opt_in_without_a_hardware_joint_count(self):
        for complete in (None, False, True):
            for state in (None, "warn", "fail", "pass"):
                with self.subTest(complete=complete, state=state):
                    payload = copy.deepcopy(self.payload)
                    if complete is not None:
                        payload["complete"] = complete
                    if state is not None:
                        payload["verdict"] = {"state": state}
                    self.write_result(payload)
                    self.load()
                    if complete is True and state == "pass":
                        current.load_identification(self.folder, "station_3_",
                                                    require_complete_pass=True)
                    else:
                        with self.assertRaises(current.InvalidCurrentModel):
                            current.load_identification(self.folder, "station_3_",
                                                        require_complete_pass=True)

    def test_return_contract_preserves_nullable_metadata_and_empty_fits(self):
        for metadata in ({}, {"friction": None, "components": None, "validation_rms_a": None}):
            with self.subTest(metadata=metadata):
                payload = payload_for(self.names)
                payload["joints"] = [{"columns": [], "parameters": [], **metadata}
                                     for name in self.names]
                self.write_result(payload)
                model = self.load()
                self.assertEqual(set(model), {"folder", "joint_names", "friction", "components",
                                              "parameters", "columns", "validation_rms_a"})
                self.assertEqual(model["folder"], str(self.folder))
                self.assertEqual(model["friction"], [{}, {}])
                self.assertEqual(model["components"], [{}, {}])
                self.assertEqual(model["validation_rms_a"], [None, None])
                np.testing.assert_array_equal(
                    current.gravity_current(model, arm_for(self.names), [0.0, 0.0]), [0.0, 0.0])

    def test_malformed_top_level_fields_are_rejected(self):
        for field, value in (
                ("joint_names", None), ("joint_names", []), ("joint_names", "station_3_slide"),
                ("joint_names", [None, self.names[1]]), ("joint_names", ["", self.names[1]]),
                ("joint_names", [self.names[0], self.names[0]]),
                ("joint_names", [self.names[0], "left_arm_joint2"]),
                ("effort_unit", "newton_metre"), ("effort_unit", None),
                ("joints", None), ("joints", {}), ("joints", []),
                ("joints", self.payload["joints"][:1]),
                ("joints", self.payload["joints"] + self.payload["joints"][:1]),
                ("joints", [None, {}]), ("joints", [[], {}]), ("verdict", "pass")):
            payload = copy.deepcopy(self.payload)
            payload[field] = value
            self.write_result(payload)
            with self.subTest(field=field, value=value):
                with self.assertRaises(current.InvalidCurrentModel):
                    self.load()
        for payload in (None, [], True, "result"):
            self.write_result(payload)
            with self.subTest(payload=payload), self.assertRaises(current.InvalidCurrentModel):
                self.load()

    def test_malformed_joint_fields_are_rejected_without_silent_zip_truncation(self):
        for field, value in (
                ("columns", None), ("columns", "01234"), ("columns", [0]),
                ("columns", [0, 1, 2, 3, "4"]), ("columns", [0, 1, 2, 3, True]),
                ("columns", [0, 1, 2, 3, 0.5]), ("columns", [0, 1, 2, 3, float("nan")]),
                ("parameters", None), ("parameters", "12345"), ("parameters", [1.0]),
                ("parameters", [1.0, 2.0, 3.0, 4.0, "5"]),
                ("parameters", [1.0, 2.0, 3.0, 4.0, True]),
                ("parameters", [1.0, 2.0, 3.0, 4.0, []]),
                ("friction", []), ("friction", [["coulomb", 0.2]]),
                ("friction", {"coulomb": "0.2"}), ("friction", {"coulomb": True}),
                ("components", []), ("components", {"offset": "true"}),
                ("components", {"stribeck_speed_deg_s": [2.0]}),
                ("validation_rms_a", -0.1), ("validation_rms_a", "0.1"),
                ("validation_rms_a", True)):
            payload = copy.deepcopy(self.payload)
            payload["joints"][0][field] = value
            self.write_result(payload)
            with self.subTest(field=field, value=value):
                with self.assertRaises(current.InvalidCurrentModel):
                    self.load()
        for field in ("columns", "parameters"):
            payload = copy.deepcopy(self.payload)
            del payload["joints"][0][field]
            self.write_result(payload)
            with self.subTest(missing=field), self.assertRaises(current.InvalidCurrentModel):
                self.load()

    def test_nonfinite_coefficients_and_metadata_are_rejected(self):
        for invalid in (float("nan"), float("inf"), -float("inf"), 10 ** 400):
            for field, value in (
                    ("parameters", [invalid, 999.0, 888.0, -0.5, 0.125]),
                    ("friction", {"coulomb": invalid}),
                    ("components", {"coulomb_transition_deg_s": invalid}),
                    ("components", {"stribeck_speed_search": [0.5, invalid]}),
                    ("validation_rms_a", invalid)):
                payload = copy.deepcopy(self.payload)
                payload["joints"][0][field] = value
                self.write_result(payload)
                with self.subTest(field=field, invalid=invalid), \
                        self.assertRaises(current.InvalidCurrentModel):
                    self.load()

    def test_each_load_is_fresh_and_nested_results_do_not_alias(self):
        self.write_result(self.payload)
        first = self.load()
        snapshot = copy.deepcopy(first)
        second = self.load()
        second["parameters"][0][0] = 20.0
        second["friction"][0]["coulomb"] = 30.0
        second["components"][0]["stribeck_speed_search"][0] = 40.0
        self.assertEqual(first, snapshot)
        self.assertEqual(self.load(), snapshot)
        changed = copy.deepcopy(self.payload)
        changed["joints"][0]["parameters"][0] = 50.0
        self.write_result(changed)
        self.assertEqual(self.load()["parameters"][0][0], 50.0)
        self.assertEqual(first, snapshot)

    def test_evaluation_preserves_readonly_inputs_and_ignores_nonrigid_columns(self):
        self.write_result(self.payload)
        model = self.load()
        original = copy.deepcopy(model)
        arm = arm_for(self.names)
        regressor = arm.torque_regressor.return_value
        original_regressor = regressor.copy()
        regressor.setflags(write=False)
        pose = np.array([1.0, 2.0])
        pose.setflags(write=False)
        expected = []
        for row in regressor:
            total = 0.0
            total += float(row[2]) * 0.25
            total += float(row[0]) * -0.5
            total += float(row[1]) * 0.125
            expected.append(total)
        np.testing.assert_array_equal(current.gravity_current(model, arm, pose), expected)
        self.assertEqual(model, original)
        np.testing.assert_array_equal(pose, [1.0, 2.0])
        np.testing.assert_array_equal(regressor, original_regressor)

    def test_pose_is_not_aliased_into_the_arm(self):
        self.write_result(self.payload)
        model = self.load()
        arm = arm_for(self.names)

        def regressor(pose):
            pose[:] = 999.0
            return np.ones((2, 3))

        arm.torque_regressor = regressor
        pose = np.array([1.0, 2.0])
        current.gravity_current(model, arm, pose)
        np.testing.assert_array_equal(pose, [1.0, 2.0])

    def test_mismatched_kinematic_identity_is_rejected_before_evaluation(self):
        self.write_result(self.payload)
        model = self.load()
        for names in (self.names[::-1], ["left_axis_pitch", "left_slide"]):
            arm = arm_for(names)
            with self.subTest(names=names), self.assertRaises(current.InvalidCurrentModel):
                current.gravity_current(model, arm, [0.0, 0.0])
            arm.torque_regressor.assert_not_called()

    def test_analytical_arm_without_joint_names_keeps_the_legacy_protocol(self):
        self.write_result(self.payload)
        model = self.load()
        arm = arm_for(self.names)
        expected = current.gravity_current(model, arm, [0.0, 0.0])
        del arm.joint_names
        np.testing.assert_array_equal(current.gravity_current(model, arm, [0.0, 0.0]), expected)

    def test_evaluation_checks_outer_and_inner_fit_lengths(self):
        self.write_result(self.payload)
        for field, value in (("columns", [[0]]), ("parameters", [[0.0]]),
                             ("columns", [[], []]), ("parameters", [[], []]),
                             ("parameters", [[float("nan")] * 5] * 2)):
            model = self.load()
            model[field] = value
            arm = arm_for(self.names)
            with self.subTest(field=field, value=value):
                with self.assertRaises(current.InvalidCurrentModel):
                    current.gravity_current(model, arm, [0.0, 0.0])
            arm.torque_regressor.assert_not_called()

    def test_malformed_and_nonfinite_poses_are_rejected_before_regression(self):
        self.write_result(self.payload)
        model = self.load()
        for pose in (None, [0.0], [[0.0, 0.0]], [float("nan"), 0.0],
                     [0.0, float("inf")], ["bad", 0.0]):
            arm = arm_for(self.names)
            with self.subTest(pose=pose), self.assertRaises(current.InvalidCurrentModel):
                current.gravity_current(model, arm, pose)
            arm.torque_regressor.assert_not_called()

    def test_malformed_and_nonfinite_regressors_are_rejected(self):
        self.write_result(self.payload)
        model = self.load()
        for regressor in (np.zeros((1, 3)), np.zeros((2, 2)), np.zeros(6),
                          np.full((2, 3), float("nan")), np.full((2, 3), float("inf")),
                          [[0.0, 1.0], [0.0]], "bad"):
            arm = arm_for(self.names)
            arm.torque_regressor.return_value = regressor
            with self.subTest(regressor=regressor), self.assertRaises(current.InvalidCurrentModel):
                current.gravity_current(model, arm, [0.0, 0.0])

    def test_invalid_parameter_counts_and_overflow_fail_closed(self):
        self.write_result(self.payload)
        model = self.load()
        for count in (-1, 1.5, True):
            arm = arm_for(self.names)
            arm.parameter_count = count
            with self.subTest(count=count), self.assertRaises(current.InvalidCurrentModel):
                current.gravity_current(model, arm, [0.0, 0.0])
        arm = arm_for(self.names)
        arm.torque_regressor.return_value = np.full((2, 3), 1e308)
        model["parameters"][0][0] = 1e308
        with self.assertRaisesRegex(current.InvalidCurrentModel, "current must be finite"):
            current.gravity_current(model, arm, [0.0, 0.0])

    def test_module_import_load_and_evaluation_have_no_ros_network_or_process_actions(self):
        self.write_result(self.payload)
        original_import = builtins.__import__

        def offline_import(name, *args, **kwargs):
            if name.split(".")[0] in {"rclpy", "rospy", "ament_index_python"}:
                raise AssertionError(f"unexpected ROS import: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=offline_import), \
                patch("socket.socket", side_effect=AssertionError("socket opened")), \
                patch("socket.create_connection", side_effect=AssertionError("connected")), \
                patch("socket.getaddrinfo", side_effect=AssertionError("name lookup")), \
                patch("subprocess.Popen", side_effect=AssertionError("process started")), \
                patch("subprocess.run", side_effect=AssertionError("process run")), \
                patch("os.system", side_effect=AssertionError("shell started")):
            spec = importlib.util.spec_from_file_location(
                "pure_current_model_test", current.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.load_identification(self.folder, "station_3_")
            module.gravity_current(model, arm_for(self.names), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()