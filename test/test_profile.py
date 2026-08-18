"""Profile tests: a different arm must be a different file, not a code change."""

import textwrap
import unittest
from pathlib import Path
import tempfile

from robot_parameter_identification import profile


def payload(**overrides):
    base = {
        "schema_version": 1,
        "name": "demo",
        "joints": {"prefix": "arm_", "count": 3},
        "limits": {
            "position_deg": [170.0, 120.0, 170.0],
            "continuous_current_a": [2.0, 3.0, 1.0],
            "peak_current_a": [3.0, 4.0, 1.5],
        },
        "envelope": {"temperature_c": 40.0},
    }
    base.update(overrides)
    return base


class LoadTest(unittest.TestCase):
    def test_bundled_template_loads(self):
        self.assertIn("example_6dof", profile.available())
        arm = profile.load("example_6dof")
        self.assertEqual(arm.joint_count, 6)

    def test_unknown_profile_names_the_alternatives(self):
        with self.assertRaises(profile.ProfileError) as caught:
            profile.load("does_not_exist")
        self.assertIn("example_6dof", str(caught.exception))

    def test_joint_count_generates_names_from_prefix(self):
        arm = profile.RobotProfile.from_dict(payload())
        self.assertEqual(arm.joint_names, ("arm_joint1", "arm_joint2", "arm_joint3"))

    def test_explicit_names_win_over_count(self):
        arm = profile.RobotProfile.from_dict(
            payload(joints={"names": ["a", "b", "c"]}))
        self.assertEqual(arm.joint_names, ("a", "b", "c"))

    def test_yaml_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.yaml"
            path.write_text(textwrap.dedent("""
                schema_version: 1
                name: demo
                joints: {prefix: "j", count: 2}
                limits:
                  position_deg: [90.0, 90.0]
                  continuous_current_a: 1.0
                  peak_current_a: 2.0
            """))
            arm = profile.RobotProfile.from_yaml(path)
        self.assertEqual(arm.joint_count, 2)
        self.assertEqual(arm.continuous_current_a, (1.0, 1.0))

    def test_missing_file_is_reported(self):
        with self.assertRaises(profile.ProfileError):
            profile.RobotProfile.from_yaml("/nonexistent/none.yaml")


class ValidationTest(unittest.TestCase):
    def test_wrong_length_vector_is_rejected(self):
        broken = payload()
        broken["limits"]["peak_current_a"] = [1.0, 2.0]
        with self.assertRaises(profile.ProfileError):
            profile.RobotProfile.from_dict(broken)

    def test_peak_below_continuous_is_rejected(self):
        broken = payload()
        broken["limits"]["peak_current_a"] = [1.0, 1.0, 1.0]
        with self.assertRaises(profile.ProfileError) as caught:
            profile.RobotProfile.from_dict(broken)
        self.assertIn("peak current", str(caught.exception))

    def test_peak_speed_below_sustained_is_rejected(self):
        broken = payload()
        broken["envelope"] = {"sustained_speed_deg_s": 20.0, "peak_speed_deg_s": 5.0}
        with self.assertRaises(profile.ProfileError):
            profile.RobotProfile.from_dict(broken)

    def test_empty_voltage_window_is_rejected(self):
        broken = payload()
        broken["envelope"] = {"minimum_voltage_v": 30.0, "maximum_voltage_v": 20.0}
        with self.assertRaises(profile.ProfileError):
            profile.RobotProfile.from_dict(broken)

    def test_non_finite_value_is_rejected(self):
        broken = payload()
        broken["limits"]["position_deg"] = [1.0, float("inf"), 1.0]
        with self.assertRaises(profile.ProfileError):
            profile.RobotProfile.from_dict(broken)

    def test_unsupported_schema_version_is_rejected(self):
        with self.assertRaises(profile.ProfileError):
            profile.RobotProfile.from_dict(payload(schema_version=99))


class TighteningTest(unittest.TestCase):
    """Measured envelopes may replace the table, but only downward."""

    def setUp(self):
        self.arm = profile.RobotProfile.from_dict(payload())

    def test_lower_temperature_is_accepted(self):
        self.assertEqual(self.arm.tightened(temperature_c=35.0).temperature_c, 35.0)

    def test_higher_temperature_is_silently_refused(self):
        self.assertEqual(self.arm.tightened(temperature_c=90.0).temperature_c, 40.0)

    def test_position_margin_only_grows(self):
        self.assertEqual(
            self.arm.tightened(position_margin_deg=1.0).position_margin_deg,
            self.arm.position_margin_deg)

    def test_currents_cannot_be_overridden_at_all(self):
        with self.assertRaises(profile.ProfileError):
            self.arm.tightened(peak_current_a=[99.0, 99.0, 99.0])


if __name__ == "__main__":
    unittest.main()


class InheritanceTest(unittest.TestCase):
    """A new robot should be a short file, not a full restatement."""

    def _write(self, directory, name, text):
        path = Path(directory) / name
        path.write_text(textwrap.dedent(text))
        return path

    def test_child_overrides_parent_and_inherits_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "base.yaml", """
                schema_version: 1
                envelope:
                  temperature_c: 45.0
                  peak_speed_deg_s: 30.0
                  minimum_voltage_v: 20.0
            """)
            child = self._write(directory, "child.yaml", """
                extends: base.yaml
                name: child
                joints: {prefix: "j", count: 2}
                limits:
                  position_deg: 90.0
                  continuous_current_a: 1.0
                  peak_current_a: 2.0
                envelope:
                  temperature_c: 35.0
            """)
            arm = profile.RobotProfile.from_yaml(child)
        self.assertEqual(arm.temperature_c, 35.0)      # overridden
        self.assertEqual(arm.peak_speed_deg_s, 30.0)   # inherited
        self.assertEqual(arm.minimum_voltage_v, 20.0)  # inherited

    def test_bundled_template_can_be_extended_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            child = self._write(directory, "child.yaml", """
                extends: manipulator
                name: child
                joints: {prefix: "j", count: 2}
                limits:
                  position_deg: 90.0
                  continuous_current_a: 1.0
                  peak_current_a: 2.0
            """)
            arm = profile.RobotProfile.from_yaml(child)
        self.assertEqual(arm.temperature_c, 45.0)
        self.assertEqual(arm.probe_current_fraction, 0.5)

    def test_lists_are_replaced_not_concatenated(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "base.yaml", """
                schema_version: 1
                joints: {names: [a, b, c, d]}
            """)
            child = self._write(directory, "child.yaml", """
                extends: base.yaml
                name: child
                joints: {names: [x, y]}
                limits:
                  position_deg: 90.0
                  continuous_current_a: 1.0
                  peak_current_a: 2.0
            """)
            arm = profile.RobotProfile.from_yaml(child)
        self.assertEqual(arm.joint_names, ("x", "y"))

    def test_inheritance_loop_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "a.yaml", "extends: b.yaml\nname: a\n")
            self._write(directory, "b.yaml", "extends: a.yaml\nname: b\n")
            with self.assertRaises(profile.ProfileError) as caught:
                profile.RobotProfile.from_yaml(Path(directory) / "a.yaml")
        self.assertIn("loops", str(caught.exception))

    def test_missing_parent_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            child = self._write(directory, "child.yaml", "extends: nope\nname: c\n")
            with self.assertRaises(profile.ProfileError):
                profile.RobotProfile.from_yaml(child)

    def test_manipulator_template_is_shipped(self):
        self.assertIn("manipulator", profile.templates())
