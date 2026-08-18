"""Deriving a profile from what the robot publishes, and refusing to guess."""

import math
import unittest

try:
    from robot_parameter_identification import autoprofile
    from fixtures import synthetic_urdf, PREFIX
except ImportError as error:
    raise unittest.SkipTest(f"needs the package: {error}") from error


NAMES = [f"{PREFIX}joint{index}" for index in range(1, 8)]


class JointLimitTest(unittest.TestCase):
    def test_limits_are_read_from_the_urdf(self):
        limits = autoprofile.joint_limits(synthetic_urdf())
        self.assertIn(NAMES[0], limits)
        self.assertAlmostEqual(limits[NAMES[0]]["lower"], -2.6)
        self.assertAlmostEqual(limits[NAMES[0]]["upper"], 2.6)

    def test_rubbish_xml_yields_nothing_rather_than_raising(self):
        self.assertEqual(autoprofile.joint_limits("<robot"), {})


class DeriveTest(unittest.TestCase):
    def test_a_profile_is_built_for_the_named_joints(self):
        profile = autoprofile.derive_profile(synthetic_urdf(), NAMES)
        self.assertEqual(list(profile.joint_names), NAMES)
        self.assertEqual(profile.joint_count, 7)

    def test_only_the_driven_subset_is_taken(self):
        profile = autoprofile.derive_profile(synthetic_urdf(), NAMES[:3])
        self.assertEqual(profile.joint_count, 3)

    def test_position_limits_come_from_the_urdf_with_margin(self):
        profile = autoprofile.derive_profile(synthetic_urdf(), NAMES)
        full = math.degrees(2.6)
        for value in profile.position_limit_deg:
            self.assertLess(value, full)
            self.assertGreater(value, full * 0.5)

    def test_speed_is_a_fraction_of_what_the_urdf_permits(self):
        profile = autoprofile.derive_profile(synthetic_urdf(), NAMES)
        self.assertLessEqual(profile.sustained_speed_deg_s,
                             autoprofile.MAXIMUM_SPEED_DEG_S)
        self.assertGreater(profile.sustained_speed_deg_s, 0.0)

    def test_current_ceilings_are_left_unset_rather_than_invented(self):
        """The URDF states newton-metres; converting needs what we are fitting."""
        profile = autoprofile.derive_profile(synthetic_urdf(), NAMES)
        self.assertTrue(all(math.isinf(v) for v in profile.continuous_current_a))
        self.assertFalse(autoprofile.current_guard_active(profile))

    def test_a_written_profile_keeps_its_current_guard(self):
        from fixtures import test_profile

        self.assertTrue(autoprofile.current_guard_active(test_profile()))

    def test_the_profile_says_it_was_derived(self):
        profile = autoprofile.derive_profile(synthetic_urdf(), NAMES)
        self.assertIn("derived", profile.notes)
        self.assertIn("current guard is off", profile.notes["derived"])

    def test_a_joint_without_limits_is_refused_by_name(self):
        urdf = synthetic_urdf().replace(
            '<limit lower="-2.6" upper="2.6" effort="60" velocity="3"/>', '', 1)
        with self.assertRaises(ValueError) as caught:
            autoprofile.derive_profile(urdf, NAMES)
        self.assertIn(NAMES[0], str(caught.exception))

    def test_an_empty_joint_list_is_refused(self):
        with self.assertRaises(ValueError):
            autoprofile.derive_profile(synthetic_urdf(), [])


if __name__ == "__main__":
    unittest.main()
