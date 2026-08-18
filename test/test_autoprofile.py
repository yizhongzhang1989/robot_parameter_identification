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

    def test_the_operator_can_raise_the_speed(self):
        profile = autoprofile.derive_profile(
            synthetic_urdf(), NAMES, speed_limit_deg_s=60.0)
        self.assertAlmostEqual(profile.sustained_speed_deg_s, 60.0)

    def test_a_raised_speed_is_still_capped_by_the_urdf_rating(self):
        rated = math.degrees(3.0)
        profile = autoprofile.derive_profile(
            synthetic_urdf(), NAMES, speed_limit_deg_s=10_000.0)
        self.assertLessEqual(profile.sustained_speed_deg_s,
                             rated * autoprofile.SPEED_CEILING_FRACTION + 1e-6)

    def test_asking_for_nothing_leaves_the_conservative_default(self):
        quiet = autoprofile.derive_profile(synthetic_urdf(), NAMES)
        asked = autoprofile.derive_profile(
            synthetic_urdf(), NAMES, speed_limit_deg_s=None)
        self.assertEqual(quiet.sustained_speed_deg_s,
                         asked.sustained_speed_deg_s)

    def test_a_non_positive_speed_is_refused(self):
        with self.assertRaises(ValueError):
            autoprofile.derive_profile(
                synthetic_urdf(), NAMES, speed_limit_deg_s=0.0)

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


class WorkspaceCapTest(unittest.TestCase):
    """The URDF describes the arm, not the stand it is bolted to."""

    def test_no_cap_leaves_the_workspace_unset(self):
        profile = autoprofile.derive_profile(synthetic_urdf(), NAMES)
        self.assertEqual(profile.workspace_limit_deg, ())

    def test_a_scalar_cap_applies_to_every_joint(self):
        profile = autoprofile.derive_profile(
            synthetic_urdf(), NAMES, workspace_limit_deg=45.0)
        self.assertEqual(list(profile.workspace_limit_deg), [45.0] * 7)

    def test_a_per_joint_cap_is_honoured(self):
        caps = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        profile = autoprofile.derive_profile(
            synthetic_urdf(), NAMES, workspace_limit_deg=caps)
        self.assertEqual(list(profile.workspace_limit_deg), caps)

    def test_a_cap_can_only_tighten_never_loosen(self):
        profile = autoprofile.derive_profile(
            synthetic_urdf(), NAMES, workspace_limit_deg=10_000.0)
        self.assertEqual(list(profile.workspace_limit_deg),
                         list(profile.position_limit_deg))

    def test_a_wrong_length_cap_is_refused(self):
        with self.assertRaises(ValueError):
            autoprofile.derive_profile(synthetic_urdf(), NAMES,
                                       workspace_limit_deg=[10.0, 20.0])

    def test_a_non_positive_cap_is_refused(self):
        with self.assertRaises(ValueError):
            autoprofile.derive_profile(synthetic_urdf(), NAMES,
                                       workspace_limit_deg=0.0)

    def test_the_cap_says_why_it_is_there(self):
        profile = autoprofile.derive_profile(
            synthetic_urdf(), NAMES, workspace_limit_deg=45.0)
        self.assertIn("workspace", profile.notes)

    def test_the_campaign_plan_inherits_the_cap(self):
        from robot_parameter_identification import campaign, identification as ident

        profile = autoprofile.derive_profile(
            synthetic_urdf(), NAMES, workspace_limit_deg=30.0)
        arm = ident.ArmModel.from_profile(synthetic_urdf(), profile)
        limits = campaign.default_plan(profile).design_limits(arm)
        self.assertTrue(all(value <= 30.0 for value in limits.upper_deg))
        self.assertTrue(all(value >= -30.0 for value in limits.lower_deg))


if __name__ == "__main__":
    unittest.main()
