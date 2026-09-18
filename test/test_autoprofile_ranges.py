"""Automatic profile bounds come from a per-call policy and the URDF."""

import inspect
import math
from typing import Any
import unittest
from unittest.mock import patch

from robot_parameter_identification import autoprofile
from robot_parameter_identification.system_config import system_defaults


JOINT_NAMES = ["joint_a"]


def simple_urdf(velocity_deg_s=300.0, lower_deg=-180.0, upper_deg=120.0):
    velocity = ("" if velocity_deg_s is None else
                f' velocity="{math.radians(velocity_deg_s)}"')
    return (
        '<robot name="policy_test"><joint name="joint_a" type="revolute">'
        f'<limit lower="{math.radians(lower_deg)}" '
        f'upper="{math.radians(upper_deg)}" effort="1"{velocity}/>'
        '</joint></robot>')


def policy_with(**overrides):
    policy = system_defaults()["profile_derivation"]
    policy.update(overrides)
    return policy


class AutoprofileRangesTest(unittest.TestCase):
    def setUp(self):
        self.defaults = system_defaults()["profile_derivation"]
        self.default_speed = min(
            300.0 * self.defaults["default_speed_fraction"],
            self.defaults["maximum_default_speed_deg_s"])

    def test_default_profile_uses_template_speed_policy(self):
        profile = autoprofile.derive_profile(simple_urdf(), JOINT_NAMES)
        self.assertEqual(profile.sustained_speed_deg_s, self.default_speed)
        self.assertEqual(profile.peak_speed_deg_s,
                         self.default_speed * self.defaults["peak_speed_multiplier"])
        self.assertEqual(profile.position_limit_deg, (108.0,))
        self.assertEqual(profile.joint_names, tuple(JOINT_NAMES))
        self.assertEqual(profile.as_dict()["limits"], {"position_deg": [108.0]})

    def test_default_speed_uses_custom_fraction_and_maximum(self):
        cases = ((0.2, 90.0, 60.0), (0.5, 75.0, 75.0))
        for fraction, maximum, expected in cases:
            with self.subTest(fraction=fraction, maximum=maximum):
                profile = autoprofile.derive_profile(
                    simple_urdf(), JOINT_NAMES,
                    policy=policy_with(default_speed_fraction=fraction,
                                       maximum_default_speed_deg_s=maximum))
                self.assertEqual(profile.sustained_speed_deg_s, expected)

    def test_requested_speed_uses_custom_fraction(self):
        profile = autoprofile.derive_profile(
            simple_urdf(), JOINT_NAMES, speed_limit_deg_s=1000.0,
            policy=policy_with(requested_speed_fraction=0.4))
        self.assertEqual(profile.sustained_speed_deg_s, 120.0)
        self.assertIn("capped at 0.4", profile.notes["speed"])

    def test_position_uses_custom_fraction_of_tighter_urdf_limit(self):
        profile = autoprofile.derive_profile(
            simple_urdf(), JOINT_NAMES,
            policy=policy_with(position_fraction=0.5))
        self.assertEqual(profile.position_limit_deg, (60.0,))

    def test_peak_speed_uses_custom_multiplier(self):
        profile = autoprofile.derive_profile(
            simple_urdf(), JOINT_NAMES,
            policy=policy_with(peak_speed_multiplier=1.5))
        self.assertEqual(profile.peak_speed_deg_s, self.default_speed * 1.5)

    def test_policies_are_independent_and_not_mutated(self):
        first_policy = policy_with(
            default_speed_fraction=0.2, maximum_default_speed_deg_s=90.0)
        second_policy = policy_with(
            default_speed_fraction=0.5, maximum_default_speed_deg_s=75.0)
        for policy, expected in ((first_policy, 60.0), (second_policy, 75.0),
                                 (first_policy, 60.0), (None, self.default_speed)):
            with self.subTest(policy=policy):
                original = None if policy is None else policy.copy()
                profile = autoprofile.derive_profile(
                    simple_urdf(), JOINT_NAMES, policy=policy)
                self.assertEqual(profile.sustained_speed_deg_s, expected)
                self.assertEqual(policy, original)

    def test_compatibility_aliases_follow_yaml_but_do_not_control_execution(self):
        defaults = system_defaults()["profile_derivation"]
        aliases = {
            "SPEED_FRACTION": "default_speed_fraction",
            "MAXIMUM_SPEED_DEG_S": "maximum_default_speed_deg_s",
            "SPEED_CEILING_FRACTION": "requested_speed_fraction",
            "POSITION_FRACTION": "position_fraction",
        }
        for alias, setting in aliases.items():
            self.assertEqual(getattr(autoprofile, alias), defaults[setting])
        policy = policy_with(default_speed_fraction=0.2,
                             maximum_default_speed_deg_s=90.0,
                             requested_speed_fraction=0.4,
                             position_fraction=0.5,
                             peak_speed_multiplier=1.5)
        with patch.multiple(autoprofile, **dict.fromkeys(aliases, 0.001)):
            profile = autoprofile.derive_profile(
                simple_urdf(), JOINT_NAMES, policy=policy)
            requested = autoprofile.derive_profile(
                simple_urdf(), JOINT_NAMES, speed_limit_deg_s=1000.0,
                policy=policy)
            default = autoprofile.derive_profile(simple_urdf(), JOINT_NAMES)
        self.assertEqual(profile.sustained_speed_deg_s, 60.0)
        self.assertEqual(profile.peak_speed_deg_s, 90.0)
        self.assertEqual(profile.position_limit_deg, (60.0,))
        self.assertEqual(requested.sustained_speed_deg_s, 120.0)
        self.assertEqual(default.sustained_speed_deg_s, self.default_speed)

    def test_existing_positional_arguments_and_keyword_only_policy(self):
        profile = autoprofile.derive_profile(
            simple_urdf(), JOINT_NAMES, "named_profile", 45.0, 80.0,
            policy=policy_with(requested_speed_fraction=0.4))
        self.assertEqual(profile.name, "named_profile")
        self.assertEqual(profile.workspace_limit_deg, (45.0,))
        self.assertEqual(profile.sustained_speed_deg_s, 80.0)
        parameter = inspect.signature(autoprofile.derive_profile).parameters["policy"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameter.default)

    def test_speed_helper_accepts_policy_and_preserves_missing_velocity(self):
        policy = policy_with(default_speed_fraction=0.5,
                             maximum_default_speed_deg_s=75.0,
                             requested_speed_fraction=0.4)
        cases = ((300.0, None, 75.0), (300.0, 200.0, 120.0),
                 (None, None, 75.0), (None, 200.0, 200.0))
        for velocity, requested, expected in cases:
            with self.subTest(velocity=velocity, requested=requested):
                radians = None if velocity is None else math.radians(velocity)
                self.assertAlmostEqual(
                    autoprofile._speed_for(radians, requested, policy=policy),
                    expected)
                profile = autoprofile.derive_profile(
                    simple_urdf(velocity), JOINT_NAMES,
                    speed_limit_deg_s=requested, policy=policy)
                self.assertEqual(profile.sustained_speed_deg_s, expected)
                self.assertNotIn("continuous_current_a", profile.as_dict()["limits"])
                self.assertNotIn("peak_current_a", profile.as_dict()["limits"])
        self.assertEqual(autoprofile._speed_for(None, None),
                         self.defaults["maximum_default_speed_deg_s"])
        self.assertEqual(autoprofile._speed_for(None, 200.0), 200.0)

    def test_unit_fractions_do_not_exceed_urdf_bounds(self):
        policy = policy_with(default_speed_fraction=1.0,
                             maximum_default_speed_deg_s=1000.0,
                             requested_speed_fraction=1.0,
                             position_fraction=1.0,
                             peak_speed_multiplier=1.0)
        cases = ((None, 30.004), (None, 30.006),
                 (1000.0, 30.004), (1000.0, 30.006))
        for requested, velocity in cases:
            with self.subTest(requested=requested, velocity=velocity):
                urdf = simple_urdf(velocity_deg_s=velocity, upper_deg=120.006)
                limits = autoprofile.joint_limits(urdf)[JOINT_NAMES[0]]
                profile = autoprofile.derive_profile(
                    urdf, JOINT_NAMES, workspace_limit_deg=1000.0,
                    speed_limit_deg_s=requested, policy=policy)
                self.assertLessEqual(profile.sustained_speed_deg_s,
                                     math.degrees(limits["velocity"]))
                self.assertLessEqual(profile.position_limit_deg[0],
                                     math.degrees(limits["upper"]))
                self.assertLessEqual(profile.workspace_limit_deg[0],
                                     profile.position_limit_deg[0])
                self.assertGreaterEqual(profile.peak_speed_deg_s,
                                        profile.sustained_speed_deg_s)

    def test_partial_policy_uses_template_for_unspecified_values(self):
        profile = autoprofile.derive_profile(
            simple_urdf(), JOINT_NAMES, policy={"position_fraction": 0.5})
        self.assertEqual(profile.position_limit_deg, (60.0,))
        self.assertEqual(profile.sustained_speed_deg_s, self.default_speed)
        self.assertEqual(profile.peak_speed_deg_s,
                         self.default_speed * self.defaults["peak_speed_multiplier"])

    def test_malformed_values_reject_before_profile_construction(self):
        invalid = (math.nan, math.inf, -math.inf, 0.0, -1.0, True, False,
                   None, "0.5", [], {}, 10 ** 400)
        cases = [(name, value)
                 for name in system_defaults()["profile_derivation"]
                 for value in invalid]
        cases.extend((name, 1.001) for name in (
            "default_speed_fraction", "requested_speed_fraction", "position_fraction"))
        cases.append(("peak_speed_multiplier", 0.5))
        for name, value in cases:
            with self.subTest(name=name, value=value):
                policy = {name: value}
                with patch.object(autoprofile.RobotProfile, "from_dict") as construct:
                    with self.assertRaisesRegex(ValueError, name):
                        autoprofile.derive_profile(
                            simple_urdf(), JOINT_NAMES, policy=policy)
                    construct.assert_not_called()
                with self.assertRaisesRegex(ValueError, name):
                    autoprofile._speed_for(None, 100.0, policy=policy)

    def test_invalid_policy_shape_and_unknown_keys_reject(self):
        invalid_policies: list[Any] = [False, [], "policy", 1, {"typo": 1.0}]
        for policy in invalid_policies:
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(ValueError, "profile_derivation"):
                    autoprofile.derive_profile(
                        simple_urdf(), JOINT_NAMES, policy=policy)
                with self.assertRaisesRegex(ValueError, "profile_derivation"):
                    autoprofile._speed_for(None, None, policy=policy)

    def test_nonfinite_or_nonpositive_requested_speed_rejects(self):
        for requested in (math.nan, math.inf, -math.inf, 0.0, -1.0):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(ValueError, "speed_limit_deg_s"):
                    autoprofile.derive_profile(
                        simple_urdf(None), JOINT_NAMES, speed_limit_deg_s=requested)
                with self.assertRaisesRegex(ValueError, "speed_limit_deg_s"):
                    autoprofile._speed_for(None, requested)

    def test_nonfinite_or_nonpositive_urdf_velocity_rejects(self):
        for velocity in (math.nan, math.inf, -math.inf, 0.0, -1.0):
            with self.subTest(velocity=velocity):
                with self.assertRaisesRegex(ValueError, "URDF velocity"):
                    autoprofile.derive_profile(simple_urdf(velocity), JOINT_NAMES)
                with self.assertRaisesRegex(ValueError, "URDF velocity"):
                    autoprofile._speed_for(velocity, 100.0)
        with self.assertRaisesRegex(ValueError, "URDF velocity"):
            autoprofile._speed_for(1e308, None)

    def test_nonfinite_urdf_position_limits_reject(self):
        for value in (math.nan, math.inf, -math.inf):
            for bound in ("lower_deg", "upper_deg"):
                with self.subTest(bound=bound, value=value):
                    with self.assertRaisesRegex(ValueError, "position limits"):
                        autoprofile.derive_profile(
                            simple_urdf(**{bound: value}), JOINT_NAMES)

    def test_nonfinite_or_zero_derived_values_reject(self):
        cases = ({"peak_speed_multiplier": 1e308},
                 {"maximum_default_speed_deg_s": 1e-6},
                 {"position_fraction": 1e-10})
        for policy in cases:
            with self.subTest(policy=policy):
                with patch.object(autoprofile.RobotProfile, "from_dict") as construct:
                    with self.assertRaises(ValueError):
                        autoprofile.derive_profile(
                            simple_urdf(), JOINT_NAMES, policy=policy)
                    construct.assert_not_called()

    def test_written_profile_keeps_motion_limits_but_ignores_legacy_current_limits(self):
        profile = autoprofile.RobotProfile.from_dict({
            "schema_version": 1,
            "name": "written_profile",
            "joints": {"names": JOINT_NAMES},
            "limits": {"position_deg": [150.0],
                       "continuous_current_a": [2.0], "peak_current_a": [4.0]},
            "envelope": {"sustained_speed_deg_s": 200.0,
                         "peak_speed_deg_s": 350.0},
        })
        autoprofile.derive_profile(
            simple_urdf(), JOINT_NAMES,
            policy=policy_with(default_speed_fraction=0.01, position_fraction=0.1))
        self.assertEqual(profile.position_limit_deg, (150.0,))
        self.assertEqual(profile.sustained_speed_deg_s, 200.0)
        self.assertEqual(profile.peak_speed_deg_s, 350.0)
        self.assertEqual(profile.as_dict()["limits"], {"position_deg": [150.0]})
        self.assertFalse(hasattr(profile, "continuous_current_a"))
        self.assertFalse(hasattr(profile, "peak_current_a"))


if __name__ == "__main__":
    unittest.main()
