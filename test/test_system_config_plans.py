"""Offline contracts for canonical and selected system plan defaults."""

from copy import deepcopy
from dataclasses import asdict, fields, replace
import math
import unittest

from robot_parameter_identification import campaign
from robot_parameter_identification.loadsweep import SweepPlan
from robot_parameter_identification.profile import RobotProfile
from robot_parameter_identification.system_config import plan_defaults, system_defaults


class CanonicalPlanDefaultsTest(unittest.TestCase):
    def test_campaign_fields_and_defaults_match_yaml_exactly(self):
        values = system_defaults()["campaign"]
        self.assertEqual(
            {entry.name for entry in fields(campaign.CampaignPlan)}, set(values))
        self.assertEqual(asdict(campaign.CampaignPlan()), plan_defaults(values))
        self.assertEqual(campaign.CampaignPlan().as_dict(), values)
        self.assertIs(campaign.CampaignPlan().derive_speed_ladders, True)

    def test_ladder_option_keeps_existing_positional_constructor_arguments(self):
        plan = campaign.CampaignPlan(22, 125, 4, 19.0, (1.0, 2.0), 5, 4)
        self.assertEqual(plan.friction_repeats, 5)
        self.assertEqual(plan.friction_postures, 4)
        self.assertIs(plan.derive_speed_ladders, True)

    def test_sweep_fields_and_defaults_match_yaml_exactly(self):
        values = system_defaults()["load_sweep"]
        self.assertEqual({entry.name for entry in fields(SweepPlan)}, set(values))
        self.assertEqual(asdict(SweepPlan()), plan_defaults(values))
        self.assertIsInstance(SweepPlan().joints, tuple)

    def test_coulomb_search_matches_canonical_campaign(self):
        expected = tuple(system_defaults()["campaign"]["coulomb_transition_search"])
        self.assertEqual(campaign.COULOMB_TRANSITION_SEARCH, expected)
        self.assertEqual(campaign.CampaignPlan().coulomb_transition_search, expected)

    def test_selected_sweep_defaults_and_speed_ladder(self):
        values = system_defaults()["load_sweep"]
        values.update(
            maximum_levels=4, speeds=7, slowest_deg_s=0.25, fastest_deg_s=12.0,
            repeats=2, arc_ceiling_deg=45.0, transit_speed_deg_s=8.0,
            drift_share=0.22, minimum_level_gap_nm=0.3, search_samples=432,
            refine_steps=17, settle_steps=19, decorrelate=True, seed=73,
            joints=[0, 1], pass_attempts=4, pass_retry_s=0.75,
            joint_failure_budget=5)
        original = deepcopy(values)
        plan = SweepPlan(**plan_defaults(values))
        self.assertEqual(asdict(plan), plan_defaults(values))
        speeds = plan.speed_ladder()
        self.assertEqual(len(speeds), 7)
        self.assertEqual((speeds[0], speeds[-1]), (0.25, 12.0))
        self.assertEqual(speeds, sorted(set(speeds)))
        self.assertEqual(values, original)
        values["joints"].append(2)
        self.assertEqual(plan.joints, (0, 1))


class SelectedCampaignDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.profile = RobotProfile(
            name="offline-config-test", joint_names=("joint1", "joint2"),
            position_limit_deg=(120.0, 120.0), workspace_limit_deg=(90.0, 45.0),
            continuous_current_a=(1.0, 1.0), peak_current_a=(2.0, 2.0),
            sustained_speed_deg_s=6.0, temperature_c=38.0, position_margin_deg=7.0)
        self.defaults = system_defaults()["campaign"]

    def test_non_ui_defaults_seed_and_start_survive_both_builders(self):
        selected = dict(
            sample_rate_hz=37.0, friction_repeats=4, friction_postures=2,
            fourier_attempts=53, optimal_fourier_amplitude_fraction=0.17,
            optimal_friction_speeds_deg_s=[0.1, 0.3, 1.0],
            gravity_probe_speeds_deg_s=[0.25, 0.75],
            coulomb_transition_search=[0.2, 0.6, 1.4],
            stribeck_speed_search=[0.4, 1.8], skip_budget=11,
            workspace_range_deg=[[-25.0, 40.0], [-10.0, 20.0]],
            seed=73, start_deg=[3.0, -2.0])
        self.defaults.update(selected)
        original = deepcopy(self.defaults)
        default = campaign.default_plan(self.profile, defaults=self.defaults)
        clamped, notes = campaign.clamp_campaign_plan(
            None, self.profile, defaults=self.defaults)
        self.assertEqual(notes, [])
        for plan in (default, clamped):
            for name, expected in plan_defaults(selected).items():
                with self.subTest(builder=type(plan).__name__, field=name):
                    self.assertEqual(getattr(plan, name), expected)
            self.assertIsInstance(plan.workspace_range_deg[0], tuple)
        self.assertEqual(self.defaults, original)
        self.defaults["workspace_range_deg"][0][0] = -100.0
        self.defaults["start_deg"][0] = 50.0
        self.assertEqual(default.workspace_range_deg[0], (-25.0, 40.0))
        self.assertEqual(default.start_deg, (3.0, -2.0))

    def test_selected_defaults_still_respect_profile_envelope(self):
        self.defaults.update(
            maximum_speed_deg_s=80.0, maximum_acceleration_deg_s2=900.0,
            temperature_ceiling_c=80.0, position_margin_deg=1.0)
        plan = campaign.default_plan(self.profile, defaults=self.defaults)
        self.assertEqual(plan.maximum_speed_deg_s, 6.0)
        self.assertEqual(plan.maximum_acceleration_deg_s2, 24.0)
        self.assertEqual(plan.temperature_ceiling_c, 38.0)
        self.assertEqual(plan.position_margin_deg, 7.0)
        self.assertEqual(plan.workspace_limit_deg, (90.0, 45.0))

    def test_explicit_workspace_intersects_profile_instead_of_being_replaced(self):
        self.defaults["workspace_limit_deg"] = [30.0, 80.0]
        for plan in (
                campaign.default_plan(self.profile, defaults=self.defaults),
                campaign.clamp_campaign_plan(
                    {}, self.profile, defaults=self.defaults)[0]):
            self.assertEqual(plan.workspace_limit_deg, (30.0, 45.0))
        self.assertEqual(self.defaults["workspace_limit_deg"], [30.0, 80.0])

    def test_explicit_workspace_survives_an_unset_profile_workspace(self):
        self.defaults["workspace_limit_deg"] = [30.0, 80.0]
        profile = replace(self.profile, workspace_limit_deg=())
        plan = campaign.default_plan(profile, defaults=self.defaults)
        self.assertEqual(plan.workspace_limit_deg, (30.0, 80.0))

    def test_mismatched_workspace_cannot_silently_drop_a_joint(self):
        self.defaults["workspace_limit_deg"] = [30.0]
        with self.assertRaises(ValueError):
            campaign.default_plan(self.profile, defaults=self.defaults)

    def test_automatic_ladders_and_amplitude_match_legacy_formula(self):
        selected = system_defaults()
        selected["ranges"]["campaign"]["maximum_speed_deg_s"]["min"] = 0.05
        selected["ranges"]["campaign"]["maximum_acceleration_deg_s2"]["min"] = 0.1
        for ceiling, amplitude in ((0.3, 20.0), (10.0, 20.0), (60.0, 70.0)):
            with self.subTest(ceiling=ceiling):
                profile = replace(
                    self.profile, sustained_speed_deg_s=ceiling,
                    peak_speed_deg_s=max(ceiling, 30.0))
                self.defaults.update(
                    derive_speed_ladders=True, maximum_speed_deg_s=ceiling,
                    friction_amplitude_deg=20.0, friction_speeds_deg_s=[0.75],
                    validation_speeds_deg_s=[1.25])
                plan = campaign.default_plan(profile, defaults=self.defaults, system_config=selected)
                expected = ((ceiling,) if ceiling < 0.5 else tuple(
                    round(0.5 * (ceiling / 0.5) ** (index / 19.0), 3)
                    for index in range(20)))
                self.assertEqual(plan.friction_speeds_deg_s, expected)
                self.assertEqual(plan.validation_speeds_deg_s, (
                    round(0.35 * ceiling, 2), round(0.65 * ceiling, 2)))
                self.assertEqual(plan.friction_amplitude_deg, amplitude)
                self.assertIs(plan.as_dict()["derive_speed_ladders"], True)

    def test_automatic_request_still_derives_ladders_and_acceleration(self):
        profile = replace(
            self.profile, sustained_speed_deg_s=60.0, peak_speed_deg_s=60.0)
        plan, notes = campaign.clamp_campaign_plan(
            {"maximum_speed_deg_s": 60.0}, profile, defaults=self.defaults)
        self.assertEqual(notes, [])
        self.assertEqual(plan.friction_speeds_deg_s, tuple(
            round(0.5 * 120.0 ** (index / 19.0), 3) for index in range(20)))
        self.assertEqual(plan.validation_speeds_deg_s, (21.0, 39.0))
        self.assertEqual(plan.friction_amplitude_deg, 70.0)
        self.assertEqual(plan.maximum_acceleration_deg_s2, 240.0)

    def test_manual_ladders_keep_configured_order_within_profile_speed(self):
        self.defaults.update(
            derive_speed_ladders=False, maximum_speed_deg_s=80.0,
            friction_speeds_deg_s=[4.0, 1.25, 8.0],
            validation_speeds_deg_s=[3.0, 2.5, 9.0])
        original = deepcopy(self.defaults)
        for plan in (
                campaign.default_plan(self.profile, defaults=self.defaults),
                campaign.clamp_campaign_plan(
                    {}, self.profile, defaults=self.defaults)[0]):
            self.assertEqual(plan.maximum_speed_deg_s, 6.0)
            self.assertEqual(plan.friction_speeds_deg_s, (4.0, 1.25))
            self.assertEqual(plan.validation_speeds_deg_s, (3.0, 2.5))
            self.assertIs(plan.as_dict()["derive_speed_ladders"], False)
        self.assertEqual(self.defaults, original)

    def test_manual_empty_or_invalid_ladders_have_bounded_positive_fallback(self):
        selected = system_defaults()
        selected["ranges"]["campaign"]["maximum_speed_deg_s"]["min"] = 0.05
        for ceiling in (0.3, 6.0):
            for speeds in ([], [0.0, -1.0, math.nan, math.inf, 100.0]):
                with self.subTest(ceiling=ceiling, speeds=speeds):
                    self.defaults.update(
                        derive_speed_ladders=False, maximum_speed_deg_s=ceiling,
                        friction_speeds_deg_s=speeds, validation_speeds_deg_s=speeds)
                    plan = campaign.default_plan(
                        self.profile, defaults=self.defaults, system_config=selected)
                    for ladder in (plan.friction_speeds_deg_s,
                                   plan.validation_speeds_deg_s):
                        self.assertTrue(ladder)
                        self.assertTrue(all(
                            math.isfinite(speed) and 0.0 < speed <= ceiling
                            for speed in ladder))
                        self.assertEqual(ladder, (min(0.5, ceiling),))

    def test_manual_request_rebounds_original_configured_ladders(self):
        self.defaults.update(
            derive_speed_ladders=False, maximum_speed_deg_s=2.0,
            friction_speeds_deg_s=[1.0, 4.0, 8.0],
            validation_speeds_deg_s=[1.5, 5.0, 12.0])
        cases = (
            (1.0, 1.0, (1.0,), (0.5,)),
            (5.0, 5.0, (1.0, 4.0), (1.5, 5.0)),
            (99.0, 6.0, (1.0, 4.0), (1.5, 5.0)))
        for requested, ceiling, friction, validation in cases:
            with self.subTest(requested=requested):
                plan, _notes = campaign.clamp_campaign_plan(
                    {"maximum_speed_deg_s": requested}, self.profile,
                    defaults=self.defaults)
                self.assertEqual(plan.maximum_speed_deg_s, ceiling)
                self.assertEqual(plan.friction_speeds_deg_s, friction)
                self.assertEqual(plan.validation_speeds_deg_s, validation)

    def test_explicit_request_ladders_remain_filtered_in_either_mode(self):
        for automatic in (True, False):
            with self.subTest(automatic=automatic):
                self.defaults["derive_speed_ladders"] = automatic
                plan, _notes = campaign.clamp_campaign_plan(
                    {"maximum_speed_deg_s": 3.0,
                     "friction_speeds_deg_s": [-1.0, 9.0, 2.5, 1.0],
                     "validation_speeds_deg_s": [0.0, 8.0, 2.0]},
                    self.profile, defaults=self.defaults)
                self.assertEqual(plan.friction_speeds_deg_s, (1.0, 2.5))
                self.assertEqual(plan.validation_speeds_deg_s, (2.0,))

    def test_manual_ladders_retain_amplitude_sizing(self):
        profile = replace(
            self.profile, sustained_speed_deg_s=60.0, peak_speed_deg_s=60.0)
        self.defaults.update(
            derive_speed_ladders=False, maximum_speed_deg_s=60.0,
            friction_amplitude_deg=20.0, friction_speeds_deg_s=[1.0],
            validation_speeds_deg_s=[2.0])
        plan = campaign.default_plan(profile, defaults=self.defaults)
        self.assertEqual(plan.friction_amplitude_deg, 70.0)

    def test_manual_ladders_reject_a_nonpositive_or_nonfinite_ceiling(self):
        for ceiling in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(ceiling=ceiling):
                plan = campaign.CampaignPlan(
                    derive_speed_ladders=False, maximum_speed_deg_s=ceiling)
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    campaign._follow_speed(plan)