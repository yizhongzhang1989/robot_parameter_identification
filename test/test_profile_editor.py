"""Editing the arm's envelope from the panel, with no file anywhere.

The first time an arm is run there is no profile YAML, and demanding one before
anything works would be a poor first experience. So the module derives one and
the panel edits that; saving is where the answer goes, not a precondition.
"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml

try:
    from robot_parameter_identification.dashboard.http_server import build_routes
    from robot_parameter_identification.dashboard.service import (
        DashboardConfig, IdentificationService)
    from robot_parameter_identification.profile import ProfileError, RobotProfile
    from fixtures import synthetic_urdf, test_profile, PREFIX
except ImportError as error:
    raise unittest.SkipTest(f"needs pinocchio: {error}") from error

JOINTS = [f"{PREFIX}joint{index}" for index in range(1, 8)]
REMOVED_CURRENT_FIELDS = (
    "continuous_current_a", "peak_current_a", "sustained_current_window_s",
    "current_slew_a_s", "probe_current_fraction", "probe_current_a",
)


def derived_service(directory: str = "") -> IdentificationService:
    """A dashboard that was pointed at a robot and given no profile at all."""
    made = IdentificationService(
        DashboardConfig(output_directory=directory or "identification_results"))
    made.adopt_description(synthetic_urdf())
    made.adopt_driven_joints(JOINTS)
    return made


class NoFileYetTest(unittest.TestCase):
    def test_an_arm_with_no_profile_file_still_has_one_to_edit(self):
        made = derived_service()
        payload = made.profile_payload()
        self.assertTrue(payload["have_profile"])
        self.assertEqual(payload["source"], "derived")
        self.assertEqual(len(payload["profile"]["joints"]["names"]), 7)

    def test_profile_api_omits_removed_current_settings_and_policies(self):
        payload = derived_service().profile_payload()
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)
        for field in REMOVED_CURRENT_FIELDS:
            with self.subTest(field=field):
                self.assertNotIn(field, payload["profile"]["limits"])
                self.assertNotIn(field, payload["profile"]["envelope"])
        self.assertNotIn("current_guard", payload)
        self.assertNotIn("calibration_current_policy", payload)
        self.assertIn("hardware_current_limits", payload)

    def test_nothing_to_edit_before_the_robot_says_who_it_is(self):
        payload = IdentificationService(DashboardConfig()).profile_payload()
        self.assertFalse(payload["have_profile"])
        self.assertIsNone(payload["profile"])
        self.assertNotIn("current_guard", payload)
        self.assertNotIn("calibration_current_policy", payload)


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.service = derived_service()
        self.payload = self.service.profile_payload()["profile"]

    def test_an_edit_takes_effect_without_any_file(self):
        self.payload["envelope"]["temperature_c"] = 38.0
        result = self.service.apply_profile(self.payload)
        self.assertTrue(result["ok"])
        self.assertEqual(self.service.profile.temperature_c, 38.0)

    def test_legacy_current_settings_are_ignored_when_applying_a_profile(self):
        self.payload["limits"]["continuous_current_a"] = [3.0] * 7
        self.payload["limits"]["peak_current_a"] = [4.0] * 7
        self.payload["envelope"].update({
            "sustained_current_window_s": 0.5, "current_slew_a_s": 0.1,
            "probe_current_fraction": 0.5, "probe_current_a": [0.1] * 7,
        })
        result = self.service.apply_profile(self.payload)
        self.assertTrue(result["ok"])
        applied = result["profile"]
        for field in REMOVED_CURRENT_FIELDS:
            with self.subTest(field=field):
                self.assertFalse(hasattr(self.service.profile, field))
                self.assertNotIn(field, applied["profile"]["limits"])
                self.assertNotIn(field, applied["profile"]["envelope"])
        self.assertEqual(applied["profile"]["limits"]["position_deg"],
                         self.payload["limits"]["position_deg"])
        self.assertNotIn("current_guard", applied)
        self.assertNotIn("calibration_current_policy", applied)

    def test_an_edited_envelope_counts_as_the_operator_s_own(self):
        applied = self.service.apply_profile(self.payload)["profile"]
        self.assertEqual(applied["source"], "configured")
        self.assertTrue(applied["edited"])

    def test_blank_and_inconsistent_legacy_current_ceilings_are_ignored(self):
        for peak in ([None] * 7, [1.0] * 7):
            with self.subTest(peak=peak):
                self.payload["limits"]["continuous_current_a"] = [5.0] * 7
                self.payload["limits"]["peak_current_a"] = peak
                result = self.service.apply_profile(self.payload)
                self.assertTrue(result["ok"])
                self.assertEqual(result["profile"]["profile"]["limits"],
                                 {"position_deg": self.payload["limits"]["position_deg"]})

    def test_an_impossible_envelope_is_refused_with_a_sentence(self):
        self.payload["envelope"]["sustained_speed_deg_s"] = 20.0
        self.payload["envelope"]["peak_speed_deg_s"] = 10.0
        with self.assertRaises(ProfileError) as caught:
            self.service.apply_profile(self.payload)
        self.assertIn("peak speed", str(caught.exception))

    def test_the_envelope_cannot_change_mid_run(self):
        self.service._state = "running"
        with self.assertRaises(RuntimeError):
            self.service.apply_profile(self.payload)

    def test_reset_goes_back_to_what_the_launch_supplied(self):
        launched = IdentificationService(DashboardConfig(),
                                         profile=test_profile())
        launched.adopt_description(synthetic_urdf())
        before = launched.profile.temperature_c
        payload = launched.profile_payload()["profile"]
        payload["envelope"]["temperature_c"] = before - 5.0
        launched.apply_profile(payload)
        self.assertEqual(launched.profile.temperature_c, before - 5.0)
        launched.reset_profile()
        self.assertEqual(launched.profile.temperature_c, before)


class SaveTest(unittest.TestCase):
    def test_a_saved_profile_reads_back_as_the_same_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            made = derived_service(directory)
            payload = made.profile_payload()["profile"]
            payload["limits"]["continuous_current_a"] = [3.0] * 7
            payload["limits"]["peak_current_a"] = [4.0] * 7
            payload["envelope"].update(dict.fromkeys(REMOVED_CURRENT_FIELDS[2:], 0.5))
            payload["envelope"]["temperature_c"] = 39.0
            made.apply_profile(payload)
            written = made.save_profile("my_arm.yaml")
            self.assertTrue(written["ok"])
            again = RobotProfile.from_yaml(written["path"])
            self.assertEqual(again.joint_names, made.profile.joint_names)
            self.assertEqual(again.position_limit_deg, made.profile.position_limit_deg)
            self.assertEqual(again.temperature_c, 39.0)
            saved = yaml.safe_load(Path(written["path"]).read_text())
            self.assertEqual(saved["envelope"], made.profile.as_dict()["envelope"])
            for field in REMOVED_CURRENT_FIELDS:
                with self.subTest(field=field):
                    self.assertNotIn(field, saved["limits"])
                    self.assertNotIn(field, saved["envelope"])
                    self.assertFalse(hasattr(again, field))

    def test_a_derived_profile_round_trip_does_not_add_current_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            made = derived_service(directory)
            path = made.save_profile("derived.yaml")["path"]
            again = RobotProfile.from_yaml(path)
            expected = made.profile.as_dict()
            actual = again.as_dict()
            expected.pop("source")
            actual.pop("source")
            self.assertEqual(actual, expected)
            self.assertEqual(set(actual["limits"]), {"position_deg"})

    def test_a_save_lands_beside_the_results_and_nowhere_else(self):
        # The web surface listens on every interface, so a path from a request
        # would be an arbitrary file write.
        with tempfile.TemporaryDirectory() as directory:
            made = derived_service(directory)
            for attempt in ("../escape.yaml", "/etc/passwd", "sub/dir.yaml",
                            "no_suffix", "profile.yml"):
                with self.assertRaises(ValueError):
                    made.save_profile(attempt)
            self.assertEqual(Path(made._save_target("ok.yaml")).parent,
                             Path(directory))

    def test_with_no_name_it_overwrites_the_file_the_launch_named(self):
        made = IdentificationService(
            DashboardConfig(profile_path="/tmp/given.yaml"))
        self.assertEqual(str(made._save_target()), "/tmp/given.yaml")


class RouteTest(unittest.TestCase):
    def test_the_panel_can_read_edit_save_and_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            made = derived_service(directory)
            routes = build_routes(made)
            self.assertEqual(routes["/api/profile"][0], "GET")
            payload = routes["/api/profile"][1]({})["profile"]
            payload["envelope"]["temperature_c"] = 41.0
            self.assertTrue(
                routes["/api/profile/apply"][1]({"profile": payload})["ok"])
            self.assertEqual(made.profile.temperature_c, 41.0)
            self.assertTrue(
                routes["/api/profile/save"][1]({"name": "a.yaml"})["ok"])
            self.assertTrue(routes["/api/profile/reset"][1]({})["ok"])


if __name__ == "__main__":
    unittest.main()
