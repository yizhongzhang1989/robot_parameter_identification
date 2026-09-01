"""Editing the arm's envelope from the panel, with no file anywhere.

The first time an arm is run there is no profile YAML, and demanding one before
anything works would be a poor first experience. So the module derives one and
the panel edits that; saving is where the answer goes, not a precondition.
"""

import math
import tempfile
import unittest
from pathlib import Path

try:
    from robot_parameter_identification import autoprofile
    from robot_parameter_identification.dashboard.http_server import build_routes
    from robot_parameter_identification.dashboard.service import (
        DashboardConfig, IdentificationService)
    from robot_parameter_identification.profile import ProfileError, RobotProfile
    from fixtures import synthetic_urdf, test_profile, PREFIX
except ImportError as error:
    raise unittest.SkipTest(f"needs pinocchio: {error}") from error

JOINTS = [f"{PREFIX}joint{index}" for index in range(1, 8)]


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

    def test_a_ceiling_nobody_supplied_travels_as_null_not_infinity(self):
        # JSON.parse rejects Infinity, so the wire form has to be null.
        payload = derived_service().profile_payload()
        self.assertEqual(
            payload["profile"]["limits"]["continuous_current_a"], [None] * 7)
        self.assertFalse(payload["current_guard"])

    def test_nothing_to_edit_before_the_robot_says_who_it_is(self):
        payload = IdentificationService(DashboardConfig()).profile_payload()
        self.assertFalse(payload["have_profile"])
        self.assertIsNone(payload["profile"])


class ApplyTest(unittest.TestCase):
    def setUp(self):
        self.service = derived_service()
        self.payload = self.service.profile_payload()["profile"]

    def test_an_edit_takes_effect_without_any_file(self):
        self.payload["envelope"]["temperature_c"] = 38.0
        result = self.service.apply_profile(self.payload)
        self.assertTrue(result["ok"])
        self.assertEqual(self.service.profile.temperature_c, 38.0)

    def test_typing_current_ceilings_arms_the_current_guard(self):
        self.payload["limits"]["continuous_current_a"] = [3.0] * 7
        self.payload["limits"]["peak_current_a"] = [4.0] * 7
        applied = self.service.apply_profile(self.payload)["profile"]
        self.assertTrue(applied["current_guard"])
        self.assertTrue(autoprofile.current_guard_active(self.service.profile))

    def test_an_edited_envelope_counts_as_the_operator_s_own(self):
        # The guards a derived profile leaves off are on once a human applies
        # the numbers, because that is the same claim a written file makes.
        applied = self.service.apply_profile(self.payload)["profile"]
        self.assertEqual(applied["source"], "configured")
        self.assertTrue(applied["edited"])

    def test_a_blank_ceiling_stays_unset_rather_than_becoming_zero(self):
        self.payload["limits"]["peak_current_a"] = [None] * 7
        self.service.apply_profile(self.payload)
        self.assertTrue(
            all(math.isinf(v) for v in self.service.profile.peak_current_a))

    def test_an_impossible_envelope_is_refused_with_a_sentence(self):
        self.payload["limits"]["continuous_current_a"] = [5.0] * 7
        self.payload["limits"]["peak_current_a"] = [1.0] * 7
        with self.assertRaises(ProfileError) as caught:
            self.service.apply_profile(self.payload)
        self.assertIn("peak current", str(caught.exception))

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
            made.apply_profile(payload)
            written = made.save_profile("my_arm.yaml")
            self.assertTrue(written["ok"])
            again = RobotProfile.from_yaml(written["path"])
            self.assertEqual(again.joint_names, made.profile.joint_names)
            self.assertEqual(again.continuous_current_a,
                             made.profile.continuous_current_a)

    def test_an_unset_ceiling_survives_the_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            made = derived_service(directory)
            path = made.save_profile("derived.yaml")["path"]
            again = RobotProfile.from_yaml(path)
            self.assertTrue(all(math.isinf(v) for v in again.peak_current_a))
            self.assertFalse(autoprofile.current_guard_active(again))

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
