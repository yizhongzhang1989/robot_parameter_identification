"""The run folder, the report inside it, and the translations they carry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import re
import tempfile
import unittest

from robot_parameter_identification import report

STATIC = (Path(report.__file__).resolve().parent / "dashboard" / "static")


@dataclass
class FakeObservation:
    phase: str
    time_s: float
    position_deg: list
    velocity_deg_s: list
    acceleration_deg_s2: list
    current_a: list
    temperature_c: list = field(default_factory=list)


def sample_payload() -> dict:
    return {
        "mode": "rehearsal",
        "joint_names": ["right_arm_joint1", "right_arm_joint2"],
        "effort_source": "current",
        "effort_unit": "ampere",
        "action": "/right_arm_controller/follow_joint_trajectory",
        "complete": True,
        "aborted": None,
        "validation_samples": 40,
        "plan": {"maximum_speed_deg_s": 60.0, "seed": 0},
        "phases": [{"phase": "A_gravity", "observations": 120,
                    "duration_s": 30.0, "peak_temperature_c": 31.0,
                    "peak_speed_deg_s": 0.4, "peak_current_a": 1.2,
                    "detail": {}, "aborted": None}],
        "joints": [
            {"joint": 0, "friction": {"coulomb": 0.65, "viscous": 0.0,
                                      "offset": 0.03},
             "condition_number": 42.0, "residual_rms_a": 0.24,
             "holdout_rms_a": 0.239, "validation_rms_a": 0.254,
             "effective_rank": 6, "samples": 900,
             "components": {"coulomb_transition_deg_s": 1.8}},
            {"joint": 1, "friction": {"coulomb": 0.66, "viscous": 0.0078,
                                      "offset": 0.03},
             "condition_number": 51.0, "residual_rms_a": 0.31,
             "holdout_rms_a": 0.316, "validation_rms_a": 0.310,
             "effective_rank": 6, "samples": 900,
             "components": {"coulomb_transition_deg_s": 1.8}},
        ],
        "verdict": {"state": "warn",
                    "joints": [{"joint": 1, "state": "pass", "reason": ""},
                               {"joint": 2, "state": "warn",
                                "reason": "condition number high"}],
                    "failed_joints": []},
        "friction_samples": [[{"speed": 1.0, "effort": 0.5},
                              {"speed": 6.0, "effort": 0.4, "sweep": True}],
                             [{"speed": -1.0, "effort": -0.5}]],
        "residual_samples": [[{"speed": 1.0, "residual": 0.01},
                              {"speed": 6.0, "residual": -0.02, "sweep": True}],
                             [{"speed": -1.0, "residual": 0.0}]],
    }


def observations(count: int = 5) -> list:
    return [FakeObservation(
        phase="A_gravity", time_s=index * 0.05,
        position_deg=[1.0 * index, 2.0 * index],
        velocity_deg_s=[0.1, 0.2], acceleration_deg_s2=[0.0, 0.0],
        current_a=[0.5, 0.6], temperature_c=[30.0, 31.0])
        for index in range(count)]


class RunFolderTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = report.write_run(
            self.temp.name, sample_payload(), observations(5), stamp="x")

    def test_the_folder_is_named_for_the_run(self):
        self.assertEqual(self.folder.name, "rehearsal-x")

    def test_it_holds_the_result_the_data_and_the_report(self):
        for name in (report.RESULT_NAME, report.OBSERVATIONS_NAME,
                     report.REPORT_NAME):
            self.assertTrue((self.folder / name).is_file(), name)

    def test_the_result_json_round_trips(self):
        loaded = json.loads(
            (self.folder / report.RESULT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(loaded["joint_names"],
                         ["right_arm_joint1", "right_arm_joint2"])

    def test_every_observation_reaches_the_csv(self):
        with open(self.folder / report.OBSERVATIONS_NAME, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 5)

    def test_the_csv_columns_are_named_after_the_urdf_joints(self):
        # A column called j1 is a code; the point is to be able to match the
        # file against the robot without a lookup table.
        with open(self.folder / report.OBSERVATIONS_NAME, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertIn("right_arm_joint1.position_deg", header)
        self.assertIn("right_arm_joint2.temperature_c", header)
        self.assertNotIn("j1.position_deg", header)

    def test_the_csv_carries_the_raw_numbers(self):
        with open(self.folder / report.OBSERVATIONS_NAME, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[2]["right_arm_joint1.position_deg"], "2.0")
        self.assertEqual(rows[0]["phase"], "A_gravity")


class ReportPageTest(unittest.TestCase):

    def setUp(self):
        self.html = report.render_report(sample_payload(), observation_rows=5,
                                         stamp="x")

    def test_no_placeholder_survives(self):
        for token in ("__DATA__", "__STRINGS__", "__TITLE__"):
            self.assertNotIn(token, self.html)

    def test_it_needs_nothing_from_a_network(self):
        # It gets copied off the robot and opened from a laptop.
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotIn("<script src", self.html)

    def test_the_embedded_data_parses(self):
        blob = re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            self.html, re.S).group(1)
        loaded = json.loads(blob)
        self.assertEqual(loaded["rows"], 5)
        self.assertEqual(loaded["names"][1], "right_arm_joint2")

    def test_a_closing_script_tag_in_the_data_cannot_end_the_block(self):
        payload = sample_payload()
        payload["action"] = "</script><script>alert(1)</script>"
        html = report.render_report(payload)
        self.assertNotIn("<script>alert(1)</script>", html)
        blob = re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            html, re.S).group(1)
        self.assertEqual(json.loads(blob)["payload"]["action"],
                         "</script><script>alert(1)</script>")

    def test_both_languages_are_in_the_page(self):
        self.assertIn("参数辨识报告", self.html)
        self.assertIn("Identification report", self.html)

    def test_the_joint_names_are_shown_not_indices(self):
        self.assertIn("right_arm_joint1", self.html)


class ServedRunsTest(unittest.TestCase):
    """The panel links to these files, so the route that serves them is a
    place where a path from the network reaches the filesystem."""

    def setUp(self):
        from robot_parameter_identification.dashboard.http_server import (
            DashboardServer)

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        report.write_run(self.temp.name, sample_payload(), observations(2),
                         stamp="x")
        outside = Path(self.temp.name).parent / "outside-the-results.txt"
        outside.write_text("secret", encoding="utf-8")
        self.addCleanup(outside.unlink)
        self.outside = outside

        temp = self.temp.name

        class Stub:
            class config:
                output_directory = temp

            def snapshot(self):
                return {}

            def runs(self):
                return []

            def viewer_state(self):
                return {}

            def collision_report(self, pose=None):
                return {}

            def start(self, mode):
                return {}

            def home(self):
                return {}

            def stop(self):
                return {}

        self.server = DashboardServer(Stub(), port=0, host="127.0.0.1")
        self.server.start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.port}"

    def _get(self, path: str) -> int:
        from urllib.error import HTTPError  # noqa: PLC0415
        from urllib.request import urlopen  # noqa: PLC0415

        try:
            with urlopen(self.base + path, timeout=5) as response:
                return response.status
        except HTTPError as error:
            return error.code

    def test_a_report_is_served(self):
        self.assertEqual(self._get(f"/runs/rehearsal-x/{report.REPORT_NAME}"),
                         200)

    def test_the_raw_data_is_served(self):
        self.assertEqual(
            self._get(f"/runs/rehearsal-x/{report.OBSERVATIONS_NAME}"), 200)

    def test_a_missing_run_is_not_found(self):
        self.assertEqual(self._get("/runs/nope/report.html"), 404)

    def test_a_path_cannot_climb_out_of_the_results_directory(self):
        self.assertEqual(self._get(f"/runs/../{self.outside.name}"), 403)

    def test_an_encoded_path_cannot_climb_out_either(self):
        self.assertEqual(self._get(f"/runs/%2e%2e%2f{self.outside.name}"), 403)


class TranslationTest(unittest.TestCase):

    def test_every_report_string_has_both_languages(self):
        for key, entry in report.TEXT.items():
            self.assertIn("en", entry, key)
            self.assertIn("zh", entry, key)
            self.assertTrue(entry["en"].strip(), key)
            self.assertTrue(entry["zh"].strip(), key)

    def test_the_chinese_is_not_a_copy_of_the_english(self):
        # A key left untranslated is worse than a missing one: it looks done.
        for key, entry in report.TEXT.items():
            if key in ("unit.deg",):
                continue
            self.assertNotEqual(entry["en"], entry["zh"], key)

    def test_every_key_the_report_asks_for_exists(self):
        used = set(re.findall(r"""data-i18n="([\w.]+)\"""", report._TEMPLATE))
        used |= set(re.findall(r"""\bt\('([\w.]+)'\)""", report._TEMPLATE))
        # Keys built by concatenation are covered by their own tests below.
        missing = {key for key in used if key not in report.TEXT}
        self.assertFalse(missing, f"no translation for {sorted(missing)}")

    def test_every_verdict_state_can_be_named_and_explained(self):
        for state in ("pass", "warn", "fail", "unknown"):
            self.assertIn(f"verdict.{state}", report.TEXT)
            self.assertIn(f"verdict.{state}.say", report.TEXT)

    def test_every_phase_has_a_name(self):
        for phase in ("A_gravity", "B_friction", "C_inertia", "D_validation"):
            self.assertIn(f"phase.{phase}", report.TEXT)

    def test_the_panel_dictionary_is_balanced(self):
        # The panel's dictionary is JavaScript, so it is checked by counting:
        # an entry added in one language only shows up as a mismatch.
        text = (STATIC / "i18n.js").read_text(encoding="utf-8")
        body = text.split("const DICT = {", 1)[1].split("\n};", 1)[0]
        self.assertEqual(len(re.findall(r"\ben:", body)),
                         len(re.findall(r"\bzh:", body)))
        self.assertGreater(len(re.findall(r"\ben:", body)), 40)

    def test_the_panel_asks_for_no_key_it_does_not_define(self):
        dictionary = (STATIC / "i18n.js").read_text(encoding="utf-8")
        defined = set(re.findall(r"^  '([\w.]+)':", dictionary, re.M))
        panel = (STATIC / "dashboard.js").read_text(encoding="utf-8")
        markup = (STATIC / "index.html").read_text(encoding="utf-8")
        used = set(re.findall(r"""\bt\('([\w.]+)'""", panel))
        used |= set(re.findall(r"""data-i18n="([\w.]+)\"""", markup))
        # A capture ending in a dot is the literal half of a key built by
        # concatenation, such as t('verdict.' + state); the families those
        # build are covered below.
        used = {key for key in used if not key.endswith(".")}
        self.assertFalse(used - defined,
                         f"no translation for {sorted(used - defined)}")

    def test_the_panel_names_every_state_it_can_reach(self):
        dictionary = (STATIC / "i18n.js").read_text(encoding="utf-8")
        defined = set(re.findall(r"^  '([\w.]+)':", dictionary, re.M))
        wanted = {f"verdict.{state}" for state in
                  ("pass", "warn", "fail", "unknown")}
        wanted |= {f"phase.{phase}" for phase in
                   ("A_gravity", "B_friction", "C_inertia", "D_validation")}
        self.assertFalse(wanted - defined,
                         f"no translation for {sorted(wanted - defined)}")


if __name__ == "__main__":
    unittest.main()
