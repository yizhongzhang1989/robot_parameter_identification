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
        "comparison": {
            "available": True, "target_met": True,
            "source": "load_sweep/sweep-x",
            "optimal_mean_validation_rms_a": 0.20,
            "sweep_mean_validation_rms_a": 0.30,
            "optimal_worst_validation_rms_a": 0.25,
            "sweep_worst_validation_rms_a": 0.35,
            "mean_improvement_percent": 33.3,
            "worst_improvement_percent": 28.6,
            "joints": [
                {"name": "right_arm_joint1",
                 "optimal_validation_rms_a": 0.18,
                 "sweep_validation_rms_a": 0.30,
                 "improvement_percent": 40.0, "optimal_better": True},
                {"name": "right_arm_joint2",
                 "optimal_validation_rms_a": 0.22,
                 "sweep_validation_rms_a": 0.30,
                 "improvement_percent": 26.7, "optimal_better": True},
            ],
        },
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

    def test_a_gravity_run_writes_a_dedicated_model_artifact(self):
        payload = sample_payload()
        payload["mode"] = "gravity"
        payload["gravity_model"] = {
            "available": True, "model_type": "pair-averaged"}
        folder = report.write_run(
            self.temp.name, payload, observations(2), stamp="gravity-model",
            model_urdf='<robot name="captured"/>')
        saved = json.loads(
            (folder / report.GRAVITY_MODEL_NAME).read_text(encoding="utf-8"))
        self.assertEqual(saved["model_type"], "pair-averaged")
        self.assertEqual(
            (folder / report.MODEL_URDF_NAME).read_text(encoding="utf-8"),
            '<robot name="captured"/>')

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

    def test_raw_csv_preserves_safety_and_alternate_effort_when_published(self):
        path = Path(self.temp.name) / "complete-raw.csv"
        frames = [{
            "phase": "D_validation", "motion": "gravity_check:p1:1:+",
            "frame": 0, "stamp_s": 10.0,
            "position_deg": [1.0, 2.0], "speed_deg_s": [0.1, 0.2],
            "current_a": [0.5, 0.6], "temperature_c": [30.0, 31.0],
            "voltage_v": [24.1, 24.2], "enabled": [True, False],
            "fault_code": [0, 7], "drive_current_a": [0.5, 0.6],
            "joint_torque_nm": [1.5, 1.6],
        }]
        report.write_raw_frames(
            path, frames, ["right_arm_joint1", "right_arm_joint2"])
        with path.open(encoding="utf-8") as handle:
            [row] = list(csv.DictReader(handle))
        self.assertEqual(row["phase"], "D_validation")
        self.assertEqual(row["right_arm_joint1.voltage_v"], "24.1")
        self.assertEqual(row["right_arm_joint1.enabled"], "1")
        self.assertEqual(row["right_arm_joint2.enabled"], "0")
        self.assertEqual(row["right_arm_joint2.fault_code"], "7")
        self.assertEqual(row["right_arm_joint1.drive_current_a"], "0.5")
        self.assertEqual(row["right_arm_joint2.joint_torque_nm"], "1.6")

    def test_raw_csv_omits_optional_channels_that_were_not_published(self):
        path = Path(self.temp.name) / "minimal-raw.csv"
        report.write_raw_frames(path, [{
            "position_deg": [0.0], "speed_deg_s": [0.0],
            "current_a": [0.0], "temperature_c": [30.0],
        }], ["joint1"])
        with path.open(encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        self.assertNotIn("joint1.voltage_v", header)
        self.assertNotIn("joint1.fault_code", header)

    def test_combined_report_names_external_raw_sources(self):
        self.assertIn("raw_frame_sources.json", report._TEMPLATE)
        self.assertIn("P.data_sources", report._TEMPLATE)
        self.assertIn("files.sources", report.TEXT)


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

    def test_the_fair_comparison_is_in_the_saved_report(self):
        self.assertIn("Optimal excitation vs load sweep", self.html)
        self.assertIn("最优激励与负载扫掠对比", self.html)
        self.assertIn("comparisonSection()", self.html)

    def test_gravity_report_leads_with_provenance_and_deployment_limits(self):
        payload = sample_payload()
        payload.update({
            "mode": "gravity",
            "provenance": {
                "source": "real_hardware", "hardware_evidence": True,
                "raw_frame_count": 198081, "fitted_observation_count": 128,
            },
            "gravity_model": {
                "available": True,
                "model_type": "pair_averaged_empirical_gravity_effort_regressor",
                "joints": [], "pairing_audit": {},
                "physical_link_parameters": {"available": False},
                "runtime": {"integrated_controller_loader": False},
            },
        })
        page = report.render_report(payload)
        self.assertIn("function provenanceSection()", page)
        self.assertIn("function gravityModelSection()", page)
        self.assertLess(page.index("provenanceSection(), verdictSection()"),
                        page.index("summarySection(), jointSection()"))
        self.assertIn("verdict.gravity.pass.say", page)
        self.assertIn("gravity.physical.missing", page)
        self.assertIn("gravity.runtime.missing", page)
        self.assertIn("files.result.gravity", page)
        self.assertIn("files.gravity_model", page)
        self.assertIn("files.model_urdf", page)
        self.assertIn("joints.head.gravity", page)
        self.assertIn("formula.head.gravity", page)
        self.assertIn("margin:0 0 18px;overflow-x:auto", page)
        self.assertIn('href="${esc(name)}"', page)
        self.assertIn("provenance.phasecheck", page)
        self.assertIn("software.pinocchio_version", page)


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


class CanvasScalingTest(unittest.TestCase):
    """The report draws its own charts, so it repeats the panel's canvas
    setup. It got it wrong once: the backing store was widened for the pixel
    ratio but not heightened, and on a 2x display the transform pushed the
    lower half of every chart outside the buffer."""

    def setUp(self):
        self.frame = report._TEMPLATE.split("function frame(canvas)", 1)[1] \
                                     .split("\n}", 1)[0]

    def test_both_dimensions_are_scaled_for_the_pixel_ratio(self):
        self.assertIn("width * ratio", self.frame)
        self.assertIn("height * ratio", self.frame)

    def test_the_logical_height_is_stashed_rather_than_read_back(self):
        # Assigning canvas.height writes the same attribute the logical height
        # would be read from, so reading it back compounds on every redraw.
        self.assertIn("logicalHeight", self.frame)
        self.assertNotIn("= canvas.height", self.frame)

    def test_the_ratio_is_capped(self):
        # A 3x phone would otherwise allocate nine times the pixels for a
        # sharpness nobody can see.
        self.assertIn("Math.min(window.devicePixelRatio || 1, 2)", self.frame)

    def test_the_css_height_stays_the_logical_one(self):
        self.assertIn("canvas.style.height = height + 'px'", self.frame)


class ChartIdentityTest(unittest.TestCase):
    """Every joint's friction curve is the same shape scaled by its own
    coefficients. Fitted to its own axis it fills the frame identically, so
    flipping through the joints showed what looked like one unchanging chart."""

    def test_the_chart_names_the_joint_it_is_drawing(self):
        self.assertIn("caption(ctx, box, `${label}", report._TEMPLATE)

    def test_the_caption_carries_the_fitted_coefficients(self):
        # Two joints differing only in magnitude are otherwise indistinguishable.
        block = report._TEMPLATE.split("function drawFriction", 1)[1]
        self.assertIn("f.coulomb", block.split("function drawResidual")[0])
        self.assertIn("f.viscous", block.split("function drawResidual")[0])

    def test_a_shared_scale_is_offered(self):
        self.assertIn('id="lock"', report._TEMPLATE)
        self.assertIn("charts.lock", report.TEXT)
        self.assertIn("frictionSpan", report._TEMPLATE)

    def test_the_axis_is_ticked_not_just_cornered(self):
        self.assertIn("function yTicks", report._TEMPLATE)
        self.assertIn("yTicks(ctx, box", report._TEMPLATE)

    def test_the_selected_joint_outlives_a_language_switch(self):
        # render() rebuilds the DOM, so a selection held only in the <select>
        # would snap back to the first joint on every switch.
        self.assertIn("let picked = 0;", report._TEMPLATE)
        self.assertIn("pick.value = String(picked);", report._TEMPLATE)
        self.assertIn("lock.checked = locked;", report._TEMPLATE)

    def test_load_and_stribeck_terms_are_drawn_with_the_fitted_shapes(self):
        panel = (STATIC / "charts.js").read_text(encoding="utf-8")
        for source in (report._TEMPLATE, panel):
            self.assertIn(
                "Math.abs(load || 0) * sign", source)
            self.assertIn(
                "const decay = sign * Math.exp(-Math.abs(velocity) / stribeckSpeed)", source)
            self.assertIn(
                "Math.abs(load || 0) * decay", source)
            self.assertNotIn("+ carried) * rev", source)

    def test_load_dependent_curves_are_drawn_as_a_cluster(self):
        panel = (STATIC / "charts.js").read_text(encoding="utf-8")
        self.assertIn("const LOAD_LEVEL_COUNT = 6", report._TEMPLATE)
        self.assertIn('id="load-legend"', report._TEMPLATE)
        self.assertIn("loadClusterIndex(s.load, loads)", report._TEMPLATE)
        self.assertIn("curves.length > 1", report._TEMPLATE)
        self.assertIn("curves[curves.length - 1]", report._TEMPLATE)
        for colour in ("#2563eb", "#06b6d4", "#22c55e",
                       "#facc15", "#f97316", "#ef4444"):
            self.assertIn(colour, report._TEMPLATE)

    def test_report_directly_lists_each_joint_numeric_formula(self):
        self.assertIn("function frictionFormula(entry)", report._TEMPLATE)
        self.assertIn("'L sgn(v)'", report._TEMPLATE)
        self.assertIn("sgn(v) exp(-|v| /", report._TEMPLATE)
        self.assertIn("L sgn(v) exp(-|v| /", report._TEMPLATE)
        self.assertIn("formulaSection(),", report._TEMPLATE)
        self.assertIn("formula.head", report.TEXT)
        self.assertIn("L = |I_rigid|", report.TEXT["formula.say"]["en"])

    def test_formula_diagnostics_distinguish_stribeck_and_current_peaks(self):
        self.assertIn("function hasStribeckPeak(entry, load)", report._TEMPLATE)
        self.assertIn("function formulaDiagnostic(entry, index)", report._TEMPLATE)
        self.assertIn("entry.peak_measured_effort", report._TEMPLATE)
        self.assertIn("formula.stribeck.nopeak", report.TEXT)
        self.assertIn("formula.totalpeak", report.TEXT)
        self.assertIn("formula.frictionpeak", report.TEXT)

    def test_controlled_low_speed_audit_is_separate_from_the_formula(self):
        self.assertIn("function steadyFrictionSection()", report._TEMPLATE)
        self.assertIn("steadyFrictionSection(), comparisonSection()",
                      report._TEMPLATE)
        self.assertIn("steady.head", report.TEXT)
        self.assertIn("do not refit the dynamic predictor",
                      report.TEXT["steady.say"]["en"])
        self.assertIn("visual peaks in the mixed", report.TEXT["steady.say"]["en"])

    def test_the_friction_chart_overlays_the_measured_steady_curve(self):
        # The fitted curve alone cannot settle whether a Stribeck peak was
        # measured. Drawing the controlled steady medians on the same axes
        # puts the measurement and the model side by side.
        self.assertIn("function steadyOverlay(", report._TEMPLATE)
        self.assertIn("steady_friction_audit", report._TEMPLATE)
        self.assertIn("charts.friction.steady", report.TEXT)
        self.assertIn("charts.friction.still", report.TEXT)
        self.assertIn("standing still", report.TEXT["charts.friction.still"]["en"])
        self.assertIn("stiction band", report.TEXT["charts.friction.still"]["en"])
        self.assertIn("still-note", report._TEMPLATE)

    def test_the_low_speed_range_gets_a_chart_of_its_own(self):
        # On an axis that runs to 35 deg/s the whole steady range is a few
        # pixels wide, so "no peak" is asserted rather than shown. The zoomed
        # chart puts the measured medians and the fitted curve side by side
        # over the speeds the Stribeck question is actually about.
        self.assertIn("function drawSteady(", report._TEMPLATE)
        self.assertIn("c-steady", report._TEMPLATE)
        self.assertIn("charts.steadylow", report.TEXT)
        self.assertIn("charts.steadylow.say", report.TEXT)


class ChartLegendTest(unittest.TestCase):
    """The residual chart drew two colours and explained neither, and the
    friction legend named its colours without saying what they were."""

    def setUp(self):
        self.section = report._TEMPLATE.split("function chartSection", 1)[1] \
                                       .split("\n}", 1)[0]

    def test_both_scatter_charts_carry_a_legend(self):
        self.assertEqual(self.section.count('<ul class="key">'), 2)

    def test_every_colour_the_friction_chart_draws_is_in_its_legend(self):
        body = report._TEMPLATE.split("const LOAD_LEVEL_COUNT", 1)[1] \
                               .split("function drawResidual", 1)[0]
        for colour in ("#2563eb", "#06b6d4", "#22c55e",
                       "#facc15", "#f97316", "#ef4444"):
            self.assertIn(colour, body)
            self.assertIn(colour, self.section)
        self.assertIn("#f8fafc", body)
        self.assertIn("#f8fafc", self.section)

    def test_every_colour_the_residual_chart_draws_is_in_its_legend(self):
        body = report._TEMPLATE.split("function drawResidual", 1)[1] \
                               .split("function drawBars", 1)[0]
        self.assertIn("rgba(226,86,90", body)      # non-sweep residuals
        self.assertIn("rgba(120,220,150", body)    # sweep residuals
        self.assertIn("var(--bad)", self.section)
        self.assertIn("var(--sweep)", self.section)

    def test_the_speed_axis_is_numbered(self):
        # It carried a name but no numbers, so a cluster could be seen without
        # being placed anywhere on the axis.
        self.assertIn("function xTicks", report._TEMPLATE)
        self.assertIn("xTicks(ctx, box, maxSpeed, sx)", report._TEMPLATE)

    def test_each_legend_entry_explains_the_colour_not_just_names_it(self):
        for key in ("charts.friction.green", "charts.friction.blue",
                    "charts.friction.curve", "charts.residual.red",
                    "charts.residual.green"):
            self.assertIn(key, report.TEXT)
            self.assertGreater(len(report.TEXT[key]["en"]), 40, key)
            self.assertGreater(len(report.TEXT[key]["zh"]), 12, key)

    def test_the_colour_is_named_in_words_as_well_as_shown(self):
        # A legend that relies on hue alone is unreadable to some readers.
        for key in ("charts.sweep", "charts.other", "charts.curve",
                    "charts.red", "charts.green"):
            self.assertIn("—", report.TEXT[key]["en"], key)

    def test_the_residual_chart_admits_it_omits_the_validation_data(self):
        # Training residuals flatter the model; saying so is the difference
        # between a diagnostic and a advertisement.
        self.assertIn("charts.residual.caveat", report.TEXT)
        self.assertIn("validation", report.TEXT["charts.residual.caveat"]["en"])
        self.assertIn("验证", report.TEXT["charts.residual.caveat"]["zh"])


class DashboardShellTest(unittest.TestCase):
    """The always-visible shell must not drift back into a folded panel."""

    @classmethod
    def setUpClass(cls):
        cls.markup = (STATIC / "index.html").read_text(encoding="utf-8")
        cls.styles = (STATIC / "dashboard.css").read_text(encoding="utf-8")
        cls.panel = (STATIC / "dashboard.js").read_text(encoding="utf-8")
        cls.translations = (STATIC / "i18n.js").read_text(encoding="utf-8")
        cls.viewer = (STATIC / "viewer.js").read_text(encoding="utf-8")

    def test_the_obstacle_editor_is_a_persistent_details_element(self):
        self.assertIn('<details class="overlay" id="edit-box" open>',
                      self.markup)
        self.assertIn("#panel details.group, #edit-box", self.panel)

    def test_the_information_strip_is_the_panel_s_unfoldable_bottom_row(self):
        panel = self.markup.index('<aside id="panel">')
        scroll = self.markup.index('id="panel-scroll"')
        strip = self.markup.index('id="activity-strip"')
        end = self.markup.index("</aside>", panel)
        self.assertLess(panel, scroll)
        self.assertLess(scroll, strip)
        self.assertLess(strip, end)
        self.assertIn("#panel {\n  display: grid;", self.styles)
        self.assertIn("grid-template-rows: minmax(0, 1fr) auto", self.styles)
        self.assertNotIn("#activity-strip {\n  position: fixed", self.styles)
        self.assertEqual(self.markup.count('id="progress-line"'), 1)

    def test_the_information_strip_is_current_state_not_a_log(self):
        self.assertNotIn('id="activity-events"', self.markup)
        self.assertNotIn('id="activity-time"', self.markup)
        self.assertNotIn("events.slice(-80)", self.panel)

    def test_the_information_strip_gives_the_current_message_room(self):
        self.assertIn("min-height: 160px", self.styles)
        self.assertIn("grid-template-columns: minmax(0, 1fr) auto", self.styles)
        self.assertIn("align-content: start", self.styles)
        self.assertIn("font-size: 13px", self.styles)
        self.assertIn("grid-column: 1 / -1", self.styles)
        self.assertIn("overflow-wrap: anywhere", self.styles)
        self.assertNotIn("-webkit-line-clamp", self.styles)

    def test_canvas_activity_is_mode_neutral(self):
        self.assertIn("onSceneActivity", self.panel)
        self.assertIn("data.scene_activity", self.viewer)
        self.assertNotIn("startsWith('gravity') && busy", self.panel)

    def test_canvas_distinguishes_target_completed_and_pending_poses(self):
        self.assertIn("const GHOST_ACTIVE = 0xffc857", self.viewer)
        self.assertIn("const GHOST_COMPLETED = 0x8fa3ad", self.viewer)
        self.assertIn("activity.completed || {}", self.viewer)
        self.assertIn("complete ? GHOST_COMPLETED : node.color", self.viewer)
        self.assertIn("focus.phase, focus.index", self.viewer)

    def test_gravity_has_a_direct_safe_report_link(self):
        self.assertIn('id="grav-report"', self.markup)
        self.assertIn("snapshot.reports?.gravity", self.panel)
        self.assertIn('target="_blank"', self.markup)
        self.assertIn('rel="noopener"', self.markup)

    def test_gravity_hardware_run_has_pause_and_resume_controls(self):
        self.assertEqual(self.markup.count('id="btn-grav-pause"'), 1)
        self.assertEqual(self.markup.count('id="btn-grav-resume"'), 1)
        self.assertIn("post('/api/pause', {})", self.panel)
        self.assertIn("post('/api/resume', {})", self.panel)
        self.assertIn("snapshot.state === 'paused'", self.panel)
        self.assertIn("'grav.paused_pose'", self.translations)

    def test_gravity_status_exists_only_in_the_information_strip(self):
        self.assertNotIn('id="grav-state"', self.markup)
        self.assertIn("function gravityStatus(snapshot)", self.panel)
        self.assertIn("passive: true", self.panel)
        self.assertIn("if (!status.passive)", self.panel)
        self.assertIn("(!active && !gravityCurrent.passive)", self.panel)
        self.assertIn("latest?.message || gravityCurrent?.message", self.panel)

    def test_activity_follows_the_latest_progress_callback(self):
        self.assertIn("progress.updated_fields", self.panel)
        self.assertIn("wasUpdated('designed')", self.panel)
        self.assertIn("wasUpdated('observations')", self.panel)
        self.assertIn("wasUpdated('pose')", self.panel)
        self.assertNotIn("&& progress.poses", self.panel)
        self.assertIn("'run.designed'", self.translations)
        self.assertIn("if (state.polling) return", self.panel)
        self.assertIn("publishLocalEvent(t('grav.rehearsal_started')", self.panel)
        self.assertIn("'phase.starting'", self.translations)
        self.assertIn("'phase.finished'", self.translations)


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
