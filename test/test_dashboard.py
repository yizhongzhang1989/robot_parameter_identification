"""The dashboard surface, exercised without ROS and without a robot."""

import json
import unittest
import urllib.error
import urllib.request

try:
    from robot_parameter_identification import identification as ident
    from robot_parameter_identification.dashboard.http_server import (
        DashboardServer, build_routes)
    from robot_parameter_identification.dashboard.service import (
        DashboardConfig, IdentificationService, _swept_here)
    from robot_parameter_identification.interfaces import (
        SignalMap, TelemetrySpec)
    from fixtures import synthetic_urdf, test_profile, PREFIX
except ImportError as error:
    raise unittest.SkipTest(f"needs pinocchio: {error}") from error


def service() -> IdentificationService:
    made = IdentificationService(DashboardConfig(), profile=test_profile())
    made.adopt_description(synthetic_urdf())
    return made


class ModelAdoptionTest(unittest.TestCase):
    def test_a_service_starts_with_no_model(self):
        self.assertFalse(IdentificationService(DashboardConfig()).have_model())

    def test_robot_description_builds_the_model_and_the_scene(self):
        made = service()
        self.assertTrue(made.have_model())
        self.assertTrue(made.frame_names())

    def test_rubbish_description_is_reported_not_raised(self):
        made = IdentificationService(DashboardConfig(), profile=test_profile())
        self.assertFalse(made.adopt_description("<robot>"))
        self.assertFalse(made.have_model())
        self.assertTrue(made.notes)

    def test_the_same_description_twice_is_ignored(self):
        made = service()
        self.assertFalse(made.adopt_description(synthetic_urdf()))

    def test_obstacles_survive_a_redescription(self):
        made = service()
        made.add_obstacle({"parent_frame": f"{PREFIX}base_link", "name": "bench"})
        made.urdf_text = ""            # force a rebuild with the same geometry
        made.adopt_description(synthetic_urdf())
        self.assertEqual([box["name"] for box in made.obstacles()], ["bench"])


class ObstacleApiTest(unittest.TestCase):
    def setUp(self):
        self.service = service()
        self.frame = f"{PREFIX}base_link"

    def test_add_then_list(self):
        box = self.service.add_obstacle({"parent_frame": self.frame})
        self.assertEqual([item["id"] for item in self.service.obstacles()],
                         [box["id"]])

    def test_update_moves_the_box(self):
        box = self.service.add_obstacle({"parent_frame": self.frame})
        moved = self.service.update_obstacle(box["id"], {"xyz_m": [1.0, 0.0, 0.0]})
        self.assertEqual(list(moved["xyz_m"]), [1.0, 0.0, 0.0])

    def test_remove_empties_the_scene(self):
        box = self.service.add_obstacle({"parent_frame": self.frame})
        self.service.remove_obstacle(box["id"])
        self.assertEqual(self.service.obstacles(), [])

    def test_an_unknown_frame_is_refused(self):
        with self.assertRaises(KeyError):
            self.service.add_obstacle({"parent_frame": "nowhere"})

    def test_collision_report_explains_a_blocked_pose(self):
        self.service.add_obstacle(
            {"parent_frame": self.frame, "size_m": [2.0, 2.0, 2.0]})
        report = self.service.collision_report([0.0] * 7)
        self.assertTrue(report["available"])
        self.assertFalse(report["clear"])
        self.assertTrue(report["contacts"])

    def test_collision_report_is_honest_without_a_model(self):
        blank = IdentificationService(DashboardConfig())
        self.assertFalse(blank.collision_report([0.0])["available"])


class RunGateTest(unittest.TestCase):
    def test_hardware_is_refused_before_a_rehearsal(self):
        made = service()
        answer = made.start("hardware")
        self.assertFalse(answer["ok"])
        self.assertIn("rehearse", answer["message"])

    def test_a_passed_rehearsal_arms_the_hardware_run(self):
        # The rehearsal gate is not ceremony: it plants known friction and must
        # find it again, and it is what caught the fit returning zero.
        made = service()
        made.rehearsal_passed = True
        made._state = "running"
        answer = made.start("hardware")
        self.assertFalse(answer["ok"])
        self.assertIn("running", answer["message"])

    def test_nothing_starts_without_a_model(self):
        blank = IdentificationService(DashboardConfig(), profile=test_profile())
        self.assertFalse(blank.start("rehearsal")["ok"])

    def test_the_snapshot_carries_what_the_page_needs(self):
        snapshot = service().snapshot()
        for key in ("state", "connection", "have_model", "obstacles", "frames",
                    "collision", "progress", "notes", "rehearsal_passed"):
            self.assertIn(key, snapshot)

    def test_no_acknowledgement_is_demanded_anywhere(self):
        self.assertNotIn("acknowledgement", service().snapshot())


class ResultProvenanceTest(unittest.TestCase):
    """A result file has to say which arm and which quantity it describes."""

    def test_the_saved_result_names_the_arm_and_the_effort(self):
        made = service()
        made.adopt_driven_joints(
            [f"{PREFIX}joint{index}" for index in range(1, 8)])
        made._finish("rehearsal", {"complete": True, "joints": []}, [])
        result = made.snapshot()["result"]
        self.assertEqual(result["joint_names"][0], f"{PREFIX}joint1")
        self.assertIn("action", result)
        self.assertEqual(result["effort_source"], "current")
        self.assertEqual(result["effort_unit"], "ampere")

    def test_a_torque_run_is_not_labelled_amperes(self):
        made = IdentificationService(
            DashboardConfig(telemetry=TelemetrySpec(
                signals=SignalMap(current=None, torque="torque",
                                  effort_source="torque"))),
            profile=test_profile())
        made.adopt_description(synthetic_urdf())
        made._finish("rehearsal", {"complete": True, "joints": []}, [])
        self.assertEqual(made.snapshot()["result"]["effort_unit"],
                         "newton_metre")


class FakePlant:
    """Enough of a hardware plant to check the homing plumbing."""

    def __init__(self, position):
        self.position = list(position)
        self.parked = False
        self.closed = False
        self.opened_with = {}

    def sample(self):
        return {"position_deg": list(self.position)}

    def park(self):
        self.parked = True
        self.position = [0.0] * len(self.position)

    def close(self):
        self.closed = True


class FakeBridge:
    def __init__(self, plant):
        self.plant = plant

    def hardware_plant(self, profile, scene, **kwargs):
        self.plant.opened_with = dict(kwargs)
        return self.plant

    def health(self):
        return {"telemetry_ok": True, "action_ok": True, "sample_age_s": 0.01,
                "description_ok": True}

    def latest_sample(self):
        return {"position_deg": list(self.plant.position)}


class HomingTest(unittest.TestCase):
    """A campaign leaves the arm off-home; this is how it gets back."""

    def homing_service(self, position):
        plant = FakePlant(position)
        made = IdentificationService(DashboardConfig(), bridge=FakeBridge(plant),
                                     profile=test_profile())
        made.adopt_description(synthetic_urdf())
        return made, plant

    def wait(self, made):
        import time

        deadline = time.monotonic() + 30
        while made.running() and time.monotonic() < deadline:
            time.sleep(0.02)

    def test_homing_runs_on_a_bare_click(self):
        made, plant = self.homing_service([5.0] * 7)
        self.assertTrue(made.home()["ok"])
        self.wait(made)
        self.assertTrue(plant.parked)

    def test_homing_is_refused_while_something_runs(self):
        made, _plant = self.homing_service([5.0] * 7)
        made._state = "running"
        made._activity = "campaign_rehearsal"
        answer = made.home()
        self.assertFalse(answer["ok"])
        self.assertIn("running", answer["message"])

    def test_homing_needs_a_model(self):
        blank = IdentificationService(DashboardConfig(), profile=test_profile())
        self.assertFalse(blank.home()["ok"])

    def test_homing_does_not_need_a_rehearsal(self):
        # It drives no identification, so the rehearsal gate would only stop
        # an operator recovering an arm the plant already refuses to arm.
        made, plant = self.homing_service([5.0] * 7)
        self.assertFalse(made.rehearsal_passed)
        self.assertTrue(made.home()["ok"])
        self.wait(made)
        self.assertTrue(plant.parked)

    def test_homing_waives_the_neutral_start_check(self):
        # That check exists to refuse campaigns off home. Homing is the one
        # job that must be allowed to run precisely then.
        made, plant = self.homing_service([40.0] * 7)
        made.home()
        self.wait(made)
        self.assertIs(plant.opened_with.get("require_neutral_start"), False)

    def test_homing_reports_where_it_started_and_ended(self):
        made, _plant = self.homing_service([5.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        made.home()
        self.wait(made)
        progress = made.snapshot()["progress"]
        self.assertEqual(progress["phase"], "homed")
        self.assertEqual(progress["from_deg"][0], 5.0)
        self.assertEqual(progress["worst_deg"], 0.0)

    def test_the_plant_is_released_afterwards(self):
        # A hardware plant owns a ROS context; not closing it leaks one per run.
        made, plant = self.homing_service([5.0] * 7)
        made.home()
        self.wait(made)
        self.assertTrue(plant.closed)

    def test_the_service_is_idle_again(self):
        made, _plant = self.homing_service([5.0] * 7)
        made.home()
        self.wait(made)
        self.assertEqual(made.snapshot()["state"], "idle")

    def test_releasing_tolerates_a_plant_that_cannot_be_closed(self):
        made, _plant = self.homing_service([0.0] * 7)
        made._release(object())
        made._release(None)

    def test_homing_is_reachable_over_http(self):
        made, plant = self.homing_service([3.0] * 7)
        routes = build_routes(made, None)
        self.assertIn("/api/home", routes)
        self.assertEqual(routes["/api/home"][0], "POST")
        self.assertTrue(routes["/api/home"][1]({})["ok"])
        self.wait(made)
        self.assertTrue(plant.parked)


class ViewerStateTest(unittest.TestCase):
    def test_viewer_reports_no_model_before_one_arrives(self):
        self.assertFalse(
            IdentificationService(DashboardConfig()).viewer_state()["have_model"])

    def test_viewer_carries_link_poses_and_boxes(self):
        made = service()
        made.add_obstacle({"parent_frame": f"{PREFIX}base_link"})
        payload = made.viewer_state()
        self.assertTrue(payload["have_model"])
        self.assertIn(f"{PREFIX}link1", payload["link_tf"])
        self.assertEqual(len(payload["link_tf"][f"{PREFIX}link1"]), 16)
        self.assertEqual(len(payload["obstacles"]), 1)


class RehearsalEndToEndTest(unittest.TestCase):
    """A whole rehearsal, start to verdict.

    The pieces all passed their own tests while the run still died in phase C
    on a mistyped trajectory call, so nothing short of running it counts.
    """

    @classmethod
    def setUpClass(cls):
        import time

        cls.service = IdentificationService(DashboardConfig())
        cls.service.adopt_description(synthetic_urdf())
        cls.service.adopt_driven_joints(
            [f"{PREFIX}joint{index}" for index in range(1, 4)])
        # Small enough to stay quick, large enough to exercise all four phases.
        cls.service.plan = cls.service.plan.__class__(
            static_poses=6, static_candidates=20, settle_samples=1,
            friction_speeds_deg_s=(2.0, 5.0), fourier_harmonics=2,
            fourier_duration_s=4.0, fourier_attempts=8, sample_rate_hz=10.0,
            validation_poses=4, validation_trajectory_s=3.0, seed=1)
        started = cls.service.start("rehearsal")
        assert started["ok"], started
        deadline = time.monotonic() + 120
        while cls.service.running() and time.monotonic() < deadline:
            time.sleep(0.2)
        cls.snapshot = cls.service.snapshot()

    def test_the_run_reached_the_end(self):
        progress = self.snapshot["progress"]
        self.assertNotEqual(progress.get("phase"), "failed",
                            msg=progress.get("traceback", ""))
        self.assertEqual(progress.get("phase"), "finished")

    def test_a_result_was_produced(self):
        result = self.snapshot["result"]
        self.assertIsNotNone(result)
        self.assertIsNone(result["aborted"])
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["joints"]), 3)

    def test_the_verdict_is_reported(self):
        self.assertIn(self.snapshot["result"]["verdict"]["state"],
                      ("pass", "warn", "fail"))

    def test_the_charts_have_something_to_draw(self):
        result = self.snapshot["result"]
        samples = result.get("friction_samples") or []
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(len(series) > 0 for series in samples))

    def test_the_residual_chart_is_not_empty(self):
        # It was, for as long as it existed: it read predicted_a and
        # measured_a, which no regression ever carried.
        residual = self.snapshot["result"].get("residual_samples") or []
        self.assertEqual(len(residual), 3)
        self.assertTrue(all(len(series) > 0 for series in residual))

    def test_residuals_carry_the_speed_they_happened_at(self):
        # Every point used to be reported at speed zero, so the one chart
        # meant to show structure against speed could not show any.
        speeds = {point["speed"]
                  for point in self.snapshot["result"]["residual_samples"][0]}
        self.assertGreater(len(speeds), 5)

    def test_the_friction_cloud_has_gravity_removed(self):
        # Raw current at rest spans the gravity of every pose visited. What
        # the friction curve claims to explain is what is left after the
        # rigid-body prediction is subtracted.
        cloud = self.snapshot["result"]["friction_samples"][0]
        self.assertIn("speed", cloud[0])
        self.assertIn("effort", cloud[0])
        still = [point["effort"] for point in cloud if abs(point["speed"]) < 1.0]
        self.assertGreater(len(still), 3)
        entry = self.snapshot["result"]["joints"][0]
        coulomb = abs(entry["friction"].get("coulomb", 0.0))
        # Near zero speed a friction model predicts one value; the spread
        # there must be small beside the Coulomb step it is meant to show.
        spread = max(still) - min(still)
        self.assertLess(spread, max(4.0 * coulomb, 0.5))

    def test_sweep_samples_are_marked_apart_from_the_rest(self):
        # The cloud holds two populations: sweep points, where one joint moves
        # about a single pose, and everything else, taken across many poses.
        # Overlaid without a mark, a reader measures the pose difference
        # between the groups and calls it a speed trend.
        #
        # No claim is made here about which group is faster. A sweep spends
        # much of its time accelerating and reversing, so it owns plenty of
        # slow samples too, and how the speeds compare is a property of the
        # plan rather than of this code.
        cloud = self.snapshot["result"]["friction_samples"][0]
        swept = [point for point in cloud if point.get("sweep")]
        rest = [point for point in cloud if not point.get("sweep")]
        self.assertTrue(swept, "no sweep samples were flagged")
        self.assertTrue(rest, "every sample was flagged as a sweep")

    def test_both_charts_agree_on_which_samples_are_sweeps(self):
        # The two charts are read against each other, so a point marked in one
        # and not the other would be worse than no mark at all.
        result = self.snapshot["result"]
        for cloud, errors in zip(result["friction_samples"],
                                 result["residual_samples"]):
            self.assertEqual([point.get("sweep") for point in cloud],
                             [point.get("sweep") for point in errors])

    def test_the_flag_is_absent_rather_than_false(self):
        # It rides on every point of a payload that is polled, so the common
        # case carries no key at all.
        cloud = self.snapshot["result"]["friction_samples"][0]
        self.assertTrue(any("sweep" not in point for point in cloud))
        self.assertTrue(all(point.get("sweep") is not False for point in cloud))

    def test_passing_a_rehearsal_unlocks_the_hardware_button(self):
        self.assertTrue(self.snapshot["rehearsal_passed"])

    def test_the_injected_friction_is_recovered(self):
        """The rehearsal is only worth running if it can catch a broken fit."""
        check = self.snapshot["result"]["rehearsal_check"]
        self.assertTrue(check["available"])
        self.assertTrue(check["passed"],
                        msg=f"worst error {check['worst_coulomb_error']} "
                            f"exceeds {check['tolerance']}: {check['joints']}")

    def test_the_planted_friction_was_not_zero(self):
        """A frictionless rehearsal recovers zero and proves nothing."""
        planted = [item["expected"]
                   for item in self.snapshot["result"]["rehearsal_check"]["joints"]]
        self.assertTrue(all(value > 0.05 for value in planted), planted)
        self.assertEqual(len(set(planted)), len(planted),
                         "joints must differ so a mix-up cannot pass")

    def test_the_coefficients_stay_physical(self):
        for entry in self.snapshot["result"]["joints"]:
            friction = entry["friction"]
            self.assertGreaterEqual(friction["coulomb"], 0.0)
            self.assertGreaterEqual(friction["viscous"], 0.0)


class StopTest(unittest.TestCase):
    """Stopping must halt the run and must not be mistaken for finishing."""

    def test_a_stopped_run_is_not_reported_as_finished(self):
        import time

        made = IdentificationService(DashboardConfig())
        made.adopt_description(synthetic_urdf())
        made.adopt_driven_joints(
            [f"{PREFIX}joint{index}" for index in range(1, 4)])
        self.assertTrue(made.start("rehearsal")["ok"])
        made.stop()
        deadline = time.monotonic() + 60
        while made.running() and time.monotonic() < deadline:
            time.sleep(0.1)
        snapshot = made.snapshot()
        self.assertEqual(snapshot["state"], "idle")
        self.assertEqual(snapshot["progress"]["phase"], "stopped")
        self.assertEqual(snapshot["result"]["aborted"], "operator stop")
        self.assertFalse(snapshot["result"]["complete"])
        # A half-measured model must never unlock the hardware button.
        self.assertFalse(snapshot["rehearsal_passed"])

    def test_starting_clears_the_previous_verdict(self):
        import time

        made = IdentificationService(DashboardConfig())
        made.adopt_description(synthetic_urdf())
        made.adopt_driven_joints(
            [f"{PREFIX}joint{index}" for index in range(1, 4)])
        made.result = {"complete": True, "verdict": {"state": "pass"}}
        self.assertTrue(made.start("rehearsal")["ok"])
        try:
            # A stale pass must not be readable as this run's outcome.
            self.assertIsNone(made.snapshot()["result"])
        finally:
            made.stop()
            deadline = time.monotonic() + 60
            while made.running() and time.monotonic() < deadline:
                time.sleep(0.1)


class SweptJointTest(unittest.TestCase):
    """The friction phase moves one joint at a time, so most of its samples
    are a record of any given joint standing still. Marking those as sweeps
    put six sevenths of them on the zero line of a seven-joint arm's chart."""

    class Record:
        def __init__(self, phase, motion="", velocity=None):
            self.phase = phase
            self.motion = motion
            self.velocity_deg_s = velocity or [0.0] * 7

    def test_the_swept_joint_is_marked(self):
        record = self.Record("B_friction", "traverse:j3:15")
        self.assertTrue(_swept_here(record, 3))

    def test_the_joints_standing_still_are_not(self):
        record = self.Record("B_friction", "traverse:j3:15")
        for joint in (0, 1, 2, 4, 5, 6):
            self.assertFalse(_swept_here(record, joint), joint)

    def test_other_phases_are_never_sweeps(self):
        record = self.Record("A_gravity", "sweep:2")
        self.assertFalse(_swept_here(record, 0))

    def test_a_run_without_motion_tags_falls_back_to_movement(self):
        # Results recorded before the tag existed still have to draw.
        moving = [0.0] * 7
        moving[2] = 20.0
        record = self.Record("B_friction", "", moving)
        self.assertTrue(_swept_here(record, 2))
        self.assertFalse(_swept_here(record, 1))

    def test_a_malformed_tag_does_not_raise(self):
        record = self.Record("B_friction", "traverse:jX:15")
        self.assertFalse(_swept_here(record, 0))


class ProgressTest(unittest.TestCase):
    """Phases report different fields; none may erase another's."""

    def service(self):
        made = IdentificationService(DashboardConfig())
        made._activity = "campaign_rehearsal"
        return made

    def test_a_pose_update_keeps_the_sample_count(self):
        made = self.service()
        made._on_progress("A_gravity", {"observations": 42})
        made._on_progress("A_gravity", {"pose": 3, "poses": 24})
        self.assertEqual(made.progress["observations"], 42)
        self.assertEqual(made.progress["pose"], 3)

    def test_a_sample_update_keeps_the_pose_index(self):
        made = self.service()
        made._on_progress("A_gravity", {"pose": 3, "poses": 24})
        made._on_progress("A_gravity", {"observations": 99})
        self.assertEqual(made.progress["pose"], 3)
        self.assertEqual(made.progress["observations"], 99)

    def test_elapsed_time_is_always_refreshed(self):
        made = self.service()
        made._started_at = 0.0
        made._on_progress("B_friction", {})
        self.assertGreater(made.progress["elapsed_s"], 0.0)

    def test_a_new_phase_drops_the_old_phase_fields(self):
        made = self.service()
        made._on_progress("A_gravity", {"pose": 24, "poses": 24})
        made._on_progress("B_friction", {"observations": 5})
        self.assertNotIn("pose", made.progress)
        self.assertEqual(made.progress["observations"], 5)


class SpeedRequestTest(unittest.TestCase):
    """Raising the ceiling must reach the sweep, not stop at the profile."""

    def service(self, speed):
        made = IdentificationService(
            DashboardConfig(maximum_speed_deg_s=speed))
        made.adopt_description(synthetic_urdf())
        made.adopt_driven_joints(
            [f"{PREFIX}joint{index}" for index in range(1, 8)])
        return made

    def test_the_request_reaches_the_profile_and_the_plan(self):
        made = self.service(60.0)
        self.assertAlmostEqual(made.profile.sustained_speed_deg_s, 60.0)
        self.assertAlmostEqual(made.plan.maximum_speed_deg_s, 60.0)

    def test_the_sweep_actually_runs_at_the_new_speed(self):
        slow = self.service(0.0)
        fast = self.service(60.0)
        self.assertGreater(max(fast.plan.friction_speeds_deg_s),
                           max(slow.plan.friction_speeds_deg_s))
        self.assertAlmostEqual(max(fast.plan.friction_speeds_deg_s), 60.0)

    def test_validation_moves_with_it(self):
        fast = self.service(60.0)
        self.assertGreater(max(fast.plan.validation_speeds_deg_s), 10.0)

    def test_zero_keeps_the_conservative_default(self):
        made = self.service(0.0)
        self.assertLessEqual(made.plan.maximum_speed_deg_s, 20.0)


class GuardTest(unittest.TestCase):
    """A named interface is not protection; the panel used to imply it was."""

    def monitor(self, **kwargs):
        from robot_parameter_identification.campaign import DriveMonitor

        return DriveMonitor(**kwargs)

    def test_a_disabled_drive_stops_the_run(self):
        trip = self.monitor().check({"enabled": [True, False, True]}, 0.0)
        self.assertIn("joint2", trip)
        self.assertIn("disabled", trip)

    def test_a_fault_word_stops_the_run(self):
        trip = self.monitor().check({"enabled": [True], "fault_code": [7]}, 0.0)
        self.assertIn("fault code 7", trip)

    def test_a_healthy_frame_passes(self):
        self.assertIsNone(self.monitor().check(
            {"enabled": [True] * 3, "fault_code": [0] * 3}, 0.0))

    def test_channels_the_robot_lacks_are_skipped_not_tripped(self):
        self.assertIsNone(self.monitor().check({}, 0.0))

    def test_voltage_is_only_checked_when_a_window_was_supplied(self):
        without = self.monitor().check({"voltage_v": [999.0]}, 0.0)
        self.assertIsNone(without)
        with_window = self.monitor(minimum_voltage_v=20.0,
                                   maximum_voltage_v=30.0)
        self.assertIn("999.0 V", with_window.check({"voltage_v": [999.0]}, 0.0))

    def test_a_derived_profile_does_not_arm_the_voltage_window(self):
        # Its window is a default, and aborting a good run on a guessed
        # threshold is worse than not checking.
        made = service()
        made.profile_source = "derived"
        self.assertNotIn("bus-voltage window", made._monitor().guards())

    def test_an_unmapped_signal_reports_its_guard_dark(self):
        made = IdentificationService(
            DashboardConfig(telemetry=TelemetrySpec(
                signals=SignalMap(enabled=None, fault_code=None))),
            profile=test_profile())
        dark = made.connection()["missing_guards"]
        self.assertIn("drive-enabled check", dark)
        self.assertIn("fault-code check", dark)

    def test_a_mapped_signal_that_never_arrives_still_reports_dark(self):
        # This is the case the old reporting got wrong: it trusted the name.
        made = IdentificationService(
            DashboardConfig(telemetry=TelemetrySpec(
                signals=SignalMap(enabled="enabled", fault_code="fault_code"))),
            bridge=SilentBridge(), profile=test_profile())
        self.assertIn("drive-enabled check", made.connection()["missing_guards"])


class SilentBridge:
    """A robot that names its interfaces but publishes none of them."""

    def health(self):
        return {"telemetry_ok": False, "action_ok": False, "sample_age_s": None,
                "description_ok": False}

    def observed_signals(self):
        return {"position", "effort"}

    def latest_sample(self):
        return None


class HttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = service()
        cls.server = DashboardServer(cls.service, port=0)
        cls.server.start()
        cls.base = f"http://127.0.0.1:{cls.server.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as reply:
            return reply.status, json.loads(reply.read())

    def post(self, path, body):
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as reply:
                return reply.status, json.loads(reply.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_index_is_served(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as reply:
            self.assertEqual(reply.status, 200)
            self.assertIn(b"<canvas", reply.read())

    def test_state_is_json(self):
        status, payload = self.get("/api/state")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["have_model"])

    def test_viewer_is_json(self):
        status, payload = self.get("/api/viewer")
        self.assertEqual(status, 200)
        self.assertTrue(payload["have_model"])

    def test_obstacles_round_trip_over_http(self):
        status, payload = self.post("/api/obstacles", {
            "action": "add",
            "obstacle": {"parent_frame": f"{PREFIX}base_link", "name": "bench"}})
        self.assertEqual(status, 200)
        identifier = payload["obstacle"]["id"]
        status, _ = self.post("/api/obstacles", {"action": "remove",
                                                 "id": identifier})
        self.assertEqual(status, 200)

    def test_a_bad_frame_answers_400_with_a_sentence(self):
        status, payload = self.post("/api/obstacles", {
            "action": "add", "obstacle": {"parent_frame": "nowhere"}})
        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])
        self.assertIn("nowhere", payload["message"])

    def test_unknown_endpoint_is_404(self):
        request = urllib.request.Request(self.base + "/api/nope", data=b"{}",
                                         method="POST")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 404)

    def test_static_traversal_is_blocked(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base + "/../service.py", timeout=5)
        self.assertIn(caught.exception.code, (400, 403, 404))

    def test_hardware_start_is_refused_over_http_too(self):
        status, payload = self.post("/api/campaign", {"mode": "hardware"})
        self.assertEqual(status, 200)
        self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()
