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
        ACKNOWLEDGEMENT, DashboardConfig, IdentificationService)
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
        answer = made.start("hardware", ACKNOWLEDGEMENT)
        self.assertFalse(answer["ok"])
        self.assertIn("rehearse", answer["message"])

    def test_hardware_is_refused_without_the_acknowledgement(self):
        made = service()
        made.rehearsal_passed = True
        answer = made.start("hardware", "please")
        self.assertFalse(answer["ok"])
        self.assertIn("acknowledgement", answer["message"])

    def test_nothing_starts_without_a_model(self):
        blank = IdentificationService(DashboardConfig(), profile=test_profile())
        self.assertFalse(blank.start("rehearsal")["ok"])

    def test_the_snapshot_carries_what_the_page_needs(self):
        snapshot = service().snapshot()
        for key in ("state", "connection", "have_model", "obstacles", "frames",
                    "collision", "progress", "notes", "acknowledgement"):
            self.assertIn(key, snapshot)


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

    def test_homing_needs_the_acknowledgement(self):
        made, plant = self.homing_service([5.0] * 7)
        answer = made.home("please")
        self.assertFalse(answer["ok"])
        self.assertIn("acknowledgement", answer["message"])
        self.assertFalse(plant.parked)

    def test_homing_is_refused_while_something_runs(self):
        made, _plant = self.homing_service([5.0] * 7)
        made._state = "running"
        made._activity = "campaign_rehearsal"
        answer = made.home(ACKNOWLEDGEMENT)
        self.assertFalse(answer["ok"])
        self.assertIn("running", answer["message"])

    def test_homing_needs_a_model(self):
        blank = IdentificationService(DashboardConfig(), profile=test_profile())
        self.assertFalse(blank.home(ACKNOWLEDGEMENT)["ok"])

    def test_homing_does_not_need_a_rehearsal(self):
        # It drives no identification, so the rehearsal gate would only stop
        # an operator recovering an arm the plant already refuses to arm.
        made, plant = self.homing_service([5.0] * 7)
        self.assertFalse(made.rehearsal_passed)
        self.assertTrue(made.home(ACKNOWLEDGEMENT)["ok"])
        self.wait(made)
        self.assertTrue(plant.parked)

    def test_homing_waives_the_neutral_start_check(self):
        # That check exists to refuse campaigns off home. Homing is the one
        # job that must be allowed to run precisely then.
        made, plant = self.homing_service([40.0] * 7)
        made.home(ACKNOWLEDGEMENT)
        self.wait(made)
        self.assertIs(plant.opened_with.get("require_neutral_start"), False)

    def test_homing_reports_where_it_started_and_ended(self):
        made, _plant = self.homing_service([5.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        made.home(ACKNOWLEDGEMENT)
        self.wait(made)
        progress = made.snapshot()["progress"]
        self.assertEqual(progress["phase"], "homed")
        self.assertEqual(progress["from_deg"][0], 5.0)
        self.assertEqual(progress["worst_deg"], 0.0)

    def test_the_plant_is_released_afterwards(self):
        # A hardware plant owns a ROS context; not closing it leaks one per run.
        made, plant = self.homing_service([5.0] * 7)
        made.home(ACKNOWLEDGEMENT)
        self.wait(made)
        self.assertTrue(plant.closed)

    def test_the_service_is_idle_again(self):
        made, _plant = self.homing_service([5.0] * 7)
        made.home(ACKNOWLEDGEMENT)
        self.wait(made)
        self.assertEqual(made.snapshot()["state"], "idle")

    def test_releasing_tolerates_a_plant_that_cannot_be_closed(self):
        made, _plant = self.homing_service([0.0] * 7)
        made._release(object())
        made._release(None)

    def test_homing_is_reachable_over_http(self):
        made, _plant = self.homing_service([0.0] * 7)
        routes = build_routes(made, None)
        self.assertIn("/api/home", routes)
        self.assertEqual(routes["/api/home"][0], "POST")
        answer = routes["/api/home"][1]({"acknowledgement": "nope"})
        self.assertFalse(answer["ok"])


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
        first = samples[0][0]
        self.assertIn("speed", first)
        self.assertIn("effort", first)

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
        status, payload = self.post("/api/campaign",
                                    {"mode": "hardware",
                                     "acknowledgement": ACKNOWLEDGEMENT})
        self.assertEqual(status, 200)
        self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()
