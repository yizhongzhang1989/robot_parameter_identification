"""Campaign tests: phase bookkeeping, guards, and an end-to-end MuJoCo run."""

import unittest
from pathlib import Path

import numpy as np

try:
    from robot_parameter_identification import campaign
    from test_identification import arm_model
    from fixtures import rm75_profile, scene_path, GAINS, COULOMB, VISCOUS
except ImportError as error:
    raise unittest.SkipTest(f"campaign needs pinocchio: {error}") from error

try:
    from robot_parameter_identification.plants import simulation
except ImportError as error:
    # Only the end-to-end simulated run needs this. Taking the whole module
    # down with it is how every guard, cap and phase test in this file stopped
    # running unnoticed when the MuJoCo plant was removed.
    simulation = None
    SIMULATION_MISSING = str(error)
else:
    SIMULATION_MISSING = ""

class ScriptedPlant:
    """Analytic stand-in so guard behaviour can be tested without a simulator."""

    def __init__(self, joints=7, temperature_c=35.0, current_a=0.0,
                 instrumented=False):
        self.joints = joints
        self.temperature_c = temperature_c
        self.current_a = current_a
        self.instrumented = instrumented
        self.calls = []

    def limits_deg(self):
        return np.full(self.joints, -120.0), np.full(self.joints, 120.0)

    def collision_free(self, pose_deg):
        return True

    def _frame(self, position, velocity):
        frame = {
            "position_deg": list(np.asarray(position, dtype=float)),
            "speed_deg_s": list(np.asarray(velocity, dtype=float)),
            "current_a": [self.current_a] * self.joints,
            "temperature_c": [self.temperature_c] * self.joints,
        }
        if self.instrumented:
            frame["voltage_v"] = [24.0] * self.joints
            frame["enabled"] = [True] * self.joints
            frame["fault_code"] = [0] * self.joints
        return frame

    def hold_pose(self, pose_deg):
        self.calls.append("hold")
        return self._frame(pose_deg, np.zeros(self.joints))

    def traverse(self, joint, start_deg, distance_deg, speed_deg_s):
        self.calls.append("traverse")
        velocity = np.zeros(self.joints)
        velocity[joint] = np.sign(distance_deg) * speed_deg_s
        for step in range(3):
            pose = np.asarray(start_deg, dtype=float).copy()
            pose[joint] += distance_deg * step / 2
            yield self._frame(pose, velocity)

    def track(self, trajectory, rate_hz):
        self.calls.append("track")
        for step in range(5):
            position, velocity, acceleration = trajectory.sample(step * 0.1)
            frame = self._frame(position, velocity)
            frame["acceleration_deg_s2"] = acceleration.tolist()
            yield frame


class StubLimits:
    """The envelope a monitor enforces, in the generic form the campaign sees."""

    def __init__(self, peak=5.0, continuous=2.0, joints=7):
        self.peak_current_a = [peak] * joints
        self.continuous_current_a = [continuous] * joints


class StubMonitor:
    """Minimal EnvelopeMonitor: trips on instantaneous over-current."""

    def __init__(self, limits):
        self.limits = limits
        self.checked = 0

    def check(self, sample, now):
        self.checked += 1
        for index, value in enumerate(sample["current_a"]):
            if abs(value) > self.limits.peak_current_a[index]:
                return (f"joint{index + 1} peak current {abs(value):.2f} A "
                        f"exceeded {self.limits.peak_current_a[index]:.2f} A")
        return None


def small_plan(**overrides):
    values = dict(
        static_poses=6, static_candidates=20, settle_samples=1,
        friction_amplitude_deg=15.0, friction_speeds_deg_s=(2.0, 5.0),
        fourier_harmonics=3, fourier_duration_s=8.0, fourier_attempts=10,
        sample_rate_hz=10.0, validation_poses=5, seed=3)
    values.update(overrides)
    return campaign.CampaignPlan(**values)


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.arm = arm_model()

    def test_design_limits_take_the_tighter_of_model_and_plant(self):
        plan = small_plan()
        plant_limits = (np.full(7, -30.0), np.full(7, 30.0))
        limits = plan.design_limits(self.arm, plant_limits)
        self.assertTrue(np.all(limits.lower_deg >= -30.0))
        self.assertTrue(np.all(limits.upper_deg <= 30.0))

    def test_plan_serialises_for_the_report(self):
        payload = small_plan().as_dict()
        self.assertEqual(payload["static_poses"], 6)
        self.assertIsInstance(payload["friction_speeds_deg_s"], list)


class PhaseBookkeepingTest(unittest.TestCase):
    def setUp(self):
        self.arm = arm_model()
        self.plant = ScriptedPlant()
        self.campaign = campaign.Campaign(
            self.arm, self.plant, small_plan())

    def test_all_four_phases_run_in_order(self):
        result = self.campaign.run()
        self.assertEqual([r["phase"] for r in result.phases], list(campaign.PHASES))
        self.assertIsNone(result.aborted)

    def test_validation_samples_are_excluded_from_training(self):
        self.campaign.run()
        phases = {record.phase for record in self.campaign.observations}
        self.assertEqual(phases, set(campaign.PHASES))
        holdout = [r for r in self.campaign.observations
                   if r.phase == campaign.PHASE_VALIDATION]
        training = [r for r in self.campaign.observations
                    if r.phase != campaign.PHASE_VALIDATION]
        seen = {tuple(np.round(r.position_deg, 6)) for r in training}
        for record in holdout:
            self.assertNotIn(tuple(np.round(record.position_deg, 6)), seen)

    def test_static_phase_records_no_motion(self):
        self.campaign.run_gravity()
        for record in self.campaign.observations:
            self.assertEqual(max(np.abs(record.velocity_deg_s)), 0.0)
            self.assertEqual(max(np.abs(record.acceleration_deg_s2)), 0.0)

    def test_validation_exercises_friction_and_inertia(self):
        """Static-only validation would leave those columns untested (F1)."""
        self.campaign.run_validation()
        records = [r for r in self.campaign.observations
                   if r.phase == campaign.PHASE_VALIDATION]
        moving = [r for r in records if max(np.abs(r.velocity_deg_s)) > 0.0]
        accelerating = [r for r in records
                        if max(np.abs(r.acceleration_deg_s2)) > 0.0]
        self.assertTrue(moving, "validation never moved: friction untested")
        self.assertTrue(accelerating, "validation never accelerated: inertia untested")

    def test_validation_speeds_are_absent_from_training(self):
        plan = small_plan()
        run = campaign.Campaign(self.arm, ScriptedPlant(), plan)
        trained = set(plan.friction_speeds_deg_s)
        for speed in run._validation_speeds():
            self.assertNotIn(speed, trained)

    def test_validation_speeds_fall_back_when_all_are_trained(self):
        plan = small_plan(friction_speeds_deg_s=(3.5, 6.5))
        run = campaign.Campaign(self.arm, ScriptedPlant(), plan)
        speeds = run._validation_speeds()
        self.assertTrue(speeds)
        self.assertTrue(all(0.0 < s <= plan.maximum_speed_deg_s for s in speeds))

    def test_friction_phase_visits_both_directions(self):
        self.campaign.run_friction()
        signs = set()
        for record in self.campaign.observations:
            for value in record.velocity_deg_s:
                if value:
                    signs.add(np.sign(value))
        self.assertEqual(signs, {-1.0, 1.0})

    def test_inertia_phase_carries_acceleration(self):
        report = self.campaign.run_inertia()
        if report.aborted:
            self.skipTest(report.aborted)
        peak = max(max(np.abs(r.acceleration_deg_s2))
                   for r in self.campaign.observations)
        self.assertGreater(peak, 0.0)


class BlindSceneRefusesUnscreenedPosturesTest(unittest.TestCase):
    """A scene whose link meshes failed to load keeps answering, and answers
    clear to everything. Home survives that because it is where the arm already
    is; postures drawn from the whole range do not."""

    class Scene:
        def __init__(self, sees_arm: bool):
            self.sees_arm = sees_arm

        def geometry_report(self) -> dict:
            return {"self_collision_checked": self.sees_arm,
                    "robot_shapes": 4 if self.sees_arm else 0}

        def collision_free(self, _pose_deg) -> bool:
            return True

    def _run(self, scene) -> dict:
        plant = ScriptedPlant()
        plant.collision_model = scene
        run = campaign.Campaign(arm_model(), plant,
                                small_plan(friction_postures=3))
        return run.run_friction().detail

    def test_a_blind_scene_falls_back_to_home_and_says_so(self):
        detail = self._run(self.Scene(sees_arm=False))
        self.assertEqual(detail["postures"], 1)
        self.assertIn("collision geometry unavailable", detail["downgraded"])

    def test_a_seeing_scene_sweeps_every_posture(self):
        detail = self._run(self.Scene(sees_arm=True))
        self.assertEqual(detail["postures"], 3)
        self.assertNotIn("downgraded", detail)

    def test_no_scene_at_all_is_treated_as_blind(self):
        plant = ScriptedPlant()
        self.assertIsNone(getattr(plant, "collision_model", None))
        run = campaign.Campaign(arm_model(), plant,
                                small_plan(friction_postures=3))
        self.assertEqual(run.run_friction().detail["postures"], 1)

    def test_the_rehearsal_plant_answers_the_same_question_as_hardware(self):
        """The rehearsal exists to exercise the run that follows it. If the two
        plants expose their scene under different names, the gate reads blind on
        one of them and the rehearsal designs a different campaign than the one
        it is meant to rehearse."""
        from robot_parameter_identification.plants.analytic import AnalyticPlant

        scene = self.Scene(sees_arm=True)
        rehearsal = AnalyticPlant(arm_model().model, rm75_profile(),
                                  collision_scene=scene)
        self.assertIs(rehearsal.collision_model, scene)
        run = campaign.Campaign(arm_model(), rehearsal,
                                small_plan(friction_postures=3))
        self.assertTrue(run._self_collision_checked())


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.arm = arm_model()

    def test_temperature_ceiling_aborts_and_keeps_the_data(self):
        plant = ScriptedPlant(temperature_c=48.0)
        run = campaign.Campaign(
            self.arm, plant, small_plan(temperature_ceiling_c=45.0))
        result = run.run()
        self.assertIsNotNone(result.aborted)
        self.assertIn("48.0", result.aborted)
        self.assertFalse(result.complete)

    def test_operator_stop_aborts(self):
        plant = ScriptedPlant()
        seen = {"count": 0}

        def should_stop():
            seen["count"] += 1
            return seen["count"] > 3

        result = campaign.Campaign(
            self.arm, plant, small_plan(), should_stop=should_stop).run()
        self.assertEqual(result.aborted, "operator stop")

    def test_progress_is_reported_per_pose(self):
        updates = []
        campaign.Campaign(
            self.arm, ScriptedPlant(), small_plan(),
            progress=lambda phase, detail: updates.append(phase)).run()
        self.assertIn(campaign.PHASE_GRAVITY, updates)


class MonitorTest(unittest.TestCase):
    """Phases A-C are position controlled, so current is watched, not commanded."""

    def setUp(self):
        self.arm = arm_model()
        self.limits = StubLimits()

    def test_measured_current_above_the_envelope_aborts(self):
        over = max(self.limits.peak_current_a) + 1.0
        plant = ScriptedPlant(current_a=over, instrumented=True)
        result = campaign.Campaign(
            self.arm, plant, small_plan(),
            monitor=StubMonitor(self.limits)).run()
        self.assertIsNotNone(result.aborted)
        self.assertIn("current", result.aborted)

    def test_current_inside_the_envelope_does_not_abort(self):
        plant = ScriptedPlant(
            current_a=min(self.limits.continuous_current_a) * 0.5,
            instrumented=True)
        result = campaign.Campaign(
            self.arm, plant, small_plan(),
            monitor=StubMonitor(self.limits)).run()
        self.assertIsNone(result.aborted)

    def test_a_measured_over_current_trips_even_when_other_signals_are_missing(self):
        """The current reading is real whether or not the bus voltage is
        published, and this arm does not publish it. Standing the over-current
        guard down because an unrelated channel is absent would disable it on
        exactly the hardware it is there to protect."""
        plant = ScriptedPlant(current_a=99.0, instrumented=False)
        result = campaign.Campaign(
            self.arm, plant, small_plan(),
            monitor=StubMonitor(self.limits)).run()
        self.assertIsNotNone(result.aborted)
        self.assertIn("current", result.aborted)

    def test_monitor_sees_every_instrumented_frame(self):
        monitor = StubMonitor(self.limits)
        plant = ScriptedPlant(current_a=0.1, instrumented=True)
        campaign.Campaign(self.arm, plant, small_plan(), monitor=monitor).run()
        self.assertGreater(monitor.checked, 0)

    def test_pose_admission_hook_filters_the_design(self):
        rejected = []

        def admissible(pose_deg):
            allowed = float(np.max(np.abs(pose_deg))) < 40.0
            if not allowed:
                rejected.append(pose_deg)
            return allowed

        run = campaign.Campaign(
            self.arm, ScriptedPlant(), small_plan(),
            pose_admissible=admissible)
        run.run()
        self.assertTrue(rejected, "hook was never consulted")
        for record in run.observations:
            self.assertLess(float(np.max(np.abs(record.position_deg))), 40.0)


@unittest.skipIf(simulation is None, f"needs the simulated plant: {SIMULATION_MISSING}")
class MujocoCampaignTest(unittest.TestCase):
    """The gate that must pass before the arm is allowed to move."""

    def setUp(self):
        self.arm = arm_model()
        # The plant reverses over the same width the plan assumes, so recovering
        # the injected Coulomb term is a real round trip rather than a fudge.
        self.simulated = simulation.SimulatedPlant(
            scene_path(), profile=rm75_profile(), current_noise_a=0.002,
            torque_to_current=GAINS, coulomb_a=COULOMB,
            coulomb_transition_deg_s=campaign.CampaignPlan()
            .coulomb_transition_deg_s,
            viscous_a_per_deg_s=VISCOUS)
        self.placement = (
            self.arm.model.jointPlacements[1].translation.copy(),
            self.arm.model.jointPlacements[1].rotation.copy(),
        )
        self.arm.align_base(*self.simulated.base_placement())
        self.plant = simulation.SimulatedCampaignPlant(self.simulated)

    def tearDown(self):
        self.arm.align_base(*self.placement)

    def test_campaign_identifies_a_model_that_generalises(self):
        run = campaign.Campaign(
            self.arm, self.plant,
            small_plan(static_poses=14, static_candidates=40,
                       fourier_duration_s=12.0, validation_poses=10))
        result = run.run()

        self.assertIsNone(result.aborted)
        self.assertTrue(result.complete)
        self.assertEqual(len(result.joints), 7)
        self.assertGreaterEqual(result.validation_samples, 8)

        for joint, entry in enumerate(result.joints):
            self.assertLessEqual(
                entry["condition_number"], campaign.MAXIMUM_CONDITION * 1.01)
            self.assertAlmostEqual(
                entry["friction"]["coulomb"],
                self.simulated.coulomb_a[joint], delta=0.02,
                msg=f"joint{joint + 1} Coulomb term")
            self.assertLess(
                entry["validation_rms_a"], 0.15,
                f"joint{joint + 1} did not generalise: "
                f"{entry['validation_rms_a']} A")

    def test_holdout_error_tracks_training_error(self):
        """Divergence between the two is the overfitting signature to catch."""
        run = campaign.Campaign(
            self.arm, self.plant,
            small_plan(static_poses=12, static_candidates=30,
                       fourier_duration_s=10.0, validation_poses=8))
        result = run.run()
        for entry in result.joints:
            self.assertIsNotNone(entry["holdout_rms_a"])
            self.assertLess(
                entry["holdout_rms_a"], entry["residual_rms_a"] + 0.05)


if __name__ == "__main__":
    unittest.main()


class DwellingPlant(ScriptedPlant):
    """A plant that can resample without being commanded again."""

    def dwell(self, pose_deg):
        self.calls.append("dwell")
        return self._frame(pose_deg, np.zeros(self.joints))


class GravityCommandsOncePerPoseTest(unittest.TestCase):
    """Repeat readings must not each cost another trajectory goal.

    A zero-distance goal still pays a full minimum segment plus settle time on
    real hardware, so re-commanding a pose the arm already holds buys nothing
    and spends minutes of arm time.
    """

    def setUp(self):
        self.arm = arm_model()

    def _gravity(self, plant, samples):
        run = campaign.Campaign(
            self.arm, plant, small_plan(settle_samples=samples))
        report = run.run_gravity()
        return run, report

    def test_one_goal_per_pose_regardless_of_sample_count(self):
        plant = DwellingPlant()
        run, report = self._gravity(plant, samples=4)
        poses = report.detail["poses"]
        self.assertEqual(plant.calls.count("hold"), poses)
        self.assertEqual(plant.calls.count("dwell"), poses * 3)
        self.assertEqual(len(run.observations), poses * 4)

    def test_single_sample_never_dwells(self):
        plant = DwellingPlant()
        run, report = self._gravity(plant, samples=1)
        self.assertEqual(plant.calls.count("dwell"), 0)
        self.assertEqual(len(run.observations), report.detail["poses"])

    def test_a_plant_without_dwell_still_works(self):
        plant = ScriptedPlant()
        run, report = self._gravity(plant, samples=3)
        poses = report.detail["poses"]
        self.assertEqual(plant.calls.count("hold"), poses * 3)
        self.assertEqual(len(run.observations), poses * 3)


class WorkspaceCapTest(unittest.TestCase):
    """Calibration may be confined inside the mechanical limits.

    The collision model cannot describe obstacles nobody modelled, so the
    envelope is the one lever that shrinks the swept volume without one.
    """

    def setUp(self):
        self.arm = arm_model()

    def test_the_cap_tightens_the_design_range(self):
        plan = small_plan(workspace_limit_deg=(30.0,) * 7)
        low, high = plan.design_limits(self.arm).usable()
        self.assertTrue(np.all(low >= -30.0))
        self.assertTrue(np.all(high <= 30.0))

    def test_the_cap_cannot_widen_beyond_the_machine(self):
        wide = small_plan(workspace_limit_deg=(1e4,) * 7)
        bare = small_plan()
        np.testing.assert_allclose(
            wide.design_limits(self.arm).usable(),
            bare.design_limits(self.arm).usable())

    def test_no_cap_leaves_the_range_alone(self):
        plan = small_plan()
        low, high = plan.design_limits(self.arm).usable()
        arm_low, arm_high = self.arm.limits_deg()
        np.testing.assert_allclose(low, arm_low + plan.position_margin_deg)
        np.testing.assert_allclose(high, arm_high - plan.position_margin_deg)

    def test_a_capped_plan_keeps_poses_inside_the_cap(self):
        plan = small_plan(workspace_limit_deg=(45.0,) * 7)
        design = campaign.excitation.design_static_poses(
            self.arm, plan.design_limits(self.arm), count=6,
            candidates=30, seed=2)
        self.assertTrue(design.poses_deg)
        for pose in design.poses_deg:
            self.assertLessEqual(max(abs(v) for v in pose), 45.0)

    def test_the_cap_survives_serialisation(self):
        plan = small_plan(workspace_limit_deg=(90.0,) * 7)
        self.assertEqual(plan.as_dict()["workspace_limit_deg"], [90.0] * 7)


class StictionCancellingProbeTest(unittest.TestCase):
    """Standing still cannot separate gravity from stiction; crossing can.

    A joint held at rest balances gravity with any value inside its stiction
    band, so the reading is gravity plus an unknowable offset. Crossing the
    pose both ways gives friction a determined sign, and the pair cancels it.
    """

    GRAVITY = 1.0
    STICTION = 0.3

    class StickyPlant(ScriptedPlant):
        def hold_pose(self, pose_deg):
            self.calls.append("hold")
            frame = self._frame(pose_deg, np.zeros(self.joints))
            bias = StictionCancellingProbeTest.STICTION
            frame["current_a"] = [
                StictionCancellingProbeTest.GRAVITY + bias] * self.joints
            return frame

        def probe_pose(self, pose_deg, delta_deg, speed_deg_s):
            self.calls.append("probe")
            frames = []
            for sign in (1.0, -1.0):
                frame = self._frame(
                    pose_deg, np.full(self.joints, sign * speed_deg_s))
                frame["current_a"] = [
                    StictionCancellingProbeTest.GRAVITY
                    + sign * StictionCancellingProbeTest.STICTION] * self.joints
                frames.append(frame)
            return frames

    def _gravity_run(self, plant):
        run = campaign.Campaign(arm_model(), plant, small_plan())
        report = run.run_gravity()
        return run, report

    def test_the_crossing_pair_averages_to_gravity(self):
        run, _ = self._gravity_run(self.StickyPlant())
        currents = np.array([o.current_a[0] for o in run.observations])
        self.assertAlmostEqual(float(currents.mean()), self.GRAVITY, places=6)
        # A standstill reading alone would sit a whole stiction band away.
        self.assertAlmostEqual(float(currents.max()),
                               self.GRAVITY + self.STICTION, places=6)

    def test_both_directions_are_recorded(self):
        run, _ = self._gravity_run(self.StickyPlant())
        speeds = np.array([o.velocity_deg_s[0] for o in run.observations])
        self.assertTrue((speeds > 0).any() and (speeds < 0).any())
        self.assertFalse((speeds == 0).any())

    def test_a_plant_that_cannot_probe_still_runs(self):
        plant = ScriptedPlant()
        run, report = self._gravity_run(plant)
        self.assertEqual(plant.calls.count("probe"), 0)
        self.assertEqual(len(run.observations),
                         report.detail["poses"] * small_plan().settle_samples)


class SlowFramesAreDroppedTest(unittest.TestCase):
    """Ramp frames near zero speed carry no usable friction sign.

    Keeping them would tell the fit "current = gravity + 0 friction" at exactly
    the moments where friction is least determined, which is the contamination
    the probe exists to remove.
    """

    class RampingPlant(ScriptedPlant):
        def probe_pose(self, pose_deg, delta_deg, speed_deg_s):
            frames = []
            for sign in (1.0, -1.0):
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                    frames.append(self._frame(
                        pose_deg,
                        np.full(self.joints, sign * fraction * speed_deg_s)))
            return frames

    def test_only_frames_above_the_floor_survive(self):
        plan = small_plan()
        run = campaign.Campaign(arm_model(), self.RampingPlant(), plan)
        kept = run._probe(np.zeros(7))
        floor = campaign.PROBE_SPEED_FRACTION * plan.gravity_probe_speed_deg_s
        self.assertTrue(kept)
        for frame in kept:
            self.assertGreaterEqual(max(abs(v) for v in frame["speed_deg_s"]),
                                    floor)

    def test_both_directions_survive_the_filter(self):
        run = campaign.Campaign(arm_model(), self.RampingPlant(), small_plan())
        speeds = [frame["speed_deg_s"][0] for frame in run._probe(np.zeros(7))]
        self.assertTrue(any(s > 0 for s in speeds))
        self.assertTrue(any(s < 0 for s in speeds))
