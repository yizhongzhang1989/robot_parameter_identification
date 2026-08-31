"""The load sweep: what it designs, and what it refuses to design.

The arm is not needed for any of this. Everything here is arithmetic on a model
plus a fake plant, which is the reason the planning was split away from the
driving in the first place.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_parameter_identification import identification as ident
from robot_parameter_identification import loadsweep
from robot_parameter_identification import loadsweep_report
from robot_parameter_identification.interfaces import MotionFailed
from robot_parameter_identification.loadsweep_run import (LoadSweepRun,
                                                          TransitBlocked)
from robot_parameter_identification.obstacles import ObstacleScene
from robot_parameter_identification.plants.ros_control import HardwareConfig

from fixtures import PREFIX, synthetic_urdf


def build_arm():
    return ident.ArmModel.from_urdf_text(synthetic_urdf(), PREFIX)


class FakePlant:
    """Answers motions without a robot, and can be told to refuse."""

    def __init__(self, joints: int, refuse=()) -> None:
        self.joints = joints
        self.refuse = set(refuse)
        self.pose = np.zeros(joints)
        self.parked = 0
        self.held: list[np.ndarray] = []
        self.driven: list[tuple] = []
        self.attempts = 0

    def sample(self):
        return {"position_deg": list(self.pose)}

    def park(self):
        self.parked += 1
        self.pose = np.zeros(self.joints)

    def hold_pose(self, pose_deg):
        self.pose = np.asarray(pose_deg, dtype=float).copy()
        self.held.append(self.pose.copy())

    def traverse(self, joint, start_deg, distance_deg, speed_deg_s):
        self.attempts += 1
        if self.attempts in self.refuse:
            raise MotionFailed("the controller rejected the trajectory")
        self.pose = np.asarray(start_deg, dtype=float).copy()
        self.pose[joint] += distance_deg
        self.driven.append((joint, float(distance_deg), float(speed_deg_s)))
        position = np.asarray(start_deg, dtype=float).copy()
        position[joint] += distance_deg / 2.0
        yield {"position_deg": list(position),
               "speed_deg_s": [speed_deg_s if i == joint else 0.0
                               for i in range(self.joints)],
               "acceleration_deg_s2": [0.0] * self.joints,
               "current_a": [0.3] * self.joints,
               "temperature_c": [35.0] * self.joints,
               "window_frames": 20, "window_fit_rms_deg": 0.001}


class BlindScene:
    """A collision screen with nothing in it, which passes everything."""

    def geometry_report(self):
        return {"self_collision_checked": False, "robot_geometry_error": ""}

    def collision_free(self, _pose):
        return True


class WallScene:
    """Refuses anything past a joint angle, so paths through it are blocked."""

    def __init__(self, joint: int, limit: float) -> None:
        self.joint, self.limit = joint, limit

    def geometry_report(self):
        return {"self_collision_checked": True}

    def collision_free(self, pose):
        return abs(float(np.asarray(pose, dtype=float)[self.joint])) <= self.limit


class LoadGeometryTest(unittest.TestCase):

    def setUp(self):
        self.arm = build_arm()

    def test_window_arc_is_capped_not_proportional_to_speed(self):
        """The fitted window is sized in degrees, so a fast pass does not
        average across a wider arc than a slow one."""
        config = HardwareConfig()
        arcs = [loadsweep.window_arc_deg(v, config)
                for v in (0.5, 2.0, 10.0, 60.0)]
        self.assertLessEqual(max(arcs), config.window_arc_deg + 1e-9)
        self.assertAlmostEqual(loadsweep.window_arc_deg(0.0, config), 0.0)

    def test_axis_gravity_angle_is_a_real_angle(self):
        for joint in range(self.arm.joint_count):
            angle = loadsweep.axis_gravity_deg(
                self.arm, np.zeros(self.arm.joint_count), joint)
            self.assertTrue(0.0 <= angle <= 90.0, angle)

    def test_load_across_reports_the_swing_not_just_the_middle(self):
        pose = np.zeros(self.arm.joint_count)
        mean, drift = loadsweep.load_across(self.arm, pose, 1, 60.0)
        self.assertGreater(drift, 0.0)
        self.assertGreaterEqual(mean, 0.0)
        _, still = loadsweep.load_across(self.arm, pose, 1, 0.0)
        self.assertEqual(still, 0.0)

    def test_path_free_checks_the_middle_not_only_the_ends(self):
        """Both ends clear of a wall does not make the straight move clear."""
        scene = WallScene(joint=0, limit=20.0)
        start = np.zeros(self.arm.joint_count)
        end = np.zeros(self.arm.joint_count)
        start[0], end[0] = -10.0, 10.0
        self.assertTrue(loadsweep.path_free(scene, start, end))
        far = np.zeros(self.arm.joint_count)
        far[0] = 90.0
        self.assertFalse(loadsweep.path_free(scene, start, far))

    def test_a_blind_screen_is_refused(self):
        """A scene that cannot see the arm passes every pose, which is
        indistinguishable from a clear path until the arm closes on itself."""
        with self.assertRaises(RuntimeError):
            loadsweep.proven_scene(BlindScene(), self.arm.joint_count)
        with self.assertRaises(RuntimeError):
            loadsweep.proven_scene(None, self.arm.joint_count)


class DesignTest(unittest.TestCase):

    def setUp(self):
        self.arm = build_arm()
        self.scene = ObstacleScene(self.arm.model, urdf_text=synthetic_urdf())
        self.config = HardwareConfig()
        self.limits = self.arm.limits_deg()

    def design(self, joint, **kwargs):
        plan = loadsweep.SweepPlan(search_samples=600, refine_steps=40,
                                   speeds=6, **kwargs)
        return loadsweep.design_joint(self.arm, self.scene, joint, plan,
                                      self.config, *self.limits)

    def test_levels_are_ordered_and_inside_the_reachable_range(self):
        design = self.design(1)
        loads = [level.load_nm for level in design.levels]
        self.assertEqual(loads, sorted(loads))
        low, high = design.reachable_nm
        for load in loads:
            self.assertGreaterEqual(load, low - 1e-6)
            self.assertLessEqual(load, high + 1e-6)

    def test_level_count_follows_the_span_not_the_setting(self):
        """A joint gravity can barely load must not be given ten levels; they
        would be ten readings of one measurement and a slope fitted to noise."""
        wide = self.design(1, minimum_level_gap_nm=0.01)
        narrow = self.design(1, minimum_level_gap_nm=1e6)
        self.assertGreater(len(wide.levels), len(narrow.levels))
        self.assertEqual(len(narrow.levels), 1)
        self.assertFalse(narrow.loadable)
        self.assertIn("below the resolution", narrow.note)

    def test_maximum_levels_is_a_ceiling(self):
        design = self.design(1, maximum_levels=3, minimum_level_gap_nm=1e-6)
        self.assertLessEqual(len(design.levels), 3)

    def test_every_pass_stays_inside_the_joint_limits(self):
        design = self.design(1)
        low, high = self.limits
        levels = {level.index: level for level in design.levels}
        for entry in design.passes:
            centre = levels[entry["level"]].pose_deg[1]
            half = entry["arc_deg"] / 2.0
            self.assertGreaterEqual(centre - half, low[1] - 1e-6)
            self.assertLessEqual(centre + half, high[1] + 1e-6)

    def test_passes_hold_their_load_still_across_the_fitted_window(self):
        design = self.design(1)
        self.assertTrue(design.passes)
        gap = design.span_nm / max(len(design.levels) - 1, 1)
        for entry in design.passes:
            self.assertLessEqual(entry["load_drift_nm"], 0.35 * gap + 1e-6)

    def test_a_collision_scene_that_refuses_everything_yields_no_passes(self):
        blocked = WallScene(joint=1, limit=-1.0)
        plan = loadsweep.SweepPlan(search_samples=400, refine_steps=20, speeds=4)
        design = loadsweep.design_joint(self.arm, blocked, 1, plan,
                                        self.config, *self.limits)
        self.assertEqual(design.passes, [])


class RunTest(unittest.TestCase):

    def setUp(self):
        self.arm = build_arm()
        self.scene = ObstacleScene(self.arm.model, urdf_text=synthetic_urdf())
        plan = loadsweep.SweepPlan(search_samples=400, refine_steps=20,
                                   speeds=3, repeats=1, maximum_levels=2)
        self.plan = plan
        self.designs = [loadsweep.design_joint(
            self.arm, self.scene, 1, plan, HardwareConfig(),
            *self.arm.limits_deg())]

    def run_sweep(self, folder, plant=None, scene=None, plan=None):
        plant = plant or FakePlant(self.arm.joint_count)
        run = LoadSweepRun(self.arm, plant, plan or self.plan, self.designs,
                           folder, scene=scene)
        return run, run.run(), plant

    def test_every_record_is_on_disk_and_readable(self):
        with tempfile.TemporaryDirectory() as folder:
            run, outcome, _ = self.run_sweep(folder)
            lines = (Path(folder) / loadsweep.RECORDS_NAME).read_text(
                encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), outcome["driven"])
            first = json.loads(lines[0])
            for field in ("joint", "level", "speed_deg_s", "direction",
                          "current_a", "load", "pose_deg", "key"):
                self.assertIn(field, first)
            self.assertIn("axial_nm", first["load"])

    def test_the_manifest_records_the_design_before_anything_moves(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_sweep(folder)
            manifest = json.loads(
                (Path(folder) / loadsweep.MANIFEST_NAME).read_text("utf-8"))
            self.assertEqual(manifest["kind"], "load_sweep")
            self.assertEqual(len(manifest["joints"]), 1)
            self.assertIn("levels", manifest["joints"][0])

    def test_a_resumed_run_drives_only_what_is_missing(self):
        """Three hours of records must not be measured twice because the run
        was interrupted in its third hour."""
        with tempfile.TemporaryDirectory() as folder:
            _, first, _ = self.run_sweep(folder)
            self.assertGreater(first["driven"], 0)
            _, second, plant = self.run_sweep(folder)
            self.assertEqual(second["driven"], 0)
            self.assertEqual(plant.driven, [])

    def test_a_refused_motion_is_retried_then_recorded_as_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            plant = FakePlant(self.arm.joint_count, refuse={1, 2, 3})
            run, outcome, _ = self.run_sweep(folder, plant=plant)
            self.assertTrue(outcome["skipped"])
            self.assertEqual(outcome["skipped"][0]["why"],
                             "the controller rejected the trajectory")
            # The pass after the refused one still ran.
            self.assertGreater(outcome["driven"], 0)

    def test_a_transient_refusal_is_survived(self):
        with tempfile.TemporaryDirectory() as folder:
            plant = FakePlant(self.arm.joint_count, refuse={1})
            _, outcome, _ = self.run_sweep(folder, plant=plant)
            self.assertEqual(outcome["skipped"], [])
            self.assertGreater(outcome["driven"], 0)

    def test_an_arm_that_accepts_goals_without_moving_is_not_measured(self):
        """The failure mode that has no error in it: the drive reports itself
        enabled and unfaulted, the controller reports the goal succeeded, and
        the joint never turns. Every part did its job and the row is fiction."""

        class Stalled(FakePlant):
            def traverse(self, joint, start_deg, distance_deg, speed_deg_s):
                for frame in super().traverse(joint, start_deg, distance_deg,
                                              speed_deg_s):
                    frame["speed_deg_s"] = [0.01] * self.joints
                    yield frame

        with tempfile.TemporaryDirectory() as folder:
            plant = Stalled(self.arm.joint_count)
            _, outcome, _ = self.run_sweep(folder, plant=plant)
            self.assertEqual(outcome["driven"], 0)
            self.assertTrue(outcome["skipped"])
            self.assertIn("not following commands", outcome["skipped"][0]["why"])
            self.assertFalse(
                (Path(folder) / loadsweep.RECORDS_NAME).read_text("utf-8").strip())

    def test_the_arm_is_parked_even_when_the_sweep_throws(self):
        class Exploding(FakePlant):
            def traverse(self, *args, **kwargs):
                raise RuntimeError("bus fell over")

        with tempfile.TemporaryDirectory() as folder:
            plant = Exploding(self.arm.joint_count)
            run = LoadSweepRun(self.arm, plant, self.plan, self.designs, folder)
            with self.assertRaises(RuntimeError):
                run.run()
            self.assertGreater(plant.parked, 0)

    def test_a_blocked_transit_is_refused_rather_than_driven_through(self):
        """Both postures clear does not make the move between them clear."""
        with tempfile.TemporaryDirectory() as folder:
            plant = FakePlant(self.arm.joint_count)
            run = LoadSweepRun(self.arm, plant, self.plan, self.designs,
                               folder, scene=WallScene(joint=1, limit=-1.0))
            outcome = run.run()
            self.assertEqual(outcome["driven"], 0)
            self.assertTrue(outcome["skipped"])
            self.assertIn("no collision free path",
                          outcome["skipped"][0]["why"])

    def test_stopping_is_honoured_mid_sweep(self):
        with tempfile.TemporaryDirectory() as folder:
            plant = FakePlant(self.arm.joint_count)
            run = LoadSweepRun(self.arm, plant, self.plan, self.designs,
                               folder, should_stop=lambda: True)
            outcome = run.run()
            self.assertEqual(outcome["driven"], 0)

    def test_a_report_is_written_beside_the_records(self):
        with tempfile.TemporaryDirectory() as folder:
            self.run_sweep(folder)
            page = Path(folder) / loadsweep_report.REPORT_NAME
            self.assertTrue(page.is_file())
            text = page.read_text(encoding="utf-8")
            self.assertIn("<canvas", text)
            # Both languages ship inside the page; neither can go stale
            # against the other because there is no second file.
            self.assertIn("负载扫掠报告", text)
            self.assertIn("Load sweep", text)

    def test_the_signed_torque_is_recorded_not_just_its_size(self):
        """joint_loads reports magnitudes, which cannot calibrate anything
        once the search picks postures either side of zero."""
        with tempfile.TemporaryDirectory() as folder:
            self.run_sweep(folder)
            first = json.loads((Path(folder) / loadsweep.RECORDS_NAME)
                               .read_text("utf-8").splitlines()[0])
            self.assertIn("axial_signed_nm", first["load"])
            self.assertGreaterEqual(first["load"]["axial_nm"], 0.0)


class ReportTest(unittest.TestCase):
    """What the page says about data it cannot honestly speak for."""

    def setUp(self):
        self.arm = build_arm()
        self.scene = ObstacleScene(self.arm.model, urdf_text=synthetic_urdf())
        self.plan = loadsweep.SweepPlan(search_samples=400, refine_steps=20,
                                        speeds=8, repeats=1, maximum_levels=2)
        self.designs = [loadsweep.design_joint(
            self.arm, self.scene, 1, self.plan, HardwareConfig(),
            *self.arm.limits_deg())]

    def story(self, folder):
        run = LoadSweepRun(self.arm, FakePlant(self.arm.joint_count),
                           self.plan, self.designs, folder)
        run.run()
        return loadsweep_report.summarise(folder)

    def test_a_torque_constant_that_does_not_fit_is_borrowed_not_invented(self):
        """The fake plant returns the same current at every posture, so its
        gravity current cannot track its gravity torque and nothing about it
        is calibrated. Reporting a slope anyway is how joint three came to be
        credited with friction worth 1370 per cent of its load."""
        with tempfile.TemporaryDirectory() as folder:
            story = self.story(folder)
            for joint in story["joints"]:
                self.assertTrue(joint["amps_per_nm_borrowed"])
                self.assertIsNone(joint["amps_per_nm"])
                self.assertGreater(joint["amps_per_nm_used"], 0.0)

    def test_the_page_stands_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            page = loadsweep_report.render(self.story(folder))
            self.assertNotIn("<script src", page)
            self.assertNotIn("http://", page)
            self.assertIn("</html>", page)

    def test_a_run_with_no_records_does_not_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            run = LoadSweepRun(self.arm, FakePlant(self.arm.joint_count),
                               self.plan, self.designs, folder,
                               should_stop=lambda: True)
            run.run()
            story = loadsweep_report.summarise(folder)
            self.assertEqual(story["records"], 0)
            self.assertTrue(loadsweep_report.render(story))

    def test_sweep_model_is_scored_on_external_validation_samples(self):
        class OneJointArm:
            joint_count = 1
            joint_names = ["joint1"]

            @staticmethod
            def inverse_dynamics(_position, _velocity, _acceleration):
                return np.array([2.0])

            @staticmethod
            def joint_loads(_position):
                return np.array([[3.0, 0.0, 0.0, 0.0]])

        friction = {"width": 1.0, "viscous": 0.0,
                    "c0": 0.2, "ck": 0.0, "d0": 0.0, "dk": 0.0,
                    "vs0": 1.0, "vsk": 0.0}
        model = {"joints": [{
            "joint": 0, "name": "joint1", "fit": friction,
            "signed_amps_per_nm": 0.5, "amps_per_nm_used": 0.5,
            "effort_offset_a": 0.1,
        }]}
        speed = 2.0
        expected = 0.5 * 2.0 + 0.1 + float(
            loadsweep_report.curve(friction, speed, 3.0))
        observations = [SimpleNamespace(
            position_deg=[0.0], velocity_deg_s=[speed],
            acceleration_deg_s2=[0.0], current_a=[expected])]

        scored = loadsweep_report.score_validation(
            OneJointArm(), observations, model)

        self.assertTrue(scored["available"])
        self.assertAlmostEqual(scored["validation_rms_a"][0], 0.0, places=12)

    def test_legacy_rows_recover_signed_torque_from_the_saved_pose(self):
        class Arm:
            @staticmethod
            def inverse_dynamics(pose_deg):
                return np.array([float(pose_deg[0])])

        rows = []
        for direction, current in (("+", 2.2), ("-", 1.8)):
            rows.append({
                "joint": 0, "level": 1, "speed_deg_s": 2.0,
                "direction": direction, "current_a": current,
                "temperature_c": 30.0, "pose_deg": [-3.0],
                "load": {"axial_nm": 3.0},
            })
        table = loadsweep_report.split(rows, arm=Arm())
        self.assertEqual(table[0]["signed"], -3.0)


if __name__ == "__main__":
    unittest.main()
