"""Experiment design has to beat random motion, or it is not worth running."""

import unittest

import numpy as np

try:
    from robot_parameter_identification import excitation
    from robot_parameter_identification import identification as ident
    from test_identification import arm_model
except ImportError as error:
    raise unittest.SkipTest(f"excitation needs pinocchio: {error}") from error


def design_limits(arm, **overrides):
    low, high = arm.limits_deg()
    return excitation.DesignLimits(low, high, **overrides)


class LimitTest(unittest.TestCase):
    def setUp(self):
        self.arm = arm_model()
        self.limits = design_limits(self.arm, margin_deg=5.0)

    def test_usable_range_keeps_the_margin(self):
        low, high = self.limits.usable()
        raw_low, raw_high = self.arm.limits_deg()
        np.testing.assert_allclose(low, raw_low + 5.0)
        np.testing.assert_allclose(high, raw_high - 5.0)

    def test_clip_never_leaves_the_usable_range(self):
        low, high = self.limits.usable()
        clipped = self.limits.clip(np.full(7, 1e4))
        self.assertTrue(np.all(clipped <= high + 1e-9))
        self.assertTrue(np.all(self.limits.clip(np.full(7, -1e4)) >= low - 1e-9))


class StaticDesignTest(unittest.TestCase):
    def setUp(self):
        self.arm = arm_model()
        self.limits = design_limits(self.arm, margin_deg=5.0)

    def test_designed_poses_beat_random_poses(self):
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=15, candidates=60, seed=1)
        self.assertEqual(len(plan.poses_deg), 15)

        rng = np.random.default_rng(5)
        low, high = self.limits.usable()
        random_rows = [self.arm.static_regressor(rng.uniform(low, high))
                       for _ in range(15)]
        self.assertLess(
            plan.condition_number,
            ident.stacked_condition_number(random_rows),
        )

    def test_collision_screening_rejects_and_reports(self):
        blocked = {"count": 0}

        def collision_free(pose):
            blocked["count"] += 1
            return blocked["count"] % 3 != 0

        plan = excitation.design_static_poses(
            self.arm, self.limits, count=5, candidates=30, seed=2,
            collision_free=collision_free)
        self.assertGreater(plan.rejected_by_collision, 0)
        self.assertLessEqual(len(plan.poses_deg), 5)

    def test_every_designed_pose_is_inside_the_usable_range(self):
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=10, candidates=40, seed=3)
        low, high = self.limits.usable()
        for pose in plan.poses_deg:
            self.assertTrue(np.all(np.asarray(pose) >= low - 1e-9))
            self.assertTrue(np.all(np.asarray(pose) <= high + 1e-9))


class FrictionDesignTest(unittest.TestCase):
    def setUp(self):
        self.arm = arm_model()
        self.limits = design_limits(self.arm, margin_deg=5.0,
                                    maximum_speed_deg_s=6.0)

    def test_each_joint_gets_a_sweep_within_limits(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0,
            speeds_deg_s=(2.0, 5.0, 20.0))
        self.assertEqual(len(sweeps), 7)
        low, high = self.limits.usable()
        for sweep in sweeps:
            self.assertNotIn(20.0, sweep.speeds_deg_s)
            start = np.asarray(sweep.start_deg)
            self.assertTrue(np.all(start >= low - 1e-9))
            self.assertLessEqual(
                start[sweep.joint] + sweep.amplitude_deg, high[sweep.joint] + 1e-6)


class MultiPostureSweepTest(unittest.TestCase):
    """A joint's gravity load depends on where the other joints are, so one
    posture measures its friction under one load out of many."""

    def setUp(self):
        self.arm = arm_model()
        self.limits = excitation.DesignLimits(
            lower_deg=np.full(self.arm.joint_count, -80.0),
            upper_deg=np.full(self.arm.joint_count, 80.0),
            margin_deg=5.0, maximum_speed_deg_s=60.0,
            maximum_acceleration_deg_s2=240.0)

    def test_one_posture_reproduces_the_old_design(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,))
        self.assertEqual(len(sweeps), self.arm.joint_count)

    def test_three_postures_give_three_sweeps_per_joint(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3)
        for joint in range(self.arm.joint_count):
            mine = [s for s in sweeps if s.joint == joint]
            self.assertEqual(len(mine), 3, f"joint {joint}")

    def test_the_postures_differ_in_load(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3)
        spread = []
        for joint in range(self.arm.joint_count):
            loads = [s.gravity_nm for s in sweeps if s.joint == joint]
            spread.append(max(loads) - min(loads))
        # Some joints genuinely cannot be loaded; at least one must vary.
        self.assertGreater(max(spread), 0.05)

    def test_every_pass_is_screened_along_its_whole_length(self):
        """A posture clear at its centre can still put the arm through an
        obstacle partway along the sweep, where it spends most of its time."""
        seen = []

        def collision_free(pose):
            seen.append(np.asarray(pose, dtype=float).copy())
            return True

        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=30.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=collision_free)
        self.assertTrue(sweeps)
        for sweep in sweeps:
            start = np.asarray(sweep.start_deg, dtype=float)
            end = start.copy()
            end[sweep.joint] += sweep.amplitude_deg
            middle = 0.5 * (start + end)
            for wanted in (start, end, middle):
                self.assertTrue(
                    any(np.allclose(wanted, pose, atol=1e-6) for pose in seen),
                    f"joint {sweep.joint} was never checked at {wanted}")

    def test_a_blocked_sweep_is_dropped_not_driven(self):
        joint = 1

        def collision_free(pose):
            # Anything that moves this joint past a third of the way is barred.
            return abs(float(np.asarray(pose, dtype=float)[joint])) < 3.0

        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=30.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=collision_free)
        self.assertFalse([s for s in sweeps if s.joint == joint])

    def test_a_refused_scene_yields_no_sweeps_rather_than_unchecked_ones(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=lambda _pose: False)
        self.assertEqual(sweeps, [])

    def test_the_choice_is_repeatable(self):
        first = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, seed=4)
        again = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, seed=4)
        self.assertEqual([s.start_deg for s in first],
                         [s.start_deg for s in again])

    def test_every_sweep_stays_inside_the_usable_range(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=40.0, speeds_deg_s=(5.0,),
            postures=3)
        low, high = self.limits.usable()
        for sweep in sweeps:
            start = np.asarray(sweep.start_deg, dtype=float)
            end = start.copy()
            end[sweep.joint] += sweep.amplitude_deg
            self.assertTrue(np.all(start >= low - 1e-6), sweep.joint)
            self.assertTrue(np.all(end <= high + 1e-6), sweep.joint)


class FourierDesignTest(unittest.TestCase):
    def setUp(self):
        self.arm = arm_model()
        self.limits = design_limits(
            self.arm, margin_deg=5.0, maximum_speed_deg_s=10.0,
            maximum_acceleration_deg_s2=20.0)

    def test_trajectory_respects_speed_and_acceleration(self):
        trajectory = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, base_frequency_hz=0.08,
            duration_s=10.0, attempts=12, seed=1)
        self.assertIsNotNone(trajectory)
        low, high = self.limits.usable()
        for step in range(60):
            position, velocity, acceleration = trajectory.sample(
                trajectory.duration_s * step / 60)
            self.assertTrue(np.all(position >= low - 1e-6))
            self.assertTrue(np.all(position <= high + 1e-6))
            self.assertLessEqual(np.max(np.abs(velocity)), 10.0 + 1e-6)
            self.assertLessEqual(np.max(np.abs(acceleration)), 20.0 + 1e-6)

    def test_trajectory_derivatives_are_consistent(self):
        trajectory = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, base_frequency_hz=0.08,
            duration_s=10.0, attempts=12, seed=4)
        self.assertIsNotNone(trajectory)
        step = 1e-5
        before, _v, _a = trajectory.sample(2.0 - step)
        after, _v, _a = trajectory.sample(2.0 + step)
        _p, velocity, _a = trajectory.sample(2.0)
        np.testing.assert_allclose(
            (after - before) / (2 * step), velocity, rtol=1e-4, atol=1e-4)

    def test_collision_blocked_search_returns_nothing(self):
        trajectory = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, duration_s=5.0, attempts=5,
            seed=6, collision_free=lambda _pose: False)
        self.assertIsNone(trajectory)


if __name__ == "__main__":
    unittest.main()


class TourOrderingTest(unittest.TestCase):
    """Ordering is free: it changes travel, never conditioning."""

    def setUp(self):
        self.arm = arm_model()
        low, high = self.arm.limits_deg()
        self.limits = excitation.DesignLimits(low, high, margin_deg=5.0)

    def test_ordering_does_not_change_conditioning(self):
        """Row order is free -- once the identifiable column set is settled.

        Conditioning is scored *after* base-parameter reduction, and that
        reduction keeps columns whose pivot exceeds a tolerance. On an arm whose
        static regressor is strongly rank-deficient the pivots pile up near that
        tolerance, and rounding alone can flip one in or out, which moves the
        reported number. So the invariant that actually holds -- and the one the
        design criterion relies on -- is over a fixed column set.
        """
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=12, candidates=40, seed=3)
        rows = [self.arm.static_regressor(pose) for pose in plan.poses_deg]
        stacked = np.vstack(rows)
        columns = ident.identifiable_columns(stacked, 1e-6)
        forward = np.linalg.cond(stacked[:, columns])
        reversed_rows = np.vstack(list(reversed(rows)))
        backward = np.linalg.cond(reversed_rows[:, columns])
        self.assertAlmostEqual(forward, backward, places=6)

    def test_tour_is_shorter_than_the_selection_order(self):
        import numpy as np

        plan = excitation.design_static_poses(
            self.arm, self.limits, count=16, candidates=50, seed=5)
        poses = [np.asarray(pose) for pose in plan.poses_deg]
        ordered = excitation._tour_length(poses)
        worst = excitation._tour_length(list(reversed(poses)))
        self.assertGreater(len(poses), 3)
        self.assertLessEqual(ordered, worst * 1.05)
        self.assertGreater(plan.travel_deg, 0.0)

    def test_travel_is_reported(self):
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=8, candidates=30, seed=1)
        self.assertIn("travel_deg", plan.as_dict())
        self.assertGreater(plan.as_dict()["travel_deg"], 0.0)

    def test_tour_starts_from_the_given_pose(self):
        import numpy as np

        start = np.full(self.arm.joint_count, 20.0)
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=10, candidates=30, seed=2,
            start_deg=start)
        first = np.asarray(plan.poses_deg[0])
        others = [np.asarray(p) for p in plan.poses_deg[1:]]
        nearest = min(float(np.max(np.abs(p - start))) for p in others)
        self.assertLessEqual(float(np.max(np.abs(first - start))), nearest + 1e-6)


class SweptPathScreeningTest(unittest.TestCase):
    """Clear endpoints do not imply a clear path between them.

    The arm hit its stand travelling between two poses that were each screened
    and accepted; the 162 deg sweep in between was never checked.
    """

    def setUp(self):
        self.arm = arm_model()
        low, high = self.arm.limits_deg()
        self.limits = excitation.DesignLimits(low, high, margin_deg=5.0)

    @staticmethod
    def _wall(lower, upper):
        """Reject poses whose first joint lies inside a forbidden band."""
        def clear(pose):
            return not (lower <= float(pose[0]) <= upper)
        return clear

    def test_a_pose_behind_a_wall_is_dropped(self):
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=8, candidates=40, seed=1,
            collision_free=self._wall(20.0, 40.0))
        for pose in plan.poses_deg:
            self.assertFalse(20.0 <= pose[0] <= 40.0)
        # Every kept pose is on the near side, because the tour starts at zero.
        self.assertTrue(all(pose[0] < 20.0 for pose in plan.poses_deg))
        self.assertGreater(plan.rejected_by_path, 0)

    def test_path_screening_is_reported(self):
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=8, candidates=40, seed=1,
            collision_free=self._wall(20.0, 40.0))
        self.assertIn("rejected_by_path", plan.as_dict())

    def test_an_open_cell_drops_nothing(self):
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=8, candidates=40, seed=1,
            collision_free=lambda pose: True)
        self.assertEqual(plan.rejected_by_path, 0)
        self.assertEqual(len(plan.poses_deg), 8)

    def test_path_free_sees_an_obstacle_between_clear_ends(self):
        start = np.zeros(self.arm.joint_count)
        end = start.copy()
        end[0] = 60.0
        self.assertTrue(self._wall(20.0, 40.0)(start))
        self.assertTrue(self._wall(20.0, 40.0)(end))
        self.assertFalse(
            excitation._path_free(start, end, self._wall(20.0, 40.0)))
