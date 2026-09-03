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


def CLEAR(_pose):  # noqa: N802 - reads as a constant at the call sites
    """A screen that exists and finds nothing wrong, which is not the same as
    having no screen: the second cannot vouch for a posture at all."""
    return True


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


class PathResolutionTest(unittest.TestCase):
    """The gap between checks is what decides whether a thin thing is stepped
    over. A fixed sample count makes that gap grow with the swing."""

    def slab(self, low_deg, high_deg):
        """A screen that refuses one narrow band of joint 1."""
        def screen(pose):
            return not (low_deg <= float(pose[0]) <= high_deg)
        return screen

    def test_a_thin_obstacle_is_not_stepped_over_on_a_long_swing(self):
        start = np.zeros(7)
        end = np.zeros(7)
        end[0] = 300.0
        # One degree wide, in the middle. Thirty-two samples over 300 deg land
        # 9.4 deg apart and miss it; the resolution rule cannot.
        screen = self.slab(149.6, 150.4)
        self.assertFalse(excitation._path_free(start, end, screen))

    def test_the_callers_step_count_is_only_a_floor(self):
        start = np.zeros(7)
        end = np.zeros(7)
        end[0] = 300.0
        screen = self.slab(149.6, 150.4)
        self.assertFalse(excitation._path_free(start, end, screen, steps=4))

    def test_a_clear_path_stays_clear(self):
        start = np.zeros(7)
        end = np.zeros(7)
        end[0] = 300.0
        self.assertTrue(excitation._path_free(start, end, CLEAR))

    def test_the_check_count_is_bounded(self):
        seen = []

        def counting(pose):
            seen.append(pose)
            return True

        start = np.zeros(7)
        end = np.zeros(7)
        end[0] = 1e5
        excitation._path_free(start, end, counting)
        self.assertLessEqual(len(seen), excitation.MAXIMUM_PATH_STEPS)


class CrossingScreenTest(unittest.TestCase):
    """The gravity probe drives +/-delta on every joint from each pose."""

    def setUp(self):
        self.arm = arm_model()
        self.limits = design_limits(self.arm, margin_deg=5.0)

    def test_a_pose_whose_crossing_is_blocked_is_not_selected(self):
        # Everything within 8 deg of the origin on joint 1 is fine; beyond it
        # the crossing would leave the allowed band.
        def screen(pose):
            return abs(float(pose[0])) <= 8.0

        plan = excitation.design_static_poses(
            self.arm, self.limits, count=4, candidates=200, seed=2,
            collision_free=screen, crossing_deg=5.0)
        for pose in plan.poses_deg:
            self.assertLessEqual(abs(pose[0]) + 5.0, 8.0 + 1e-6)

    def test_the_crossing_rejections_are_counted_separately(self):
        def screen(pose):
            return abs(float(pose[0])) <= 8.0

        plan = excitation.design_static_poses(
            self.arm, self.limits, count=4, candidates=60, seed=2,
            collision_free=screen, crossing_deg=5.0)
        self.assertGreater(plan.rejected_by_crossing, 0)
        self.assertIn("rejected_by_crossing", plan.as_dict())

    def test_without_a_crossing_the_design_is_unchanged(self):
        plain = excitation.design_static_poses(
            self.arm, self.limits, count=6, candidates=60, seed=3,
            collision_free=CLEAR)
        same = excitation.design_static_poses(
            self.arm, self.limits, count=6, candidates=60, seed=3,
            collision_free=CLEAR, crossing_deg=0.0)
        self.assertEqual(plain.poses_deg, same.poses_deg)


class StandingStartTest(unittest.TestCase):
    """The tour is planned from where the arm is, not from where zero is."""

    def setUp(self):
        self.arm = arm_model()
        self.limits = design_limits(self.arm, margin_deg=5.0)

    def design(self, start_deg):
        return excitation.design_static_poses(
            self.arm, self.limits, count=6, candidates=60, seed=5,
            collision_free=CLEAR, start_deg=start_deg)

    def test_the_first_pose_is_the_one_nearest_the_start(self):
        # The first transit is the longest and it is screened from here, so
        # "here" has to be where the arm actually is.
        start = np.full(self.arm.joint_count, -40.0)
        plan = self.design(start)
        away = [float(np.max(np.abs(np.asarray(pose) - start)))
                for pose in plan.poses_deg]
        self.assertEqual(away[0], min(away))

    def test_a_different_start_reorders_the_same_poses(self):
        far = self.design(np.full(self.arm.joint_count, -40.0))
        home = self.design(None)
        self.assertEqual(sorted(map(tuple, far.poses_deg)),
                         sorted(map(tuple, home.poses_deg)))
        self.assertNotEqual(far.poses_deg, home.poses_deg)

    def test_the_design_says_where_it_leaves_the_arm(self):
        plan = self.design(None)
        self.assertEqual(plan.final_deg, plan.poses_deg[-1])
        self.assertIn("final_deg", plan.as_dict())

    def test_a_crossing_leaves_the_arm_at_the_sweep_start(self):
        # The arm is left where the crossing began, not on the pose centre.
        plan = excitation.design_static_poses(
            self.arm, self.limits, count=4, candidates=60, seed=5,
            collision_free=CLEAR, crossing_deg=5.0)
        low, high = self.limits.usable()
        self.assertTrue(np.allclose(
            plan.final_deg,
            np.clip(np.asarray(plan.poses_deg[-1]) - 5.0, low, high)))
        self.assertFalse(np.allclose(plan.final_deg, plan.poses_deg[-1]))


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

    def test_without_a_screen_only_home_is_swept(self):
        """Nothing could have made a drawn posture safe, so none is driven."""
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3)
        self.assertEqual(len(sweeps), self.arm.joint_count)
        for sweep in sweeps:
            posture = np.asarray(sweep.start_deg, dtype=float)
            posture[sweep.joint] = 0.0
            self.assertTrue(np.allclose(posture, 0.0),
                            f"joint {sweep.joint} left home unscreened")

    def test_one_posture_reproduces_the_old_design(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,))
        self.assertEqual(len(sweeps), self.arm.joint_count)

    def test_three_postures_give_three_sweeps_per_joint(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=CLEAR)
        for joint in range(self.arm.joint_count):
            mine = [s for s in sweeps if s.joint == joint]
            self.assertEqual(len(mine), 3, f"joint {joint}")

    def test_the_postures_differ_in_load(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=CLEAR)
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
            postures=3, collision_free=CLEAR, seed=4)
        again = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=CLEAR, seed=4)
        self.assertEqual([s.start_deg for s in first],
                         [s.start_deg for s in again])

    def test_every_sweep_stays_inside_the_usable_range(self):
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=40.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=CLEAR)
        low, high = self.limits.usable()
        for sweep in sweeps:
            start = np.asarray(sweep.start_deg, dtype=float)
            end = start.copy()
            end[sweep.joint] += sweep.amplitude_deg
            self.assertTrue(np.all(start >= low - 1e-6), sweep.joint)
            self.assertTrue(np.all(end <= high + 1e-6), sweep.joint)

    def test_the_recorded_load_is_the_load_at_the_posture(self):
        """The sweep starts half an amplitude before the posture it was chosen
        for, so the load has to be read at the posture, not at the start."""
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=CLEAR)
        for sweep in sweeps:
            posture = np.asarray(sweep.start_deg, dtype=float)
            posture[sweep.joint] += sweep.amplitude_deg / 2.0
            expected = self.arm.joint_loads(posture)[sweep.joint]
            self.assertTrue(np.allclose(sweep.load, expected, atol=1e-9),
                            f"joint {sweep.joint}: {sweep.load} vs {expected}")

    def test_postures_spread_load_a_torque_alone_would_miss(self):
        """Choosing on gravity torque picks arbitrarily wherever that torque is
        the same in every pose, so require every joint to move in some term."""
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=CLEAR)
        for joint in range(self.arm.joint_count):
            loads = np.array([s.load for s in sweeps if s.joint == joint])
            span = loads.max(axis=0) - loads.min(axis=0)
            self.assertGreater(span.max(), 0.0, f"joint {joint} never moved")

    def test_postures_reach_the_heaviest_load_the_workspace_allows(self):
        """Spreading is not the same as reaching. Friction rises with the load
        pressing the surfaces together, so a run that spreads three postures
        across the light end describes only the light end."""
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=CLEAR, seed=5)
        low, high = self.limits.usable()
        rng = np.random.default_rng(11)
        for joint in range(self.arm.joint_count):
            here = [s for s in sweeps if s.joint == joint]
            if not here:
                continue
            centre = (here[0].start_deg[joint] + here[0].amplitude_deg / 2.0)
            poses = rng.uniform(low, high, size=(400, self.arm.joint_count))
            poses[:, joint] = centre
            reachable = max(abs(self.arm.joint_loads(pose)[joint][0])
                            for pose in poses)
            if reachable < 1e-6:
                continue        # gravity cannot load this joint in any pose
            heaviest = max(abs(s.load[0]) for s in here)
            self.assertGreater(heaviest, 0.7 * reachable,
                               f"joint {joint}: {heaviest} of {reachable}")

    def test_the_heaviest_candidate_is_kept_even_when_spread_would_drop_it(self):
        # Two postures extreme in the lesser terms win the farthest-point
        # distance, and the one carrying five times the axial torque -- the
        # load the motor actually works against -- goes unswept.
        loads = np.array([
            [0.0, 0.0, 0.0, 0.0],       # home
            [5.0, 0.0, 0.0, 0.0],       # heaviest axial torque
            [0.5, 10.0, 0.0, 0.0],
            [0.4, 0.0, 10.0, 0.0],
        ])
        chosen = excitation._choose_by_load(loads, 3)
        self.assertIn(1, chosen)
        # A middle rung is what tells a straight load line from a curve.
        picked = sorted(abs(loads[index][0]) for index in chosen)
        self.assertLess(picked[1], 0.9 * picked[2])
        self.assertGreater(picked[1], picked[0])

    def test_a_joint_gravity_cannot_load_still_gets_spread_postures(self):
        # Joint seven carries no axial torque in any pose, so ranking on it
        # would pick arbitrarily; the terms that do move have to decide.
        loads = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 1.2, 0.0, 0.0],
        ])
        chosen = excitation._choose_by_load(loads, 2)
        self.assertEqual(sorted(chosen), [0, 1])

    def test_the_swept_joint_may_be_centred_away_from_home(self):
        # The joint's own angle is part of the load on it, so a centre pinned
        # at home caps how heavily it can ever be measured.
        sweeps = excitation.design_friction_sweeps(
            self.arm, self.limits, amplitude_deg=20.0, speeds_deg_s=(5.0,),
            postures=3, collision_free=CLEAR, seed=3)
        centres = {}
        for sweep in sweeps:
            centres.setdefault(sweep.joint, set()).add(round(
                sweep.start_deg[sweep.joint] + sweep.amplitude_deg / 2.0, 6))
        self.assertTrue(any(len(seen) > 1 for seen in centres.values()),
                        centres)


class JointLoadTest(unittest.TestCase):
    """What a joint carries is four numbers, and they do not move together."""

    def setUp(self):
        self.arm = arm_model()

    def test_the_axial_term_is_the_inverse_dynamics_torque(self):
        rng = np.random.default_rng(0)
        for _ in range(8):
            pose = rng.uniform(-60.0, 60.0, self.arm.joint_count)
            axial = self.arm.joint_loads(pose)[:, 0]
            expected = np.abs(self.arm.inverse_dynamics(pose))
            self.assertTrue(np.allclose(axial, expected, atol=1e-9),
                            f"{axial} vs {expected}")

    def test_the_terms_are_reported_per_joint(self):
        loads = self.arm.joint_loads(np.zeros(self.arm.joint_count))
        self.assertEqual(loads.shape, (self.arm.joint_count, 4))
        self.assertTrue(np.all(loads >= 0.0))

    def test_the_force_only_changes_direction_not_size(self):
        """The force through a joint is the weight of everything past it, the
        same in every pose. Its size is therefore useless for telling postures
        apart; what moves is how it is shared between radial and thrust, which
        is why both are kept and neither is summed away."""
        rng = np.random.default_rng(1)
        loads = np.array([self.arm.joint_loads(
            rng.uniform(-80.0, 80.0, self.arm.joint_count)) for _ in range(40)])
        size = np.hypot(loads[:, :, 1], loads[:, :, 2])
        spread = size.max(axis=0) - size.min(axis=0)
        self.assertLess(spread.max(), 1e-6, f"force size moved by {spread}")
        split = loads[:, :, 1].max(axis=0) - loads[:, :, 1].min(axis=0)
        self.assertGreater(split.max(), 1e-3, "no joint changed its load split")


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

    def test_random_centres_explore_more_than_neutral(self):
        trajectory = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, duration_s=5.0, attempts=12,
            seed=8, randomize_centre=True)
        self.assertIsNotNone(trajectory)
        self.assertGreater(np.max(np.abs(trajectory.centre_deg)), 1.0)

    def test_transit_to_trajectory_start_is_collision_screened(self):
        def wall(pose):
            return not 20.0 <= float(pose[0]) <= 40.0

        centre = np.zeros(self.arm.joint_count)
        centre[0] = 70.0
        without_transit = excitation.design_fourier_trajectory(
            self.arm, self.limits, centre_deg=centre, harmonics=1,
            duration_s=5.0, attempts=12, seed=9, collision_free=wall)
        with_transit = excitation.design_fourier_trajectory(
            self.arm, self.limits, centre_deg=centre, harmonics=1,
            duration_s=5.0, attempts=12, seed=9, collision_free=wall,
            start_deg=np.zeros(self.arm.joint_count))
        self.assertIsNotNone(without_transit)
        self.assertIsNone(with_transit)

    def test_higher_amplitude_floor_strengthens_the_same_candidate(self):
        gentle = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, base_frequency_hz=0.1,
            duration_s=10.0, attempts=1, seed=12,
            minimum_amplitude_fraction=0.2)
        stronger = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, base_frequency_hz=0.1,
            duration_s=10.0, attempts=1, seed=12,
            minimum_amplitude_fraction=0.45)

        self.assertIsNotNone(gentle)
        self.assertIsNotNone(stronger)
        self.assertTrue(np.all(
            np.asarray(stronger.amplitudes_deg)
            >= np.asarray(gentle.amplitudes_deg)))
        self.assertGreater(
            np.linalg.norm(stronger.amplitudes_deg),
            np.linalg.norm(gentle.amplitudes_deg))

    def test_joint_backoff_scales_amplitude_without_shrinking_the_workspace(self):
        normal = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, base_frequency_hz=0.1,
            duration_s=10.0, attempts=1, seed=14)
        scale = np.ones(self.arm.joint_count)
        scale[-1] = 0.25
        backed_off = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, base_frequency_hz=0.1,
            duration_s=10.0, attempts=1, seed=14,
            joint_amplitude_scale=scale)

        self.assertIsNotNone(normal)
        self.assertIsNotNone(backed_off)
        before = np.asarray(normal.amplitudes_deg)
        after = np.asarray(backed_off.amplitudes_deg)
        np.testing.assert_allclose(after[:, :-1], before[:, :-1])
        np.testing.assert_allclose(after[:, -1], 0.25 * before[:, -1])
        np.testing.assert_allclose(backed_off.centre_deg, normal.centre_deg)

    def test_amplitude_floor_must_be_a_fraction(self):
        with self.assertRaises(ValueError):
            excitation.design_fourier_trajectory(
                self.arm, self.limits, minimum_amplitude_fraction=1.1)

    def test_time_scaling_starts_and_ends_at_rest(self):
        base = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, base_frequency_hz=0.1,
            duration_s=10.0, attempts=1, seed=2)
        ramped = excitation.ramp_fourier_trajectory(base, ramp_s=4.0)

        start = ramped.sample(0.0)
        finish = ramped.sample(ramped.duration_s)

        np.testing.assert_allclose(start[0], base.sample(0.0)[0])
        np.testing.assert_allclose(finish[0], base.sample(base.duration_s)[0])
        np.testing.assert_allclose(start[1], np.zeros(self.arm.joint_count))
        np.testing.assert_allclose(start[2], np.zeros(self.arm.joint_count))
        np.testing.assert_allclose(finish[1], np.zeros(self.arm.joint_count),
                                   atol=1e-12)
        np.testing.assert_allclose(finish[2], np.zeros(self.arm.joint_count),
                                   atol=1e-12)
        self.assertEqual(ramped.duration_s, 12.5)

    def test_time_scaled_derivatives_match_the_position_path(self):
        base = excitation.design_fourier_trajectory(
            self.arm, self.limits, harmonics=2, base_frequency_hz=0.1,
            duration_s=10.0, attempts=1, seed=3)
        ramped = excitation.ramp_fourier_trajectory(base, ramp_s=4.0)
        step = 1e-4
        for moment in (0.5, 3.5, 5.0, 10.5, 13.5):
            before = ramped.sample(moment - step)[0]
            position, velocity, _acceleration = ramped.sample(moment)
            after = ramped.sample(moment + step)[0]
            np.testing.assert_allclose(
                (after - before) / (2.0 * step), velocity,
                rtol=2e-4, atol=2e-4,
                err_msg=f"velocity at t={moment}: {position}")


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
