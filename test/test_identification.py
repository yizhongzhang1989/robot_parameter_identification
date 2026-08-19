"""Identification must generalise, not just fit the data it was given."""

import unittest

import numpy as np

pytest = None
try:
    from robot_parameter_identification import identification as ident
except ImportError as error:  # pinocchio ships with ROS, not with pip
    raise unittest.SkipTest(f"identification needs pinocchio: {error}") from error

MODEL_CACHE = {}


def arm_model():
    if "arm" not in MODEL_CACHE:
        from fixtures import synthetic_urdf, PREFIX

        MODEL_CACHE["arm"] = ident.ArmModel.from_urdf_text(
            synthetic_urdf(), PREFIX)
    return MODEL_CACHE["arm"]


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.arm = arm_model()

    def test_one_arm_is_isolated_from_a_multi_arm_urdf(self):
        self.assertEqual(self.arm.joint_count, 7)
        self.assertTrue(
            all(name.startswith("right_arm_") for name in self.arm.joint_names))

    def test_regressor_reproduces_inverse_dynamics(self):
        rng = np.random.default_rng(0)
        theta = self.arm.inertial_parameters()
        for _ in range(5):
            q = rng.uniform(-40, 40, 7)
            v = rng.uniform(-10, 10, 7)
            a = rng.uniform(-15, 15, 7)
            regressor = self.arm.torque_regressor(q, v, a)
            np.testing.assert_allclose(
                regressor @ theta, self.arm.inverse_dynamics(q, v, a), atol=1e-9)

    def test_static_regressor_has_no_velocity_terms(self):
        q = np.array([10.0, 20.0, -15.0, 30.0, -25.0, 12.0, 40.0])
        np.testing.assert_allclose(
            self.arm.static_regressor(q), self.arm.torque_regressor(q))

    def test_align_base_changes_gravity_torque(self):
        q = np.array([10.0, 20.0, -15.0, 30.0, -25.0, 12.0, 40.0])
        before = self.arm.inverse_dynamics(q).copy()
        placement = self.arm.model.jointPlacements[1]
        original = (placement.translation.copy(), placement.rotation.copy())
        try:
            self.arm.align_base(original[0], np.eye(3))
            self.assertFalse(np.allclose(before, self.arm.inverse_dynamics(q)))
        finally:
            self.arm.align_base(*original)


class TruncationTest(unittest.TestCase):
    def test_truncation_caps_the_condition_number(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=(200, 4))
        # A fifth column that is almost a copy of the first is unidentifiable.
        matrix = np.column_stack([base, base[:, 0] + 1e-9 * rng.normal(size=200)])
        target = matrix @ np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        solution, rank, condition = ident.truncated_solve(matrix, target, 1e3)
        self.assertLessEqual(condition, 1e3 + 1)
        self.assertLess(rank, matrix.shape[1])
        self.assertEqual(solution.shape, (5,))

    def test_identifiable_columns_drops_dead_columns(self):
        rng = np.random.default_rng(2)
        matrix = np.column_stack([
            rng.normal(size=50), np.zeros(50), rng.normal(size=50)])
        self.assertEqual(ident.identifiable_columns(matrix), [0, 2])

    def test_condition_number_is_scored_after_reduction(self):
        rng = np.random.default_rng(3)
        rows = [np.column_stack([rng.normal(size=7), np.zeros(7)])
                for _ in range(10)]
        self.assertTrue(np.isfinite(ident.stacked_condition_number(rows)))


class GeneralisationTest(unittest.TestCase):
    """The failure mode that matters: a fit that only works on its own data."""

    def setUp(self):
        self.arm = arm_model()
        self.theta = self.arm.inertial_parameters()
        self.gains = np.array([0.42, 0.37, 0.96, 1.05, 1.06, 0.90, 1.20])
        self.coulomb = np.array([0.05, 0.20, 0.04, 0.15, 0.03, 0.03, 0.02])
        self.viscous = np.array([0.004, 0.006, 0.003, 0.005, 0.002, 0.002, 0.001])

    def sample(self, count, seed, noise_a=0.002, spread_deg=50.0,
               speed_deg_s=10.0):
        rng = np.random.default_rng(seed)
        regressors, velocities, currents = [], [], []
        for _ in range(count):
            q = rng.uniform(-spread_deg, spread_deg, 7)
            v = rng.uniform(-speed_deg_s, speed_deg_s, 7)
            a = rng.uniform(-1.5 * speed_deg_s, 1.5 * speed_deg_s, 7)
            regressor = self.arm.torque_regressor(q, v, a)
            current = (regressor @ self.theta) / self.gains
            current = current + self.coulomb * np.sign(v) + self.viscous * v
            if noise_a > 0.0:
                current = current + rng.normal(0.0, noise_a, 7)
            regressors.append(regressor)
            velocities.append(v)
            currents.append(current)
        return regressors, velocities, currents

    def test_capped_fit_recovers_friction_and_generalises(self):
        regressors, velocities, currents = self.sample(320, seed=4)
        holdout = self.sample(80, seed=99)
        for joint in range(7):
            fit = ident.fit_joint(
                joint, regressors, [v[joint] for v in velocities],
                [c[joint] for c in currents], maximum_condition=1e3)
            self.assertLessEqual(fit.condition_number, 1.01e3)
            self.assertAlmostEqual(
                fit.friction["coulomb"], self.coulomb[joint], delta=0.01)
            self.assertAlmostEqual(
                fit.friction["viscous"], self.viscous[joint], delta=0.001)

            predicted = np.array([
                ident.predict_joint(fit, regressor, velocity[joint])
                for regressor, velocity in zip(holdout[0], holdout[1])])
            truth = np.array([c[joint] for c in holdout[2]])
            error = float(np.sqrt(np.mean((predicted - truth) ** 2)))
            self.assertLess(error, 0.2, f"joint{joint + 1} did not generalise")

    def test_uncapped_fit_blows_up_on_weakly_excited_data(self):
        """Narrow motion makes columns nearly dependent, which is the real risk."""
        regressors, velocities, currents = self.sample(
            120, seed=4, noise_a=0.002, spread_deg=4.0, speed_deg_s=1.0)
        holdout = self.sample(60, seed=99, noise_a=0.0)

        def holdout_error(cap):
            worst = 0.0
            for joint in range(3):
                fit = ident.fit_joint(
                    joint, regressors, [v[joint] for v in velocities],
                    [c[joint] for c in currents], maximum_condition=cap)
                predicted = np.array([
                    ident.predict_joint(fit, regressor, velocity[joint])
                    for regressor, velocity in zip(holdout[0], holdout[1])])
                truth = np.array([c[joint] for c in holdout[2]])
                worst = max(
                    worst, float(np.sqrt(np.mean((predicted - truth) ** 2))))
            return worst

        self.assertLess(holdout_error(1e3), holdout_error(1e12))

    def test_fit_reports_its_own_holdout_error(self):
        regressors, velocities, currents = self.sample(200, seed=5)
        fit = ident.fit_joint(
            0, regressors, [v[0] for v in velocities],
            [c[0] for c in currents], maximum_condition=1e3)
        self.assertIsNotNone(fit.holdout_rms_a)
        self.assertLess(fit.holdout_rms_a, 0.2)
        self.assertGreater(fit.effective_rank, 0)
        self.assertIn("coulomb", fit.as_dict()["friction"])

    def test_unexcited_data_is_refused(self):
        regressor = np.zeros((7, 70))
        with self.assertRaises(ValueError):
            ident.fit_joint(0, [regressor] * 5, [0.0] * 5, [0.0] * 5,
                            include_friction=False)

    def test_excluding_friction_actually_excludes_it(self):
        """The flag existed and was ignored, so both calls returned the same."""
        regressors, velocities, currents = self.sample(
            160, seed=11, noise_a=0.0, speed_deg_s=20.0)
        fit = ident.fit_joint(
            0, regressors, [v[0] for v in velocities],
            [c[0] for c in currents], maximum_condition=1e3)
        self.assertGreater(abs(fit.friction["coulomb"]), 0.01)

        moving = next(index for index, v in enumerate(velocities)
                      if abs(v[0]) > 5.0)
        whole = ident.predict_joint(fit, regressors[moving], velocities[moving][0])
        rigid = ident.predict_joint(fit, regressors[moving], velocities[moving][0],
                                    include_friction=False)
        self.assertNotAlmostEqual(whole, rigid, places=3)
        # What the two differ by is exactly the friction the curve draws.
        expected = (fit.friction["coulomb"] * np.tanh(
            velocities[moving][0] / 1.8)
            + fit.friction["viscous"] * velocities[moving][0]
            + fit.friction["offset"])
        self.assertAlmostEqual(whole - rigid, expected, places=3)


if __name__ == "__main__":
    unittest.main()
