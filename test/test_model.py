"""Optional model columns: they must earn their place, not just exist."""

import unittest

import numpy as np

try:
    from robot_parameter_identification import consistency, identification as ident, model
    from test_identification import arm_model
    from fixtures import rm75_profile, scene_path, GAINS, COULOMB, VISCOUS
except ImportError as error:
    raise unittest.SkipTest(f"needs pinocchio: {error}") from error

try:
    from robot_parameter_identification.plants import simulation
except ImportError as error:
    # Only the simulated round trips need this. Taking the module down with it
    # is how the rest of these tests stopped running unnoticed.
    simulation = None
    SIMULATION_MISSING = str(error)
else:
    SIMULATION_MISSING = ""


class ComponentsTest(unittest.TestCase):
    def test_default_is_coulomb_viscous_offset(self):
        self.assertEqual(
            model.ModelComponents().column_names(),
            ("coulomb", "viscous", "offset"))

    def test_columns_appear_in_a_stable_order(self):
        components = model.ModelComponents(actuator_inertia=True, stribeck=True)
        self.assertEqual(
            components.column_names(),
            ("coulomb", "viscous", "stribeck", "actuator_inertia", "offset"))

    def test_disabling_everything_leaves_no_columns(self):
        components = model.ModelComponents(friction=False, offset=False)
        self.assertEqual(components.column_names(), ())
        self.assertEqual(model.extra_row(1.0, 1.0, components).size, 0)

    def test_row_width_matches_the_names(self):
        for components in (
            model.ModelComponents(),
            model.ModelComponents(actuator_inertia=True),
            model.ModelComponents(stribeck=True),
            model.ModelComponents(actuator_inertia=True, stribeck=True),
        ):
            self.assertEqual(
                model.extra_row(3.0, 4.0, components).size,
                len(components.column_names()))

    def test_actuator_inertia_column_is_the_acceleration(self):
        components = model.ModelComponents(
            friction=False, offset=False, actuator_inertia=True)
        self.assertEqual(model.extra_row(0.0, 7.5, components)[0], 7.5)

    def test_stribeck_decays_with_speed_and_flips_with_direction(self):
        components = model.ModelComponents(
            friction=False, offset=False, stribeck=True, stribeck_speed_deg_s=2.0)
        slow = model.extra_row(0.5, 0.0, components)[0]
        fast = model.extra_row(20.0, 0.0, components)[0]
        back = model.extra_row(-0.5, 0.0, components)[0]
        self.assertGreater(slow, fast)
        self.assertAlmostEqual(back, -slow)

    def test_stribeck_speed_must_be_positive(self):
        with self.assertRaises(ValueError):
            model.ModelComponents.from_dict({"stribeck_speed_deg_s": 0.0})

    def test_round_trip_through_a_dict(self):
        original = model.ModelComponents(stribeck=True, actuator_inertia=True)
        self.assertEqual(
            model.ModelComponents.from_dict(original.as_dict()), original)


class SmoothCoulombTest(unittest.TestCase):
    """The Coulomb column reverses over a finite speed, not instantly."""

    def column(self, velocity, width):
        components = model.ModelComponents(
            offset=False, coulomb_transition_deg_s=width)
        return model.extra_row(velocity, 0.0, components)[0]

    def test_zero_width_keeps_the_hard_sign(self):
        self.assertEqual(self.column(0.01, 0.0), 1.0)
        self.assertEqual(self.column(-9.0, 0.0), -1.0)

    def test_a_finite_width_ramps_instead_of_stepping(self):
        near = self.column(0.2, 1.8)
        far = self.column(9.0, 1.8)
        self.assertLess(near, 0.2)
        self.assertGreater(far, 0.99)
        self.assertLess(near, far)

    def test_the_ramp_is_odd_in_velocity(self):
        self.assertAlmostEqual(self.column(1.3, 1.8), -self.column(-1.3, 1.8))
        self.assertEqual(self.column(0.0, 1.8), 0.0)

    def test_a_narrower_width_reverses_faster(self):
        self.assertGreater(self.column(1.0, 0.5), self.column(1.0, 4.0))

    def test_it_does_not_add_a_column(self):
        widened = model.ModelComponents(coulomb_transition_deg_s=1.8)
        self.assertEqual(
            widened.column_names(), model.ModelComponents().column_names())

    def test_a_negative_width_is_rejected(self):
        with self.assertRaises(ValueError):
            model.ModelComponents.from_dict({"coulomb_transition_deg_s": -1.0})

    def test_the_shape_stays_reachable_with_a_non_negative_coefficient(self):
        """Why this beats Stribeck: one column, no cancelling pair needed."""
        speeds = np.linspace(-9.0, 9.0, 400)
        smooth = np.array([self.column(speed, 1.8) for speed in speeds])
        target = 0.7 * np.tanh(speeds / 1.8)
        coefficient = float(smooth @ target / (smooth @ smooth))
        self.assertGreater(coefficient, 0.0)
        self.assertLess(np.abs(coefficient * smooth - target).max(), 1e-9)


class LoadFrictionShapeTest(unittest.TestCase):
    def test_load_column_keeps_direction_while_coulomb_reverses_smoothly(self):
        components = model.ModelComponents(
            offset=False, load_friction=True,
            coulomb_transition_deg_s=1.8)
        names = components.column_names()
        near = model.extra_row(0.001, 0.0, components, load_a=2.5)
        back = model.extra_row(-0.001, 0.0, components, load_a=2.5)

        self.assertLess(abs(near[names.index("coulomb")]), 0.001)
        self.assertEqual(near[names.index("load_friction")], 2.5)
        self.assertEqual(back[names.index("load_friction")], -2.5)

    def test_load_stribeck_scales_the_decay_by_rigid_load(self):
        components = model.ModelComponents(
            friction=False, offset=False, load_stribeck=True,
            stribeck_speed_deg_s=1.0)
        names = components.column_names()
        slow = model.extra_row(0.2, 0.0, components, load_a=2.5)
        unloaded = model.extra_row(0.2, 0.0, components, load_a=0.0)
        back = model.extra_row(-0.2, 0.0, components, load_a=2.5)

        column = names.index("load_stribeck")
        self.assertGreater(slow[column], 0.0)
        self.assertEqual(unloaded[column], 0.0)
        self.assertAlmostEqual(back[column], -slow[column])


class ConsistencyTest(unittest.TestCase):
    def test_the_urdf_parameters_are_physically_feasible(self):
        verdicts = consistency.check_parameters(arm_model().inertial_parameters())
        summary = consistency.summarise(verdicts)
        self.assertTrue(summary["feasible"], summary["infeasible_links"])
        self.assertEqual(summary["links"], 7)

    def test_negative_mass_is_rejected(self):
        values = np.array(arm_model().inertial_parameters()[:10], dtype=float)
        values[0] = -1.0
        verdict = consistency.check_link(values)
        self.assertFalse(verdict.feasible)
        self.assertIn("mass", verdict.reasons[0])

    def test_impossible_inertia_is_rejected(self):
        values = np.zeros(10)
        values[0] = 1.0
        values[4], values[6], values[9] = 1.0, 1.0, 50.0  # violates the triangle
        self.assertFalse(consistency.check_link(values).feasible)

    def test_verdict_survives_positive_scaling(self):
        """Per-joint fits are scaled by an unknown positive torque constant."""
        values = np.array(arm_model().inertial_parameters()[:10], dtype=float)
        self.assertEqual(
            consistency.check_link(values).feasible,
            consistency.check_link(values * 3.7).feasible)

    def test_non_finite_parameters_are_rejected(self):
        values = np.zeros(10)
        values[0] = np.nan
        self.assertFalse(consistency.check_link(values).feasible)

    def test_wrong_length_is_reported(self):
        with self.assertRaises(ValueError):
            consistency.check_parameters(np.zeros(13))

    def test_projection_is_declared_unavailable_rather_than_faked(self):
        summary = consistency.summarise(
            consistency.check_parameters(np.zeros(10)))
        self.assertIn("not implemented", summary["projection"])


@unittest.skipIf(simulation is None, f"needs the simulated plant: {SIMULATION_MISSING}")
class StribeckRecoveryTest(unittest.TestCase):
    """The column must recover an effect the plant genuinely has.

    Stribeck decays as exp(-|v|/v_s), so it is only visible in data taken near
    standstill. A friction sweep that includes slow passes supplies that; one
    that stays fast does not, which the last test here pins down.
    """

    JOINT = 1

    @classmethod
    def setUpClass(cls):
        profile = rm75_profile()
        cls.arm = arm_model()
        cls.excess = tuple(1.0 * value for value in COULOMB)
        cls.plant = simulation.SimulatedPlant(
            scene_path(), profile=profile, torque_to_current=GAINS,
            coulomb_a=COULOMB, viscous_a_per_deg_s=VISCOUS,
            stribeck_a=cls.excess, stribeck_speed_deg_s=2.0)
        cls.sweep = cls._collect(cls, ladder=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0))
        # A sweep that never goes slow: exp(-|v|/2) is already ~0.05 at 6 deg/s,
        # so there is nothing left for the column to see.
        cls.fast_only = cls._collect(cls, ladder=(6.0, 8.0, 10.0, 12.0))

    def _collect(self, ladder):
        rng = np.random.default_rng(3)
        rows, velocities, accelerations, currents = [], [], [], []
        joints = 7
        # What a friction phase actually does: hold a base pose, walk one joint
        # across a stroke at a fixed speed, repeat for several speeds and both
        # directions.
        bases = [rng.uniform(-50, 50, joints) for _ in range(4)]
        for base in bases:
            for magnitude in ladder:
                for direction in (1.0, -1.0):
                    for step in range(8):
                        pose = base.copy()
                        pose[self.JOINT] = np.clip(
                            base[self.JOINT] - 20.0 + 5.0 * step, -80.0, 80.0)
                        speed = np.zeros(joints)
                        speed[self.JOINT] = direction * magnitude
                        sample = self.plant.hold(pose, speed)
                        rows.append(self.arm.torque_regressor(pose, speed))
                        velocities.append(speed)
                        accelerations.append(np.zeros(joints))
                        currents.append(np.asarray(sample["current_a"]))
        return rows, velocities, accelerations, currents

    def _fit(self, joint, components, data=None):
        rows, velocities, accelerations, currents = data or self.sweep
        return ident.fit_joint(
            joint, rows, [v[joint] for v in velocities],
            [c[joint] for c in currents],
            accelerations=[a[joint] for a in accelerations],
            components=components)

    def test_plant_shows_more_friction_at_low_speed(self):
        pose = np.zeros(7)
        joint = 1
        still = np.asarray(self.plant.hold(pose)["current_a"])[joint]

        def excess_over_coulomb(speed):
            drawn = np.asarray(self.plant.hold(pose, np.full(7, speed))["current_a"])
            return (drawn[joint] - still
                    - COULOMB[joint] - VISCOUS[joint] * speed)

        self.assertGreater(excess_over_coulomb(0.1), self.excess[joint] * 0.9)
        self.assertLess(excess_over_coulomb(20.0), self.excess[joint] * 0.1)

    def test_the_column_recovers_the_injected_excess(self):
        fit = self._fit(self.JOINT, model.ModelComponents(
            stribeck=True, stribeck_speed_deg_s=2.0))
        self.assertAlmostEqual(
            fit.friction["stribeck"], self.excess[self.JOINT], delta=0.02)
        self.assertAlmostEqual(fit.friction["coulomb"], COULOMB[self.JOINT], delta=0.02)

    def test_coulomb_only_model_fits_worse(self):
        plain = self._fit(self.JOINT, model.ModelComponents())
        with_stribeck = self._fit(self.JOINT, model.ModelComponents(
            stribeck=True, stribeck_speed_deg_s=2.0))
        self.assertLess(with_stribeck.residual_rms_a, plain.residual_rms_a)

    def test_stribeck_needs_low_speed_data(self):
        """Excitation, not the estimator, is what makes this term knowable."""
        components = model.ModelComponents(stribeck=True, stribeck_speed_deg_s=2.0)
        from_sweep = self._fit(self.JOINT, components).friction["stribeck"]
        from_fast = self._fit(
            self.JOINT, components, data=self.fast_only).friction["stribeck"]
        self.assertLess(abs(from_sweep - self.excess[self.JOINT]), 0.02)
        self.assertGreater(abs(from_fast - self.excess[self.JOINT]), 0.05)

    def test_fit_records_which_model_it_used(self):
        components = model.ModelComponents(stribeck=True, actuator_inertia=True)
        entry = self._fit(0, components).as_dict()
        self.assertTrue(entry["components"]["stribeck"])
        self.assertTrue(entry["components"]["actuator_inertia"])


@unittest.skipIf(simulation is None, f"needs the simulated plant: {SIMULATION_MISSING}")
class LoadFrictionRecoveryTest(unittest.TestCase):
    """Friction proportional to transmitted load needs an iterated fit.

    The column is |load| * sign(v) and the load is itself being identified, so
    the first pass has nothing to build it from. Only the refit sees it.
    """

    JOINT = 1
    RATIO = 0.30

    @classmethod
    def setUpClass(cls):
        cls.arm = arm_model()
        cls.plant = simulation.SimulatedPlant(
            scene_path(), profile=rm75_profile(), torque_to_current=GAINS,
            coulomb_a=COULOMB, viscous_a_per_deg_s=VISCOUS,
            load_friction_ratio=cls.RATIO)
        rng = np.random.default_rng(5)
        rows, velocities, currents = [], [], []
        # Poses spread widely so the joint sees a range of gravity loads; a
        # constant load would make the column indistinguishable from Coulomb.
        for _ in range(120):
            pose = rng.uniform(-80, 80, 7)
            for direction in (1.0, -1.0):
                for magnitude in (1.0, 4.0, 8.0):
                    speed = np.zeros(7)
                    speed[cls.JOINT] = direction * magnitude
                    sample = cls.plant.hold(pose, speed)
                    rows.append(cls.arm.torque_regressor(pose, speed))
                    velocities.append(speed)
                    currents.append(np.asarray(sample["current_a"]))
        cls.data = (rows, velocities, currents)

    def _fit(self, components):
        rows, velocities, currents = self.data
        return ident.fit_joint(
            self.JOINT, rows, [v[self.JOINT] for v in velocities],
            [c[self.JOINT] for c in currents], components=components)

    def test_the_column_recovers_the_injected_ratio(self):
        fit = self._fit(model.ModelComponents(load_friction=True))
        self.assertAlmostEqual(
            fit.friction["load_friction"], self.RATIO, delta=0.05)

    def test_a_velocity_only_model_fits_worse(self):
        # Only by a little: the Coulomb column absorbs the *mean* load term and
        # leaves the column just its variation across poses.
        plain = self._fit(model.ModelComponents())
        loaded = self._fit(model.ModelComponents(load_friction=True))
        self.assertLess(loaded.residual_rms_a, plain.residual_rms_a)

    def test_a_single_pass_cannot_see_the_column(self):
        original = ident.LOAD_FRICTION_PASSES
        try:
            ident.LOAD_FRICTION_PASSES = 1
            once = self._fit(model.ModelComponents(load_friction=True))
        finally:
            ident.LOAD_FRICTION_PASSES = original
        # With no load estimate the column is all zeros and gets dropped.
        self.assertNotIn("load_friction", once.friction)

    def test_prediction_rebuilds_the_load_it_was_fitted_with(self):
        rows, velocities, currents = self.data
        fit = self._fit(model.ModelComponents(load_friction=True))
        errors = [
            ident.predict_joint(fit, row, velocity[self.JOINT])
            - current[self.JOINT]
            for row, velocity, current in zip(rows, velocities, currents)]
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.square(errors)))),
            fit.residual_rms_a, delta=1e-9)


class LoadColumnSelectionTest(unittest.TestCase):
    """Choosing the load column, on data built here rather than simulated.

    The recovery tests below need the simulated plant and are skipped without
    it, which is how two tests written for this behaviour ran nowhere at all.
    """

    JOINT = 1
    COULOMB = 0.30
    RATIO = 0.20

    def setUp(self):
        arm = arm_model()
        rng = np.random.default_rng(0)
        self.rigid, self.speeds, self.target = [], [], []
        for _ in range(300):
            pose = rng.uniform(-70.0, 70.0, arm.joint_count)
            speed = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 60.0))
            velocity = np.zeros(arm.joint_count)
            velocity[self.JOINT] = speed
            row = np.asarray(
                arm.torque_regressor(pose, velocity)[self.JOINT], dtype=float)
            load = float(row @ arm.inertial_parameters())
            self.rigid.append(row)
            self.speeds.append(speed)
            self.target.append(
                load + np.sign(speed) * (self.COULOMB + self.RATIO * abs(load))
                + float(rng.normal(0.0, 0.002)))
        self.rigid = np.asarray(self.rigid, dtype=float)
        self.speeds = np.asarray(self.speeds, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        self.blank = np.zeros_like(self.speeds)

    def test_a_zero_load_leaves_the_column_empty(self):
        """The trap the fold scoring fell into: with no load estimate the
        column is identically zero, so it can never look useful."""
        stacked = ident._stack(
            self.rigid, self.speeds, self.blank,
            model.ModelComponents(load_friction=True), self.blank)
        self.assertTrue(np.allclose(stacked[:, -2], 0.0))

    def test_the_search_keeps_a_column_the_data_needs(self):
        picked = ident._choose_column(
            "load_friction", "load_friction_search", self.rigid, self.speeds,
            self.blank, model.ModelComponents(load_friction_search=True),
            self.target, 1e-6, 1.0e3, self.rigid.shape[1], 0)
        self.assertTrue(picked.load_friction, "declined a column it needs")
        self.assertFalse(picked.load_friction_search)

    def test_the_search_drops_a_column_the_data_does_not_need(self):
        plain = self.target - np.array(
            [np.sign(s) * self.RATIO * abs(r @ arm_model().inertial_parameters())
             for s, r in zip(self.speeds, self.rigid)])
        picked = ident._choose_column(
            "load_friction", "load_friction_search", self.rigid, self.speeds,
            self.blank, model.ModelComponents(load_friction_search=True),
            plain, 1e-6, 1.0e3, self.rigid.shape[1], 0)
        self.assertFalse(picked.load_friction, "kept a column it does not need")

    def test_the_recovered_ratio_is_the_injected_one(self):
        fit = ident.fit_joint(
            self.JOINT, list(self.rigid[:, None, :].repeat(
                arm_model().joint_count, axis=1)),
            list(self.speeds), list(self.target),
            components=model.ModelComponents(load_friction=True))
        self.assertAlmostEqual(fit.friction["load_friction"], self.RATIO,
                               delta=0.05)


@unittest.skipIf(simulation is None, f"needs the simulated plant: {SIMULATION_MISSING}")
class PhysicalFrictionSignTest(unittest.TestCase):
    """Friction coefficients that physics forbids must never be returned.

    The Stribeck column is nearly collinear with the Coulomb column over any
    narrow speed range, so an unconstrained fit answers with a large cancelling
    pair: it scores well and describes nothing. Real hardware did exactly that
    (Coulomb 1.31 A against Stribeck -1.26 A, and a *negative* friction at
    standstill on J4).
    """

    JOINT = 1

    @classmethod
    def setUpClass(cls):
        cls.arm = arm_model()
        cls.plant = simulation.SimulatedPlant(
            scene_path(), profile=rm75_profile(), torque_to_current=GAINS,
            coulomb_a=COULOMB, viscous_a_per_deg_s=VISCOUS)
        rng = np.random.default_rng(11)
        rows, velocities, currents = [], [], []
        # A narrow speed band is what makes the two columns collinear.
        for _ in range(60):
            pose = rng.uniform(-60, 60, 7)
            for direction in (1.0, -1.0):
                for magnitude in (2.0, 2.4):
                    speed = np.zeros(7)
                    speed[cls.JOINT] = direction * magnitude
                    sample = cls.plant.hold(pose, speed)
                    rows.append(cls.arm.torque_regressor(pose, speed))
                    velocities.append(speed)
                    currents.append(np.asarray(sample["current_a"]))
        cls.data = (rows, velocities, currents)

    def _fit(self, components):
        rows, velocities, currents = self.data
        return ident.fit_joint(
            self.JOINT, rows, [v[self.JOINT] for v in velocities],
            [c[self.JOINT] for c in currents], components=components)

    def test_every_constrained_coefficient_stays_non_negative(self):
        fit = self._fit(model.ModelComponents(stribeck=True))
        for name in ("coulomb", "viscous", "stribeck"):
            if name in fit.friction:
                self.assertGreaterEqual(
                    fit.friction[name], 0.0, f"{name} went negative")

    def test_friction_at_standstill_cannot_be_negative(self):
        fit = self._fit(model.ModelComponents(stribeck=True))
        standstill = fit.friction.get("coulomb", 0.0) + fit.friction.get(
            "stribeck", 0.0)
        self.assertGreater(standstill, 0.0)

    def test_the_unconstrained_solve_is_what_needs_constraining(self):
        # Straight at the solver, because the pathology is a property of the
        # normal equations rather than of any particular plant: two nearly
        # parallel columns whose best free answer is a big cancelling pair.
        rng = np.random.default_rng(3)
        speeds = rng.uniform(2.0, 2.5, 400)
        first = np.sign(speeds)
        second = np.sign(speeds) * np.exp(-speeds / 2.0)
        matrix = np.column_stack([first, second])
        # A truth that only the forbidden direction can express, which is what
        # the hardware fit found: Stribeck negative, i.e. static < dynamic.
        target = 0.30 * first - 0.10 * second + rng.normal(0.0, 0.01, speeds.size)

        free, _rank, _cond = ident.truncated_solve(matrix, target, 1.0e3)
        bounded, _rank, _cond = ident.physical_solve(
            matrix, target, 1.0e3, bounded=[0, 1])

        self.assertLess(free[1], 0.0, "expected the free fit to go negative")
        self.assertGreaterEqual(bounded.min(), 0.0)
        self.assertAlmostEqual(bounded[1], 0.0, delta=1e-9)

    def test_a_well_posed_fit_is_left_alone(self):
        # No bound is active here, so the constrained path must not perturb it.
        original = ident.NONNEGATIVE_COLUMNS
        try:
            ident.NONNEGATIVE_COLUMNS = ()
            free = self._fit(model.ModelComponents())
        finally:
            ident.NONNEGATIVE_COLUMNS = original
        constrained = self._fit(model.ModelComponents())
        self.assertAlmostEqual(
            free.residual_rms_a, constrained.residual_rms_a, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
