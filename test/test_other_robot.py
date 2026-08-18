"""A different robot must be a different YAML file, and nothing else.

This is the whole point of the package, so it is tested against an arm that
shares nothing with the workspace robot: different joint count, different joint
names, different limits, built from a URDF written here.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import robot_parameter_identification as ri
except ImportError as error:
    raise unittest.SkipTest(f"needs pinocchio: {error}") from error

JOINTS = 3
COULOMB = 0.06
VISCOUS = 0.005
NOISE = 0.002


def three_joint_urdf() -> str:
    links = ['<?xml version="1.0"?><robot name="tri"><link name="base"/>']
    for index in range(1, JOINTS + 1):
        parent = "base" if index == 1 else f"l{index - 1}"
        axis = "0 1 0" if index % 2 else "1 0 0"
        links.append(f"""
        <link name="l{index}"><inertial><origin xyz="0 0 0.15"/>
        <mass value="{2.0 - 0.4 * index}"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.01"/>
        </inertial></link>
        <joint name="ax{index}" type="revolute">
        <parent link="{parent}"/><child link="l{index}"/>
        <origin xyz="0 0 {0.3 if index > 1 else 0.1}"/><axis xyz="{axis}"/>
        <limit lower="-2.6" upper="2.6" effort="50" velocity="3"/></joint>""")
    return "".join(links) + "</robot>"


PROFILE_YAML = """
extends: manipulator
schema_version: 1
name: tri-arm
joints: {names: [ax1, ax2, ax3]}
limits:
  position_deg: 149.0
  continuous_current_a: [2.0, 1.5, 1.0]
  peak_current_a: [3.0, 2.0, 1.4]
envelope: {temperature_c: 50.0, sustained_speed_deg_s: 12.0}
"""


def load_profile():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tri.yaml"
        path.write_text(PROFILE_YAML)
        return ri.RobotProfile.from_yaml(path)


class ConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile()

    def test_joint_count_comes_from_the_file(self):
        self.assertEqual(self.profile.joint_count, JOINTS)
        self.assertEqual(self.profile.joint_names, ("ax1", "ax2", "ax3"))

    def test_scalar_limit_expands_to_every_joint(self):
        self.assertEqual(self.profile.position_limit_deg, (149.0,) * JOINTS)

    def test_envelope_overrides_the_template(self):
        self.assertEqual(self.profile.temperature_c, 50.0)
        self.assertEqual(self.profile.sustained_speed_deg_s, 12.0)

    def test_unstated_envelope_fields_are_inherited(self):
        self.assertEqual(self.profile.minimum_voltage_v, 20.0)
        self.assertEqual(self.profile.maximum_voltage_v, 30.0)
        self.assertEqual(self.profile.probe_current_fraction, 0.5)

    def test_campaign_bounds_follow_this_arm(self):
        bounds = ri.campaign_bounds(self.profile)
        self.assertEqual(bounds["maximum_speed_deg_s"], (1.0, 12.0))
        self.assertEqual(bounds["temperature_ceiling_c"], (30.0, 50.0))

    def test_default_plan_respects_this_arm(self):
        plan = ri.default_plan(self.profile)
        self.assertLessEqual(plan.maximum_speed_deg_s, 12.0)
        self.assertLessEqual(plan.temperature_ceiling_c, 50.0)
        for speed in plan.friction_speeds_deg_s:
            self.assertLessEqual(speed, plan.maximum_speed_deg_s)

    def test_the_sweep_reaches_the_speed_ceiling(self):
        plan = ri.default_plan(self.profile)
        self.assertAlmostEqual(max(plan.friction_speeds_deg_s),
                               plan.maximum_speed_deg_s)

    def test_raising_the_ceiling_actually_moves_the_sweep(self):
        slow = ri.default_plan(self.profile)
        fast, _notes = ri.clamp_campaign_plan(
            {"maximum_speed_deg_s": 12.0}, self.profile)
        self.assertGreater(max(fast.friction_speeds_deg_s),
                           max(slow.friction_speeds_deg_s))
        self.assertAlmostEqual(max(fast.friction_speeds_deg_s), 12.0)

    def test_an_explicit_sweep_speed_still_wins(self):
        plan, _notes = ri.clamp_campaign_plan(
            {"maximum_speed_deg_s": 12.0, "friction_speeds_deg_s": [4.0]},
            self.profile)
        self.assertEqual(plan.friction_speeds_deg_s, (4.0,))

    def test_a_raised_ceiling_cannot_escape_the_envelope(self):
        plan, notes = ri.clamp_campaign_plan(
            {"maximum_speed_deg_s": 500.0}, self.profile)
        self.assertLessEqual(plan.maximum_speed_deg_s, 12.0)
        self.assertTrue(any("maximum_speed_deg_s" in note for note in notes))


class SweepSizingTest(unittest.TestCase):
    """A fast sweep needs room, or the ramp is violent and nothing cruises."""

    def test_amplitude_grows_with_speed(self):
        slow = ri.sweep_amplitude_deg(10.0, 20.0)
        fast = ri.sweep_amplitude_deg(60.0, 20.0)
        self.assertGreater(fast, slow)

    def test_a_slow_sweep_keeps_the_requested_amplitude(self):
        self.assertEqual(ri.sweep_amplitude_deg(10.0, 20.0), 20.0)

    def test_amplitude_never_exceeds_the_operator_bound(self):
        high = ri.campaign_bounds(load_profile())["friction_amplitude_deg"][1]
        self.assertLessEqual(ri.sweep_amplitude_deg(500.0, 20.0), high)

    def test_the_implied_ramp_stays_within_the_design_target(self):
        # The plant ramps over a quarter of the pass, so acceleration is
        # 4*speed^2/distance. Sizing must keep that near the target.
        for speed in (10.0, 30.0, 60.0):
            amplitude = ri.sweep_amplitude_deg(speed, 20.0)
            implied = 4.0 * speed ** 2 / amplitude
            self.assertLessEqual(implied, 400.0, f"{speed} deg/s")

    def test_speeds_span_the_ceiling_without_duplicates(self):
        speeds = ri.sweep_speeds(60.0, (0.1, 0.3, 0.6, 1.0))
        self.assertEqual(speeds, tuple(sorted(set(speeds))))
        self.assertAlmostEqual(max(speeds), 60.0)
        self.assertLess(min(speeds), 10.0)


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile()
        self.arm = ri.ArmModel.from_profile(three_joint_urdf(), self.profile)

    def test_model_has_the_profile_joints(self):
        self.assertEqual(self.arm.joint_count, JOINTS)
        self.assertEqual(self.arm.joint_names, ["ax1", "ax2", "ax3"])
        self.assertEqual(self.arm.parameter_count, 10 * JOINTS)

    def test_missing_joint_is_reported_by_name(self):
        broken = ri.RobotProfile.from_dict({
            "schema_version": 1, "name": "wrong",
            "joints": {"names": ["ax1", "nope"]},
            "limits": {"position_deg": 90.0, "continuous_current_a": 1.0,
                       "peak_current_a": 2.0},
        })
        with self.assertRaises(ValueError) as caught:
            ri.ArmModel.from_profile(three_joint_urdf(), broken)
        self.assertIn("nope", str(caught.exception))


class EndToEndTest(unittest.TestCase):
    """Identify a robot the package has never seen, and validate independently."""

    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile()
        cls.arm = ri.ArmModel.from_profile(three_joint_urdf(), cls.profile)
        cls.theta = cls.arm.inertial_parameters()

    def _gather(self, count, seed):
        rng = np.random.default_rng(seed)
        rows, velocities, accelerations, currents = [], [], [], []
        for _ in range(count):
            q = rng.uniform(-80, 80, JOINTS)
            v = rng.uniform(-6, 6, JOINTS)
            a = rng.uniform(-10, 10, JOINTS)
            regressor = self.arm.torque_regressor(q, v, a)
            current = (regressor @ self.theta + COULOMB * np.sign(v)
                       + VISCOUS * v + rng.normal(0.0, NOISE, JOINTS))
            rows.append(regressor)
            velocities.append(v)
            accelerations.append(a)
            currents.append(current)
        return rows, velocities, accelerations, currents

    def test_friction_and_prediction_hold_on_an_unseen_robot(self):
        train = self._gather(260, 1)
        holdout = self._gather(90, 99)
        for joint in range(JOINTS):
            fit = ri.fit_joint(
                joint, train[0], [v[joint] for v in train[1]],
                [c[joint] for c in train[3]],
                accelerations=[a[joint] for a in train[2]])
            self.assertAlmostEqual(fit.friction["coulomb"], COULOMB, delta=0.01)
            self.assertAlmostEqual(fit.friction["viscous"], VISCOUS, delta=0.001)

            predicted = np.array([
                ri.predict_joint(fit, regressor, velocity[joint],
                                 acceleration=acceleration[joint])
                for regressor, velocity, acceleration
                in zip(holdout[0], holdout[1], holdout[2])])
            truth = np.array([c[joint] for c in holdout[3]])
            error = float(np.sqrt(np.mean((predicted - truth) ** 2)))
            self.assertLess(error, 4.0 * NOISE, f"joint{joint + 1} error {error}")


if __name__ == "__main__":
    unittest.main()
