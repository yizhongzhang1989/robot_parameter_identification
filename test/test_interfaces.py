"""The robot contract: what is required, what degrades, what is refused."""

import unittest

from robot_parameter_identification.interfaces import (
    CommandSpec, JOINT_STATE_MAP, SignalMap, TelemetrySpec)


class SignalMapTest(unittest.TestCase):
    def test_defaults_target_a_ros2_control_arm(self):
        signals = SignalMap()
        self.assertEqual(signals.position, "position")
        self.assertEqual(signals.effort, "current")

    def test_position_and_effort_cannot_be_blank(self):
        for field in ("position", "effort"):
            with self.assertRaises(ValueError, msg=field):
                SignalMap(**{field: ""})

    def test_an_unknown_effort_unit_is_refused(self):
        with self.assertRaises(ValueError):
            SignalMap(effort_unit="furlong")

    def test_optional_signals_may_be_absent(self):
        signals = SignalMap(temperature=None, enabled=None, fault_code=None)
        self.assertEqual(signals.optional_interfaces(), {"velocity": "velocity"})

    def test_absent_signals_name_the_guards_they_cost(self):
        guards = SignalMap(temperature=None).missing_guards()
        self.assertIn("temperature ceiling", guards)

    def test_a_fully_instrumented_arm_loses_no_guard(self):
        signals = SignalMap(temperature="temperature", voltage="voltage",
                            enabled="enabled", fault_code="fault_code")
        self.assertEqual(signals.missing_guards(), ())

    def test_blank_strings_from_a_form_mean_absent(self):
        signals = SignalMap.from_dict(
            {"temperature": "", "voltage": "none", "enabled": "  "})
        self.assertIsNone(signals.temperature)
        self.assertIsNone(signals.voltage)
        self.assertIsNone(signals.enabled)

    def test_round_trip_through_a_dict(self):
        original = SignalMap(effort="effort", effort_unit="newton_metre",
                             temperature=None)
        self.assertEqual(SignalMap.from_dict(original.as_dict()), original)

    def test_the_joint_state_preset_is_torque_in_newton_metres(self):
        self.assertEqual(JOINT_STATE_MAP.effort, "effort")
        self.assertEqual(JOINT_STATE_MAP.effort_unit, "newton_metre")
        self.assertIsNone(JOINT_STATE_MAP.temperature)


class TelemetrySpecTest(unittest.TestCase):
    def test_dynamic_joint_states_is_preferred_when_both_exist(self):
        spec = TelemetrySpec()
        self.assertEqual(spec.transport(), "dynamic_joint_states")
        self.assertEqual(spec.topic(), "/dynamic_joint_states")

    def test_joint_states_is_used_when_it_is_all_there_is(self):
        spec = TelemetrySpec(dynamic_joint_state_topic="",
                             signals=JOINT_STATE_MAP)
        self.assertEqual(spec.transport(), "joint_states")
        self.assertEqual(spec.topic(), "/joint_states")

    def test_no_topic_at_all_is_refused(self):
        spec = TelemetrySpec(joint_state_topic="",
                             dynamic_joint_state_topic="")
        with self.assertRaises(ValueError):
            spec.transport()

    def test_the_description_tells_the_operator_what_is_missing(self):
        spec = TelemetrySpec(dynamic_joint_state_topic="",
                             signals=JOINT_STATE_MAP)
        described = spec.describe()
        self.assertEqual(described["transport"], "joint_states")
        self.assertEqual(described["effort_unit"], "newton_metre")
        self.assertIn("temperature ceiling", described["missing_guards"])

    def test_round_trip_through_a_dict(self):
        original = TelemetrySpec(signals=JOINT_STATE_MAP, stale_after_s=1.5)
        self.assertEqual(TelemetrySpec.from_dict(original.as_dict()), original)


class CommandSpecTest(unittest.TestCase):
    def test_the_only_command_path_is_a_standard_action(self):
        self.assertTrue(CommandSpec().follow_joint_trajectory_action
                        .endswith("/follow_joint_trajectory"))

    def test_round_trip_through_a_dict(self):
        original = CommandSpec(follow_joint_trajectory_action="/a/b",
                               robot_description_topic="/desc")
        self.assertEqual(CommandSpec.from_dict(original.as_dict()), original)


if __name__ == "__main__":
    unittest.main()
