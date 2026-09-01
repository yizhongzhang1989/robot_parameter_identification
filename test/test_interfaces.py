"""The robot contract: what is required, what degrades, what is refused."""

import unittest

from robot_parameter_identification.interfaces import (
    CommandSpec, JOINT_STATE_MAP, SignalMap, TelemetrySpec)


class SignalMapTest(unittest.TestCase):
    def test_defaults_target_a_ros2_control_arm(self):
        signals = SignalMap()
        self.assertEqual(signals.position, "position")
        self.assertEqual(signals.effort, "current")

    def test_both_effort_channels_are_looked_for_without_being_configured(self):
        self.assertEqual(SignalMap().effort_channels(),
                         {"current": "current", "torque": "effort"})

    def test_position_cannot_be_blank(self):
        with self.assertRaises(ValueError):
            SignalMap(position="")

    def test_the_selected_effort_channel_must_be_mapped(self):
        with self.assertRaises(ValueError):
            SignalMap(current=None)
        with self.assertRaises(ValueError):
            SignalMap(torque=None, effort_source="torque")

    def test_an_unknown_effort_source_is_refused(self):
        with self.assertRaises(ValueError):
            SignalMap(effort_source="furlongs")

    def test_the_unit_follows_the_source_it_cannot_disagree(self):
        amps = SignalMap(current="current", effort_source="current")
        newtons = SignalMap(torque="torque", effort_source="torque")
        self.assertEqual(amps.effort_unit, "ampere")
        self.assertEqual(amps.effort, "current")
        self.assertEqual(newtons.effort_unit, "newton_metre")
        self.assertEqual(newtons.effort, "torque")

    def test_a_drive_reporting_both_records_the_one_it_does_not_fit(self):
        both = SignalMap(current="current", torque="torque",
                         effort_source="current")
        self.assertEqual(both.effort, "current")
        self.assertEqual(both.effort_channels(),
                         {"current": "current", "torque": "torque"})

    def test_a_drive_that_publishes_the_preference_keeps_it(self):
        signals = SignalMap()
        self.assertIs(signals.settled_among(["current", "torque"]), signals)
        self.assertIs(signals.settled_among(["current"]), signals)

    def test_a_drive_publishing_only_the_other_channel_overrides_it(self):
        settled = SignalMap().settled_among(["torque"])
        self.assertEqual(settled.effort_source, "torque")
        self.assertEqual(settled.effort, "effort")
        self.assertEqual(settled.effort_unit, "newton_metre")

    def test_nothing_arriving_leaves_the_preference_alone(self):
        signals = SignalMap()
        self.assertIs(signals.settled_among([]), signals)

    def test_a_torque_only_drive_is_supported(self):
        signals = SignalMap(current=None, torque="effort",
                            effort_source="torque")
        self.assertEqual(signals.required_interfaces(), ("position",))
        self.assertEqual(signals.effort_channels(), {"torque": "effort"})
        self.assertNotIn("current", signals.optional_interfaces())

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
        original = SignalMap(current=None, torque="effort",
                             effort_source="torque", temperature=None)
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

    def test_a_signal_may_arrive_on_a_topic_of_its_own(self):
        spec = TelemetrySpec(
            extra_dynamic_joint_state_topics=("/arm/motor_currents",))
        self.assertIn("/arm/motor_currents", spec.describe()["extra_topics"])

    def test_blank_extra_topics_are_dropped_rather_than_subscribed(self):
        spec = TelemetrySpec(
            extra_dynamic_joint_state_topics=["", "  ", "/arm/currents"])
        self.assertEqual(spec.extra_dynamic_joint_state_topics,
                         ("/arm/currents",))

    def test_extra_topics_survive_a_round_trip(self):
        original = TelemetrySpec(
            extra_dynamic_joint_state_topics=("/arm/currents",))
        self.assertEqual(TelemetrySpec.from_dict(original.as_dict()), original)


class CommandSpecTest(unittest.TestCase):
    def test_the_only_command_path_is_a_standard_action(self):
        self.assertTrue(CommandSpec().follow_joint_trajectory_action
                        .endswith("/follow_joint_trajectory"))

    def test_naming_the_controller_is_enough(self):
        spec = CommandSpec.for_controller("left_arm_jtc")
        self.assertEqual(spec.follow_joint_trajectory_action,
                         "/left_arm_jtc/follow_joint_trajectory")
        self.assertEqual(spec.controller_state_topic,
                         "/left_arm_jtc/controller_state")

    def test_a_leading_slash_on_the_controller_is_not_a_second_one(self):
        self.assertEqual(
            CommandSpec.for_controller("/left_arm_jtc"),
            CommandSpec.for_controller("left_arm_jtc"))

    def test_a_blank_controller_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            CommandSpec.for_controller("  ")

    def test_the_controller_state_topic_follows_a_full_action_path(self):
        spec = CommandSpec(
            follow_joint_trajectory_action="/odd/place/follow_joint_trajectory")
        self.assertEqual(spec.controller_state_topic,
                         "/odd/place/controller_state")

    def test_round_trip_through_a_dict(self):
        original = CommandSpec(follow_joint_trajectory_action="/a/b",
                               robot_description_topic="/desc")
        self.assertEqual(CommandSpec.from_dict(original.as_dict()), original)


if __name__ == "__main__":
    unittest.main()
