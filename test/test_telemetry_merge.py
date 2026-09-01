"""Reading a signal that arrives on a topic of its own.

The bridge is the one file that touches ROS, but the two decisions that matter
-- which source wins, and what counts as a complete frame -- are ordinary
functions, so they are tested without a graph.
"""

import unittest

from robot_parameter_identification.interfaces import SignalMap, TelemetrySpec

try:
    from robot_parameter_identification.dashboard.node import (
        DashboardNode, merge_by_joint)
except ImportError:  # pragma: no cover - rclpy is not on this interpreter
    DashboardNode = None


class Assembler:
    """Enough of the bridge for :meth:`DashboardNode._assemble` to run."""

    def __init__(self, signals: SignalMap) -> None:
        self._spec = TelemetrySpec(signals=signals)
        self.stored = None
        self.complaints = []

    def _store(self, rows, count) -> None:
        self.stored = (rows, count)

    def _complain(self, message) -> None:
        self.complaints.append(message)


@unittest.skipIf(DashboardNode is None, "rclpy not available")
class MergeTest(unittest.TestCase):
    def test_the_topic_the_operator_named_wins(self):
        merged = merge_by_joint(
            {"j1": {"position": 0.5, "current": 0.0}},
            [(10.0, {"j1": {"current": 2.5}})], now=10.1, stale_after_s=0.5)
        self.assertEqual(merged["j1"], {"position": 0.5, "current": 2.5})

    def test_a_source_that_stopped_publishing_drops_out(self):
        merged = merge_by_joint(
            {"j1": {"position": 0.5}},
            [(10.0, {"j1": {"current": 2.5}})], now=11.0, stale_after_s=0.5)
        self.assertNotIn("current", merged["j1"])

    def test_the_main_topic_is_left_untouched(self):
        primary = {"j1": {"position": 0.5}}
        merge_by_joint(primary, [(10.0, {"j1": {"current": 2.5}})],
                       now=10.0, stale_after_s=0.5)
        self.assertEqual(primary, {"j1": {"position": 0.5}})


@unittest.skipIf(DashboardNode is None, "rclpy not available")
class AssembleTest(unittest.TestCase):
    def test_a_frame_missing_the_effort_channel_is_not_a_sample(self):
        bridge = Assembler(SignalMap(temperature=None, enabled=None,
                                     fault_code=None))
        DashboardNode._assemble(bridge, {"j1": {"position": 0.5}}, ["j1"])
        self.assertIsNone(bridge.stored)

    def test_the_complaint_names_what_was_wanted_and_what_arrived(self):
        bridge = Assembler(SignalMap())
        DashboardNode._assemble(
            bridge, {"j1": {"position": 0.5, "amps": 2.5}}, ["j1"])
        self.assertIsNone(bridge.stored)
        complaint = bridge.complaints[0]
        self.assertIn("'current', 'effort'", complaint)
        self.assertIn("'amps'", complaint)

    def test_an_arm_publishing_only_effort_still_makes_a_frame(self):
        bridge = Assembler(SignalMap(velocity=None, temperature=None,
                                     enabled=None, fault_code=None))
        DashboardNode._assemble(
            bridge, {"j1": {"position": 0.5, "effort": 1.25}}, ["j1"])
        rows, count = bridge.stored
        self.assertEqual(count, 1)
        self.assertEqual(rows["torque"], [1.25])
        self.assertNotIn("current", rows)

    def test_an_arm_publishing_both_channels_keeps_both(self):
        bridge = Assembler(SignalMap(velocity=None, temperature=None,
                                     enabled=None, fault_code=None))
        DashboardNode._assemble(
            bridge, {"j1": {"position": 0.5, "current": 2.5, "effort": 1.25}},
            ["j1"])
        rows, _ = bridge.stored
        self.assertEqual(rows["current"], [2.5])
        self.assertEqual(rows["torque"], [1.25])

    def test_an_unmapped_optional_signal_does_not_hold_up_the_frame(self):
        bridge = Assembler(SignalMap(velocity=None, temperature=None,
                                     enabled=None, fault_code=None))
        DashboardNode._assemble(
            bridge, {"j1": {"position": 0.5, "current": 2.5}}, ["j1"])
        rows, count = bridge.stored
        self.assertEqual(count, 1)
        self.assertEqual(rows["current"], [2.5])
        self.assertNotIn("velocity", rows)

    def test_a_joint_the_controller_drives_but_nobody_publishes_is_refused(self):
        bridge = Assembler(SignalMap(velocity=None, temperature=None,
                                     enabled=None, fault_code=None))
        DashboardNode._assemble(
            bridge, {"j1": {"position": 0.5, "current": 2.5}}, ["j1", "j2"])
        self.assertIsNone(bridge.stored)


if __name__ == "__main__":
    unittest.main()
