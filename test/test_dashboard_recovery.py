"""Offline recovery evidence tests; no ROS graph or hardware access."""

import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from robot_parameter_identification.dashboard.recovery import (
    CURRENT_CONTROLLER, RIGHT_JOINTS, RecoveryMonitor, SAFE_FLAGS)


ACTION = "/right_arm_joint_trajectory_controller/follow_joint_trajectory"


class RecoveryMonitorTest(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.sequence = 0
        self.monitor = RecoveryMonitor(clock=lambda: self.now)
        self.inventory = {
            "available": True, "age_s": 0.0, "error": "", "items": [
                {"name": CURRENT_CONTROLLER, "state": "inactive",
                 "claimed_interfaces": []},
                {"name": "right_arm_joint_trajectory_controller", "state": "active",
                 "type": "joint_trajectory_controller/JointTrajectoryController",
                 "claimed_interfaces": [f"{joint}/position" for joint in RIGHT_JOINTS]},
            ]}
        self.frame = {joint: dict(SAFE_FLAGS, position=0.0, velocity=0.0,
                                  telemetry_sequence=0.0) for joint in RIGHT_JOINTS}
        self.monitor.action_status(ACTION, [])

    def feed(self, advance=True):
        if advance:
            self.sequence += 1
            for row in self.frame.values():
                row["telemetry_sequence"] = self.sequence
        self.monitor.observe(self.frame, RIGHT_JOINTS, ACTION, self.inventory)

    def healthy(self):
        for _index in range(23):
            self.now += 0.05
            self.feed()

    def status(self):
        return self.monitor.status(RIGHT_JOINTS, ACTION, self.inventory)

    def test_complete_stable_advancing_evidence(self):
        self.feed()
        self.assertFalse(self.status()["ready"])
        self.healthy()
        self.assertEqual(set(self.status()), {"ready", "reason"})
        self.assertTrue(self.status()["ready"])

    def test_stale_and_frozen_sequences_reset_window(self):
        for publish in (False, True):
            with self.subTest(publish=publish):
                self.setUp()
                self.healthy()
                self.now += 0.11
                if publish:
                    self.feed(advance=False)
                self.assertFalse(self.status()["ready"])
                self.feed()
                self.assertFalse(self.status()["ready"])
                self.healthy()
                self.assertTrue(self.status()["ready"])

    def test_regression_requires_new_window(self):
        self.healthy()
        self.sequence = 0
        self.feed()
        self.assertFalse(self.status()["ready"])
        self.assertIn("regressed", self.status()["reason"])
        self.healthy()
        self.assertTrue(self.status()["ready"])

    def test_partial_flags_and_nonfinite_fields_fail_closed(self):
        for field in (*SAFE_FLAGS, "position", "velocity", "telemetry_sequence"):
            for invalid in (None, math.nan):
                with self.subTest(field=field, invalid=invalid):
                    self.setUp()
                    self.healthy()
                    if invalid is None:
                        del self.frame[RIGHT_JOINTS[-1]][field]
                    else:
                        self.frame[RIGHT_JOINTS[-1]][field] = invalid
                    self.feed(advance=False)
                    self.assertFalse(self.status()["ready"])

    def test_faults_enabled_and_stop_flags_are_exact(self):
        for field, expected in SAFE_FLAGS.items():
            with self.subTest(field=field):
                self.setUp()
                self.healthy()
                self.frame[RIGHT_JOINTS[0]][field] = 1 - expected
                self.feed()
                self.assertFalse(self.status()["ready"])
                self.frame[RIGHT_JOINTS[0]][field] = expected
                self.feed()
                self.assertFalse(self.status()["ready"])

    def test_speed_and_cumulative_travel(self):
        self.frame[RIGHT_JOINTS[0]]["velocity"] = math.radians(1.01)
        self.healthy()
        self.assertFalse(self.status()["ready"])
        self.setUp()
        for index in range(23):
            self.now += 0.05
            self.frame[RIGHT_JOINTS[0]]["position"] = math.radians(0.2 * (index % 2))
            self.feed()
        self.assertFalse(self.status()["ready"])

    def test_unknown_active_and_terminal_goals(self):
        for statuses in (None, [0], [1], [2], [3], [7], [4, 2]):
            with self.subTest(statuses=statuses):
                self.setUp()
                self.monitor.action_status(ACTION, statuses)
                self.healthy()
                self.assertFalse(self.status()["ready"])
                self.monitor.action_status(ACTION, [4, 5, 6])
                self.feed()
                self.assertFalse(self.status()["ready"])
                self.healthy()
                self.assertTrue(self.status()["ready"])

    def test_completed_motion_can_use_hardware_evidence_before_first_action_status(self):
        self.monitor = RecoveryMonitor(clock=lambda: self.now)
        self.healthy()
        self.assertTrue(self.monitor.status(
            RIGHT_JOINTS, ACTION, self.inventory, require_goal_status=False)["ready"])
        self.assertFalse(self.status()["ready"])
        self.monitor.action_status(ACTION, [2])
        self.healthy()
        self.assertFalse(self.monitor.status(
            RIGHT_JOINTS, ACTION, self.inventory, require_goal_status=False)["ready"])

    def test_action_update_revokes_ready_without_telemetry(self):
        self.healthy()
        self.monitor.action_status(ACTION, [2])
        self.monitor.action_status(ACTION, [])
        self.assertFalse(self.status()["ready"])

    def test_inventory_refusals_reset_window(self):
        variants = []
        for patch in ({"available": False}, {"age_s": 3.0}, {"age_s": None},
                      {"age_s": math.nan}, {"error": "timeout"}, {"items": []}):
            variants.append(dict(self.inventory, **patch))
        for index, field, value in ((0, "state", "active"), (0, "state", "unconfigured"),
                                    (0, "name", "other_current_controller"),
                                    (1, "state", "inactive"), (1, "name", "wrong_jtc"),
                                    (1, "type", "other/Controller"),
                                    (1, "claimed_interfaces", [])):
            inventory = copy.deepcopy(self.inventory)
            inventory["items"][index][field] = value
            variants.append(inventory)
        for inventory in variants:
            with self.subTest(inventory=inventory):
                self.setUp()
                self.healthy()
                good = self.inventory
                self.inventory = inventory
                self.assertFalse(self.status()["ready"])
                self.inventory = good
                self.feed()
                self.assertFalse(self.status()["ready"])
                self.healthy()
                self.assertTrue(self.status()["ready"])

    def test_wrong_arm_or_action_refused(self):
        self.healthy()
        self.assertFalse(self.monitor.status(
            RIGHT_JOINTS[:-1], ACTION, self.inventory)["ready"])
        self.assertFalse(self.monitor.status(
            RIGHT_JOINTS, "/other/follow_joint_trajectory", self.inventory)["ready"])

    def test_one_frozen_joint_is_not_masked_by_six_advancing(self):
        self.healthy()
        frozen = self.frame[RIGHT_JOINTS[-1]]["telemetry_sequence"]
        for _index in range(4):
            self.now += 0.05
            self.sequence += 1
            for row in self.frame.values():
                row["telemetry_sequence"] = self.sequence
            self.frame[RIGHT_JOINTS[-1]]["telemetry_sequence"] = frozen
            self.feed(advance=False)
        self.assertFalse(self.status()["ready"])

    def test_new_packet_after_unobserved_gap_cannot_preserve_window(self):
        self.healthy()
        self.now += 0.11
        self.feed()
        self.assertFalse(self.status()["ready"])

    def test_inventory_invalid_during_callback_resets_without_status_poll(self):
        self.healthy()
        self.inventory["available"] = False
        self.feed()
        self.inventory["available"] = True
        self.feed()
        self.assertFalse(self.status()["ready"])

    def test_one_second_must_be_observed_not_extrapolated(self):
        self.feed()
        for _index in range(19):
            self.now += 0.05
            self.feed()
        self.now += 0.06
        self.assertFalse(self.status()["ready"])
        self.feed()
        self.assertTrue(self.status()["ready"])

    def test_no_status_message_is_not_assumed_idle(self):
        self.monitor = RecoveryMonitor(clock=lambda: self.now)
        self.healthy()
        self.assertFalse(self.status()["ready"])
        self.assertIn("unknown", self.status()["reason"])

    def test_exact_speed_and_total_travel_limits_are_allowed(self):
        self.feed()
        self.frame[RIGHT_JOINTS[0]]["position"] = math.radians(0.5)
        self.frame[RIGHT_JOINTS[0]]["velocity"] = math.radians(1.0)
        self.healthy()
        self.assertTrue(self.status()["ready"])

    def test_missing_joint_and_invalid_sequences(self):
        self.healthy()
        del self.frame[RIGHT_JOINTS[-1]]
        self.feed()
        self.assertFalse(self.status()["ready"])
        for sequence in (-1, 2.5, math.inf):
            with self.subTest(sequence=sequence):
                self.setUp()
                self.healthy()
                self.frame[RIGHT_JOINTS[-1]]["telemetry_sequence"] = sequence
                self.feed(advance=False)
                self.assertFalse(self.status()["ready"])

    def test_controller_is_selected_from_configured_action(self):
        selected_action = "/another_jtc/follow_joint_trajectory"
        self.inventory["items"][1]["name"] = "another_jtc"
        self.monitor.action_status(selected_action, [])
        for _index in range(23):
            self.now += 0.05
            self.sequence += 1
            for row in self.frame.values():
                row["telemetry_sequence"] = self.sequence
            self.monitor.observe(self.frame, RIGHT_JOINTS, selected_action, self.inventory)
        self.assertTrue(self.monitor.status(
            RIGHT_JOINTS, selected_action, self.inventory)["ready"])

    def test_concurrent_telemetry_goal_updates_and_reads(self):
        self.healthy()

        def telemetry():
            for _index in range(100):
                self.monitor.observe(self.frame, RIGHT_JOINTS, ACTION, self.inventory)

        def goals():
            for _index in range(100):
                self.monitor.action_status(ACTION, [2])
                self.monitor.action_status(ACTION, [])

        def reads():
            for _index in range(100):
                result = self.status()
                self.assertIsInstance(result["ready"], bool)
                self.assertIsInstance(result["reason"], str)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(operation) for operation in (telemetry, goals, reads)]
            for future in futures:
                future.result(timeout=5)
        self.assertFalse(self.status()["ready"])


def node_slice():
    path = (Path(__file__).parents[1] / "robot_parameter_identification"
            / "dashboard" / "node.py")
    source = ast.parse(path.read_text())
    methods = {"_on_dynamic_state", "_subscribe_recovery_status",
               "_on_recovery_action_status", "recovery_status"}
    node = next(item for item in source.body
                if isinstance(item, ast.ClassDef) and item.name == "DashboardNode")
    node.bases = []
    node.body = [item for item in node.body
                 if isinstance(item, ast.FunctionDef) and item.name in methods]
    source.body = [item for item in source.body if (
        isinstance(item, ast.ImportFrom) and item.module == "__future__") or (
        isinstance(item, ast.FunctionDef) and item.name == "_by_joint") or (
        isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "DESCRIPTION_QOS"
            for target in item.targets))] + [node]
    namespace = {
        "QoSProfile": lambda **kwargs: SimpleNamespace(**kwargs),
        "ReliabilityPolicy": SimpleNamespace(RELIABLE="reliable"),
        "DurabilityPolicy": SimpleNamespace(TRANSIENT_LOCAL="transient_local"),
        "HistoryPolicy": SimpleNamespace(KEEP_LAST="keep_last"),
    }
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["DashboardNode"]


class RecoveryNodeWiringTest(unittest.TestCase):
    def setUp(self):
        self.fixture = RecoveryMonitorTest()
        self.fixture.setUp()
        self.node = node_slice()()
        self.node._lock = threading.Lock()
        self.node._recovery = self.fixture.monitor
        self.node._spec = SimpleNamespace(signals=SimpleNamespace(position="position"))
        self.node.service = SimpleNamespace(
            driven_joints=list(RIGHT_JOINTS), config=SimpleNamespace(
                commands=SimpleNamespace(follow_joint_trajectory_action=ACTION)))
        self.node._controllers = mock.Mock(spec=["snapshot"])
        self.node._controllers.snapshot.side_effect = lambda: self.fixture.inventory
        self.node._joint_names = lambda: list(RIGHT_JOINTS)
        self.node._merge_extra = lambda frame: frame
        self.node._note_everything_else = mock.Mock()
        self.node._assemble = mock.Mock()

    def message(self):
        return SimpleNamespace(
            joint_names=list(self.fixture.frame), interface_values=[
                SimpleNamespace(interface_names=list(row), values=list(row.values()))
                for row in self.fixture.frame.values()])

    def test_raw_frame_is_checked_before_assemble_and_service_callbacks(self):
        self.fixture.healthy()
        del self.fixture.frame[RIGHT_JOINTS[0]]["direct_current_stop_confirmed"]

        def check_reset(*_args):
            self.assertFalse(self.node._lock.locked())
            self.assertFalse(self.node.recovery_status()["ready"])

        self.node._note_everything_else.side_effect = check_reset
        self.node._assemble.side_effect = check_reset
        self.node._on_dynamic_state(self.message())
        self.node._assemble.assert_called_once()
        self.node._note_everything_else.assert_called_once()

    def test_subscription_path_qos_and_status_callback(self):
        self.node.create_subscription = mock.Mock()
        message_type = type("GoalStatusArray", (), {})
        with mock.patch.dict(sys.modules, {
                "action_msgs": SimpleNamespace(),
                "action_msgs.msg": SimpleNamespace(GoalStatusArray=message_type)}):
            self.node._subscribe_recovery_status(self.node.service.config.commands)
        message_class, topic, callback, qos = self.node.create_subscription.call_args.args
        self.assertIs(message_class, message_type)
        self.assertEqual(topic, ACTION + "/_action/status")
        self.assertEqual(vars(qos), {"depth": 1, "reliability": "reliable",
                                     "durability": "transient_local", "history": "keep_last"})
        self.fixture.healthy()
        callback(SimpleNamespace(status_list=[SimpleNamespace(status=2)]))
        self.assertFalse(self.node.recovery_status()["ready"])
        callback(SimpleNamespace(status_list=[]))
        self.fixture.healthy()
        self.assertTrue(self.node.recovery_status()["ready"])

    def test_status_only_reads_inventory_and_latest_monitor(self):
        self.fixture.healthy()
        self.assertTrue(self.node.recovery_status()["ready"])
        self.assertEqual(self.node._controllers.mock_calls, [mock.call.snapshot()])
        self.node.service.config.commands.follow_joint_trajectory_action = (
            "/different_controller/follow_joint_trajectory")
        self.assertFalse(self.node.recovery_status()["ready"])


if __name__ == "__main__":
    unittest.main()
