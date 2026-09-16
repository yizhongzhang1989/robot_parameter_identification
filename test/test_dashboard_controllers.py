"""Controller inventory tests without ROS or robot connections."""

from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest import mock

from robot_parameter_identification.dashboard.controllers import ControllerInventory


class ControllerInventoryTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.future = Future()
        self.client = mock.Mock()
        self.client.service_is_ready.return_value = True
        self.client.call_async.side_effect = lambda _request: self.future
        self.inventory = ControllerInventory(
            self.client, object, "/controller_manager", clock=lambda: self.now)

    def reply(self, items):
        self.inventory.poll()
        pending = self.future
        self.future = Future()
        pending.set_result(SimpleNamespace(controller=[
            SimpleNamespace(name=name, type="example/Controller", state=state,
                            claimed_interfaces=claims)
            for name, state, claims in items]))
        self.now += 1.0
        self.inventory.poll()

    def test_all_states_and_active_broadcasters_are_retained(self):
        self.assertFalse(self.inventory.snapshot()["available"])
        self.reply([("z_inactive", "inactive", []), ("b_broadcaster", "active", []),
                    ("a_motion", "active", ["joint/position"]),
                    ("c_new", "unconfigured", [])])
        snapshot = self.inventory.snapshot()
        self.assertTrue(snapshot["available"])
        self.assertEqual([item["name"] for item in snapshot["items"]],
                         ["a_motion", "b_broadcaster", "c_new", "z_inactive"])
        self.assertEqual([item["name"] for item in snapshot["items"]
                          if item["state"] == "active"], ["a_motion", "b_broadcaster"])

    def test_empty_reply_is_available_not_disconnected(self):
        self.reply([])
        self.assertTrue(self.inventory.snapshot()["available"])
        self.assertEqual(self.inventory.snapshot()["items"], [])

    def test_fresh_snapshot_rejects_earlier_request_and_queries_again(self):
        self.inventory.poll()
        old = self.future
        self.now = 1.0
        old.set_result(SimpleNamespace(controller=[SimpleNamespace(
            name="old", type="example/Controller", state="inactive", claimed_interfaces=[],
            required_command_interfaces=[])]))

        def answer(_request):
            self.now += 0.01
            future = Future()
            future.set_result(SimpleNamespace(controller=[SimpleNamespace(
                name="new", type="example/Controller", state="active",
                claimed_interfaces=["joint/position"],
                required_command_interfaces=["joint/position"])]))
            return future

        self.client.call_async.side_effect = answer
        snapshot = self.inventory.fresh_snapshot()
        self.assertEqual(snapshot["items"][0]["name"], "new")
        self.assertGreaterEqual(snapshot["requested_at_monotonic"], 1.0)
        self.assertEqual(snapshot["items"][0]["required_command_interfaces"], ["joint/position"])
        self.assertGreaterEqual(self.client.call_async.call_count, 2)

    def test_fresh_snapshot_fails_when_service_is_unavailable(self):
        self.client.service_is_ready.return_value = False
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.inventory.fresh_snapshot()

    def test_connection_exposes_inventory_and_no_bridge_is_unavailable(self):
        from robot_parameter_identification.dashboard.service import (
            DashboardConfig, IdentificationService)

        service = IdentificationService(DashboardConfig())
        self.assertFalse(service.connection()["controllers"]["available"])
        self.reply([("motion", "active", [])])
        bridge = mock.Mock()
        bridge.observed_signals.return_value = set()
        bridge.health.return_value = {"controllers": self.inventory.snapshot()}
        service.bridge = bridge
        self.assertEqual(service.connection()["controllers"], self.inventory.snapshot())

    def test_stale_reply_does_not_leave_active_controllers_visible(self):
        self.reply([("motion", "active", [])])
        self.now += 3.0
        self.assertFalse(self.inventory.snapshot()["available"])
        self.assertEqual(self.inventory.snapshot()["items"], [])

    def test_timeout_cancels_request_and_recovers(self):
        self.inventory.poll()
        timed_out = self.future
        self.now = 2.1
        self.future = Future()
        self.inventory.poll()
        self.client.remove_pending_request.assert_called_once_with(timed_out)
        self.assertTrue(timed_out.cancelled())
        self.assertEqual(self.inventory.snapshot()["error"], "timeout")
        self.reply([("motion", "inactive", [])])
        self.assertTrue(self.inventory.snapshot()["available"])

    def test_unavailable_service_and_failed_reply(self):
        self.client.service_is_ready.return_value = False
        self.inventory.poll()
        self.client.call_async.assert_not_called()
        self.assertEqual(self.inventory.snapshot()["error"], "unavailable")
        self.client.service_is_ready.return_value = True
        self.inventory.poll()
        self.future.set_exception(RuntimeError("manager exited"))
        self.inventory.poll()
        self.assertEqual(self.inventory.snapshot()["error"], "query_failed")


if __name__ == "__main__":
    unittest.main()
