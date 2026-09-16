"""Fresh post-result arrival confirmation without extra trajectory commands."""

import threading
import time
import unittest
from unittest.mock import Mock

from robot_parameter_identification.plants.ros_control import HardwarePlant, MotionFailed


class HoldArrivalTest(unittest.TestCase):
    def plant(self):
        plant = HardwarePlant.__new__(HardwarePlant)
        plant.joint_count = 7
        plant._lock = threading.Lock()
        plant._latest_at = time.monotonic() - 0.2
        plant._latest = {"position_deg": [0.0] * 7}
        plant._raise_if_stop_requested = Mock()
        plant._raise_if_monitor_tripped = Mock()
        plant._client = Mock()
        plant._spin_once = Mock()
        return plant

    def test_waits_for_new_telemetry_and_original_tolerance_without_resending(self):
        plant = self.plant()
        target = [0.0] * 7
        values = [None, [1.8] * 7, [0.2] * 7]

        def spin(_timeout):
            value = values.pop(0)
            if value is not None:
                plant._latest = {"position_deg": value}
                plant._latest_at = time.monotonic()

        plant._spin_once.side_effect = spin
        result = plant.wait_for_position(target)
        self.assertEqual(result["position_deg"], [0.2] * 7)
        self.assertEqual(plant._spin_once.call_count, 3)
        plant._client.send_goal_async.assert_not_called()

    def test_old_sample_at_target_cannot_verify_arrival(self):
        plant = self.plant()
        with self.assertRaisesRegex(MotionFailed, "fresh post-trajectory"):
            plant.wait_for_position([0.0] * 7, timeout_s=0.01)
        plant._client.send_goal_async.assert_not_called()

    def test_new_off_target_sample_does_not_relax_tolerance(self):
        plant = self.plant()

        def spin(_timeout):
            plant._latest = {"position_deg": [1.01] + [0.0] * 6}
            plant._latest_at = time.monotonic()

        plant._spin_once.side_effect = spin
        with self.assertRaises(MotionFailed):
            plant.wait_for_position([0.0] * 7, timeout_s=0.01)

    def test_stop_during_arrival_wait_propagates(self):
        plant = self.plant()
        plant._raise_if_stop_requested.side_effect = [None, RuntimeError("operator stop")]
        with self.assertRaisesRegex(RuntimeError, "operator stop"):
            plant.wait_for_position([0.0] * 7)
        plant._client.send_goal_async.assert_not_called()


if __name__ == "__main__":
    unittest.main()
