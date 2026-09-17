import unittest
from unittest.mock import Mock, patch

import numpy as np

from fixtures import rm75_profile
from robot_parameter_identification.interfaces import DriveLimitExceeded, MotionFailed
from robot_parameter_identification.plants import ros_control as hardware


class StationaryMoveStartTest(unittest.TestCase):
    def setUp(self):
        self.plant = hardware.HardwarePlant(rm75_profile())
        self.now = 100.0
        self.started = self.now
        self.base = np.arange(self.plant.joint_count, dtype=float)
        self.moving = False
        self.frozen = False
        self.stale = False
        self.plant._attempt = Mock()
        self.plant._spin_once = self.tick
        self.clock = patch.object(hardware.time, "monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def tick(self, _timeout):
        self.now += 0.01
        position = self.base + (self.now - self.started if self.moving else 0.0)
        self.plant._latest = {
            "position_deg": position.tolist(), "speed_deg_s": [7.8] * self.plant.joint_count,
            "stamp_s": self.started if self.frozen else self.now,
        }
        self.plant._latest_at = self.now - (0.2 if self.stale else 0.0)

    def test_fresh_stationary_start_ignores_quantized_velocity(self):
        target = self.base.copy()
        target[2] += 2
        self.plant.move_to(target)
        points, callback = self.plant._attempt.call_args.args
        self.assertIsNone(callback)
        self.assertEqual(len(points), 2)
        np.testing.assert_array_equal(points[0][0], self.base)
        np.testing.assert_array_equal(points[0][1], np.zeros(7))
        self.assertEqual(points[0][2], 0.0)
        np.testing.assert_array_equal(points[1][0], target)
        np.testing.assert_array_equal(points[1][1], np.zeros(7))
        self.assertAlmostEqual(points[1][2], max(
            hardware.MINIMUM_SEGMENT_S, hardware.PEAK_TO_AVERAGE * 2 / 10))
        self.assertGreaterEqual(self.now - self.started, hardware.STATIONARY_START_WINDOW_S)

    def test_moving_frozen_or_stale_start_never_sends_a_goal(self):
        for field in ("moving", "frozen", "stale"):
            with self.subTest(field=field):
                setattr(self, field, True)
                with self.assertRaisesRegex(hardware.TelemetryUnavailable, "stationary start"):
                    self.plant.move_to(self.base)
                setattr(self, field, False)
                self.plant._attempt.assert_not_called()

    def test_stop_pause_and_guard_trip_prevent_zero_velocity_start(self):
        self.plant.set_stop_requested(lambda: True)
        with self.assertRaisesRegex(RuntimeError, "stop requested"):
            self.plant.move_to(self.base)
        self.plant.set_stop_requested(None)
        self.plant.set_pause_requested(lambda: True)
        with self.assertRaises(hardware.MotionPaused):
            self.plant.move_to(self.base)
        self.plant.set_pause_requested(None)
        self.plant._monitor_trip = "joint3 peak current"
        self.plant._monitor_trip_detail = {"joint": 2, "kind": "peak_current"}
        with self.assertRaises(DriveLimitExceeded):
            self.plant.move_to(self.base)
        self.plant._attempt.assert_not_called()

    def test_retry_rebuilds_start_and_duration_from_new_position(self):
        target = self.base + 30
        starts = []

        def attempt(points, _callback):
            starts.append(points)
            if len(starts) == 1:
                self.base += 5
                raise MotionFailed("rejected")

        self.plant._attempt.side_effect = attempt
        self.plant.move_to(target)
        self.assertEqual(len(starts), 2)
        np.testing.assert_array_equal(starts[1][0][0] - starts[0][0][0], np.full(7, 5))
        self.assertLess(starts[1][-1][2], starts[0][-1][2])

    def test_explicit_sweep_points_are_not_rewritten(self):
        points = [(self.base, np.ones(7), 1.0), (self.base + 1, np.zeros(7), 2.0)]
        self.plant._execute(points)
        self.assertIs(self.plant._attempt.call_args.args[0], points)
        self.assertEqual(self.now, self.started)

    def test_invalid_target_never_spins_or_sends(self):
        for target in ([0] * 6, [float("nan")] * 7, [[0]] * 7):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.plant.move_to(target)
        self.plant._attempt.assert_not_called()
        self.assertEqual(self.now, self.started)


if __name__ == "__main__":
    unittest.main()
