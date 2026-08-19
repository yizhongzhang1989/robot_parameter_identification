"""Collapsing a burst of frames into one high-quality sample."""

from __future__ import annotations

import math
import unittest

import numpy as np

from robot_parameter_identification.plants.ros_control import (
    fit_window, pick_windows)

RATE = 200.0
QUANTUM_DEG = 0.001


def burst(count, speed, start=0.0, acceleration=0.0, t0=100.0,
          joints=2, quantise=True, current=0.5):
    """Frames as the driver would publish them: 200 Hz, position quantised."""
    frames = []
    for index in range(count):
        t = index / RATE
        travel = start + speed * t + 0.5 * acceleration * t * t
        if quantise:
            travel = round(travel / QUANTUM_DEG) * QUANTUM_DEG
        frames.append({
            "stamp_s": t0 + t,
            "position_deg": [travel] + [0.0] * (joints - 1),
            "speed_deg_s": [0.0] * joints,
            "current_a": [current] * joints,
            "temperature_c": [30.0] * joints,
            "voltage_v": [],
            "enabled": [True] * joints,
            "fault_code": [0] * joints,
        })
    return frames


class WindowFitTest(unittest.TestCase):

    def test_a_steady_speed_is_recovered(self):
        fitted = fit_window(burst(21, speed=37.5), 2)
        self.assertAlmostEqual(fitted["speed_deg_s"][0], 37.5, delta=0.05)

    def test_a_stationary_joint_reads_as_stationary(self):
        # The drive's own channel reports up to 7.8 deg/s here.
        fitted = fit_window(burst(21, speed=0.0), 2)
        self.assertLess(abs(fitted["speed_deg_s"][0]), 0.05)

    def test_acceleration_comes_out_of_the_same_fit(self):
        fitted = fit_window(burst(41, speed=10.0, acceleration=200.0), 2)
        self.assertAlmostEqual(fitted["acceleration_deg_s2"][0], 200.0, delta=5.0)

    def test_it_beats_the_quantised_velocity_channel(self):
        """The point of fitting: the drive quantises speed to 0.024 deg/s and
        reads up to 7.8 while provably still."""
        errors = []
        for speed in (0.5, 2.0, 7.5, 30.0, 60.0):
            fitted = fit_window(burst(21, speed=speed), 2)
            errors.append(abs(fitted["speed_deg_s"][0] - speed))
        self.assertLess(max(errors), 0.024,
                        f"worst error {max(errors):.4f} deg/s")

    def test_the_untouched_joints_stay_at_zero(self):
        fitted = fit_window(burst(21, speed=45.0), 2)
        self.assertAlmostEqual(fitted["speed_deg_s"][1], 0.0, places=6)

    def test_current_is_averaged_over_the_window(self):
        frames = burst(21, speed=10.0)
        for index, frame in enumerate(frames):
            frame["current_a"] = [1.0 if index % 2 else 0.0, 0.0]
        fitted = fit_window(frames, 2)
        self.assertAlmostEqual(fitted["current_a"][0], 10 / 21, places=6)

    def test_the_fit_reports_how_straight_the_motion_was(self):
        steady = fit_window(burst(21, speed=20.0), 2)
        ramping = fit_window(burst(21, speed=20.0, acceleration=4000.0), 2)
        self.assertLess(steady["window_fit_rms_deg"], 0.001)
        # A quadratic absorbs constant acceleration, so the residual stays
        # small; what it catches is a frame gap or a jerk, not a clean ramp.
        self.assertLess(ramping["window_fit_rms_deg"], 0.01)

    def test_a_frame_gap_shows_up_in_the_residual(self):
        frames = burst(21, speed=20.0)
        frames[10]["position_deg"] = [frames[10]["position_deg"][0] + 0.5, 0.0]
        fitted = fit_window(frames, 2)
        self.assertGreater(fitted["window_fit_rms_deg"], 0.01)

    def test_too_few_frames_is_refused_rather_than_guessed(self):
        self.assertIsNone(fit_window(burst(2, speed=10.0), 2))

    def test_frames_with_no_time_between_them_are_refused(self):
        frames = burst(10, speed=10.0)
        for frame in frames:
            frame["stamp_s"] = 100.0
        self.assertIsNone(fit_window(frames, 2))

    def test_the_window_records_what_it_was_made_of(self):
        fitted = fit_window(burst(21, speed=10.0), 2)
        self.assertEqual(fitted["window_frames"], 21)
        self.assertAlmostEqual(fitted["window_span_s"], 20 / RATE, places=4)


class WindowPickingTest(unittest.TestCase):

    def test_windows_come_from_the_middle_not_the_ramps(self):
        frames = burst(400, speed=10.0)
        span = frames[-1]["stamp_s"] - frames[0]["stamp_s"]
        for window in pick_windows(frames, 3, 0.1):
            for frame in window:
                offset = frame["stamp_s"] - frames[0]["stamp_s"]
                self.assertGreater(offset, 0.15 * span)
                self.assertLess(offset, 0.85 * span)

    def test_the_requested_number_is_produced(self):
        self.assertEqual(len(pick_windows(burst(400, speed=10.0), 3, 0.1)), 3)

    def test_windows_do_not_overlap(self):
        windows = pick_windows(burst(400, speed=10.0), 3, 0.1)
        for earlier, later in zip(windows, windows[1:]):
            self.assertLess(earlier[-1]["stamp_s"], later[0]["stamp_s"])

    def test_each_window_is_about_the_requested_length(self):
        for window in pick_windows(burst(400, speed=10.0), 3, 0.1):
            span = window[-1]["stamp_s"] - window[0]["stamp_s"]
            self.assertLessEqual(span, 0.1 + 1e-9)
            self.assertGreater(span, 0.05)

    def test_a_short_motion_still_yields_something(self):
        windows = pick_windows(burst(12, speed=10.0), 3, 0.1)
        self.assertTrue(windows)

    def test_nothing_to_pick_from_is_not_an_error(self):
        self.assertEqual(pick_windows([], 3, 0.1), [])


class PrecisionClaimTest(unittest.TestCase):
    """The redesign rests on the fitted speed being far better than the
    reported one. That claim is checked here rather than asserted in prose."""

    REPORTED_NOISE_DEG_S = 1.696   # measured mean on a stationary joint

    def test_a_tenth_of_a_second_resolves_speed_a_hundredfold_better(self):
        worst = 0.0
        for speed in (0.0, 1.0, 5.0, 25.0, 60.0):
            fitted = fit_window(burst(21, speed=speed), 2)
            worst = max(worst, abs(fitted["speed_deg_s"][0] - speed))
        self.assertLess(worst, self.REPORTED_NOISE_DEG_S / 100.0,
                        f"fitted speed error {worst:.5f} deg/s is not a "
                        f"hundredth of the reported channel's noise")


if __name__ == "__main__":
    unittest.main()
