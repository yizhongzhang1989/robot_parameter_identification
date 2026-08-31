"""Hardware plant tests: the ROS edges are faked, the logic is real."""

import unittest

import numpy as np

try:
    from robot_parameter_identification.plants import ros_control as hardware
    from fixtures import rm75_profile
except ImportError as error:
    raise unittest.SkipTest(f"hardware module unavailable: {error}") from error

try:
    from control_msgs.action import FollowJointTrajectory
except ImportError:  # message packages come from ROS
    FollowJointTrajectory = None

JOINTS = 7


def frame(position=None, speed=None, current=0.5, temperature=35.0,
          voltage=24.0, enabled=True, fault=0):
    return {
        "position_deg": list(np.zeros(JOINTS) if position is None else position),
        "speed_deg_s": list(np.zeros(JOINTS) if speed is None else speed),
        "current_a": [current] * JOINTS,
        "temperature_c": [temperature] * JOINTS,
        "voltage_v": [voltage] * JOINTS,
        "enabled": [enabled] * JOINTS,
        "fault_code": [fault] * JOINTS,
        "arm_status": None,
    }


class DifferentiateTest(unittest.TestCase):
    def test_recovers_a_known_derivative(self):
        times = np.linspace(0.0, 2.0, 201)
        values = np.column_stack([np.sin(2.0 * times)] * 2)
        derivative = hardware.differentiate(times, values, window=5)
        expected = 2.0 * np.cos(2.0 * times)
        interior = slice(5, -5)
        error = np.max(np.abs(derivative[interior, 0] - expected[interior]))
        self.assertLess(error, 0.02)

    def test_short_series_returns_zeros(self):
        self.assertEqual(
            hardware.differentiate([0.0, 0.1], [[1.0], [2.0]]).tolist(),
            [[0.0], [0.0]])

    def test_repeated_timestamps_do_not_divide_by_zero(self):
        derivative = hardware.differentiate(
            [0.0, 0.0, 0.0], [[1.0], [2.0], [3.0]])
        self.assertTrue(np.all(np.isfinite(derivative)))


class AverageTest(unittest.TestCase):
    def test_numeric_channels_are_averaged(self):
        frames = [frame(current=1.0), frame(current=3.0)]
        averaged = hardware._average(frames, JOINTS)
        self.assertAlmostEqual(averaged["current_a"][0], 2.0)

    def test_discrete_channels_take_the_worst_case(self):
        frames = [frame(enabled=True, fault=0), frame(enabled=False, fault=7)]
        averaged = hardware._average(frames, JOINTS)
        self.assertFalse(averaged["enabled"][0])
        self.assertEqual(averaged["fault_code"][0], 7)

    def test_single_frame_passes_through(self):
        only = frame(current=1.25)
        self.assertIs(hardware._average([only], JOINTS), only)


class ScreenTest(unittest.TestCase):
    def setUp(self):
        self.plant = hardware.HardwarePlant(rm75_profile(), )

    def test_limits_come_from_the_commissioned_table(self):
        low, high = self.plant.limits_deg()
        self.assertEqual(len(low), JOINTS)
        self.assertTrue(np.all(low < 0) and np.all(high > 0))

    def test_out_of_range_pose_is_rejected(self):
        self.assertFalse(self.plant.collision_free(np.full(JOINTS, 1e4)))

    def test_collision_model_is_consulted_when_supplied(self):
        asked = []

        class Twin:
            def collision_free(self, pose_deg):
                asked.append(np.asarray(pose_deg).copy())
                return False

        plant = hardware.HardwarePlant(rm75_profile(), collision_model=Twin())
        self.assertFalse(plant.collision_free(np.zeros(JOINTS)))
        self.assertEqual(len(asked), 1)

    def test_without_a_twin_only_joint_limits_apply(self):
        self.assertTrue(self.plant.collision_free(np.zeros(JOINTS)))


class RawFrameGuardTest(unittest.TestCase):
    class Monitor:
        last_trip = None

        def check(self, sample, now):
            if sample["current_a"][3] <= 4.0:
                return None
            self.last_trip = {"joint": 3, "kind": "peak_current"}
            return "joint4 peak current"

    def test_raw_frame_trip_is_latched_until_the_motion_checks_it(self):
        plant = hardware.HardwarePlant(rm75_profile())
        plant.set_monitor(self.Monitor())
        over = frame()
        over["current_a"][3] = 4.1

        plant._check_monitor(over, 2.0)

        with self.assertRaises(hardware.DriveLimitExceeded) as caught:
            plant._raise_if_monitor_tripped()
        self.assertIn("joint4 peak current", str(caught.exception))
        self.assertEqual(caught.exception.joint, 3)
        self.assertEqual(caught.exception.kind, "peak_current")


class PositionRateGuardTest(unittest.TestCase):
    def test_rate_uses_position_and_publisher_time(self):
        plant = hardware.HardwarePlant(rm75_profile())
        first = plant._position_speed(np.zeros(JOINTS), 10.0)
        moved = np.zeros(JOINTS)
        moved[2] = 0.02
        second = plant._position_speed(moved, 10.01)

        np.testing.assert_allclose(first, np.zeros(JOINTS))
        self.assertAlmostEqual(second[2], 2.0, places=6)

    def test_invalid_timestamp_gap_is_not_reported_as_motion(self):
        plant = hardware.HardwarePlant(rm75_profile())
        plant._position_speed(np.zeros(JOINTS), 10.0)
        moved = np.full(JOINTS, 20.0)

        repeated = plant._position_speed(moved, 10.0)

        np.testing.assert_allclose(repeated, np.zeros(JOINTS))

    def test_isolated_quantized_jump_is_smoothed_below_the_limit(self):
        plant = hardware.HardwarePlant(rm75_profile())
        values = (0.0, 0.146, 0.146, 0.452, 0.452)
        speeds = []
        for index, value in enumerate(values):
            position = np.zeros(JOINTS)
            position[3] = value
            speeds.append(plant._position_speed(position, 10.0 + 0.005 * index))

        self.assertGreater((values[3] - values[2]) / 0.005, 60.0)
        self.assertLess(max(abs(speed[3]) for speed in speeds), 60.0)

    def test_sustained_overspeed_is_detected_within_five_frames(self):
        plant = hardware.HardwarePlant(rm75_profile())
        found = []
        for index in range(5):
            position = np.zeros(JOINTS)
            position[3] = 70.0 * 0.005 * index
            found.append(plant._position_speed(position, 10.0 + 0.005 * index))

        self.assertGreater(found[-1][3], 69.9)


class FakeFuture:
    def __init__(self, value, spins_until_done=0):
        self._value = value
        self._remaining = spins_until_done

    def done(self):
        if self._remaining > 0:
            self._remaining -= 1
            return False
        return True

    def result(self):
        return self._value


class FakeHandle:
    def __init__(self, result, spins_until_done=0):
        self.accepted = True
        self.cancelled = False
        self._result = result
        self._spins = spins_until_done

    def get_result_async(self):
        return FakeFuture(self._result, self._spins)

    def cancel_goal_async(self):
        self.cancelled = True
        return FakeFuture(None)


class FakeClient:
    def __init__(self, result, spins_until_done=0):
        self.goals = []
        self._result = result
        self._spins = spins_until_done

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return FakeFuture(FakeHandle(self._result, self._spins))


class FakeRclpy:
    """Pumps a scripted telemetry stream every time the plant spins.

    ``frames`` may be a list to replay or a callable taking the tick index, in
    which case the stream never runs out. A finite list does: hold_pose alone
    can spin tens of thousands of times, and a fake that then repeats its last
    frame forever hands the fit a motionless arm.
    """

    def __init__(self, plant, frames, period_s=0.005):
        self.plant = plant
        self.make = frames if callable(frames) else None
        self.frames = [] if self.make else list(frames)
        self.index = 0
        self.period_s = period_s

    def _tick(self):
        if self.make is None:
            if not self.frames:
                return
            arrived = dict(self.frames[min(self.index, len(self.frames) - 1)])
        else:
            arrived = dict(self.make(self.index))
        # The real callback stamps each frame from the message header and, while
        # a motion is being captured, appends it to the buffer the fit reads.
        # A fake that only sets _latest leaves every windowed phase with nothing
        # to fit, which is not a stand-in for the driver but for a dead topic.
        arrived.setdefault("stamp_s", self.index * self.period_s)
        self.plant._latest = arrived
        self.plant._latest_at = __import__("time").monotonic()
        with self.plant._lock:
            if self.plant._buffering:
                self.plant._buffer.append(arrived)
        self.index += 1

    def spin_once(self, _node, timeout_sec=0.0):
        self._tick()

    def spin_until_future_complete(self, _node, _future, timeout_sec=0.0):
        self._tick()

    def ok(self):
        return True


@unittest.skipIf(FollowJointTrajectory is None, "control_msgs is unavailable")
class MotionTest(unittest.TestCase):
    def setUp(self):
        self.plant = hardware.HardwarePlant(rm75_profile(), 
            hardware.HardwareConfig(settle_s=0.0, settle_samples=1,
                                    stream_rate_hz=1000.0))
        self.result = type("R", (), {
            "result": type("Inner", (), {
                "error_code": FollowJointTrajectory.Result.SUCCESSFUL})()})()
        self.plant._action_type = FollowJointTrajectory
        self.plant._client = FakeClient(self.result)
        self.plant._node = object()
        self.plant._rclpy = FakeRclpy(self.plant, [frame()])
        self.plant._latest = frame()
        self.plant._latest_at = __import__("time").monotonic()

    def test_duration_scales_with_distance_and_speed(self):
        target = np.zeros(JOINTS)
        target[0] = 37.5
        duration = self.plant._duration_for(target, 10.0)
        self.assertAlmostEqual(
            duration, hardware.PEAK_TO_AVERAGE * 37.5 / 10.0, places=6)

    def test_duration_has_a_floor(self):
        self.assertEqual(
            self.plant._duration_for(np.zeros(JOINTS), 10.0),
            hardware.MINIMUM_SEGMENT_S)

    def test_goal_carries_radians_and_split_time(self):
        goal = self.plant._goal([(np.full(JOINTS, 90.0), np.zeros(JOINTS), 2.5)])
        point = goal.trajectory.points[0]
        self.assertAlmostEqual(point.positions[0], np.pi / 2, places=9)
        self.assertEqual(point.time_from_start.sec, 2)
        self.assertEqual(point.time_from_start.nanosec, 500000000)
        self.assertEqual(goal.trajectory.joint_names[0], "right_arm_joint1")

    def test_traverse_cruises_at_constant_speed(self):
        list(self.plant.traverse(2, np.zeros(JOINTS), 20.0, 5.0))
        goal = self.plant._client.goals[-1]
        self.assertEqual(len(goal.trajectory.points), 3)
        cruise = [np.degrees(point.velocities[2])
                  for point in goal.trajectory.points[:2]]
        self.assertAlmostEqual(cruise[0], 5.0, places=6)
        self.assertAlmostEqual(cruise[1], 5.0, places=6)
        self.assertAlmostEqual(
            np.degrees(goal.trajectory.points[-1].velocities[2]), 0.0, places=9)

    def test_traverse_reverses_the_sign_for_negative_travel(self):
        list(self.plant.traverse(1, np.zeros(JOINTS), -20.0, 4.0))
        goal = self.plant._client.goals[-1]
        self.assertAlmostEqual(
            np.degrees(goal.trajectory.points[0].velocities[1]), -4.0, places=6)

    def test_a_crawl_is_commanded_at_the_speed_it_was_asked_for(self):
        # A silent floor here runs the pass at 0.1 deg/s while the tag, the
        # plan and the report all say 0.02, which is worse than refusing it.
        list(self.plant.traverse(2, np.zeros(JOINTS), 0.4, 0.02))
        goal = self.plant._client.goals[-1]
        self.assertAlmostEqual(
            np.degrees(goal.trajectory.points[0].velocities[2]), 0.02,
            places=6)

    def test_the_window_grows_for_a_crawl_but_stays_inside_the_cruise(self):
        ceiling = self.plant.config.window_maximum_span_s
        self.assertLessEqual(self.plant._window_span(5.0), ceiling)
        self.assertLessEqual(self.plant._window_span(1.0, cruise_s=4.0),
                             ceiling)
        crawl = self.plant._window_span(0.02, cruise_s=20.0)
        self.assertGreater(crawl, ceiling)
        self.assertLessEqual(crawl, 10.0)
        # A window longer than the cruise would average the ramps back in.
        self.assertLessEqual(self.plant._window_span(0.02, cruise_s=2.0), 1.0)

    def test_crawl_produces_a_steady_position_fitted_observation(self):
        period_s = 0.005
        speed_deg_s = 0.1

        def crawl(index):
            position = np.zeros(JOINTS)
            position[0] = speed_deg_s * index * period_s
            return frame(position=position, speed=np.full(JOINTS, 8.0))

        self.plant.hold_pose = lambda _pose: frame()
        self.plant._client = FakeClient(self.result, spins_until_done=1000)
        self.plant._rclpy = FakeRclpy(
            self.plant, crawl, period_s=period_s)

        observations = list(self.plant.traverse(
            0, np.zeros(JOINTS), 0.4, speed_deg_s))

        self.assertEqual(len(observations), 1)
        self.assertAlmostEqual(
            observations[0]["speed_deg_s"][0], speed_deg_s, places=6)
        self.assertLess(
            abs(observations[0]["acceleration_deg_s2"][0]), 1e-6)
        self.assertGreater(observations[0]["window_frames"], 100)

    def test_rejected_goal_raises(self):
        class Rejecting(FakeClient):
            def send_goal_async(self, goal):
                handle = FakeHandle(None)
                handle.accepted = False
                return FakeFuture(handle)

        self.plant._client = Rejecting(self.result)
        with self.assertRaises(RuntimeError):
            self.plant.hold_pose(np.zeros(JOINTS))

    def test_track_fits_acceleration_out_of_the_captured_window(self):
        """Phase C exists to measure inertia, and inertia is read off the
        acceleration. It now comes from the same quadratic fit as the speed
        rather than from differentiating the drive's quantised velocity
        channel, so drive a known parabola through and check it comes back."""
        period_s = 0.005
        wanted = 4.0  # deg/s^2

        def parabola(index):
            time_s = index * period_s
            return frame(position=np.full(JOINTS, 0.5 * wanted * time_s ** 2),
                         speed=np.full(JOINTS, wanted * time_s))

        self.plant._client = FakeClient(self.result, spins_until_done=80)
        self.plant._rclpy = FakeRclpy(self.plant, parabola, period_s=period_s)

        class Design:
            duration_s = 1.0

            def sample(self, time_s):
                position = np.full(JOINTS, 0.5 * wanted * time_s ** 2)
                return position, np.full(JOINTS, wanted * time_s), np.zeros(JOINTS)

        frames = list(self.plant.track(Design(), rate_hz=10.0))
        self.assertGreaterEqual(len(frames), 3, "too few windows to fit")
        for fitted in frames:
            self.assertIn("acceleration_deg_s2", fitted)
            found = np.asarray(fitted["acceleration_deg_s2"], dtype=float)
            self.assertTrue(np.all(np.isfinite(found)))
            self.assertTrue(np.allclose(found, wanted, atol=1e-6),
                            f"fitted {found[0]:.4f}, drove {wanted}")

    def test_track_refuses_to_report_too_few_frames(self):
        """Silently returning nothing would poison the inertia fit."""
        self.plant._rclpy = FakeRclpy(self.plant, [])
        self.plant._latest = None

        class Design:
            duration_s = 1.0

            def sample(self, time_s):
                return np.zeros(JOINTS), np.zeros(JOINTS), np.zeros(JOINTS)

        with self.assertRaises(Exception):
            list(self.plant.track(Design(), rate_hz=10.0))


class RefusedGoalTest(unittest.TestCase):
    """Arms drop and refuse the odd goal. Retrying costs seconds; giving up
    costs whatever the run had already measured."""

    def setUp(self):
        self.plant = hardware.HardwarePlant(
            rm75_profile(),
            hardware.HardwareConfig(settle_s=0.0, settle_samples=1,
                                    stream_rate_hz=1000.0,
                                    motion_retry_s=0.0))
        self.result = type("R", (), {
            "result": type("Inner", (), {
                "error_code": FollowJointTrajectory.Result.SUCCESSFUL})()})()
        self.plant._action_type = FollowJointTrajectory
        self.plant._node = object()
        self.plant._rclpy = FakeRclpy(self.plant, [frame()])
        self.plant._latest = frame()
        self.plant._latest_at = __import__("time").monotonic()

    class Balky:
        """Refuses its first `refusals` goals, then behaves."""

        def __init__(self, result, refusals=0, silent=0):
            self.result = result
            self.refusals = refusals
            self.silent = silent
            self.sent = 0

        def send_goal_async(self, _goal):
            self.sent += 1
            if self.sent <= self.silent:
                return FakeFuture(None)      # no answer at all
            handle = FakeHandle(self.result)
            if self.sent <= self.silent + self.refusals:
                handle.accepted = False
            return FakeFuture(handle)

    def _move(self):
        self.plant._execute([(np.zeros(JOINTS), np.zeros(JOINTS), 1.0)])

    def test_a_refused_goal_is_retried_not_fatal(self):
        self.plant._client = self.Balky(self.result, refusals=1)
        self._move()
        self.assertEqual(self.plant._client.sent, 2)

    def test_an_unanswered_goal_is_retried_too(self):
        """A goal the controller never acknowledged is not a goal it refused,
        and the five-second wait expiring says nothing about the arm."""
        self.plant._client = self.Balky(self.result, silent=1)
        self._move()
        self.assertEqual(self.plant._client.sent, 2)

    def test_the_two_failures_are_not_reported_as_the_same_thing(self):
        seen = []
        for kwargs in ({"refusals": 9}, {"silent": 9}):
            self.plant._client = self.Balky(self.result, **kwargs)
            with self.assertRaises(hardware.MotionFailed) as caught:
                self._move()
            seen.append(str(caught.exception))
        self.assertNotEqual(seen[0], seen[1], seen)
        self.assertIn("rejected", seen[0])
        self.assertIn("did not answer", seen[1])

    def test_it_gives_up_after_the_configured_attempts(self):
        self.plant.config.motion_attempts = 3
        self.plant._client = self.Balky(self.result, refusals=99)
        with self.assertRaises(hardware.MotionFailed):
            self._move()
        self.assertEqual(self.plant._client.sent, 3)

    def test_a_drive_limit_goes_to_the_amplitude_backoff_without_recommanding(self):
        calls = 0

        def trip(_points, _on_frame=None):
            nonlocal calls
            calls += 1
            raise hardware.DriveLimitExceeded(
                "joint7 peak current", joint=6, kind="peak_current")

        self.plant._attempt = trip
        with self.assertRaises(hardware.DriveLimitExceeded) as caught:
            self.plant._execute([])
        self.assertEqual(caught.exception.joint, 6)
        self.assertEqual(calls, 1)

    def test_a_faulted_arm_is_not_re_commanded(self):
        """Retrying into a fault is how a bad moment becomes a worse one."""
        faulted = frame()
        faulted["fault_code"] = [0, 0, 7, 0, 0, 0, 0]
        self.plant._rclpy = FakeRclpy(self.plant, [faulted])
        self.plant._latest = faulted
        self.plant._client = self.Balky(self.result, refusals=99)
        with self.assertRaises(RuntimeError) as caught:
            self._move()
        self.assertIn("fault code 7", str(caught.exception))
        self.assertEqual(self.plant._client.sent, 1, "retried into a fault")

    def test_a_disabled_drive_is_not_re_commanded(self):
        off = frame()
        off["enabled"] = [True, True, True, True, False, True, True]
        self.plant._rclpy = FakeRclpy(self.plant, [off])
        self.plant._latest = off
        self.plant._client = self.Balky(self.result, refusals=99)
        with self.assertRaises(RuntimeError) as caught:
            self._move()
        self.assertIn("joint5 is not enabled", str(caught.exception))
        self.assertEqual(self.plant._client.sent, 1)


class StaleTelemetryTest(unittest.TestCase):
    def test_stale_sample_is_not_returned(self):
        plant = hardware.HardwarePlant(rm75_profile(), )
        plant._latest = frame()
        plant._latest_at = 0.0
        self.assertIsNone(plant.sample(maximum_age_s=0.5))

    def test_missing_sample_raises_on_require(self):
        plant = hardware.HardwarePlant(rm75_profile(), )
        with self.assertRaises(hardware.TelemetryUnavailable):
            plant._require_sample()


if __name__ == "__main__":
    unittest.main()


class ContextOwnershipTest(unittest.TestCase):
    """Closing a plant must never touch a context it did not create.

    Two bugs lived here. Shutting down the global context killed the host
    process mid-run; sharing the host's context let two executors mutate one
    wait set from two threads, which crashed rclpy with "wait set index too
    big". The plant now owns an isolated context and executor.
    """

    class FakeRclpy:
        def __init__(self):
            self.shutdowns = []

        def ok(self, context=None):
            return context not in self.shutdowns

        def shutdown(self, context=None):
            self.shutdowns.append(context)

    def _plant(self, own_context):
        plant = hardware.HardwarePlant(
            rm75_profile(), hardware.HardwareConfig())
        fake = self.FakeRclpy()
        plant._rclpy = fake
        plant._owns_node = True
        plant._owns_context = own_context
        plant._context = object() if own_context else None
        return plant, fake

    def test_only_the_plants_own_context_is_shut_down(self):
        plant, fake = self._plant(own_context=True)
        context = plant._context
        plant.close()
        self.assertEqual(fake.shutdowns, [context])

    def test_a_borrowed_context_is_left_alone(self):
        plant, fake = self._plant(own_context=False)
        plant.close()
        self.assertEqual(fake.shutdowns, [])

    def test_close_is_idempotent(self):
        plant, fake = self._plant(own_context=True)
        plant.close()
        plant.close()
        self.assertEqual(len(fake.shutdowns), 1)

    def test_close_shuts_the_executor_down(self):
        plant, _ = self._plant(own_context=True)
        closed = []
        plant._executor = type("E", (), {"shutdown": lambda self: closed.append(1)})()
        plant.close()
        self.assertEqual(closed, [1])
        self.assertIsNone(plant._executor)


class TelemetryRefreshTest(unittest.TestCase):
    """Planning work between moves must not look like a dead telemetry link.

    Subscription callbacks only run while the plant spins. A campaign that
    spends a second designing poses leaves the cached frame older than the
    freshness window, and the next move used to abort the whole run.
    """

    def _plant(self, arrives_after_spin):
        plant = hardware.HardwarePlant(
            rm75_profile(), hardware.HardwareConfig())
        plant.spins = 0

        def spin(seconds):
            plant.spins += 1
            if arrives_after_spin:
                plant._latest = frame()
                plant._latest_at = hardware.time.monotonic()

        plant._spin_for = spin
        return plant

    def test_a_stale_frame_is_refreshed_instead_of_raising(self):
        plant = self._plant(arrives_after_spin=True)
        plant._latest = frame()
        plant._latest_at = hardware.time.monotonic() - 5.0
        self.assertIsNotNone(plant._require_sample())
        self.assertEqual(plant.spins, 1)

    def test_a_genuinely_dead_link_still_raises(self):
        plant = self._plant(arrives_after_spin=False)
        with self.assertRaises(hardware.TelemetryUnavailable):
            plant._require_sample()
        self.assertEqual(plant.spins, 1)

    def test_a_fresh_frame_costs_no_spin(self):
        plant = self._plant(arrives_after_spin=True)
        plant._latest = frame()
        plant._latest_at = hardware.time.monotonic()
        plant._require_sample()
        self.assertEqual(plant.spins, 0)


class CruiseSpeedTest(unittest.TestCase):
    """The cruise segment must actually run at the speed that was asked for.

    The ramps already cover speed*ramp of the distance. Giving the cruise
    segment the full distance/speed as its duration made it run slower, which
    on a +/-2 deg probe cost a quarter of the commanded speed and left samples
    sitting near standstill where friction has no determined sign.
    """

    def _points(self, distance_deg, speed_deg_s):
        plant = hardware.HardwarePlant(
            rm75_profile(), hardware.HardwareConfig())
        captured = {}
        plant._execute = lambda points, on_frame=None: captured.setdefault(
            "points", points)
        start = np.zeros(JOINTS)
        end = start.copy()
        end[0] = distance_deg
        plant._sweep(start, end, speed_deg_s)
        return captured["points"]

    def _cruise_speed(self, distance_deg, speed_deg_s):
        points = self._points(distance_deg, speed_deg_s)
        (entry, _, t_entry), (exit_pose, _, t_exit) = points[0], points[1]
        return abs(exit_pose[0] - entry[0]) / (t_exit - t_entry)

    def test_short_probe_reaches_the_commanded_speed(self):
        self.assertAlmostEqual(self._cruise_speed(10.0, 2.0), 2.0, places=6)

    def test_a_very_short_move_still_reaches_it(self):
        self.assertAlmostEqual(self._cruise_speed(4.0, 2.0), 2.0, places=6)

    def test_a_fast_long_sweep_reaches_it(self):
        self.assertAlmostEqual(self._cruise_speed(20.0, 8.0), 8.0, places=6)

    def test_the_waypoint_velocity_matches_the_geometry(self):
        points = self._points(10.0, 2.0)
        self.assertAlmostEqual(abs(points[0][1][0]), 2.0, places=6)
        self.assertAlmostEqual(abs(points[1][1][0]), 2.0, places=6)
        self.assertAlmostEqual(abs(points[2][1][0]), 0.0, places=6)
