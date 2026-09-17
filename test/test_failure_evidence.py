import json
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fixtures import rm75_profile
from robot_parameter_identification.plants.ros_control import (
    DriveLimitExceeded, HardwarePlant,
)


def test_first_trip_survives_rollback_and_mutation_until_new_monitor():
    monitor = Mock(last_trip={"kind": "peak_current", "joint": 6})
    monitor.check.side_effect = lambda sample, now: (
        "J7 peak current 0.804 > 0.800 A" if sample["current_a"][6] > 0.8 else None)
    plant = HardwarePlant(rm75_profile(), monitor=monitor)
    assert plant.failure_evidence() == {}
    checkpoint = plant.raw_frame_checkpoint()
    sample = {"current_a": [0.0] * 7, "position_deg": [0.0] * 7,
              "speed_deg_s": [8.0] * 7, "safety_speed_deg_s": [0.1] * 7,
              "voltage_v": [float("nan")] * 7}
    for index in range(45):
        sample["stamp_s"] = float(index)
        plant._check_monitor(sample, 100.0 + index)
    sample["current_a"][6] = 0.804
    sample["stamp_s"] = 45.0
    plant._check_monitor(sample, 145.0)
    with pytest.raises(DriveLimitExceeded) as caught:
        plant._raise_if_monitor_tripped()
    assert (caught.value.kind, caught.value.joint) == ("peak_current", 6)
    original = plant.failure_evidence()
    trip = original["first_trip"]
    assert trip["guard"] == {"kind": "peak_current", "joint": 6,
                             "reason": "J7 peak current 0.804 > 0.800 A"}
    assert trip["received_monotonic_s"] == 145.0
    assert trip["sample"]["stamp_s"] == 45.0
    assert trip["sample"]["current_a"][6] == 0.804
    assert trip["sample"]["speed_deg_s"] == [8.0] * 7
    assert trip["sample"]["safety_speed_deg_s"] == [0.1] * 7
    assert "desired" not in trip and "target" not in trip
    assert len(trip["prior_raw_frames"]) == 40
    assert trip["prior_raw_frames"][0]["sample"]["stamp_s"] == 5.0
    json.dumps(original, allow_nan=False)
    plant.raw_frames.append(sample)
    plant.rollback_raw_frames(checkpoint)
    assert plant.raw_frames == []
    sample["current_a"][6] = 0.0
    plant._check_monitor(sample, 146.0)
    monitor.last_trip.update(kind="speed", joint=0)
    sample["current_a"][6] = 1.0
    plant._check_monitor(sample, 147.0)
    exposed = plant.failure_evidence()
    exposed["first_trip"]["sample"]["current_a"][6] = 99.0
    exposed["first_trip"]["prior_raw_frames"].clear()
    assert plant.failure_evidence() == original
    plant.set_monitor(monitor)
    assert plant.failure_evidence() == {}
    plant._check_monitor(sample, 148.0)
    assert plant.failure_evidence()["first_trip"]["prior_raw_frames"] == []


@pytest.mark.parametrize("published_velocity", [False, True])
def test_callback_keeps_published_and_position_derived_velocity_distinct(published_velocity):
    monitor = Mock(last_trip={"kind": "peak_current", "joint": 6})
    monitor.check.side_effect = [None, "J7 peak current"]
    plant = HardwarePlant(rm75_profile(), monitor=monitor)
    reading = SimpleNamespace(
        interface_names=["position", "current"] + (["velocity"] if published_velocity else []),
        values=[0.0, 0.0] + ([0.4] if published_velocity else []))
    message = SimpleNamespace(
        joint_names=plant.joint_names, interface_values=[reading] * 7,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0)))
    plant._on_state(message)
    reading.values[:2] = [0.01, 0.804]
    message.header.stamp.nanosec = 100000000
    plant._on_state(message)
    trip = plant.failure_evidence()["first_trip"]
    sample = trip["sample"]
    assert sample["stamp_s"] == 10.1
    assert sample["safety_speed_deg_s"] == pytest.approx([math.degrees(0.1)] * 7)
    assert sample["publisher_speed_deg_s"] == (
        pytest.approx([math.degrees(0.4)] * 7) if published_velocity else None)
    assert sample["speed_deg_s"] == (
        sample["publisher_speed_deg_s"] if published_velocity else sample["safety_speed_deg_s"])
    assert trip["received_monotonic_s"] >= trip["prior_raw_frames"][0]["received_monotonic_s"]