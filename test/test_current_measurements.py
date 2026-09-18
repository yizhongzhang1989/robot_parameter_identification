import json
import math

from robot_parameter_identification.current_measurements import CurrentMeasurements


def test_full_stream_statistics_preserve_signed_values():
    measured = CurrentMeasurements()
    for stamp, value in enumerate((-4.0, 1.0, 2.0)):
        measured.observe({"stamp_s": stamp, "current_a": [value]})
    payload = measured.as_dict(["left_joint1"])
    entry = payload["joints"][0]
    assert payload["policy"] == "record_only"
    assert entry["joint"] == "left_joint1"
    assert entry["samples"] == 3
    assert entry["minimum_a"] == -4.0
    assert entry["maximum_a"] == 2.0
    assert entry["peak_abs_a"] == 4.0
    assert entry["peak_stamp_s"] == 0.0
    assert entry["rms_a"] == math.sqrt(7.0)
    assert not any("reference" in key for key in entry)


def test_missing_and_invalid_channels_are_not_reported_as_zero():
    measured = CurrentMeasurements()
    measured.observe({})
    assert measured.as_dict()["joints"] == []
    measured.observe({"current_a": [math.nan, None]})
    payload = measured.as_dict()
    assert payload["joints"][0]["samples"] == 0
    assert payload["joints"][0]["invalid_samples"] == 1
    assert payload["joints"][0]["rms_a"] is None
    json.dumps(payload, allow_nan=False)


def test_torque_effort_is_never_mislabeled_as_current():
    measured = CurrentMeasurements(channel="drive_current_a")
    measured.observe({"current_a": [99.0]})
    assert measured.as_dict()["joints"] == []
    measured.observe({"current_a": [99.0], "drive_current_a": [2.0]})
    assert measured.as_dict()["joints"][0]["peak_abs_a"] == 2.0


def test_transit_frames_are_measured_without_capture_and_survive_rollback():
    from fixtures import rm75_profile
    from robot_parameter_identification.campaign import DriveMonitor
    from robot_parameter_identification.plants.ros_control import HardwarePlant

    monitor = DriveMonitor()
    plant = HardwarePlant(rm75_profile())
    plant.set_monitor(monitor)
    plant._check_monitor({"stamp_s": 1.0, "current_a": [-8.0] * 7}, 1.0)
    plant._check_monitor({"stamp_s": 2.0, "current_a": [2.0] * 7}, 2.0)
    plant.rollback_raw_frames(0)
    plant._raise_if_monitor_tripped()
    assert plant.raw_frames == []
    entry = monitor.current_measurements.as_dict()["joints"][0]
    assert entry["samples"] == 2
    assert entry["peak_abs_a"] == 8.0


def test_all_trajectory_operations_record_current_without_current_guards():
    from fixtures import rm75_profile
    from robot_parameter_identification.dashboard.service import (
        DashboardConfig, IdentificationService)

    service = IdentificationService(DashboardConfig(), profile=rm75_profile())
    monitor = service._monitor()
    for stamp in (0.0, 10.0):
        assert monitor.check({"current_a": [99.0] * 7}, stamp) is None
    assert "peak-current ceiling" not in monitor.guards()
    assert "sustained-current ceiling" not in monitor.guards()
    assert "temperature ceiling" in monitor.guards()
    assert monitor.current_measurements is not None


def test_large_current_is_kept_by_campaign_without_backoff_or_double_counting():
    from test_campaign import ScriptedPlant, arm_model, small_plan
    from robot_parameter_identification import campaign

    monitor = campaign.DriveMonitor()

    class MeasuredPlant(ScriptedPlant):
        def _frame(self, position, velocity):
            sample = super()._frame(position, velocity)
            monitor.current_measurements.observe(sample)
            return sample

    run = campaign.Campaign(
        arm_model(), MeasuredPlant(current_a=-9.0), small_plan(), monitor=monitor)
    phase = run.run_gravity()
    assert phase.aborted is None
    assert phase.observations > 0
    assert run.skipped == []
    assert all(scale == 1.0 for scale in run.joint_amplitude_scale)
    payload = run.fit().as_dict()
    entry = payload["current_measurements"]["joints"][0]
    assert entry["samples"] == phase.observations
    assert entry["peak_abs_a"] == entry["rms_a"] == 9.0


def test_gravity_entry_preserves_transit_current_when_fault_precedes_first_observation(
        tmp_path, monkeypatch):
    from unittest.mock import Mock
    from fixtures import rm75_profile
    from robot_parameter_identification.dashboard import service
    from robot_parameter_identification.plants.ros_control import HardwarePlant

    made = service.IdentificationService(
        service.DashboardConfig(output_directory=str(tmp_path)), profile=rm75_profile())
    for name in ("_gravity_plan", "_remember_gravity", "_gravity_signature"):
        monkeypatch.setattr(made, name, Mock(return_value={}))
    plant = HardwarePlant(rm75_profile())
    run = Mock(observations=[], skipped=[])

    def fail():
        assert plant.monitor.current_measurements is not None
        plant._check_monitor({"stamp_s": 1.0, "current_a": [-9.0] * 7}, 1.0)
        plant._raise_if_monitor_tripped()
        plant._check_monitor({"stamp_s": 2.0, "current_a": [2.0] * 7,
                              "fault_code": [12] * 7}, 2.0)
        plant._raise_if_monitor_tripped()

    run.run.side_effect = fail
    monkeypatch.setattr(service.campaign_module, "GravityCampaign", Mock(return_value=run))
    monkeypatch.setattr(made, "_build_plant", Mock(return_value=plant))
    monkeypatch.setattr(made, "_release", Mock())
    made._run(service.GRAVITY_MODE)
    folder = tmp_path / made._reports[service.GRAVITY_MODE]["name"]
    payload = json.loads((folder / "result.json").read_text())
    assert "fault" in payload["aborted"]
    assert payload["failure_evidence"]["first_trip"]["guard"]["kind"] == "fault"
    entry = payload["current_measurements"]["joints"][0]
    assert entry["samples"] == 2
    assert entry["peak_abs_a"] == 9.0
    assert entry["joint"] == rm75_profile().joint_names[0]
    assert '"current_measurements":' in (folder / "report.html").read_text()
