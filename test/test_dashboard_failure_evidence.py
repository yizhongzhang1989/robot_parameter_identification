import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fixtures import rm75_profile
from robot_parameter_identification.dashboard import service
from robot_parameter_identification.plants.ros_control import HardwarePlant


@pytest.mark.parametrize("mode", ["hardware", service.GRAVITY_MODE])
@pytest.mark.parametrize("outcome", ["aborted", "empty_exception", "salvaged", "fit_error", "build_error"])
def test_failure_report_is_saved_before_release(tmp_path, monkeypatch, outcome, mode):
    made = service.IdentificationService(
        service.DashboardConfig(output_directory=str(tmp_path)), profile=rm75_profile())
    for name in ("_gravity_plan", "_remember_gravity", "_gravity_signature", "_gravity_model_payload"):
        monkeypatch.setattr(made, name, Mock(return_value={}))
    monitor = Mock(last_trip={"kind": "peak_current", "joint": 6})
    monitor.check.return_value = "J7 peak current 0.804 > 0.800 A"
    plant = HardwarePlant(rm75_profile())
    committed = outcome in ("salvaged", "fit_error")
    observations = [SimpleNamespace(phase="A_gravity")] if committed else []
    plant.raw_frames = [{"stamp_s": 1.0}] if committed else []
    checkpoint = plant.raw_frame_checkpoint()
    result = {"mode": mode, "complete": False, "aborted": "guard trip", "joints": []}
    run = Mock(observations=observations, skipped=[])
    run.fit.return_value = SimpleNamespace(as_dict=lambda: dict(result), fits=None)
    if outcome == "fit_error":
        run.fit.side_effect = ValueError("incomplete fit")

    def fail():
        sample = {"stamp_s": 2.0, "current_a": [0.0] * 6 + [0.804]}
        plant._check_monitor(sample, 10.0)
        plant.raw_frames.append(sample)
        plant.rollback_raw_frames(checkpoint)
        if outcome == "aborted":
            return result
        raise RuntimeError("guard trip")

    run.run.side_effect = fail
    monkeypatch.setattr(service.campaign_module, "Campaign", Mock(return_value=run))
    monkeypatch.setattr(service.campaign_module, "GravityCampaign", Mock(return_value=run))
    monkeypatch.setattr(made, "_build_plant", Mock(return_value=plant))
    if outcome == "build_error":
        made._build_plant.side_effect = RuntimeError("plant unavailable")
    monkeypatch.setattr(made, "_monitor", Mock(return_value=monitor))
    older = service.report_module.write_run(tmp_path, {"mode": mode}, stamp="previous")
    made._remember_report(mode, older)
    previous_bytes = (older / "result.json").read_bytes()

    def release(active):
        assert active is (None if outcome == "build_error" else plant)
        folder = tmp_path / made._reports[mode]["name"]
        evidence = json.loads((folder / "failure_evidence.json").read_text())
        saved = json.loads((folder / "result.json").read_text())
        assert evidence == saved["failure_evidence"] == plant.failure_evidence()
        assert saved == made.result
        assert '"failure_evidence":' in (folder / "report.html").read_text()
        assert len((folder / "raw_frames.csv").read_text().splitlines()) == 1 + committed
        assert len((folder / "observations.csv").read_text().splitlines()) == 1 + committed
        assert (older / "result.json").read_bytes() == previous_bytes
        json.dumps(evidence, allow_nan=False)

    released = Mock(side_effect=release)
    monkeypatch.setattr(made, "_release", released)
    made._run(mode)
    released.assert_called_once()
    assert run.fit.call_count == int(committed)
    assert run.observations == observations