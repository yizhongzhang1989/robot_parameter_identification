"""The service and browser receive the same system defaults without hardware."""

from html.parser import HTMLParser
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from robot_parameter_identification.dashboard.http_server import build_routes
from robot_parameter_identification.dashboard.service import (
    DashboardConfig, GRAVITY_HOLD_TEST, GRAVITY_TEST_ACKNOWLEDGEMENT,
    IdentificationService,
)
from robot_parameter_identification.system_config import load_system_config
from fixtures import synthetic_urdf, test_profile as make_profile


def configured_service(tmp_path):
    path = tmp_path / "system.yaml"
    path.write_text(
        "ros:\n  output_directory: " + str(tmp_path / "runs") + "\n"
        "dashboard:\n  drag_test:\n    maximum_speed_deg_s: 80.0\n"
        "  hold_test:\n    poses: 4\n    seconds: 2.0\n    transit_speed_deg_s: 4.0\n"
        "campaign:\n  static_poses: 28\n  transit_speed_deg_s: 7.0\n"
        "  optimal_training_trajectories: 8\n"
        "load_sweep:\n  repeats: 2\n")
    settings = load_system_config(path)
    made = IdentificationService(
        DashboardConfig.from_system_config(settings), profile=make_profile())
    made.adopt_description(synthetic_urdf())
    return made, path


def test_api_resolves_custom_defaults_and_does_not_modify_them(tmp_path):
    made, path = configured_service(tmp_path)
    route = build_routes(made)["/api/system-config"]
    assert route[0] == "GET"
    payload = route[1]({})
    assert payload["path"] == str(path)
    assert payload["controls"]["gravtest-speed-stop"]["value"] == 80.0
    assert payload["controls"]["grav-poses"]["value"] == 28
    assert made.config.output_directory == str(tmp_path / "runs")
    payload["values"]["dashboard"]["drag_test"]["maximum_speed_deg_s"] = 1.0
    assert made.system_config_payload()["controls"]["gravtest-speed-stop"]["value"] == 80.0


def test_every_html_config_control_has_a_backend_default(tmp_path):
    made, _ = configured_service(tmp_path)
    identifiers = set()

    class Controls(HTMLParser):
        def handle_starttag(self, tag, attributes):
            attributes = dict(attributes)
            if "data-system-config" in attributes:
                identifiers.add(attributes["id"])
                assert "value" not in attributes and "checked" not in attributes

    static = Path(__file__).resolve().parents[1] / "robot_parameter_identification/dashboard/static/index.html"
    Controls().feed(static.read_text())
    assert identifiers == set(made.system_config_payload()["controls"])


def test_campaign_and_sweep_defaults_reach_plans(tmp_path):
    made, _ = configured_service(tmp_path)
    assert made.gravity_defaults()["static_poses"] == 28
    assert made._gravity_plan({}).transit_speed_deg_s == 7.0
    assert made._optimal_plan().optimal_training_trajectories == 8
    assert made._sweep_plan().repeats == 2


def test_saved_cell_gravity_overrides_system_without_writing_system(tmp_path):
    made, path = configured_service(tmp_path)
    original = path.read_bytes()
    cell = tmp_path / "cell.json"
    cell.write_text(json.dumps({"schema_version": 1, "gravity": {"static_poses": 32}}))
    made.config.config_file_path = str(cell)
    made._restore_settings()
    assert made.gravity_defaults()["static_poses"] == 32
    assert made._gravity_plan({}).static_poses == 32
    assert made._gravity_plan({"static_poses": 30}).static_poses == 30
    made.save_config()
    assert path.read_bytes() == original
    assert made.system_config_payload()["values"]["campaign"]["static_poses"] == 28


def test_hold_api_uses_configured_defaults_when_options_are_absent(tmp_path):
    made, _ = configured_service(tmp_path)
    with patch.object(made, "_start_planned_hold", Mock(return_value={"ok": True})) as start:
        made.start_gravity_test(GRAVITY_HOLD_TEST, {
            "acknowledgement": GRAVITY_TEST_ACKNOWLEDGEMENT, "plan_id": "reviewed"})
    start.assert_called_once_with("reviewed", 4, 2.0, 4.0)


def test_system_config_cannot_be_overwritten_by_cell_or_profile_save(tmp_path):
    made, path = configured_service(tmp_path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="system_config must be separate"):
        IdentificationService(DashboardConfig.from_system_config(
            load_system_config(path), config_file_path=str(path)))
    made.config.output_directory = str(path.parent)
    with pytest.raises(ValueError, match="system_config must be separate"):
        made.save_profile(path.name)
    assert path.read_bytes() == original