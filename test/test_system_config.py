"""System configuration is independent of saved cell state and ROS."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from robot_parameter_identification.system_config import (
    default_system_config_path,
    load_system_config,
    resolved_controls,
    system_defaults,
)


def test_missing_config_creates_complete_defaults(tmp_path):
    path = tmp_path / "new" / "system_config.yaml"
    loaded = load_system_config(path)
    assert loaded.path == path
    assert yaml.safe_load(path.read_text()) == system_defaults()
    assert loaded.values["ros"]["output_directory"] == "identification_results"
    assert loaded.values["dashboard"]["drag_test"]["maximum_speed_deg_s"] == 120.0


def test_default_location_uses_user_config_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    expected = tmp_path / "robot_parameter_identification" / "system_config.yaml"
    assert default_system_config_path() == expected
    assert load_system_config().path == expected


def test_selected_configs_override_defaults_without_mutating_them(tmp_path):
    first = tmp_path / "first.yaml"
    original = "ros:\n  output_directory: /tmp/another-cell\ndashboard:\n  drag_test:\n    maximum_speed_deg_s: 80.0\n"
    first.write_text(original)
    loaded = load_system_config(first)
    assert loaded.values["ros"]["output_directory"] == "/tmp/another-cell"
    assert loaded.values["dashboard"]["drag_test"]["maximum_speed_deg_s"] == 80.0
    assert loaded.values["dashboard"]["hold_test"]["poses"] == 5
    loaded.values["dashboard"]["hold_test"]["poses"] = 2
    second = load_system_config(tmp_path / "second.yaml")
    assert second.values["dashboard"]["hold_test"]["poses"] == 5
    assert first.read_text() == original


@pytest.mark.parametrize("document", [
    "[]", "", "ros: [", "schema_version: 2", "schema_version: true",
    "output_directory: other", "ros:\n  port: wrong", "ros:\n  port: true",
    "ros:\n  port: 70000", "ros:\n  telemetry_stale_s: .nan",
    "ros:\n  extra_telemetry_topics: [/topic, 12]",
    "gravity:\n  static_poses: 20",
])
def test_invalid_config_is_rejected_without_overwriting(tmp_path, document):
    path = tmp_path / "system_config.yaml"
    path.write_text(document)
    with pytest.raises(ValueError):
        load_system_config(path)
    assert path.read_text() == document


def test_concurrent_startup_publishes_one_complete_file(tmp_path):
    path = tmp_path / "system_config.yaml"
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: load_system_config(path), range(8)))
    assert all(result.values == system_defaults() for result in results)
    assert list(tmp_path.iterdir()) == [path]


def test_controls_resolve_one_source_of_values(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("dashboard:\n  drag_test:\n    maximum_speed_deg_s: 75.0\ncampaign:\n  static_poses: 30\n")
    controls = resolved_controls(load_system_config(path).values)
    assert len(controls) == 36
    assert controls["gravtest-speed-stop"] == {"value": 75.0, "min": 1.0, "max": 120.0, "step": 1.0}
    assert controls["grav-poses"]["value"] == 30
    assert controls["show-mesh"]["value"] is True


@pytest.mark.parametrize("document", [
    "dashboard:\n  drag_test:\n    maximum_speed_deg_s: 121.0",
    "ranges:\n  drag_test:\n    maximum_speed_deg_s: {min: 130.0, max: 120.0}",
    "runtime:\n  jog_poll_s: 0.0",
    "ui:\n  polling:\n    state_ms: 0",
    "ui:\n  scene:\n    camera_position_m: [1.0]",
    "campaign:\n  static_poses: -5",
    "controls:\n  home-speed:\n    source: wrong.path",
])
def test_invalid_operating_defaults_fail_before_startup(tmp_path, document):
    path = tmp_path / "invalid.yaml"
    path.write_text(document)
    with pytest.raises(ValueError, match="system_config"):
        load_system_config(path)