"""Selected configuration owns software limits, independently of factory values."""

import pytest

from robot_parameter_identification.system_config import (
    checked_value, configured_range, load_system_config, resolved_controls,
)


def test_selected_ranges_can_raise_and_lower_factory_bounds(tmp_path):
    path = tmp_path / "system.yaml"
    path.write_text(
        "ranges:\n  motion:\n    transit_speed_deg_s: {min: 0.2, max: 90.0}\n"
        "  hold_test:\n    poses: {min: 2, max: 12}\n"
        "  campaign:\n    static_poses: {min: 3, max: 80}\n",
        encoding="utf-8")
    selected = load_system_config(path).values
    assert configured_range(selected, "motion.transit_speed_deg_s") == (0.2, 90.0)
    assert configured_range(selected, "hold_test.poses") == (2, 12)
    assert configured_range(selected, "campaign.static_poses") == (3, 80)


@pytest.mark.parametrize("bounds", [
    "{min: 10.0, max: 1.0}", "{min: .nan, max: 20.0}",
    "{min: false, max: 20.0}", "{min: 0.1, max: .inf}",
])
def test_invalid_range_is_refused(tmp_path, bounds):
    path = tmp_path / "invalid.yaml"
    path.write_text(f"ranges:\n  motion:\n    transit_speed_deg_s: {bounds}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="system_config"):
        load_system_config(path)


def test_ui_and_execution_share_selected_range(tmp_path):
    path = tmp_path / "selected.yaml"
    path.write_text("ranges:\n  motion:\n    transit_speed_deg_s: {min: 0.2, max: 90.0}\n")
    values = load_system_config(path).values
    assert checked_value(80, values, "motion.transit_speed_deg_s") == 80
    for name in ("home-speed", "jog-speed", "grav-transit-speed", "gravtest-transit-speed"):
        control = resolved_controls(values)[name]
        assert (control["min"], control["max"]) == (0.2, 90.0)
    with pytest.raises(ValueError, match=r"\[0.2, 90\]"):
        checked_value(0.1, values, "motion.transit_speed_deg_s")


def test_legacy_controls_migrate_without_overriding_explicit_ranges(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("controls:\n  home-speed: {min: 0.1, max: 70.0}\n  jog-speed: {min: 0.1, max: 60.0}\n")
    assert configured_range(load_system_config(path).values, "motion.transit_speed_deg_s") == (0.1, 70)
    path.write_text(path.read_text() + "ranges:\n  motion:\n    transit_speed_deg_s: {max: 85.0}\n")
    assert configured_range(load_system_config(path).values, "motion.transit_speed_deg_s") == (0.1, 85)


def test_fractional_integer_request_is_not_silently_truncated(tmp_path):
    values = load_system_config(tmp_path / "system.yaml").values
    with pytest.raises(ValueError, match="integer"):
        checked_value(2.5, values, "hold_test.poses", integer=True)


def test_null_maximum_removes_only_the_software_ceiling(tmp_path):
    path = tmp_path / "unbounded.yaml"
    path.write_text("ranges:\n  motion:\n    transit_speed_deg_s: {max: null}\n"
                    "  campaign:\n    static_poses: {max: null}\n")
    values = load_system_config(path).values
    assert checked_value(100, values, "campaign.static_poses", integer=True) == 100
    assert "max" not in resolved_controls(values)["grav-poses"]
    with pytest.raises(ValueError, match=r"\[0.1, 30\]"):
        checked_value(40, values, "motion.transit_speed_deg_s", ceiling=30)


@pytest.mark.parametrize("document", [
    "ranges:\n  hold_test:\n    poses: {min: 0, max: 20}",
    "ranges:\n  campaign:\n    friction_repeats: {min: 1, max: 2.5}",
    "ranges:\n  campaign:\n    static_poses: {min: -1, max: 80}",
    "ranges:\n  motion:\n    transit_speed_deg_s: {min: 0.1, max: 0.0}",
    "dashboard:\n  hold_test:\n    path_samples: 1",
    "dashboard:\n  hold_test:\n    target_timeout_s: 0.0",
])
def test_invalid_range_structure_and_hold_tolerances_are_refused(tmp_path, document):
    path = tmp_path / "invalid.yaml"
    path.write_text(document)
    with pytest.raises(ValueError, match="system_config"):
        load_system_config(path)


def test_service_uses_selected_limits_and_retains_robot_constraints(tmp_path):
    from types import SimpleNamespace
    from robot_parameter_identification.dashboard.service import DashboardConfig, IdentificationService

    path = tmp_path / "selected.yaml"
    path.write_text("ranges:\n  motion:\n    transit_speed_deg_s: {min: 0.2, max: 90.0}\n")
    made = IdentificationService(DashboardConfig.from_system_config(load_system_config(path)))
    made.profile = SimpleNamespace(sustained_speed_deg_s=100.0)
    assert made._motion_speed({"transit_speed_deg_s": 80.0}, 10.0) == 80.0
    made.profile.sustained_speed_deg_s = 20.0
    with pytest.raises(ValueError, match=r"\[0.2, 20\]"):
        made._motion_speed({"transit_speed_deg_s": 21.0}, 10.0)


def test_service_campaign_options_use_selected_pose_range(tmp_path):
    from robot_parameter_identification.dashboard.service import DashboardConfig, IdentificationService
    from fixtures import synthetic_urdf, test_profile as make_profile

    path = tmp_path / "selected.yaml"
    path.write_text("ranges:\n  campaign:\n    static_poses: {min: 4, max: 80}\n")
    made = IdentificationService(DashboardConfig.from_system_config(load_system_config(path)), profile=make_profile())
    made.adopt_description(synthetic_urdf())
    assert made._gravity_plan({"static_poses": 72}).static_poses == 72
    assert made._gravity_plan({"static_poses": 90}).static_poses == 80
    assert made.system_config_payload()["controls"]["grav-poses"]["max"] == 80
    assert made.snapshot()["control_ranges"]["grav-poses"]["max"] == 80
    speed = made.snapshot()["control_ranges"]["gravtest-transit-speed"]
    assert speed["max"] == made.profile.sustained_speed_deg_s
    assert speed["source"] == "ranges.motion.transit_speed_deg_s"


def test_current_profile_limit_is_distinct_from_configured_ui_limit(tmp_path):
    from robot_parameter_identification.dashboard.service import DashboardConfig, IdentificationService
    from fixtures import synthetic_urdf, test_profile as make_profile

    settings = load_system_config(tmp_path / "system.yaml")
    made = IdentificationService(DashboardConfig.from_system_config(settings), profile=make_profile())
    made.adopt_description(synthetic_urdf())
    payload = made.system_config_payload()
    assert payload["controls"]["home-speed"]["max"] == 60.0
    assert payload["control_ranges"]["home-speed"]["max"] == make_profile().sustained_speed_deg_s
    assert payload["constraints"]["profile_source"] == "configured"


def test_sweep_ranges_validate_inputs_then_intersect_robot_speed(tmp_path):
    from robot_parameter_identification.dashboard.service import DashboardConfig, IdentificationService
    from fixtures import synthetic_urdf, test_profile as make_profile

    path = tmp_path / "system.yaml"
    path.write_text("ranges:\n  load_sweep:\n    repeats: {min: 1, max: 4}\n")
    made = IdentificationService(DashboardConfig.from_system_config(load_system_config(path)), profile=make_profile())
    made.adopt_description(synthetic_urdf())
    assert made._sweep_plan().fastest_deg_s == made.profile.sustained_speed_deg_s
    for repeats in (2.5, 5, True):
        made._options = {"repeats": repeats}
        with pytest.raises(ValueError, match="repeats"):
            made._sweep_plan()