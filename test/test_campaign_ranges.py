"""Selected operator ranges remain bounded by the actual robot profile."""

from copy import deepcopy
from dataclasses import replace
import math

import pytest

from robot_parameter_identification import campaign
from robot_parameter_identification.campaign import (
    campaign_bounds, clamp_campaign_plan, default_plan, sweep_amplitude_deg)
from robot_parameter_identification.system_config import (
    configured_range, system_defaults)
from test_other_robot import load_profile


@pytest.mark.parametrize("low, high, requested, expected", [
    (4, 100, 90, 90),
    (4, 20, 90, 20),
    (30, 60, 10, 30),
    (2, 60, 1, 2),
])
def test_selected_static_pose_range(low, high, requested, expected):
    selected = system_defaults()
    selected["ranges"]["campaign"]["static_poses"] = {
        "min": low, "max": high}

    plan, notes = clamp_campaign_plan(
        {"static_poses": requested}, load_profile(), system_config=selected)

    assert plan.static_poses == expected
    assert any("static_poses" in note for note in notes) == (requested != expected)


@pytest.mark.parametrize("low, high, requested, speed, expected", [
    (0.5, 4.0, 1.0, 0.1, 1.0),
    (90.0, 140.0, 100.0, 12.0, 100.0),
    (90.0, 140.0, 150.0, 12.0, 140.0),
    (90.0, 140.0, 1.0, 12.0, 90.0),
    (0.5, None, 200.0, 12.0, 200.0),
])
def test_amplitude_bounds_survive_defaulting_and_rederivation(
        low, high, requested, speed, expected):
    selected = system_defaults()
    selected["ranges"]["campaign"]["friction_amplitude_deg"] = {
        "min": low, "max": high}
    selected["ranges"]["campaign"]["maximum_speed_deg_s"]["min"] = 0.01
    selected["campaign"].update(
        friction_amplitude_deg=requested, maximum_speed_deg_s=speed)
    profile = load_profile()

    assert sweep_amplitude_deg(
        speed, requested, system_config=selected) == expected
    assert default_plan(
        profile, selected["campaign"],
        system_config=selected).friction_amplitude_deg == expected
    plan, _notes = clamp_campaign_plan(
        {"maximum_speed_deg_s": speed, "friction_amplitude_deg": requested},
        profile, system_config=selected)
    assert plan.friction_amplitude_deg == expected


@pytest.mark.parametrize("low, high, profile_speed, requested, expected", [
    (0.1, 100.0, 120.0, 90.0, 90.0),
    (0.1, 100.0, 75.0, 90.0, 75.0),
    (0.1, 100.0, 12.0, 90.0, 12.0),
    (0.02, 100.0, 12.0, 0.04, 0.04),
    (15.0, 100.0, 120.0, 10.0, 15.0),
    (0.1, 40.0, 120.0, 90.0, 40.0),
])
def test_transit_range_intersects_actual_profile(
        low, high, profile_speed, requested, expected):
    selected = system_defaults()
    selected["ranges"]["motion"]["transit_speed_deg_s"] = {
        "min": low, "max": high}
    selected["campaign"]["transit_speed_deg_s"] = requested
    profile = replace(load_profile(), sustained_speed_deg_s=profile_speed,
                      peak_speed_deg_s=2.0 * profile_speed)

    assert campaign_bounds(profile, system_config=selected)[
        "transit_speed_deg_s"] == (low, min(high, profile_speed))
    assert default_plan(
        profile, system_config=selected).transit_speed_deg_s == expected
    plan, _notes = clamp_campaign_plan(
        {"transit_speed_deg_s": requested}, profile, system_config=selected)
    assert plan.transit_speed_deg_s == expected


@pytest.mark.parametrize("low, high, requested, expected", [
    (1.0, None, 500.0, 12.0),
    (1.0, 500.0, 500.0, 12.0),
    (1.0, 6.0, 12.0, 6.0),
    (0.05, 6.0, 0.1, 0.1),
    (3.0, 6.0, 1.0, 3.0),
])
def test_speed_range_cannot_relax_actual_profile(low, high, requested, expected):
    selected = system_defaults()
    selected["ranges"]["campaign"]["maximum_speed_deg_s"] = {
        "min": low, "max": high}
    selected["campaign"]["maximum_speed_deg_s"] = requested
    profile = load_profile()

    assert default_plan(
        profile, system_config=selected).maximum_speed_deg_s == expected
    plan, _notes = clamp_campaign_plan(
        {"maximum_speed_deg_s": requested}, profile, system_config=selected)
    assert plan.maximum_speed_deg_s == expected
    assert max(plan.friction_speeds_deg_s) == expected
    assert all(speed <= expected for speed in plan.validation_speeds_deg_s)


@pytest.mark.parametrize("automatic", [True, False])
def test_slow_profile_works_when_selected_ranges_allow_it(automatic):
    selected = system_defaults()
    selected["ranges"]["campaign"]["maximum_speed_deg_s"]["min"] = 0.05
    selected["ranges"]["campaign"]["maximum_acceleration_deg_s2"]["min"] = 0.1
    selected["campaign"].update(
        maximum_speed_deg_s=0.3, derive_speed_ladders=automatic,
        friction_speeds_deg_s=[], validation_speeds_deg_s=[])
    profile = replace(load_profile(), sustained_speed_deg_s=0.3)

    plan = default_plan(profile, system_config=selected)
    assert plan.maximum_speed_deg_s == 0.3
    assert plan.transit_speed_deg_s == 0.3
    assert plan.maximum_acceleration_deg_s2 == pytest.approx(1.2)
    assert plan.friction_speeds_deg_s == (0.3,)
    assert plan.validation_speeds_deg_s == ((0.1, 0.2) if automatic else (0.3,))

    plan, _notes = clamp_campaign_plan(
        {"maximum_speed_deg_s": 0.2}, profile, system_config=selected)
    assert plan.maximum_speed_deg_s == 0.2
    assert plan.maximum_acceleration_deg_s2 == pytest.approx(0.8)
    assert plan.friction_speeds_deg_s == (0.2,)
    assert plan.validation_speeds_deg_s == ((0.07, 0.13) if automatic else (0.2,))


@pytest.mark.parametrize("coefficient, low, high, speed, expected", [
    (0.5, 0.1, None, 4.0, 2.0),
    (3.0, 18.0, None, 4.0, 18.0),
    (3.0, 2.0, 8.0, 4.0, 8.0),
    (7.0, 2.0, None, 12.0, 84.0),
])
def test_selected_acceleration_policy_and_range(
        coefficient, low, high, speed, expected, monkeypatch):
    selected = system_defaults()
    selected["planning"]["acceleration_per_speed_s_inv"] = coefficient
    selected["ranges"]["campaign"]["maximum_acceleration_deg_s2"] = {
        "min": low, "max": high}
    selected["campaign"]["maximum_acceleration_deg_s2"] = 1000.0
    profile = load_profile()
    ceiling = min(math.inf if high is None else high,
                  coefficient * profile.sustained_speed_deg_s)
    monkeypatch.setattr(campaign, "ACCELERATION_PER_SPEED", 9999.0)

    assert campaign_bounds(profile, system_config=selected)[
        "maximum_acceleration_deg_s2"] == (low, ceiling)
    assert default_plan(
        profile, system_config=selected).maximum_acceleration_deg_s2 == ceiling
    plan, _notes = clamp_campaign_plan(
        {"maximum_speed_deg_s": speed}, profile, system_config=selected)
    assert plan.maximum_acceleration_deg_s2 == expected
    for requested, bounded in ((low - 1.0, low), (ceiling + 1.0, ceiling)):
        plan, _notes = clamp_campaign_plan(
            {"maximum_speed_deg_s": speed,
             "maximum_acceleration_deg_s2": requested},
            profile, system_config=selected)
        assert plan.maximum_acceleration_deg_s2 == bounded


@pytest.mark.parametrize("low, high, requested, expected", [
    (20.0, 100.0, 90.0, 50.0),
    (20.0, 100.0, 10.0, 20.0),
    (20.0, 40.0, 90.0, 40.0),
])
def test_temperature_range_intersects_actual_profile(
        low, high, requested, expected):
    selected = system_defaults()
    selected["ranges"]["campaign"]["temperature_ceiling_c"] = {
        "min": low, "max": high}
    selected["campaign"]["temperature_ceiling_c"] = requested
    profile = load_profile()

    assert default_plan(
        profile, system_config=selected).temperature_ceiling_c == expected
    plan, _notes = clamp_campaign_plan(
        {"temperature_ceiling_c": requested}, profile, system_config=selected)
    assert plan.temperature_ceiling_c == expected


@pytest.mark.parametrize("configured_minimum, expected", [(2.0, 12.0), (20.0, 20.0)])
def test_position_margin_keeps_actual_profile_minimum(configured_minimum, expected):
    selected = system_defaults()
    selected["ranges"]["campaign"]["position_margin_deg"] = {
        "min": configured_minimum, "max": 40.0}
    profile = replace(load_profile(), position_margin_deg=12.0)

    assert campaign_bounds(profile, system_config=selected)[
        "position_margin_deg"] == (expected, 40.0)
    assert default_plan(
        profile, system_config=selected).position_margin_deg == expected
    plan, _notes = clamp_campaign_plan(
        {"position_margin_deg": 1.0}, profile, system_config=selected)
    assert plan.position_margin_deg == expected


@pytest.mark.parametrize("endpoint", ["min", "max"])
def test_default_plan_bounds_every_configured_numeric_field(endpoint):
    selected = system_defaults()
    selected["ranges"]["campaign"]["static_poses"] = {"min": 6, "max": 100}
    selected["ranges"]["campaign"]["friction_amplitude_deg"] = {
        "min": 0.5, "max": 140.0}
    selected["ranges"]["campaign"]["maximum_speed_deg_s"]["min"] = 0.01
    selected["campaign"]["maximum_speed_deg_s"] = 0.1
    profile = load_profile()
    bounds = campaign_bounds(profile, system_config=selected)

    for name, (low, high) in bounds.items():
        if endpoint == "max" and high == float("inf"):
            continue
        requested = low - 100.0 if endpoint == "min" else high + 100.0
        defaults = {**selected["campaign"], name: requested}
        plan = default_plan(profile, defaults, system_config=selected)
        assert getattr(plan, name) == (low if endpoint == "min" else high), name
        if type(selected["campaign"][name]) is int:
            assert type(getattr(plan, name)) is int, name
        assert all(lower <= getattr(plan, field) <= upper
                   for field, (lower, upper) in bounds.items())


@pytest.mark.parametrize("name", [
    "maximum_speed_deg_s", "transit_speed_deg_s",
    "maximum_acceleration_deg_s2", "temperature_ceiling_c", "position_margin_deg",
])
def test_disjoint_profile_ranges_are_rejected(name):
    selected = system_defaults()
    profile = load_profile()
    overrides = {
        "maximum_speed_deg_s": {"min": profile.sustained_speed_deg_s + 1.0},
        "transit_speed_deg_s": {"min": profile.sustained_speed_deg_s + 1.0},
        "maximum_acceleration_deg_s2": {
            "min": selected["planning"]["acceleration_per_speed_s_inv"]
            * profile.sustained_speed_deg_s + 1.0},
        "temperature_ceiling_c": {"min": profile.temperature_c + 1.0},
        "position_margin_deg": {"max": profile.position_margin_deg - 1.0},
    }
    group = "motion" if name == "transit_speed_deg_s" else "campaign"
    selected["ranges"][group][name].update(overrides[name])

    with pytest.raises(ValueError, match=rf"{name}.*no overlap.*profile"):
        campaign_bounds(profile, system_config=selected)
    with pytest.raises(ValueError, match=rf"{name}.*no overlap.*profile"):
        default_plan(profile, system_config=selected)
    with pytest.raises(ValueError, match=rf"{name}.*no overlap.*profile"):
        clamp_campaign_plan({}, profile, system_config=selected)


@pytest.mark.parametrize("group, name", [
    ("campaign", "static_poses"),
    ("campaign", "friction_amplitude_deg"),
    ("motion", "transit_speed_deg_s"),
])
def test_inverted_configured_ranges_are_rejected(group, name):
    selected = system_defaults()
    selected["ranges"][group][name] = {"min": 10, "max": 5}
    profile = load_profile()

    with pytest.raises(ValueError, match=rf"{name}.*inverted"):
        campaign_bounds(profile, system_config=selected)
    with pytest.raises(ValueError, match=rf"{name}.*inverted"):
        default_plan(profile, system_config=selected)
    with pytest.raises(ValueError, match=rf"{name}.*inverted"):
        clamp_campaign_plan({}, profile, system_config=selected)
    if name == "friction_amplitude_deg":
        with pytest.raises(ValueError, match=rf"{name}.*inverted"):
            sweep_amplitude_deg(1.0, 5.0, system_config=selected)


def test_explicit_config_does_not_read_canonical_fallback(monkeypatch):
    selected = system_defaults()
    selected["campaign"]["static_poses"] = 40
    profile = load_profile()

    def unexpected_fallback():
        raise AssertionError("selected config must not consult canonical defaults")

    monkeypatch.setattr(campaign, "system_defaults", unexpected_fallback)
    assert campaign_bounds(profile, system_config=selected)["static_poses"] == (
        configured_range(selected, "campaign.static_poses"))
    assert default_plan(profile, system_config=selected).static_poses == 40
    plan, _notes = clamp_campaign_plan(
        {"maximum_speed_deg_s": 12.0}, profile, system_config=selected)
    assert plan.static_poses == 40
    assert sweep_amplitude_deg(1.0, 20.0, system_config=selected) == 20.0


def test_selected_explicit_speed_ladders_are_restored_before_rederiving():
    selected = system_defaults()
    selected["ranges"]["campaign"]["friction_amplitude_deg"]["max"] = 140.0
    selected["campaign"].update(
        derive_speed_ladders=False, maximum_speed_deg_s=2.0,
        friction_amplitude_deg=120.0,
        friction_speeds_deg_s=[1.0, 50.0, 90.0],
        validation_speeds_deg_s=[3.0, 80.0])
    profile = replace(load_profile(), sustained_speed_deg_s=120.0,
                      peak_speed_deg_s=240.0)

    plan, _notes = clamp_campaign_plan(
        {"maximum_speed_deg_s": 90.0}, profile, system_config=selected)
    assert plan.friction_speeds_deg_s == (1.0, 50.0, 90.0)
    assert plan.validation_speeds_deg_s == (3.0, 80.0)
    assert plan.friction_amplitude_deg == 120.0

    plan, _notes = clamp_campaign_plan(
        {"maximum_speed_deg_s": 90.0}, profile,
        {"friction_speeds_deg_s": [2.0, 25.0, 120.0]}, system_config=selected)
    assert plan.friction_speeds_deg_s == (2.0, 25.0)
    assert plan.validation_speeds_deg_s == (3.0, 80.0)
    assert plan.friction_amplitude_deg == 120.0


def test_multiple_configs_and_canonical_fallback_are_independent():
    canonical = system_defaults()
    first = deepcopy(canonical)
    second = deepcopy(canonical)
    first["ranges"]["campaign"]["static_poses"]["max"] = 100
    first["ranges"]["campaign"]["friction_amplitude_deg"]["max"] = 140.0
    first["ranges"]["motion"]["transit_speed_deg_s"]["max"] = 100.0
    first["planning"]["acceleration_per_speed_s_inv"] = 3.0
    second["ranges"]["campaign"]["static_poses"]["max"] = 12
    second["ranges"]["campaign"]["friction_amplitude_deg"]["max"] = 10.0
    second["ranges"]["motion"]["transit_speed_deg_s"]["max"] = 20.0
    second["planning"]["acceleration_per_speed_s_inv"] = 1.0
    snapshots = deepcopy((first, second))
    profile = replace(load_profile(), sustained_speed_deg_s=120.0,
                      peak_speed_deg_s=240.0)
    request = {"static_poses": 90, "friction_amplitude_deg": 100.0,
               "maximum_speed_deg_s": 12.0, "transit_speed_deg_s": 90.0}

    first_plan, _notes = clamp_campaign_plan(request, profile, system_config=first)
    second_plan, _notes = clamp_campaign_plan(request, profile, system_config=second)
    canonical_plan, _notes = clamp_campaign_plan(request, profile)
    repeated_plan, _notes = clamp_campaign_plan(request, profile, system_config=first)
    assert first_plan.as_dict() == repeated_plan.as_dict()
    assert (first_plan.static_poses, first_plan.friction_amplitude_deg,
            first_plan.transit_speed_deg_s,
            first_plan.maximum_acceleration_deg_s2) == (90, 100.0, 90.0, 36.0)
    assert (second_plan.static_poses, second_plan.friction_amplitude_deg,
            second_plan.transit_speed_deg_s,
            second_plan.maximum_acceleration_deg_s2) == (12, 10.0, 20.0, 12.0)
    explicit_canonical, _notes = clamp_campaign_plan(
        request, profile, system_config=canonical)
    assert canonical_plan.as_dict() == explicit_canonical.as_dict()
    assert (first, second) == snapshots
    assert system_defaults() == canonical
    assert campaign.ACCELERATION_PER_SPEED == canonical["planning"][
        "acceleration_per_speed_s_inv"]