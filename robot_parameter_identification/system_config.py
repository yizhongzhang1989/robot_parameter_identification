"""System defaults shared by the ROS entry point and the dashboard."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import os
import re
import tempfile

import yaml


TEMPLATE_PATH = Path(__file__).parent / "config" / "system_config.yaml"


@dataclass(frozen=True)
class SystemConfig:
    path: Path | None
    values: dict


def default_system_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "robot_parameter_identification" / "system_config.yaml"


@lru_cache(maxsize=1)
def _template_defaults() -> dict:
    return yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))


def system_defaults() -> dict:
    return deepcopy(_template_defaults())


def system_default(section: str, name: str):
    return deepcopy(_template_defaults()[section][name])


def plan_defaults(values: dict) -> dict:
    def convert(value):
        if isinstance(value, list):
            return tuple(convert(entry) for entry in value)
        return value
    return {name: convert(value) for name, value in values.items()}


def default_system_config() -> SystemConfig:
    return SystemConfig(None, system_defaults())


def configured_range(values: dict, name: str) -> tuple[float, float]:
    entry = values["ranges"]
    try:
        for part in name.split("."):
            entry = entry[part]
        return entry["min"], math.inf if entry["max"] is None else entry["max"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown system_config range: {name}") from error


def checked_value(value, values: dict, name: str, *, integer=False, ceiling=None):
    low, high = configured_range(values, name)
    if ceiling is not None:
        high = min(high, ceiling)
    label = name.rsplit(".", 1)[-1]
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} is not numeric") from None
    if (isinstance(value, bool) or not math.isfinite(number)
            or not low <= number <= high or (integer and not number.is_integer())):
        suffix = " and be an integer" if integer else ""
        raise ValueError(f"{label} must be in [{low:g}, {high:g}]{suffix}")
    return int(number) if integer else number


def validate_ranges(values: dict) -> None:
    defaults = system_defaults()

    def check(entry, path):
        if "min" in entry or "max" in entry:
            if set(entry) != {"min", "max"}:
                raise ValueError(f"{path} requires min and max")
            endpoints = [entry["min"]] + ([] if entry["max"] is None else [entry["max"]])
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in endpoints):
                raise ValueError(f"{path} requires finite numeric endpoints")
            if entry["max"] is not None and entry["min"] > entry["max"]:
                raise ValueError(f"{path}.min must not exceed max")
            if entry["min"] < 0:
                raise ValueError(f"{path}.min must be non-negative")
            parts = path.split(".")[2:]
            default_value = None
            if parts[0] in ("campaign", "load_sweep"):
                default_value = defaults[parts[0]].get(parts[-1])
            elif parts[0] in defaults["dashboard"]:
                default_value = defaults["dashboard"][parts[0]].get(parts[-1])
            if type(default_value) is int and any(not float(value).is_integer() for value in endpoints):
                raise ValueError(f"{path} must have integer endpoints")
            if type(default_value) is int and parts[-1] not in ("seed", "skip_budget") and entry["min"] < 1:
                raise ValueError(f"{path} must allow at least one item")
        else:
            for name, child in entry.items():
                if not isinstance(child, dict):
                    raise ValueError(f"{path}.{name} must be a range mapping")
                check(child, f"{path}.{name}")
    check(values["ranges"], "system_config.ranges")


def _merge(default, supplied, name: str):
    if supplied is None and name.startswith("system_config.ranges.") and name.endswith(".max"):
        return None
    if isinstance(default, dict):
        if not isinstance(supplied, dict):
            raise ValueError(f"{name} must be a mapping")
        unknown = set(supplied).difference(default)
        if unknown:
            raise ValueError(f"unknown system_config keys in {name}: {sorted(unknown)}")
        return {
            key: _merge(value, supplied[key], f"{name}.{key}")
            if key in supplied else deepcopy(value)
            for key, value in default.items()
        }
    if isinstance(default, list):
        if not isinstance(supplied, list):
            raise ValueError(f"{name} must be a list")
        if not default:
            if name.endswith("workspace_range_deg"):
                if any(not isinstance(pair, list) or len(pair) != 2 for pair in supplied):
                    raise ValueError(f"{name} must contain lower/upper pairs")
                bounds = [[_merge(0.0, value, name) for value in pair] for pair in supplied]
                if any(low >= high for low, high in bounds):
                    raise ValueError(f"{name} has an invalid range")
                return bounds
            prototype = 0 if name.endswith(".joints") else 0.0
            return [_merge(prototype, value, f"{name}[{index}]")
                    for index, value in enumerate(supplied)]
        return [_merge(default[0], value, f"{name}[{index}]")
                for index, value in enumerate(supplied)] if default else deepcopy(supplied)
    if default is None:
        if supplied is None:
            return None
        return _merge(0.0, supplied, name)
    if isinstance(default, float):
        if isinstance(supplied, bool) or not isinstance(supplied, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(supplied):
            raise ValueError(f"{name} must be a finite number")
        return float(supplied)
    if type(supplied) is not type(default):
        raise ValueError(f"{name} must be {type(default).__name__}")
    return supplied


def _legacy_control_ranges(supplied: dict, defaults: dict) -> dict:
    supplied = deepcopy(supplied)
    controls = supplied.get("controls")
    if not isinstance(controls, dict):
        return supplied
    pending = {}
    for name, descriptor in controls.items():
        if not isinstance(descriptor, dict):
            continue
        reference = defaults["controls"].get(name, {}).get("range")
        if not reference:
            continue
        for endpoint in ("min", "max"):
            if endpoint in descriptor:
                pending.setdefault((reference, endpoint), []).append(descriptor.pop(endpoint))
    for (reference, endpoint), candidates in pending.items():
        path = reference.split(".")
        explicit = supplied.get("ranges", {})
        factory = defaults["ranges"]
        for part in path:
            explicit = explicit.get(part, {}) if isinstance(explicit, dict) else {}
            factory = factory[part]
        if isinstance(explicit, dict) and endpoint in explicit:
            continue
        changed = [value for value in candidates if value != factory[endpoint]]
        chosen = changed[0] if changed else candidates[0]
        if any(value != chosen for value in changed):
            raise ValueError(f"conflicting legacy controls; set ranges.{reference}.{endpoint} explicitly")
        target = supplied.setdefault("ranges", {})
        for part in path:
            if not isinstance(target, dict):
                raise ValueError("system_config.ranges must be a mapping")
            target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"system_config.ranges.{reference} must be a mapping")
        target[endpoint] = chosen
    return supplied


def resolved_controls(values: dict) -> dict:
    controls = {}
    for name, descriptor in values["controls"].items():
        value = values
        try:
            for key in descriptor["source"].split("."):
                value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"invalid system_config control source: {name}") from error
        if type(value) not in (bool, int, float, str):
            raise ValueError(f"system_config control {name} must have a scalar value")
        bounds = {key: descriptor[key] for key in ("min", "max", "step")
                  if key in descriptor}
        if descriptor.get("range"):
            low, high = configured_range(values, descriptor["range"])
            bounds["min"] = low
            if math.isfinite(high):
                bounds["max"] = high
        if bounds.get("step", 1) <= 0 or bounds.get("min", 0) > bounds.get("max", math.inf):
            raise ValueError(f"invalid system_config control bounds: {name}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or not bounds.get("min", -math.inf) <= value <= bounds.get("max", math.inf):
                raise ValueError(f"system_config default is outside control bounds: {name}")
        controls[name] = {"value": value, **bounds}
    return controls


def validate_system_config(values: dict) -> None:
    validate_ranges(values)
    derivation = values["profile_derivation"]
    for name in ("default_speed_fraction", "requested_speed_fraction", "position_fraction"):
        if not 0 < derivation[name] <= 1:
            raise ValueError(f"system_config.profile_derivation.{name} must be in (0, 1]")
    if derivation["maximum_default_speed_deg_s"] <= 0 or derivation["peak_speed_multiplier"] < 1:
        raise ValueError("system_config.profile_derivation requires positive speed and peak multiplier >= 1")
    if values["planning"]["acceleration_per_speed_s_inv"] <= 0:
        raise ValueError("system_config.planning.acceleration_per_speed_s_inv must be positive")
    hold = values["dashboard"]["hold_test"]
    for name in ("planning_drift_deg", "target_tolerance_deg", "target_timeout_s", "child_timeout_margin_s"):
        if hold[name] <= 0:
            raise ValueError(f"system_config.dashboard.hold_test.{name} must be positive")
    if hold["path_samples"] < 2 or hold["replan_attempts"] < 1:
        raise ValueError("system_config hold planning needs >=2 path samples and >=1 attempt")
    for group in ("hold_test", "drag_test"):
        for name in values["ranges"][group]:
            default = values["dashboard"][group].get(name)
            if default is not None:
                try:
                    checked_value(default, values, f"{group}.{name}", integer=type(default) is int)
                except ValueError as error:
                    raise ValueError(f"system_config.dashboard.{group}.{name}: {error}") from error
    for name, suffix in (("cell_config_filename", "json"), ("profile_filename", "yaml")):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\." + suffix, values["storage"][name]):
            raise ValueError(f"system_config.storage.{name} must be a simple .{suffix} filename")
    if not 1 <= values["ros"]["port"] <= 65535:
        raise ValueError("system_config.ros.port must be in [1, 65535]")
    if not values["ros"]["output_directory"].strip():
        raise ValueError("system_config.ros.output_directory must not be blank")
    for name in ("telemetry_stale_s",):
        if values["ros"][name] <= 0:
            raise ValueError(f"system_config.ros.{name} must be positive")
    for name in ("safety_margin_m", "maximum_speed_deg_s"):
        if not math.isfinite(values["ros"][name]) or values["ros"][name] < 0:
            raise ValueError(f"system_config.ros.{name} must not be negative")
    for section, names in ((values["dashboard"]["gravity"], ("gravity_probe_speeds_deg_s",)),
                           (values["campaign"], ("friction_speeds_deg_s", "validation_speeds_deg_s",
                                                 "optimal_friction_speeds_deg_s", "coulomb_transition_search"))):
        for name in names:
            if not section[name] or any(value <= 0 for value in section[name]):
                raise ValueError(f"system_config {name} must contain positive values")
    if values["ros"]["effort_source"] not in ("current", "torque"):
        raise ValueError("system_config.ros.effort_source must be current or torque")
    for section in ("runtime", "rehearsal", "campaign", "load_sweep"):
        for name, value in values[section].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value < 0 or (value == 0 and name not in (
                        "seed", "fourier_ramp_s", "coulomb_transition_deg_s",
                        "skip_budget", "noise", "recovery_tolerance",
                        "gravity_holdout_tolerance", "coulomb_base",
                        "coulomb_per_joint", "viscous_base", "viscous_per_joint")):
                    raise ValueError(f"system_config.{section}.{name} is outside its range")
    for group in ("polling", "telemetry", "editor"):
        if any(value <= 0 for value in values["ui"][group].values()):
            raise ValueError(f"system_config.ui.{group} values must be positive")
    scene = values["ui"]["scene"]
    if not 0 < scene["camera_fov_deg"] < 180 or scene["smoothing_tau_s"] < 0:
        raise ValueError("system_config.ui.scene has invalid camera or smoothing values")
    for name in ("pixel_ratio_max", "grid_size_m", "grid_divisions", "axes_size_m"):
        if scene[name] <= 0:
            raise ValueError(f"system_config.ui.scene.{name} must be positive")
    for vector in (scene["camera_position_m"], scene["camera_target_m"],
                   values["ui"]["obstacle"]["size_m"],
                   values["ui"]["obstacle"]["xyz_m"],
                   values["ui"]["obstacle"]["rpy_deg"]):
        if len(vector) != 3:
            raise ValueError("system_config scene vectors must have three components")
    if any(value <= 0 for value in values["ui"]["obstacle"]["size_m"]):
        raise ValueError("system_config obstacle dimensions must be positive")
    plot = values["ui"]["plot"]
    if plot["mode"] not in ("off", "normal", "big") or not plot["signal"]:
        raise ValueError("system_config.ui.plot has an invalid mode or signal")
    if not 0 < plot["height"]["normal"] <= plot["height"]["big"]:
        raise ValueError("system_config.ui.plot has invalid heights")
    tour = scene["tour"]
    if any(value <= 0 for value in tour.values()) or tour["min_segment_ms"] > tour["max_segment_ms"]:
        raise ValueError("system_config.ui.scene.tour has invalid timing")
    resolved_controls(values)


def _create_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(TEMPLATE_PATH.read_text(encoding="utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_system_config(path: str | Path | None = None) -> SystemConfig:
    selected = Path(path).expanduser() if path else default_system_config_path()
    selected = selected.resolve()
    if not selected.exists():
        _create_config(selected)
    try:
        supplied = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid system_config YAML at {selected}: {error}") from error
    defaults = system_defaults()
    if not isinstance(supplied, dict):
        raise ValueError(f"system_config at {selected} must be a mapping")
    version = supplied.get("schema_version", defaults["schema_version"])
    if type(version) is not int or version != defaults["schema_version"]:
        raise ValueError(f"unsupported system_config schema_version: {version!r}")
    values = _merge(defaults, _legacy_control_ranges(supplied, defaults), "system_config")
    validate_system_config(values)
    return SystemConfig(selected, values)


def write_system_config_snapshot(directory: str | Path, values: dict) -> Path:
    validate_system_config(values)
    path = Path(directory) / "system_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, allow_unicode=True, sort_keys=False)
    return path