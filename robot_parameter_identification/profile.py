"""Robot profile: every number that is specific to one arm, in one place.

The identification method is generic; the envelope is not. Keeping the two
apart is what lets the same calibration run on a different robot by pointing at
a different YAML file instead of editing code.

A profile carries no method parameters. It answers only "what is this arm, and
what is it allowed to do".
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import yaml

SCHEMA_VERSION = 1


class ProfileError(ValueError):
    """Raised when a profile is missing or internally inconsistent."""


def _sequence(payload: dict, key: str, count: int, name: str) -> tuple[float, ...]:
    value = payload.get(key)
    if value is None:
        raise ProfileError(f"{name}: '{key}' is required")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return tuple(float(value) for _ in range(count))
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ProfileError(
            f"{name}: '{key}' must be a scalar or {count} values, got {value!r}")
    out = []
    for index, entry in enumerate(value):
        try:
            number = float(entry)
        except (TypeError, ValueError) as error:
            raise ProfileError(f"{name}: '{key}[{index}]' is not a number") from error
        if not math.isfinite(number):
            raise ProfileError(f"{name}: '{key}[{index}]' is not finite")
        out.append(number)
    return tuple(out)


def _scalar(payload: dict, key: str, name: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ProfileError(f"{name}: '{key}' is required")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ProfileError(f"{name}: '{key}' is not a number") from error
    if not math.isfinite(number):
        raise ProfileError(f"{name}: '{key}' is not finite")
    return number


def _deep_merge(base: dict, override: dict) -> dict:
    """Mappings merge key by key; everything else is replaced outright.

    Lists are replaced rather than concatenated: a joint list that silently
    grew by inheritance would be far worse than one that has to be restated.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_extends(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Load one YAML and fold it onto whatever it extends."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in (*seen, path))
        raise ProfileError(f"profile inheritance loops: {chain}")
    if not path.is_file():
        raise ProfileError(f"profile not found: {path}")
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ProfileError(f"{path}: top level must be a mapping")

    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    reference = Path(str(parent))
    if not reference.is_absolute():
        candidate = (path.parent / reference)
        if not candidate.is_file():
            candidate = TEMPLATE_DIRECTORY / f"{parent}.yaml"
        reference = candidate
    return _deep_merge(_resolve_extends(reference, (*seen, path)), payload)


@dataclass(frozen=True)
class RobotProfile:
    """One arm's identity and its hardware envelope."""

    name: str
    joint_names: tuple[str, ...]
    joint_prefix: str = ""

    position_limit_deg: tuple[float, ...] = ()
    # Calibration may be confined inside the mechanical limits when the
    # cell contains obstacles the collision model does not describe.
    workspace_limit_deg: tuple[float, ...] = ()
    continuous_current_a: tuple[float, ...] = ()
    peak_current_a: tuple[float, ...] = ()

    temperature_c: float = 45.0
    sustained_speed_deg_s: float = 15.0
    peak_speed_deg_s: float = 30.0
    current_slew_a_s: float = 4.0
    sender_gap_s: float = 0.02
    telemetry_stale_s: float = 0.25
    position_margin_deg: float = 5.0
    minimum_voltage_v: float = 20.0
    maximum_voltage_v: float = 30.0
    sustained_speed_window_s: float = 0.1
    sustained_current_window_s: float = 0.5
    probe_current_fraction: float = 0.5

    source: str = "<built-in>"
    notes: dict = field(default_factory=dict)

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def probe_current_a(self) -> tuple[float, ...]:
        return tuple(
            round(value * self.probe_current_fraction, 4)
            for value in self.continuous_current_a
        )

    def __post_init__(self) -> None:
        count = len(self.joint_names)
        if count == 0:
            raise ProfileError(f"{self.name}: joint_names must not be empty")
        for label in ("position_limit_deg", "continuous_current_a", "peak_current_a"):
            values = getattr(self, label)
            if len(values) != count:
                raise ProfileError(
                    f"{self.name}: '{label}' has {len(values)} entries but there "
                    f"are {count} joints")
        for index in range(count):
            if self.peak_current_a[index] < self.continuous_current_a[index]:
                raise ProfileError(
                    f"{self.name}: joint{index + 1} peak current "
                    f"{self.peak_current_a[index]} is below its continuous "
                    f"{self.continuous_current_a[index]}")
        if self.peak_speed_deg_s < self.sustained_speed_deg_s:
            raise ProfileError(
                f"{self.name}: peak speed is below the sustained speed")
        if self.maximum_voltage_v <= self.minimum_voltage_v:
            raise ProfileError(f"{self.name}: voltage window is empty")

    @classmethod
    def from_dict(cls, payload: dict, source: str = "<dict>") -> "RobotProfile":
        version = payload.get("schema_version", SCHEMA_VERSION)
        if int(version) != SCHEMA_VERSION:
            raise ProfileError(
                f"{source}: schema_version {version} is not supported "
                f"(expected {SCHEMA_VERSION})")

        name = str(payload.get("name") or Path(source).stem)
        joints = payload.get("joints") or {}
        names = joints.get("names")
        prefix = str(joints.get("prefix", ""))
        if not names:
            count = joints.get("count")
            if not count:
                raise ProfileError(f"{source}: joints.names or joints.count required")
            names = [f"{prefix}joint{index}" for index in range(1, int(count) + 1)]
        names = tuple(str(entry) for entry in names)
        count = len(names)

        limits = payload.get("limits") or {}
        envelope = payload.get("envelope") or {}
        optional = {
            key: float(envelope[key])
            for key in (
                "temperature_c", "sustained_speed_deg_s", "peak_speed_deg_s",
                "current_slew_a_s", "sender_gap_s", "telemetry_stale_s",
                "position_margin_deg", "minimum_voltage_v", "maximum_voltage_v",
                "sustained_speed_window_s", "sustained_current_window_s",
                "probe_current_fraction",
            )
            if key in envelope
        }
        return cls(
            name=name,
            joint_names=names,
            joint_prefix=prefix,
            position_limit_deg=_sequence(limits, "position_deg", count, source),
            workspace_limit_deg=(
                _sequence(limits, "workspace_deg", count, source)
                if limits.get("workspace_deg") is not None else ()),
            continuous_current_a=_sequence(
                limits, "continuous_current_a", count, source),
            peak_current_a=_sequence(limits, "peak_current_a", count, source),
            source=source,
            notes=dict(payload.get("notes") or {}),
            **optional,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotProfile":
        path = Path(path)
        if not path.is_file():
            raise ProfileError(f"profile not found: {path}")
        return cls.from_dict(_resolve_extends(path), source=str(path))

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "source": self.source,
            "joints": {
                "prefix": self.joint_prefix,
                "names": list(self.joint_names),
                "count": self.joint_count,
            },
            "limits": {
                "position_deg": list(self.position_limit_deg),
                "continuous_current_a": list(self.continuous_current_a),
                "peak_current_a": list(self.peak_current_a),
                **({"workspace_deg": list(self.workspace_limit_deg)}
                   if self.workspace_limit_deg else {}),
            },
            "envelope": {
                "temperature_c": self.temperature_c,
                "sustained_speed_deg_s": self.sustained_speed_deg_s,
                "peak_speed_deg_s": self.peak_speed_deg_s,
                "current_slew_a_s": self.current_slew_a_s,
                "sender_gap_s": self.sender_gap_s,
                "telemetry_stale_s": self.telemetry_stale_s,
                "position_margin_deg": self.position_margin_deg,
                "minimum_voltage_v": self.minimum_voltage_v,
                "maximum_voltage_v": self.maximum_voltage_v,
                "sustained_speed_window_s": self.sustained_speed_window_s,
                "sustained_current_window_s": self.sustained_current_window_s,
                "probe_current_fraction": self.probe_current_fraction,
            },
            "notes": dict(self.notes),
        }

    def tightened(self, **overrides) -> "RobotProfile":
        """A copy that may only be stricter; loosening is refused.

        Measured envelopes replace the built-in table over time, but that must
        never become a way to quietly raise a ceiling.
        """
        stricter = {
            "temperature_c": min, "sustained_speed_deg_s": min,
            "peak_speed_deg_s": min, "current_slew_a_s": min,
            "position_margin_deg": max, "maximum_voltage_v": min,
            "minimum_voltage_v": max,
        }
        payload = self.as_dict()
        current = {**payload["envelope"]}
        for key, value in overrides.items():
            if key not in stricter:
                raise ProfileError(f"{key} cannot be overridden")
            chooser = stricter[key]
            current[key] = chooser(float(value), current[key])
        payload["envelope"] = current
        return RobotProfile.from_dict(payload, source=f"{self.source} (tightened)")


PROFILE_DIRECTORY = Path(__file__).resolve().parent / "profiles"
TEMPLATE_DIRECTORY = PROFILE_DIRECTORY / "templates"


def available() -> list[str]:
    if not PROFILE_DIRECTORY.is_dir():
        return []
    return sorted(path.stem for path in PROFILE_DIRECTORY.glob("*.yaml"))


def templates() -> list[str]:
    if not TEMPLATE_DIRECTORY.is_dir():
        return []
    return sorted(path.stem for path in TEMPLATE_DIRECTORY.glob("*.yaml"))


def load(name: str) -> RobotProfile:
    """Load a bundled profile by name, e.g. ``load("example_6dof")``."""
    path = PROFILE_DIRECTORY / f"{name}.yaml"
    if not path.is_file():
        raise ProfileError(
            f"unknown profile {name!r}; available: {', '.join(available()) or 'none'}")
    return RobotProfile.from_yaml(path)
