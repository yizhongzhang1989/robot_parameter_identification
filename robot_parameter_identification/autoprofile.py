"""Build a usable profile from what the robot already publishes.

Asking an operator to hand-write a profile before anything works is a poor
first experience, and most of it is already stated in the URDF. So: take the
joint list from the trajectory controller, the position and speed limits from
the URDF. Current is a measured outcome of position control, not a profile limit.

Position and speed margins come from the system configuration's
``profile_derivation`` policy, optionally supplied for each call. Written
robot profiles are independent of this automatic derivation policy.

"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree

from .profile import RobotProfile
from .system_config import system_default, system_defaults

SPEED_FRACTION = system_default("profile_derivation", "default_speed_fraction")
MAXIMUM_SPEED_DEG_S = system_default(
    "profile_derivation", "maximum_default_speed_deg_s")
SPEED_CEILING_FRACTION = system_default(
    "profile_derivation", "requested_speed_fraction")
POSITION_FRACTION = system_default("profile_derivation", "position_fraction")


def joint_limits(urdf_text: str) -> dict[str, dict]:
    """lower/upper/velocity per named joint, in radians, from the URDF."""
    try:
        root = ElementTree.fromstring(urdf_text)
    except ElementTree.ParseError:
        return {}
    found = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        entry = {}
        for key in ("lower", "upper", "velocity", "effort"):
            raw = limit.get(key)
            if raw is None:
                continue
            try:
                entry[key] = float(raw)
            except ValueError:
                continue
        if entry:
            found[name] = entry
    return found


def derive_profile(urdf_text: str, joint_names, name: str = "derived",
                   workspace_limit_deg=None,
                   speed_limit_deg_s=None, *,
                   policy: dict | None = None) -> RobotProfile:
    """A profile good enough to run, built from the URDF and a joint list.

    ``workspace_limit_deg`` caps how far the campaign may swing each joint,
    regardless of what the URDF permits. The URDF describes the arm, not the
    room it stands in: a stand, a bench or a cable tray is invisible to it. Cap
    the workspace, or model the obstruction as a box, or both.

    ``speed_limit_deg_s`` replaces the conservative default speed, up to
    the policy's ``requested_speed_fraction`` of each joint's URDF rating.
    It exists because viscous friction is invisible at crawling speeds.

    ``policy`` overrides the system template's ``profile_derivation`` values
    for this call only. Fractions must be in (0, 1], the default speed ceiling
    must be positive, and the peak multiplier must be at least one. All
    policy values must be finite numbers.

    Raises ``ValueError`` for an invalid policy or unusable URDF limits,
    because guessing a range for a joint we are about to move is not acceptable.
    """
    derivation = _derivation_policy(policy)
    names = [str(entry) for entry in joint_names]
    if not names:
        raise ValueError("no joint names supplied")
    requested = _requested_speed(speed_limit_deg_s)
    limits = joint_limits(urdf_text)

    positions, speeds = [], []
    missing = []
    for joint in names:
        entry = limits.get(joint) or {}
        lower, upper = entry.get("lower"), entry.get("upper")
        if lower is None or upper is None:
            missing.append(joint)
            continue
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError(f"joint {joint} position limits must be finite")
        # Symmetric because the campaign designs poses about zero; the tighter
        # side wins so the asymmetric half is never exceeded.
        reach = min(abs(lower), abs(upper))
        reach_deg = math.degrees(reach)
        positions.append(_positive_finite(
            min(round(reach_deg * derivation["position_fraction"], 2), reach_deg),
            f"joint {joint} derived position limit"))
        speeds.append(_speed_for(entry.get("velocity"), requested,
                                 policy=derivation))
    if missing:
        raise ValueError(
            "these joints have no position limit in the URDF, so a profile "
            f"cannot be derived: {', '.join(missing)}")

    sustained = _positive_finite(_bounded_round(min(speeds)),
                                 "derived sustained_speed_deg_s")
    peak = _positive_finite(
        max(sustained, _bounded_round(sustained * derivation["peak_speed_multiplier"])),
        "derived peak_speed_deg_s")
    workspace = _workspace(workspace_limit_deg, positions)
    payload = {
        "schema_version": 1,
        "name": name,
        "joints": {"names": names},
        "limits": {
            "position_deg": positions,
        },
        "envelope": {
            "sustained_speed_deg_s": sustained,
            "peak_speed_deg_s": peak,
        },
        "notes": {
            "derived": "Built from /robot_description and the trajectory "
                       "controller's joint list. Position and speed limits come "
                       "from the URDF. Position-controlled motion records "
                       "current without software current limits.",
        },
    }
    if workspace:
        payload["limits"]["workspace_deg"] = workspace
        payload["notes"]["workspace"] = (
            "Capped by the operator because the URDF does not describe what "
            "stands around the arm.")
    if requested is not None:
        payload["notes"]["speed"] = (
            f"Operator raised the campaign speed to {sustained} deg/s "
            f"(asked {requested}, capped at "
            f"{derivation['requested_speed_fraction']:g} of what "
            "the URDF rates each joint for).")
    return RobotProfile.from_dict(
        payload, source="<derived from /robot_description>")


def _workspace(requested, positions: list[float]) -> list[float]:
    """The cap, broadcast to every joint, never looser than the URDF allows."""
    if requested is None:
        return []
    values = ([float(requested)] * len(positions)
              if isinstance(requested, (int, float))
              else [float(entry) for entry in requested])
    if len(values) != len(positions):
        raise ValueError(
            f"workspace_limit_deg needs 1 or {len(positions)} values, "
            f"got {len(values)}")
    if any(value <= 0.0 for value in values):
        raise ValueError("workspace_limit_deg must be positive")
    return [_bounded_round(min(cap, reach)) for cap, reach in zip(values, positions)]


def _bounded_round(value: float) -> float:
    """Round a limit without widening its underlying bound."""
    return min(round(value, 2), value)


def _requested_speed(speed_limit_deg_s) -> float | None:
    if speed_limit_deg_s is None:
        return None
    return _positive_finite(speed_limit_deg_s, "speed_limit_deg_s")


def _positive_finite(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _derivation_policy(policy: dict | None) -> dict:
    """Resolve and validate per-call overrides without mutating shared defaults."""
    values = system_defaults()["profile_derivation"]
    if policy is not None:
        if not isinstance(policy, dict):
            raise ValueError("profile_derivation policy must be a dictionary")
        for name, value in policy.items():
            if name not in values:
                raise ValueError(f"unknown profile_derivation.{name}")
            values[name] = value
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"profile_derivation.{name} must be a finite number")
        values[name] = _positive_finite(value, f"profile_derivation.{name}")
    for name in ("default_speed_fraction", "requested_speed_fraction",
                 "position_fraction"):
        if values[name] > 1.0:
            raise ValueError(f"profile_derivation.{name} must be in (0, 1]")
    if values["peak_speed_multiplier"] < 1.0:
        raise ValueError("profile_derivation.peak_speed_multiplier must be >= 1")
    return values


def _speed_for(velocity_rad_s, requested: float | None, *,
               policy: dict | None = None) -> float:
    """Use policy-capped URDF speed, or the default/request if velocity is absent."""
    derivation = _derivation_policy(policy)
    requested = _requested_speed(requested)
    if velocity_rad_s is None:
        return (derivation["maximum_default_speed_deg_s"]
                if requested is None else requested)
    rated = _positive_finite(
        math.degrees(_positive_finite(velocity_rad_s, "URDF velocity")),
        "URDF velocity in deg/s")
    if requested is None:
        return _positive_finite(
            min(derivation["maximum_default_speed_deg_s"],
                rated * derivation["default_speed_fraction"]),
            "derived default speed")
    return _positive_finite(
        min(requested, rated * derivation["requested_speed_fraction"]),
        "derived requested speed")
