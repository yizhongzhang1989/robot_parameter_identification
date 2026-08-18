"""Build a usable profile from what the robot already publishes.

Asking an operator to hand-write a profile before anything works is a poor
first experience, and most of it is already stated in the URDF. So: take the
joint list from the trajectory controller, the position and speed limits from
the URDF, and be explicit about the one thing that cannot be derived.

**Current ceilings are not derivable.** The URDF states effort in newton-metres;
converting that to a drive current needs the torque constant, which is one of
the quantities being identified. Inventing a number is wrong in both directions
-- too low aborts good runs, too high protects nothing -- so a derived profile
leaves the current guard off and says so. Supply a written profile to turn it on.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree

from .profile import RobotProfile

# Identification is a slow, deliberate exercise; there is no reason to run it
# anywhere near the speeds the URDF permits for ordinary motion.
SPEED_FRACTION = 0.15
MAXIMUM_SPEED_DEG_S = 20.0
POSITION_FRACTION = 0.9


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


def derive_profile(urdf_text: str, joint_names, name: str = "derived") -> RobotProfile:
    """A profile good enough to run, built from the URDF and a joint list.

    Raises ``ValueError`` when a named joint has no position limit, because
    guessing a range for a joint we are about to move is not acceptable.
    """
    names = [str(entry) for entry in joint_names]
    if not names:
        raise ValueError("no joint names supplied")
    limits = joint_limits(urdf_text)

    positions, speeds = [], []
    missing = []
    for joint in names:
        entry = limits.get(joint) or {}
        lower, upper = entry.get("lower"), entry.get("upper")
        if lower is None or upper is None:
            missing.append(joint)
            continue
        # Symmetric because the campaign designs poses about zero; the tighter
        # side wins so the asymmetric half is never exceeded.
        reach = min(abs(lower), abs(upper))
        positions.append(round(math.degrees(reach) * POSITION_FRACTION, 2))
        velocity = entry.get("velocity")
        speeds.append(MAXIMUM_SPEED_DEG_S if velocity is None else
                      min(MAXIMUM_SPEED_DEG_S,
                          math.degrees(velocity) * SPEED_FRACTION))
    if missing:
        raise ValueError(
            "these joints have no position limit in the URDF, so a profile "
            f"cannot be derived: {', '.join(missing)}")

    sustained = round(min(speeds), 2) if speeds else MAXIMUM_SPEED_DEG_S
    return RobotProfile.from_dict({
        "schema_version": 1,
        "name": name,
        "joints": {"names": names},
        "limits": {
            "position_deg": positions,
            # Infinity, not a guess: see the module docstring.
            "continuous_current_a": [math.inf] * len(names),
            "peak_current_a": [math.inf] * len(names),
        },
        "envelope": {
            "sustained_speed_deg_s": sustained,
            "peak_speed_deg_s": round(sustained * 2.0, 2),
        },
        "notes": {
            "derived": "Built from /robot_description and the trajectory "
                       "controller's joint list. Position and speed limits come "
                       "from the URDF; current ceilings are NOT set, so the "
                       "current guard is off. Supply a written profile to "
                       "enable it.",
        },
    }, source="<derived from /robot_description>")


def current_guard_active(profile: RobotProfile) -> bool:
    """False when the profile carries no usable current ceiling."""
    values = list(profile.continuous_current_a) + list(profile.peak_current_a)
    return bool(values) and all(math.isfinite(value) for value in values)
