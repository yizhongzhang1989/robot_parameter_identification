"""The contract between this module and any robot.

Everything the identification needs arrives over standard ROS interfaces. This
file states exactly which quantities are required, which are optional, and how
a robot's own naming is mapped onto them, so that supporting a new arm is a
configuration change rather than a code change.

Two transports are understood, both standard:

``sensor_msgs/JointState``
    Universal. Carries ``position``, ``velocity`` and ``effort`` only.

``control_msgs/DynamicJointState``
    The ros2_control state broadcaster. Carries arbitrarily named interfaces,
    so anything a driver exposes -- motor current, winding temperature, a fault
    word -- can be read without a custom message.

A robot that publishes neither, or that lacks a quantity, is not special-cased
here: the user is expected to republish what is missing onto one of these two
topics. That keeps this module free of per-robot transport code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace

# The regression needs a commanded-effort-like measurement per joint. Which
# physical quantity that is depends on the drive, and it only sets the units of
# the identified parameters, so the module stays agnostic and just records it.
EFFORT_UNITS = ("ampere", "newton_metre")

# Without these two the campaign cannot run at all.
REQUIRED_SIGNALS = ("position", "effort")
# These improve the fit or the safety envelope but each has a fallback.
OPTIONAL_SIGNALS = ("velocity", "temperature", "voltage", "enabled",
                    "fault_code")
SIGNALS = REQUIRED_SIGNALS + OPTIONAL_SIGNALS


@dataclass(frozen=True)
class SignalMap:
    """Which published interface name carries each quantity we need.

    ``None`` means the robot does not publish it. Only ``velocity`` has a
    numerical fallback (differentiating position); the rest simply degrade the
    safety envelope, and :meth:`missing_guards` says which guards go dark.
    """

    position: str = "position"
    effort: str = "current"
    velocity: str | None = "velocity"
    temperature: str | None = "temperature"
    voltage: str | None = None
    enabled: str | None = None
    fault_code: str | None = None
    # Only labels the identified parameters; no arithmetic depends on it.
    effort_unit: str = "ampere"
    # Angles are radians on every standard ROS topic. A driver that publishes
    # degrees anyway can say so here instead of us guessing from magnitudes.
    position_in_degrees: bool = False

    def __post_init__(self) -> None:
        for name in REQUIRED_SIGNALS:
            value = getattr(self, name)
            if not value or not str(value).strip():
                raise ValueError(f"{name} is required and cannot be blank")
        if self.effort_unit not in EFFORT_UNITS:
            raise ValueError(
                f"effort_unit must be one of {EFFORT_UNITS}, "
                f"got {self.effort_unit!r}")

    def required_interfaces(self) -> tuple[str, ...]:
        """Interface names a sample must carry before it is usable."""
        return tuple(getattr(self, name) for name in REQUIRED_SIGNALS)

    def optional_interfaces(self) -> dict[str, str]:
        """Signal -> interface name, for the optional signals that are mapped."""
        return {name: getattr(self, name) for name in OPTIONAL_SIGNALS
                if getattr(self, name)}

    def missing_guards(self) -> tuple[str, ...]:
        """Safety guards that cannot run because their signal is unmapped.

        Surfaced to the operator rather than silently skipped: a campaign with
        no temperature feed is legitimate, but it must be an informed choice.
        """
        absent = []
        if not self.temperature:
            absent.append("temperature ceiling")
        if not self.enabled:
            absent.append("drive-enabled check")
        if not self.fault_code:
            absent.append("fault-code check")
        if not self.voltage:
            absent.append("bus-voltage window")
        return tuple(absent)

    def as_dict(self) -> dict:
        return asdict(self)

    def with_overrides(self, **overrides) -> "SignalMap":
        known = {key: value for key, value in overrides.items()
                 if key in self.__dataclass_fields__}
        return replace(self, **known)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "SignalMap":
        data = dict(payload or {})
        known = {key: data[key] for key in cls.__dataclass_fields__
                 if key in data}
        for name in OPTIONAL_SIGNALS:
            # Explicit "" and "none" both mean "this robot does not have it".
            value = known.get(name)
            if isinstance(value, str) and value.strip().lower() in ("", "none"):
                known[name] = None
        return cls(**known)


JOINT_STATE_MAP = SignalMap(
    position="position", velocity="velocity", effort="effort",
    temperature=None, voltage=None, enabled=None, fault_code=None,
    effort_unit="newton_metre")
"""Mapping for a robot that only offers ``sensor_msgs/JointState``."""


@dataclass(frozen=True)
class TelemetrySpec:
    """Where joint state comes from and how to read it."""

    # Either topic may be used; dynamic_joint_states wins when both are set,
    # because it is the only one that can carry current or temperature.
    joint_state_topic: str = "/joint_states"
    dynamic_joint_state_topic: str = "/dynamic_joint_states"
    signals: SignalMap = SignalMap()
    # Samples older than this are treated as no sample at all.
    stale_after_s: float = 0.5

    def transport(self) -> str:
        if self.dynamic_joint_state_topic:
            return "dynamic_joint_states"
        if self.joint_state_topic:
            return "joint_states"
        raise ValueError("no telemetry topic configured")

    def topic(self) -> str:
        return (self.dynamic_joint_state_topic
                if self.transport() == "dynamic_joint_states"
                else self.joint_state_topic)

    def describe(self) -> dict:
        """A short self-description for the dashboard's connection panel."""
        return {
            "transport": self.transport(),
            "topic": self.topic(),
            "required": list(self.signals.required_interfaces()),
            "optional": self.signals.optional_interfaces(),
            "missing_guards": list(self.signals.missing_guards()),
            "effort_unit": self.signals.effort_unit,
        }

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["signals"] = self.signals.as_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict | None) -> "TelemetrySpec":
        data = dict(payload or {})
        signals = SignalMap.from_dict(data.get("signals"))
        known = {key: data[key] for key in cls.__dataclass_fields__
                 if key in data and key != "signals"}
        return cls(signals=signals, **known)


@dataclass(frozen=True)
class CommandSpec:
    """How motion is commanded. One standard action, nothing else.

    Identification only ever asks for positions: the effort is the quantity
    being measured, so commanding it would beg the question.
    """

    follow_joint_trajectory_action: str = (
        "/joint_trajectory_controller/follow_joint_trajectory")
    # Optional, and only used to report which controller is active; the module
    # never switches controllers on its own.
    controller_manager: str = "/controller_manager"
    robot_description_topic: str = "/robot_description"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CommandSpec":
        data = dict(payload or {})
        known = {key: data[key] for key in cls.__dataclass_fields__
                 if key in data}
        return cls(**known)
