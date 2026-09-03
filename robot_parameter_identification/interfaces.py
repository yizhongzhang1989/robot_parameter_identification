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

# The regression needs one effort-like measurement per joint. Drives differ:
# some report motor current, some joint torque, some both. Both are mapped
# separately and ``effort_source`` says which one the fit uses, so the unit is
# derived rather than configured and cannot disagree with the channel read.
EFFORT_SOURCES = ("current", "torque")
EFFORT_UNIT_BY_SOURCE = {"current": "ampere", "torque": "newton_metre"}
EFFORT_UNITS = tuple(EFFORT_UNIT_BY_SOURCE.values())

# Without position and the selected effort channel the campaign cannot run.
REQUIRED_SIGNALS = ("position",)


class MotionFailed(RuntimeError):
    """One motion did not run, and the arm is still fit to attempt the next.

    Kept apart from the errors that mean stop, because most of what an arm does
    wrong is momentary: a goal refused while the controller is between
    trajectories, or an acknowledgement that arrives late. Drives reporting a
    fault is not this, and raises through.
    """


class MotionPaused(RuntimeError):
    """A pause was requested and the current controller goal has stopped.

    This is control flow rather than a failed motion.  Campaigns use it to
    discard the unfinished measurement transaction, wait, and retry the same
    target after resume.
    """


class DriveLimitExceeded(MotionFailed):
    """One motion exceeded a drive limit that a smaller motion may satisfy."""

    def __init__(self, message: str, joint: int | None = None,
                 kind: str = "") -> None:
        super().__init__(message)
        self.joint = joint
        self.kind = kind

# These improve the fit or the safety envelope but each has a fallback.
OPTIONAL_SIGNALS = ("velocity", "temperature", "voltage", "enabled",
                    "fault_code")
SIGNALS = REQUIRED_SIGNALS + EFFORT_SOURCES + OPTIONAL_SIGNALS


@dataclass(frozen=True)
class SignalMap:
    """Which published interface name carries each quantity we need.

    ``None`` means the robot does not publish it. Only ``velocity`` has a
    numerical fallback (differentiating position); the rest simply degrade the
    safety envelope, and :meth:`missing_guards` says which guards go dark.
    """

    position: str = "position"
    # Both are mapped by default so a drive publishing either one is readable
    # with no configuration: "effort" is the only effort-like name ros2_control
    # standardises on Humble, and "current" is what the later releases call the
    # other one. A drive publishing both has both recorded.
    current: str | None = "current"
    torque: str | None = "effort"
    # Which of the two the identification regresses against when both arrive.
    # A drive that publishes only the other one overrides this; see
    # :meth:`settled_among`.
    effort_source: str = "current"
    velocity: str | None = "velocity"
    temperature: str | None = "temperature"
    voltage: str | None = None
    enabled: str | None = None
    fault_code: str | None = None
    # Angles are radians on every standard ROS topic. A driver that publishes
    # degrees anyway can say so here instead of us guessing from magnitudes.
    position_in_degrees: bool = False

    def __post_init__(self) -> None:
        for name in REQUIRED_SIGNALS:
            value = getattr(self, name)
            if not value or not str(value).strip():
                raise ValueError(f"{name} is required and cannot be blank")
        if self.effort_source not in EFFORT_SOURCES:
            raise ValueError(
                f"effort_source must be one of {EFFORT_SOURCES}, "
                f"got {self.effort_source!r}")
        if not getattr(self, self.effort_source):
            raise ValueError(
                f"effort_source is {self.effort_source!r} but no "
                f"{self.effort_source} interface is mapped")

    @property
    def effort(self) -> str:
        """The interface the identification actually regresses against."""
        return getattr(self, self.effort_source)

    @property
    def effort_unit(self) -> str:
        return EFFORT_UNIT_BY_SOURCE[self.effort_source]

    def effort_channels(self) -> dict[str, str]:
        """Effort source -> interface name, for every channel that is mapped.

        Ordered by ``EFFORT_SOURCES``, which is also the preference order when
        a drive turns out to publish both.
        """
        return {source: getattr(self, source) for source in EFFORT_SOURCES
                if getattr(self, source)}

    def settled_among(self, sources) -> "SignalMap":
        """This map with ``effort_source`` set to a channel that arrived.

        The configured source is a preference, not a demand. A drive that
        publishes only the other channel gets to veto it, because a source that
        never arrives is not a choice between two quantities -- it is a stall
        with no error message. Callers announce the switch, since the unit of
        every identified parameter changes with it.
        """
        present = [source for source in EFFORT_SOURCES if source in sources]
        if not present or self.effort_source in present:
            return self
        return replace(self, effort_source=present[0])

    def required_interfaces(self) -> tuple[str, ...]:
        """Interface names every sample must carry.

        The effort channels are alternatives rather than requirements -- one of
        them is enough -- so they are listed by :meth:`effort_channels`.
        """
        return (self.position,)

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
        payload = asdict(self)
        # Derived, so they travel with the map instead of being recomputed.
        payload["effort"] = self.effort
        payload["effort_unit"] = self.effort_unit
        return payload

    def with_overrides(self, **overrides) -> "SignalMap":
        known = {key: value for key, value in overrides.items()
                 if key in self.__dataclass_fields__}
        return replace(self, **known)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "SignalMap":
        data = dict(payload or {})
        known = {key: data[key] for key in cls.__dataclass_fields__
                 if key in data}
        for name in OPTIONAL_SIGNALS + EFFORT_SOURCES:
            # Explicit "" and "none" both mean "this robot does not have it".
            value = known.get(name)
            if isinstance(value, str) and value.strip().lower() in ("", "none"):
                known[name] = None
        return cls(**known)


JOINT_STATE_MAP = SignalMap(
    position="position", velocity="velocity",
    # JointState.effort is documented as a torque, so that is what it maps to.
    current=None, torque="effort", effort_source="torque",
    temperature=None, voltage=None, enabled=None, fault_code=None)
"""Mapping for a robot that only offers ``sensor_msgs/JointState``."""


@dataclass(frozen=True)
class TelemetrySpec:
    """Where joint state comes from and how to read it."""

    # Either topic may be used; dynamic_joint_states wins when both are set,
    # because it is the only one that can carry current or temperature.
    joint_state_topic: str = "/joint_states"
    dynamic_joint_state_topic: str = "/dynamic_joint_states"
    # Signals the robot publishes somewhere other than its main state topic:
    # more DynamicJointState topics, merged by joint name. A drive whose
    # current only reaches ROS through a vendor node is republished onto one of
    # these instead of being special-cased here. They win over the main topic,
    # because naming one is an explicit statement of where a signal comes from.
    extra_dynamic_joint_state_topics: tuple[str, ...] = ()
    signals: SignalMap = SignalMap()
    # Samples older than this are treated as no sample at all.
    stale_after_s: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "extra_dynamic_joint_state_topics",
            tuple(str(topic).strip()
                  for topic in self.extra_dynamic_joint_state_topics
                  if str(topic).strip()))

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
            "extra_topics": list(self.extra_dynamic_joint_state_topics),
            "required": list(self.signals.required_interfaces()),
            # Alternatives, not requirements: whichever of these the robot
            # publishes is the one that gets fitted.
            "effort_options": self.signals.effort_channels(),
            "optional": self.signals.optional_interfaces(),
            "missing_guards": list(self.signals.missing_guards()),
            "effort_source": self.signals.effort_source,
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

    @property
    def controller(self) -> str:
        """The controller the action belongs to, e.g. ``/left_arm_jtc``."""
        return self.follow_joint_trajectory_action.rsplit(
            "/follow_joint_trajectory", 1)[0]

    @property
    def controller_state_topic(self) -> str:
        """Where that controller says which joints it actually drives."""
        return f"{self.controller}/controller_state"

    @classmethod
    def for_controller(cls, controller: str, **rest) -> "CommandSpec":
        """Naming the controller is enough; the action follows from it."""
        name = str(controller).strip().strip("/")
        if not name:
            raise ValueError("controller name cannot be blank")
        return cls(follow_joint_trajectory_action=
                   f"/{name}/follow_joint_trajectory", **rest)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "CommandSpec":
        data = dict(payload or {})
        known = {key: data[key] for key in cls.__dataclass_fields__
                 if key in data}
        return cls(**known)
