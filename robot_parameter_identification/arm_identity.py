"""Validated instance names for identical seven-axis arm workflows."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math
import re
from typing import Sequence
import xml.etree.ElementTree as ET

DEFAULT_MAXIMUM_COMMAND_A = (3.0, 4.1, 3.0, 3.1, 1.1, 1.15, 0.6)
DEFAULT_CONTINUOUS_CURRENT_A = DEFAULT_MAXIMUM_COMMAND_A
MAXIMUM_CONTINUOUS_CURRENT_A = (3.0, 4.1, 3.0, 3.1, 1.5, 1.5, 1.5)
DEFAULT_PEAK_CURRENT_A = (4.0, 5.0, 4.0, 4.0, 1.5, 1.5, 0.8)
MAXIMUM_PEAK_CURRENT_A = (4.0, 5.0, 4.0, 4.0, 1.5, 1.5, 2.0)


@dataclass(frozen=True)
class ArmIdentity:
    name: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError("arm instance must be a valid ROS name component")

    @property
    def prefix(self) -> str:
        return f"{self.name}_"

    @property
    def model_prefix(self) -> str:
        return f"{self.name}_arm_"

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(f"{self.name}_arm_joint{index}" for index in range(1, 8))

    @property
    def trajectory_controller(self) -> str:
        return f"{self.name}_arm_joint_trajectory_controller"

    @property
    def current_controller(self) -> str:
        return f"{self.name}_arm_forward_current_controller"

    @classmethod
    def from_joint_names(cls, joint_names: Sequence[str]) -> ArmIdentity:
        names = tuple(joint_names)
        suffix = "_arm_joint1"
        if not names or not isinstance(names[0], str) or not names[0].endswith(suffix):
            raise ValueError("expected seven ordered joints from one arm instance")
        identity = cls(names[0][:-len(suffix)])
        if names != identity.joint_names:
            raise ValueError("expected seven ordered joints from one arm instance")
        return identity


@dataclass(frozen=True)
class ArmBinding:
    identity: ArmIdentity
    hardware_name: str
    host: str
    port: int
    guard_port: int
    maximum_command_a: tuple[float, ...] = DEFAULT_MAXIMUM_COMMAND_A
    continuous_current_a: tuple[float, ...] = DEFAULT_CONTINUOUS_CURRENT_A
    peak_current_a: tuple[float, ...] = DEFAULT_PEAK_CURRENT_A

    @property
    def current_limits(self) -> dict[str, list[float]]:
        return {"maximum_command_a": list(self.maximum_command_a),
                "continuous_current_a": list(self.continuous_current_a),
                "peak_current_a": list(self.peak_current_a)}

    @classmethod
    def from_description(cls, identity: ArmIdentity, description: str) -> ArmBinding:
        root = ET.fromstring(description)
        systems = root.findall("ros2_control")
        owners = [system for system in systems if
                  set(identity.joint_names).intersection(
                      joint.get("name") for joint in system.findall("joint"))]
        if len(owners) != 1:
            raise ValueError("selected arm must belong to exactly one hardware system")
        system = owners[0]
        joints = system.findall("joint")
        if tuple(joint.get("name") for joint in joints) != identity.joint_names:
            raise ValueError("hardware must own exactly the selected seven ordered joints")
        if system.findtext("hardware/plugin") != "rm_control/RMSystemHardware":
            raise ValueError("selected hardware is not an integrated RealMan current system")
        params = {param.get("name"): param.text or ""
                  for param in system.findall("hardware/param")}
        if (params.get("direct_current") != "true" or
                params.get("read_only") != "false" or
                params.get("direct_current_ack") != "I_ACCEPT_DIRECT_CURRENT_CONTROLLER_RISK"):
            raise ValueError("selected hardware has not enabled acknowledged direct current")
        for joint in joints:
            interfaces = {item.get("name") for item in joint.findall("command_interface")}
            if not {"position", "actuator_current"}.issubset(interfaces):
                raise ValueError("selected hardware lacks position/current command interfaces")
        host = str(ipaddress.ip_address(params.get("ip", "")))
        address = ipaddress.ip_address(host)
        if address.is_unspecified or address.is_multicast:
            raise ValueError("hardware endpoint must be a unicast IP address")
        port = int(params.get("port", "0"))
        guard_port = int(params.get("direct_current_guard_port", "0"))
        if not 1 <= port <= 65535 or not 1 <= guard_port <= 65535:
            raise ValueError("hardware and guard ports must be valid TCP ports")
        for other in systems:
            if (other is system or
                    other.findtext("hardware/plugin") != "rm_control/RMSystemHardware"):
                continue
            peer = {param.get("name"): param.text or ""
                    for param in other.findall("hardware/param")}
            if peer.get("ip") == host and int(peer.get("port", "8080")) == port:
                raise ValueError("multiple arm identities share the same hardware endpoint")
            if (peer.get("direct_current") == "true" and
                    int(peer.get("direct_current_guard_port", "0")) == guard_port):
                raise ValueError("current-enabled arms must have independent guard ports")

        def current_vector(name, default, ceiling):
            values = (tuple(float(value) for value in params[name].split(","))
                      if name in params else default)
            if len(values) != 7 or any(
                    not math.isfinite(value) or not 0 < value <= maximum
                    for value, maximum in zip(values, ceiling)):
                raise ValueError(f"{name} must contain seven finite supported current limits")
            return values

        command = current_vector(
            "direct_current_maximum_command_a",
            DEFAULT_MAXIMUM_COMMAND_A, DEFAULT_MAXIMUM_COMMAND_A)
        continuous = current_vector(
            "direct_current_continuous_a",
            DEFAULT_CONTINUOUS_CURRENT_A, MAXIMUM_CONTINUOUS_CURRENT_A)
        peak = current_vector("direct_current_peak_a", DEFAULT_PEAK_CURRENT_A,
                              MAXIMUM_PEAK_CURRENT_A)
        if any(command_limit > sustained or sustained > peak_limit
               for command_limit, sustained, peak_limit in zip(command, continuous, peak)):
            raise ValueError("current limits require command <= continuous <= peak")
        return cls(identity, system.get("name", ""), host, port, guard_port,
                   command, continuous, peak)
