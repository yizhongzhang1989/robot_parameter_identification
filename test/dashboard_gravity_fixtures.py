"""Local calibration and declared-hardware fixtures; never contact a robot."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from fixtures import synthetic_urdf
from robot_parameter_identification.arm_identity import ArmIdentity


def gravity_source(directory, name="right"):
    identity = ArmIdentity(name)
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "complete": True, "verdict": {"state": "pass"}, "effort_unit": "ampere",
        "joint_names": list(identity.joint_names),
        "joints": [{"columns": [0], "parameters": [0.1]} for _index in range(7)],
    }
    (folder / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return folder


def gravity_urdf(name="right"):
    identity = ArmIdentity(name)
    root = ET.fromstring(synthetic_urdf(prefix=identity.model_prefix))
    system = ET.SubElement(root, "ros2_control", name=f"{name}_hardware", type="system")
    hardware = ET.SubElement(system, "hardware")
    ET.SubElement(hardware, "plugin").text = "rm_control/RMSystemHardware"
    for key, value in {
        "direct_current": "true", "read_only": "false",
        "direct_current_ack": "I_ACCEPT_DIRECT_CURRENT_CONTROLLER_RISK",
        "ip": "192.0.2.1", "port": "8080", "direct_current_guard_port": "18080",
    }.items():
        ET.SubElement(hardware, "param", name=key).text = value
    for name in identity.joint_names:
        joint = ET.SubElement(system, "joint", name=name)
        for interface in ("position", "actuator_current"):
            ET.SubElement(joint, "command_interface", name=interface)
    return ET.tostring(root, encoding="unicode")