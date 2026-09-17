import unittest
import xml.etree.ElementTree as ET

from robot_parameter_identification.arm_identity import ArmBinding, ArmIdentity


def description():
    root = ET.Element("robot", name="instances")
    for number, name in enumerate(("right", "left", "station_3"), 1):
        system = ET.SubElement(root, "ros2_control", name=f"{name}_system", type="system")
        hardware = ET.SubElement(system, "hardware")
        ET.SubElement(hardware, "plugin").text = "rm_control/RMSystemHardware"
        for key, value in {
            "direct_current": "true", "read_only": "false",
            "direct_current_ack": "I_ACCEPT_DIRECT_CURRENT_CONTROLLER_RISK",
            "ip": f"192.0.2.{number}", "port": "8080",
            "direct_current_guard_port": str(18000 + number),
        }.items():
            ET.SubElement(hardware, "param", name=key).text = value
        for joint_name in ArmIdentity(name).joint_names:
            joint = ET.SubElement(system, "joint", name=joint_name)
            for interface in ("position", "actuator_current"):
                ET.SubElement(joint, "command_interface", name=interface)
    return root


class ArmBindingTest(unittest.TestCase):
    def test_left_peak_override_does_not_change_right_or_continuous_limits(self):
        root = description()
        hardware = root.findall("ros2_control")[1].find("hardware")
        ET.SubElement(hardware, "param", name="direct_current_peak_a").text = "4,5,4,4,1.5,1.5,2"
        xml = ET.tostring(root, encoding="unicode")
        left = ArmBinding.from_description(ArmIdentity("left"), xml)
        right = ArmBinding.from_description(ArmIdentity("right"), xml)
        third = ArmBinding.from_description(ArmIdentity("station_3"), xml)
        self.assertEqual(left.peak_current_a[-1], 2.0)
        self.assertEqual(right.peak_current_a[-1], 0.8)
        self.assertEqual(third.peak_current_a[-1], 0.8)
        self.assertEqual(left.peak_current_a[:-1], right.peak_current_a[:-1])
        self.assertEqual(left.continuous_current_a, right.continuous_current_a)
        self.assertEqual(left.maximum_command_a, right.maximum_command_a)
        self.assertEqual(left.maximum_command_a[-1], 0.6)
        values = left.current_limits
        values["peak_current_a"][-1] = 999
        self.assertEqual(left.peak_current_a[-1], 2.0)

    def test_invalid_or_unsupported_instance_currents_are_refused(self):
        cases = [
            ("direct_current_peak_a", "4,5,4,4,1.5,1.5,2.001"),
            ("direct_current_peak_a", "4,5,4,4,1.5,1.5,nan"),
            ("direct_current_peak_a", "4,5,4,4,1.5,1.5,0"),
            ("direct_current_peak_a", "4,5,4,4,1.5,1.5,0.5"),
            ("direct_current_peak_a", "4,5,4,4,1.5,1.5"),
            ("direct_current_continuous_a", "3,4.1,3,3.1,1.1,1.15,1.501"),
            ("direct_current_maximum_command_a", "3,4.1,3,3.1,1.1,1.15,2"),
        ]
        for name, value in cases:
            root = description()
            hardware = root.findall("ros2_control")[1].find("hardware")
            ET.SubElement(hardware, "param", name=name).text = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                ArmBinding.from_description(ArmIdentity("left"),
                                            ET.tostring(root, encoding="unicode"))

    def test_every_instance_resolves_its_own_live_hardware_endpoint(self):
        xml = ET.tostring(description(), encoding="unicode")
        for number, name in enumerate(("right", "left", "station_3"), 1):
            identity = ArmIdentity(name)
            binding = ArmBinding.from_description(identity, xml)
            self.assertEqual(binding.host, f"192.0.2.{number}")
            self.assertEqual(binding.guard_port, 18000 + number)
            self.assertEqual(identity.model_prefix, f"{name}_arm_")

    def test_sustained_limits_can_increase_without_increasing_commands(self):
        root = description()
        for index, peak in ((0, "4,5,4,4,1.5,1.5,1.5"),
                            (1, "4,5,4,4,1.5,1.5,2")):
            hardware = root.findall("ros2_control")[index].find("hardware")
            ET.SubElement(hardware, "param", name="direct_current_continuous_a").text = (
                "3,4.1,3,3.1,1.5,1.5,1.5")
            ET.SubElement(hardware, "param", name="direct_current_peak_a").text = peak
        xml = ET.tostring(root, encoding="unicode")
        for name, peak in (("right", 1.5), ("left", 2.0)):
            with self.subTest(name=name):
                binding = ArmBinding.from_description(ArmIdentity(name), xml)
                self.assertEqual(binding.continuous_current_a,
                                 (3.0, 4.1, 3.0, 3.1, 1.5, 1.5, 1.5))
                self.assertEqual(binding.maximum_command_a,
                                 (3.0, 4.1, 3.0, 3.1, 1.1, 1.15, 0.6))
                self.assertEqual(binding.peak_current_a[-1], peak)
        legacy = ArmBinding.from_description(ArmIdentity("station_3"), xml)
        self.assertEqual(legacy.continuous_current_a[-1], 0.6)

    def test_continuous_cannot_exceed_configured_peak(self):
        root = description()
        hardware = root.findall("ros2_control")[0].find("hardware")
        ET.SubElement(hardware, "param", name="direct_current_continuous_a").text = (
            "3,4.1,3,3.1,1.5,1.5,1.5")
        with self.assertRaisesRegex(ValueError, "continuous <= peak"):
            ArmBinding.from_description(ArmIdentity("right"),
                                        ET.tostring(root, encoding="unicode"))

    def test_missing_wrong_or_unacknowledged_hardware_fails_closed(self):
        for key, value in (("direct_current", "false"), ("read_only", "true"),
                           ("direct_current_ack", ""), ("ip", "0.0.0.0"),
                           ("port", "0"), ("direct_current_guard_port", "65536")):
            root = description()
            root.find(f"ros2_control/hardware/param[@name='{key}']").text = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ArmBinding.from_description(ArmIdentity("right"),
                                            ET.tostring(root, encoding="unicode"))

    def test_duplicate_endpoint_or_guard_cannot_bind_another_robot(self):
        for key in ("ip", "direct_current_guard_port"):
            root = description()
            systems = root.findall("ros2_control")
            first = systems[0].find(f"hardware/param[@name='{key}']")
            systems[1].find(f"hardware/param[@name='{key}']").text = first.text
            with self.subTest(key=key), self.assertRaises(ValueError):
                ArmBinding.from_description(ArmIdentity("left"),
                                            ET.tostring(root, encoding="unicode"))

    def test_other_arm_or_mixed_joint_list_cannot_satisfy_binding(self):
        root = description()
        root.findall("ros2_control")[1].find("joint").set("name", "right_arm_joint1")
        for name in ("left", "right", "missing"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ArmBinding.from_description(ArmIdentity(name),
                                            ET.tostring(root, encoding="unicode"))


if __name__ == "__main__":
    unittest.main()
