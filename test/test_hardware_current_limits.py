from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from dashboard_gravity_fixtures import gravity_urdf
from robot_parameter_identification.arm_identity import ArmIdentity
from robot_parameter_identification.dashboard.service import DashboardConfig, IdentificationService


class HardwareCurrentLimitsTest(unittest.TestCase):
    def service(self, name="left", peak="4,5,4,4,1.5,1.5,2"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        made = IdentificationService(DashboardConfig(
            output_directory=str(folder), config_file_path=str(folder / "cell.json")))
        identity = ArmIdentity(name)
        made.adopt_driven_joints(identity.joint_names)
        root = ET.fromstring(gravity_urdf(name))
        ET.SubElement(root.find("ros2_control/hardware"), "param",
                      name="direct_current_peak_a").text = peak
        made.adopt_description(ET.tostring(root, encoding="unicode"))
        return made

    def test_profile_exposes_selected_hardware_not_shared_peak_defaults(self):
        for name, peak in (("left", 2.0), ("right", 0.8), ("station_3", 0.8)):
            with self.subTest(name=name):
                made = self.service(name, f"4,5,4,4,1.5,1.5,{peak}")
                limits = made.profile_payload()["hardware_current_limits"]
                self.assertEqual(limits["peak_current_a"][-1], peak)
                self.assertEqual(limits["continuous_current_a"][-1], 0.6)
                self.assertEqual(limits["maximum_command_a"][-1], 0.6)
                limits["peak_current_a"][-1] = 99
                self.assertEqual(made.profile_payload()[
                    "hardware_current_limits"]["peak_current_a"][-1], peak)

    def test_invalid_hardware_limits_are_not_exposed_as_valid(self):
        for peak in ("4,5,4,4,1.5,1.5,2.01", "4,5,4,4,1.5,1.5,nan"):
            with self.subTest(peak=peak):
                made = self.service(peak=peak)
                self.assertIsNone(made.profile_payload()["hardware_current_limits"])

    def test_position_profile_does_not_raise_direct_current_hardware_limits(self):
        made = self.service("left")
        original = made.profile_payload()["hardware_current_limits"]
        profile = made.profile_payload()["profile"]
        profile["limits"]["continuous_current_a"] = [4.0, 5.1, 4.0, 4.1, 1.5, 1.5, 1.5]
        profile["limits"]["peak_current_a"] = [5.0, 6.0, 5.0, 5.0, 1.5, 1.5, 2.0]
        profile["envelope"]["sustained_current_window_s"] = 0.5
        made.apply_profile(profile)
        payload = made.profile_payload()
        self.assertEqual(payload["hardware_current_limits"], original)
        monitor = made._monitor()
        self.assertEqual(monitor.continuous_current_a, (4.0, 5.1, 4.0, 4.1, 1.5, 1.5, 1.5))
        self.assertEqual(monitor.peak_current_a, (5.0, 6.0, 5.0, 5.0, 1.5, 1.5, 2.0))
        frame = {"current_a": [0.0, 0.0, 4.11, 0.0, 0.0, 0.0, 0.0]}
        self.assertIsNone(monitor.check(frame, 0.0))
        self.assertIn("continuous current", monitor.check(frame, 0.5))
        frame["current_a"][2] = 5.001
        self.assertIn("peak current", monitor.check(frame, 1.0))
        self.assertEqual(original["maximum_command_a"][2], 3.0)
        self.assertEqual(original["peak_current_a"][2], 4.0)

    def test_unknown_controller_or_missing_description_has_no_hardware_limits(self):
        made = self.service()
        made.driven_joints = list(ArmIdentity("right").joint_names)
        self.assertIsNone(made.profile_payload()["hardware_current_limits"])
        made.driven_joints = list(ArmIdentity("left").joint_names)
        made.urdf_text = ""
        self.assertIsNone(made.profile_payload()["hardware_current_limits"])


if __name__ == "__main__":
    unittest.main()
