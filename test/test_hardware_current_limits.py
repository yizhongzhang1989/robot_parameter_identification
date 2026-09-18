import inspect
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from dashboard_gravity_fixtures import gravity_urdf
from robot_parameter_identification.arm_identity import ArmBinding, ArmIdentity
from robot_parameter_identification.dashboard.service import DashboardConfig, IdentificationService
from robot_parameter_identification.plants.ros_control import HardwarePlant


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
                binding = ArmBinding.from_description(ArmIdentity(name), made.urdf_text)
                self.assertEqual(limits, binding.current_limits)
                limits["peak_current_a"][-1] = 99
                self.assertEqual(made.profile_payload()[
                    "hardware_current_limits"]["peak_current_a"][-1], peak)

    def test_invalid_hardware_limits_are_not_exposed_as_valid(self):
        for peak in ("4,5,4,4,1.5,1.5,2.01", "4,5,4,4,1.5,1.5,nan"):
            with self.subTest(peak=peak):
                made = self.service(peak=peak)
                self.assertIsNone(made.profile_payload()["hardware_current_limits"])
                profile = made.profile_payload()["profile"]
                profile["limits"]["continuous_current_a"] = [1.0] * 7
                profile["limits"]["peak_current_a"] = [2.0] * 7
                self.assertTrue(made.apply_profile(profile)["ok"])
                self.assertIsNone(made.profile_payload()["hardware_current_limits"])

    def test_legacy_profile_limits_cannot_raise_or_lower_hardware_limits(self):
        for name in ("left", "right", "station_3"):
            made = self.service(name)
            original = made.profile_payload()["hardware_current_limits"]
            self.assertEqual(original["maximum_command_a"][2], 3.0)
            self.assertEqual(original["peak_current_a"][2], 4.0)
            for current in (0.001, 99.0):
                with self.subTest(name=name, current=current):
                    profile = made.profile_payload()["profile"]
                    profile["limits"]["continuous_current_a"] = [current] * 7
                    profile["limits"]["peak_current_a"] = [current * 2.0] * 7
                    profile["envelope"].update({
                        "sustained_current_window_s": 0.5,
                        "current_slew_a_s": 0.01,
                        "probe_current_fraction": 0.01,
                        "probe_current_a": [0.01] * 7,
                    })
                    result = made.apply_profile(profile)
                    self.assertTrue(result["ok"])
                    payload = result["profile"]
                    self.assertEqual(payload["hardware_current_limits"], original)
                    self.assertEqual(ArmBinding.from_description(
                        ArmIdentity(name), made.urdf_text).current_limits, original)
                    self.assertNotIn("current_guard", payload)
                    self.assertNotIn("calibration_current_policy", payload)
                    monitor = made._monitor()
                    self.assertEqual(dict(inspect.signature(made._monitor).parameters), {})
                    for field in ("continuous_current_a", "peak_current_a",
                                  "sustained_current_window_s", "current_slew_a_s",
                                  "probe_current_fraction", "probe_current_a"):
                        self.assertFalse(hasattr(made.profile, field), field)
                        self.assertFalse(hasattr(monitor, field), field)
                        self.assertNotIn(field, payload["profile"]["limits"])
                        self.assertNotIn(field, payload["profile"]["envelope"])
                    self.assertIsNone(monitor.check({"current_a": [99.0] * 7}, 0.0))
                    self.assertIsNone(monitor.check({"current_a": [-99.0] * 7}, 10.0))

    def test_jtc_records_current_for_derived_and_configured_profiles(self):
        for configured in (False, True):
            with self.subTest(configured=configured):
                made = self.service()
                if configured:
                    profile = made.profile_payload()["profile"]
                    profile["limits"]["continuous_current_a"] = [0.001] * 7
                    profile["limits"]["peak_current_a"] = [0.002] * 7
                    self.assertTrue(made.apply_profile(profile)["ok"])
                monitor = made._monitor()
                self.assertNotIn("peak-current ceiling", monitor.guards())
                self.assertNotIn("sustained-current ceiling", monitor.guards())
                self.assertIn("temperature ceiling", monitor.guards())
                plant = HardwarePlant(made.profile, monitor=monitor)
                for stamp, current in ((0.0, 99.0), (10.0, -99.0)):
                    plant._check_monitor({"stamp_s": stamp, "current_a": [current] * 7},
                                         stamp)
                    plant._raise_if_monitor_tripped()
                plant.rollback_raw_frames(0)
                measured = monitor.current_measurements.as_dict(made.driven_joints)
                self.assertEqual(measured["policy"], "record_only")
                self.assertEqual(measured["source"], "full_telemetry")
                self.assertEqual(measured["channel"], "current_a")
                self.assertEqual(len(measured["joints"]), 7)
                for name, entry in zip(made.driven_joints, measured["joints"]):
                    self.assertEqual(entry["joint"], name)
                    self.assertEqual(entry["samples"], 2)
                    self.assertEqual(entry["minimum_a"], -99.0)
                    self.assertEqual(entry["maximum_a"], 99.0)
                    self.assertEqual(entry["peak_abs_a"], 99.0)
                    self.assertAlmostEqual(entry["rms_a"], 99.0)
                plant._check_monitor({"temperature_c": [made.profile.temperature_c] * 7},
                                     11.0)
                self.assertEqual(monitor.last_trip["kind"], "temperature")
                with self.assertRaisesRegex(RuntimeError, "temperature"):
                    plant._raise_if_monitor_tripped()

    def test_unknown_controller_or_missing_description_has_no_hardware_limits(self):
        made = self.service()
        made.driven_joints = list(ArmIdentity("right").joint_names)
        self.assertIsNone(made.profile_payload()["hardware_current_limits"])
        made.driven_joints = list(ArmIdentity("left").joint_names)
        made.urdf_text = ""
        self.assertIsNone(made.profile_payload()["hardware_current_limits"])


if __name__ == "__main__":
    unittest.main()
