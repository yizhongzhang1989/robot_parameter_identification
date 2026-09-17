from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_parameter_identification.dashboard.service import DashboardConfig, IdentificationService
from fixtures import synthetic_urdf
from test_identification import TWO_ARM_URDF


class ControllerPreviewTest(unittest.TestCase):
    def make_service(self, urdf, joints):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        made = IdentificationService(DashboardConfig(
            output_directory=str(folder), config_file_path=str(folder / "cell.json")))
        made.adopt_driven_joints(joints)
        made.adopt_description(urdf)
        return made

    def preview(self, made, angles):
        return made._build_preview("gravity", {"phases": [{
            "phase": "A_gravity", "detail": {"poses_deg": [angles]}}]})

    def test_exact_controller_joints_choose_the_tip_on_each_chain(self):
        for selected in ("mine", "theirs"):
            with self.subTest(selected=selected):
                joints = [f"{selected}_joint1"]
                made = self.make_service(TWO_ARM_URDF, joints)
                preview = self.preview(made, [25.0])
                pose = preview["groups"][0]["poses"][0]
                tip = np.asarray(made.arm.link_transforms([25.0])[
                    f"{selected}_link1"]).reshape(4, 4)[:3, 3]
                self.assertEqual(preview["joint_names"], joints)
                self.assertEqual(pose["paths"], [pose["points"]])
                np.testing.assert_allclose(pose["paths"][0][-1], tip, atol=1e-5)

    def test_other_robot_names_and_joint_counts_use_the_given_controller(self):
        for prefix, count in (("positioner_", 2), ("toolhead_", 3), ("manipulator_", 6)):
            with self.subTest(prefix=prefix, count=count):
                joints = [f"{prefix}joint{index}" for index in range(1, count + 1)]
                made = self.make_service(synthetic_urdf(count, prefix), joints[::-1])
                angles = [15.0] * count
                preview = self.preview(made, angles)
                pose = preview["groups"][0]["poses"][0]
                tip = np.asarray(made.arm.link_transforms(angles)[
                    f"{prefix}link{count}"]).reshape(4, 4)[:3, 3]
                self.assertEqual(preview["joint_names"], joints)
                self.assertEqual(len(pose["points"]), count + 1)
                np.testing.assert_allclose(pose["points"][-1], tip, atol=1e-5)

    def test_branched_controller_publishes_paths_without_a_cross_chain_polyline(self):
        made = self.make_service(TWO_ARM_URDF, ["mine_joint1", "theirs_joint1"])
        pose = self.preview(made, [0.0, 0.0])["groups"][0]["poses"][0]
        self.assertEqual(pose["points"], [])
        self.assertEqual(pose["paths"], [
            [[0.0, 0.2, 0.0], [0.0, 0.2, 0.0]],
            [[0.0, -0.2, 0.0], [0.0, -0.2, 0.0]],
        ])

    def test_controller_change_invalidates_previous_preview_and_geometry(self):
        made = self.make_service(TWO_ARM_URDF, ["mine_joint1"])
        made.preview = self.preview(made, [20.0])
        made._designed = {"A_gravity": {"poses_deg": [[20.0]]}}
        token = made.preview_token
        made.adopt_driven_joints(["theirs_joint1"])
        self.assertFalse(made.preview_payload()["available"])
        self.assertGreater(made.preview_token, token)
        self.assertEqual(made._designed, {})
        preview = self.preview(made, [-20.0])
        self.assertEqual(preview["joint_names"], ["theirs_joint1"])
        self.assertEqual(preview["groups"][0]["poses"][0]["points"][-1], [0.0, -0.2, 0.0])

    def test_mismatched_profile_cannot_draw_a_different_controller(self):
        made = self.make_service(TWO_ARM_URDF, ["mine_joint1"])
        made.driven_joints = ["theirs_joint1"]
        self.assertFalse(self.preview(made, [0.0])["available"])


if __name__ == "__main__":
    unittest.main()
