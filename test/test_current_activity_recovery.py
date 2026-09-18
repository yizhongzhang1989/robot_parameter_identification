import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from dashboard_gravity_fixtures import gravity_urdf
from robot_parameter_identification.arm_identity import ArmBinding, ArmIdentity
from robot_parameter_identification.dashboard.service import (
    GRAVITY_DRAG_TEST, GRAVITY_TEST_ACKNOWLEDGEMENT, IdentificationService, PAUSED, RUNNING,
)


class CurrentActivityRecoveryTest(unittest.TestCase):
    def test_failed_drag_restoration_keeps_every_instance_blocked(self):
        for name in ("left", "right", "station_3"):
            for stop_verified, errors in ((False, []), (True, ["position restore failed"])):
                with self.subTest(arm=name, stop=stop_verified, errors=errors):
                    identity = ArmIdentity(name)
                    with tempfile.TemporaryDirectory() as directory:
                        service = object.__new__(IdentificationService)
                        service.config = SimpleNamespace(
                            output_directory=directory,
                            commands=SimpleNamespace(follow_joint_trajectory_action=(
                                f"/{identity.trajectory_controller}/follow_joint_trajectory")))
                        service.driven_joints = list(identity.joint_names)
                        service.arm = SimpleNamespace(joint_names=list(identity.joint_names))
                        service._lock = threading.RLock()
                        service._state = RUNNING
                        service._activity = GRAVITY_DRAG_TEST
                        service._worker = threading.current_thread()
                        service._hold_recovery_required = False
                        service._hold_recovery_selection = None
                        service._hold_current_started = False
                        service._hold_plan = {}
                        service.publish_event = lambda *_args, **_kwargs: None
                        summary = Path(directory) / "summary.json"
                        summary.write_text(json.dumps({
                            "result": "FAIL", "reason": "motion envelope exceeded",
                            "stop_verified": stop_verified, "restore_errors": errors,
                            "controller_switch_attempted": True,
                        }))
                        process = SimpleNamespace(stdout=(), wait=lambda: 1)

                        service._run_gravity_test(GRAVITY_DRAG_TEST, process, summary)

                        self.assertEqual(service._state, PAUSED)
                        self.assertEqual(service._activity, GRAVITY_DRAG_TEST)
                        self.assertTrue(service._hold_recovery_required)
                        self.assertEqual(service._hold_recovery_selection, (
                            identity.joint_names,
                            service.config.commands.follow_joint_trajectory_action))
                        self.assertEqual(service.result["result"], "FAIL")
                        self.assertIsNone(service._worker)


class PositionRecoveryRequestTest(unittest.TestCase):
    def make_service(self, directory, name="left"):
        identity = ArmIdentity(name)
        service = object.__new__(IdentificationService)
        service.config = SimpleNamespace(
            output_directory=directory, commands=SimpleNamespace(
                controller_manager="/controller_manager", follow_joint_trajectory_action=(
                    f"/{identity.trajectory_controller}/follow_joint_trajectory")))
        service.driven_joints = list(identity.joint_names)
        service.arm = SimpleNamespace(joint_names=list(identity.joint_names))
        service.urdf_text = gravity_urdf(name)
        service._lock = threading.RLock()
        service._state, service._activity = PAUSED, GRAVITY_DRAG_TEST
        service._worker = service._external_process = None
        service._hold_recovery_required = True
        service._hold_recovery_selection = (
            identity.joint_names, service.config.commands.follow_joint_trajectory_action)
        service._hold_current_started = True
        service._current_recovery_running = False
        service._current_recovery_report = None
        service._hold_plan = {}
        service.planning = False
        service._abort = threading.Event()
        service.progress = {"phase": "failed"}
        service.result = {"result": "FAIL", "reason": "original failure"}
        service.bridge = None
        service.publish_event = mock.Mock()
        service._process_launcher = mock.Mock(side_effect=AssertionError("unexpected child"))
        service.connection = mock.Mock(return_value={"telemetry_ok": True, "controllers": {
            "manager": "/controller_manager", "available": True, "age_s": 0.01,
            "items": [
                {"name": identity.current_controller, "state": "inactive", "claimed_interfaces": []},
                {"name": identity.trajectory_controller, "state": "inactive", "claimed_interfaces": [],
                 "type": "joint_trajectory_controller/JointTrajectoryController"},
            ]}})
        return service

    def test_request_is_explicit_and_never_launches_for_wrong_selection_or_stale_state(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            options = {"acknowledgement": GRAVITY_TEST_ACKNOWLEDGEMENT}
            self.assertFalse(service.recover_position({})["ok"])
            service.driven_joints = list(ArmIdentity("right").joint_names)
            self.assertFalse(service.recover_position(options)["ok"])
            service.driven_joints = list(ArmIdentity("left").joint_names)
            service.connection.return_value["controllers"]["age_s"] = 4
            self.assertFalse(service.recover_position(options)["ok"])
            service._process_launcher.assert_not_called()

    def test_accepted_request_reserves_the_owner_and_rejects_duplicate_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            options = {"acknowledgement": GRAVITY_TEST_ACKNOWLEDGEMENT}
            with mock.patch("threading.Thread") as worker:
                self.assertTrue(service.recover_position(options)["ok"])
                self.assertTrue(service.current_recovery_payload()["running"])
                self.assertFalse(service.recover_position(options)["ok"])
                worker.assert_called_once()
                worker.return_value.start.assert_called_once()
            self.assertEqual(service.result["result"], "FAIL")

    def test_instance_recovery_uses_shared_entrypoint_and_preserves_original_failure(self):
        import hashlib

        for name in ("right", "left", "station_3"):
            with self.subTest(arm=name), tempfile.TemporaryDirectory() as directory:
                service = self.make_service(directory, name)
                original = dict(service.result)
                binding = ArmBinding.from_description(ArmIdentity(name), service.urdf_text)
                digest = hashlib.sha256(service.urdf_text.encode()).hexdigest()

                def launch(command, **_kwargs):
                    self.assertEqual(command[:6], ["ros2", "run", "rm_control",
                                                   "recover_position", "--arm", name])
                    self.assertNotIn("--source", command)
                    report = {"result": "PASS", "test": "position_recovery", "stop_verified": True,
                              "restore_errors": [], "model": None, "published_commands": 0,
                              "robot": {"arm": name, "host": binding.host, "port": binding.port,
                                        "hardware": binding.hardware_name, "guard_port": binding.guard_port,
                                        "description_sha256": digest},
                              "current_limits": binding.current_limits,
                              "controller_final_states": {binding.identity.trajectory_controller: "active",
                                                          binding.identity.current_controller: "inactive"},
                              "post_stop_still": {"distinct_udp_samples": 20}}
                    Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
                    return SimpleNamespace(stdout=(), wait=lambda: 0, poll=lambda: 0)

                service._process_launcher = launch
                service._run_position_recovery(binding, service._hold_recovery_selection, digest)
                self.assertEqual(service._current_recovery_report["error"], "")
                self.assertEqual(service.result, original)
                self.assertEqual(service._state, PAUSED)
                self.assertTrue(service._hold_recovery_required)