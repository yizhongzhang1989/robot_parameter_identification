"""Network-free drag stop checks using ros2run and owned Python fixtures."""

import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from robot_parameter_identification.dashboard.service import (
    DashboardConfig, GRAVITY_DRAG_TEST, GRAVITY_TEST_ACKNOWLEDGEMENT,
    IdentificationService,
)
from fixtures import synthetic_urdf, test_profile

try:
    from ros2run.api import run_executable
except ImportError:
    run_executable = None


WRAPPER = """
import signal
import sys
from ros2run.api import run_executable

def interrupted(number, frame):
    print('wrapper SIGINT', flush=True)
    signal.default_int_handler(number, frame)

signal.signal(signal.SIGINT, interrupted)
print('wrapper ready', flush=True)
sys.exit(run_executable(path=sys.executable, argv=['-u', '-c', sys.argv[1]]))
"""

CHILD = """
import signal
import sys

def interrupted(number, frame):
    print('child SIGINT', flush=True)
    sys.exit(0)

signal.signal(signal.SIGINT, interrupted)
print('child ready', flush=True)
while True:
    signal.pause()
"""

INDEPENDENT = """
import signal
import sys

def interrupted(number, frame):
    print('independent SIGINT', flush=True)
    sys.exit(2)

signal.signal(signal.SIGINT, interrupted)
print('independent ready', flush=True)
for command in sys.stdin:
    if command.strip() == 'ping':
        print('independent alive', flush=True)
    elif command.strip() == 'quit':
        sys.exit(0)
"""


@unittest.skipUnless(os.name == "posix" and run_executable is not None,
                     "requires POSIX and ros2run.api.run_executable")
class DragProcessGroupTest(unittest.TestCase):
    def read_ready_line(self, process):
        deadline = time.monotonic() + 5.0
        received = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while not received.endswith(b"\n"):
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0, "fixture readiness timed out")
                self.assertTrue(selector.select(remaining),
                                "fixture readiness timed out")
                chunk = os.read(process.stdout.fileno(), 1)
                self.assertTrue(chunk, f"fixture exited: {received!r}")
                received.extend(chunk)
        return received.decode().strip()

    def check_group_stop(self, *, during_launch):
        processes = []
        messages = []
        with tempfile.TemporaryDirectory() as directory:
            made = IdentificationService(
                DashboardConfig(output_directory=directory),
                profile=test_profile())
            made.adopt_description(synthetic_urdf())
            made.publish_event = lambda message, **kwargs: messages.append(message)

            def launch(command, **kwargs):
                self.assertEqual(command[:4],
                                 ["ros2", "run", "rm_control", "manual_drag"])
                self.assertTrue(kwargs["start_new_session"])
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", WRAPPER, CHILD], **kwargs)
                processes.append(process)
                self.assertEqual(os.getpgid(process.pid), process.pid)
                self.assertEqual(self.read_ready_line(process), "wrapper ready")
                self.assertEqual(self.read_ready_line(process), "child ready")
                folder = Path(command[command.index("--status-file") + 1]).parent
                (folder / "gravity_test_summary.json").write_text(
                    '{"result": "STOPPED", "stop_verified": true}',
                    encoding="utf-8")
                if during_launch:
                    self.assertIsNone(made._external_process)
                    made.stop()
                return process

            made._process_launcher = launch
            try:
                independent = subprocess.Popen(
                    [sys.executable, "-u", "-c", INDEPENDENT],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, start_new_session=True)
                processes.append(independent)
                self.assertEqual(os.getpgid(independent.pid), independent.pid)
                self.assertEqual(self.read_ready_line(independent),
                                 "independent ready")
                answer = made.start_gravity_test(GRAVITY_DRAG_TEST, {
                    "acknowledgement": GRAVITY_TEST_ACKNOWLEDGEMENT,
                })
                self.assertTrue(answer["ok"], answer)
                if not during_launch:
                    made.stop()
                worker = made._worker
                if worker is not None:
                    worker.join(5.0)
                    self.assertFalse(worker.is_alive(),
                                     "ros2run or its child did not stop")
                self.assertIn("wrapper SIGINT", messages)
                self.assertIn("child SIGINT", messages)
                self.assertEqual(processes[-1].returncode, 0)
                self.assertEqual(made.result["result"], "STOPPED")
                self.assertEqual(made.snapshot()["state"], "idle")
                independent.stdin.write("ping\n")
                independent.stdin.flush()
                self.assertEqual(self.read_ready_line(independent),
                                 "independent alive")
                self.assertIsNone(independent.poll())
                independent.stdin.write("quit\n")
                independent.stdin.flush()
                self.assertEqual(independent.wait(timeout=5.0), 0)
            finally:
                for process in processes:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5.0)
                worker = made._worker
                if worker is not None:
                    worker.join(5.0)
                for process in processes:
                    if process.stdin is not None:
                        process.stdin.close()
                    if process.stdout is not None:
                        process.stdout.close()

    def test_stop_reaches_ros2run_and_child_only(self):
        self.check_group_stop(during_launch=False)

    def test_stop_during_launch_reaches_ros2run_and_child_only(self):
        self.check_group_stop(during_launch=True)