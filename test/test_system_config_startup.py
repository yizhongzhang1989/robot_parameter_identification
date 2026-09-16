"""Real dashboard node startup on an isolated ROS domain, without a robot."""

import json
import socket
import urllib.request

import pytest
import yaml


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.fixture
def ros(monkeypatch):
    module = pytest.importorskip("rclpy")
    monkeypatch.setenv("ROS_DOMAIN_ID", "97")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    yield module
    if module.ok():
        module.shutdown()


def start_node(ros, path, port, *overrides):
    from robot_parameter_identification.dashboard.node import DashboardNode

    ros.init(args=["--ros-args", "-p", f"system_config:={path}",
                   "-p", f"port:={port}", *overrides])
    return DashboardNode()


def read_config(node):
    with urllib.request.urlopen(
            f"http://127.0.0.1:{node.server.port}/api/system-config", timeout=5) as response:
        return json.load(response)


def test_first_node_startup_creates_complete_config(ros, tmp_path):
    path = tmp_path / "fresh" / "system_config.yaml"
    port = available_port()
    node = start_node(ros, path, port)
    try:
        payload = read_config(node)
        assert payload["path"] == str(path)
        assert payload["values"]["ros"]["port"] == port
        assert payload["controls"]["gravtest-speed-stop"]["value"] == 120.0
        written = yaml.safe_load(path.read_text())
        assert written["ros"]["port"] == 8300
        assert set(written) == set(payload["values"])
    finally:
        node.destroy_node()


def test_custom_config_and_ros_overrides_reach_http(ros, tmp_path):
    path = tmp_path / "cell_system.yaml"
    path.write_text(
        "ros:\n  output_directory: /tmp/from-file\n"
        "  controller: right_arm_joint_trajectory_controller\n"
        "  follow_joint_trajectory_action: /right_arm_joint_trajectory_controller/follow_joint_trajectory\n"
        "dashboard:\n  drag_test:\n    maximum_speed_deg_s: 72.0\n")
    original = path.read_bytes()
    output = str(tmp_path / "from-cli")
    node = start_node(ros, path, available_port(),
                      "-p", f"output_directory:={output}",
                      "-p", "controller:=left_arm_joint_trajectory_controller")
    try:
        payload = read_config(node)
        assert payload["controls"]["gravtest-speed-stop"]["value"] == 72.0
        assert payload["values"]["ros"]["output_directory"] == output
        assert node.service.config.commands.follow_joint_trajectory_action == (
            "/left_arm_joint_trajectory_controller/follow_joint_trajectory")
        assert path.read_bytes() == original
    finally:
        node.destroy_node()