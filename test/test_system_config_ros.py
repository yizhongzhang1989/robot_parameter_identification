"""Configuration precedence without starting a controller or a ROS node."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from robot_parameter_identification.system_config import load_system_config, system_defaults


ROOT = Path(__file__).resolve().parents[1]


def node_method(name="_declare"):
    source = ROOT / "robot_parameter_identification/dashboard/node.py"
    tree = ast.parse(source.read_text())
    node = next(entry for entry in tree.body if isinstance(entry, ast.ClassDef)
                and entry.name == "DashboardNode")
    method = next(entry for entry in node.body if isinstance(entry, ast.FunctionDef)
                  and entry.name == name)
    from robot_parameter_identification.interfaces import CommandSpec

    namespace = {"system_defaults": system_defaults, "CommandSpec": CommandSpec,
                 "DEFAULT_ACTION": system_defaults()["ros"]["follow_joint_trajectory_action"]}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def test_ros_parameters_use_selected_file_then_explicit_override(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("ros:\n  port: 8450\n  output_directory: /tmp/custom-results\n")
    values = {"port": 8451}
    node = SimpleNamespace(system_config=load_system_config(path))
    node.declare_parameter = lambda name, default: values.setdefault(name, default)
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    declare = node_method()
    assert declare(node, "port") == 8451
    assert declare(node, "output_directory") == "/tmp/custom-results"


def test_empty_array_defaults_preserve_ros_parameter_types(tmp_path):
    path = tmp_path / "arrays.yaml"
    path.write_text("ros:\n  extra_telemetry_topics: []\n  workspace_limit_deg: []\n")
    values = {}
    node = SimpleNamespace(system_config=load_system_config(path))
    node.declare_parameter = lambda name, default: values.setdefault(name, default)
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    declare = node_method()
    assert declare(node, "extra_telemetry_topics") == [""]
    assert declare(node, "workspace_limit_deg") == [0.0]


def launch_module():
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    specification = importlib.util.spec_from_file_location(
        "system_config_dashboard_launch", ROOT / "launch/dashboard.launch.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_launch_forwards_only_explicit_overrides():
    module = launch_module()
    from launch import LaunchContext

    context = LaunchContext()
    context.launch_configurations.update(
        {name: module.INHERIT for name in system_defaults()["ros"]})
    context.launch_configurations.update({
        "system_config": "/tmp/selected.yaml", "port": "8451",
        "signal.voltage": "", "workspace_limit_deg": "[15.0, 20.0]",
    })
    with patch.object(module, "Node") as node:
        module._dashboard(context)
    parameters = node.call_args.kwargs["parameters"][0]
    evaluated = {name: value.evaluate(context) for name, value in parameters.items()}
    assert evaluated == {
        "system_config": "/tmp/selected.yaml", "port": 8451,
        "signal.voltage": "", "workspace_limit_deg": [15.0, 20.0],
    }


def test_all_configured_ros_parameters_remain_launch_arguments():
    module = launch_module()
    from launch.actions import DeclareLaunchArgument

    names = {entry.name for entry in module.generate_launch_description().entities
             if isinstance(entry, DeclareLaunchArgument)}
    assert names == set(system_defaults()["ros"]) | {"system_config"}


@pytest.mark.parametrize("overrides, expected", [
    ({"controller": "left_arm_joint_trajectory_controller"},
     "/left_arm_joint_trajectory_controller/follow_joint_trajectory"),
    ({"controller": "left_arm_joint_trajectory_controller",
      "follow_joint_trajectory_action": "/custom_controller/follow_joint_trajectory"},
     "/custom_controller/follow_joint_trajectory"),
    ({}, "/right_arm_joint_trajectory_controller/follow_joint_trajectory"),
])
def test_controller_override_cannot_inherit_another_arms_action(tmp_path, overrides, expected):
    path = tmp_path / "right.yaml"
    path.write_text("ros:\n  controller: right_arm_joint_trajectory_controller\n"
                    "  follow_joint_trajectory_action: /right_arm_joint_trajectory_controller/follow_joint_trajectory\n")
    values = dict(overrides)
    node = SimpleNamespace(system_config=load_system_config(path), _parameter_overrides=overrides)
    node.declare_parameter = lambda name, default: values.setdefault(name, default)
    node.get_parameter = lambda name: SimpleNamespace(value=values[name])
    node._declare = lambda name: node_method()(node, name)
    assert node_method("_resolve_action")(node) == expected
    assert node.system_config.values["ros"]["follow_joint_trajectory_action"] == expected