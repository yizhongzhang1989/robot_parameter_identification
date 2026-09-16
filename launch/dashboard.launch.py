"""Launch the identification dashboard against whatever arm is already running.

This launches nothing but the dashboard. Bring the robot up however you
normally do, then point this at its controller and its telemetry topic. Every
robot-specific value is an argument; nothing here names a robot.

Switching to another arm -- the other half of a dual-arm, or a different robot
entirely -- is a change of arguments, not of code::

    ros2 launch robot_parameter_identification dashboard.launch.py \\
        controller:=left_arm_joint_trajectory_controller \\
        port:=8301 config_file_path:=/home/me/left_arm_cell.json

Naming the controller is enough: the trajectory action and the controller-state
topic both follow from it. ``follow_joint_trajectory_action`` remains for a
controller whose action does not sit under its own name.

A signal that reaches ROS on its own topic rather than the robot's state topic
is added rather than special-cased::

    extra_telemetry_topics:="['/right_arm/motor_currents']" \\
        signal.current:=motor_current

Those topics are ``control_msgs/DynamicJointState`` and are merged by joint
name. If the robot publishes a quantity on no topic at all, republish it onto
one of these; that boundary is what keeps robot-specific code out of here.

Types matter here. A launch substitution is a string, and rclpy refuses a
string where the node declared a double, so non-string arguments are wrapped in
ParameterValue with an explicit value_type rather than passed raw.
"""

from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from robot_parameter_identification.system_config import (
    default_system_config_path, system_defaults,
)


INHERIT = "__system_config__"


def _dashboard(context):
    parameters = {"system_config": ParameterValue(
        LaunchConfiguration("system_config"), value_type=str)}
    for name, default in system_defaults()["ros"].items():
        value = LaunchConfiguration(name).perform(context)
        if value == INHERIT:
            continue
        value_type = type(default)
        if isinstance(default, list):
            value_type = List[str] if isinstance(default[0], str) else List[float]
        parameters[name] = ParameterValue(LaunchConfiguration(name), value_type=value_type)
    return [Node(
        package="robot_parameter_identification",
        executable="dashboard",
        name="robot_parameter_identification",
        output="screen",
        parameters=[parameters],
    )]


def generate_launch_description() -> LaunchDescription:
    declarations = [DeclareLaunchArgument(
        "system_config", default_value=str(default_system_config_path()),
        description="system defaults YAML; created from the package template if missing")]
    declarations.extend(
        DeclareLaunchArgument(
            name, default_value=INHERIT,
            description=f"override ros.{name} in system_config (factory default: {default!r})")
        for name, default in system_defaults()["ros"].items()
    )
    return LaunchDescription(declarations + [OpaqueFunction(function=_dashboard)])
