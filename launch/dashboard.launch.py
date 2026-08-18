"""Launch the identification dashboard against whatever arm is running.

Every robot-specific value is an argument. Nothing here names a robot.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ARGUMENTS = (
    ("port", "8300", "web port"),
    ("profile_path", "", "robot profile YAML; required before anything moves"),
    ("output_directory", "identification_results", "where results are written"),
    ("joint_state_topic", "/joint_states", "sensor_msgs/JointState source"),
    ("dynamic_joint_state_topic", "/dynamic_joint_states",
     "control_msgs/DynamicJointState source; blank to use joint_states"),
    ("robot_description_topic", "/robot_description", "URDF source"),
    ("follow_joint_trajectory_action",
     "/joint_trajectory_controller/follow_joint_trajectory",
     "the only path used to command motion"),
    ("effort_unit", "ampere", "ampere or newton_metre"),
    ("signal.position", "position", "interface carrying joint position"),
    ("signal.velocity", "velocity", "interface carrying joint velocity"),
    ("signal.effort", "current", "interface carrying the measured effort"),
    ("signal.temperature", "temperature",
     "interface carrying joint temperature; blank disables the thermal guard"),
    ("signal.voltage", "", "blank disables the bus-voltage guard"),
    ("signal.enabled", "", "blank disables the drive-enabled guard"),
    ("signal.fault_code", "", "blank disables the fault guard"),
)


def generate_launch_description() -> LaunchDescription:
    declarations = [
        DeclareLaunchArgument(name, default_value=default, description=text)
        for name, default, text in ARGUMENTS
    ]
    parameters = {name: LaunchConfiguration(name) for name, _d, _t in ARGUMENTS}
    parameters["port"] = LaunchConfiguration("port")
    return LaunchDescription(declarations + [
        Node(
            package="robot_parameter_identification",
            executable="dashboard",
            name="robot_parameter_identification",
            output="screen",
            parameters=[parameters],
        ),
    ])
