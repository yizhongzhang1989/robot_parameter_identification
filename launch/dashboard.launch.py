"""Launch the identification dashboard against whatever arm is already running.

This launches nothing but the dashboard. Bring the robot up however you
normally do, then point this at its controller and its telemetry topic. Every
robot-specific value is an argument; nothing here names a robot.

Switching to another arm -- the other half of a dual-arm, or a different robot
entirely -- is a change of arguments, not of code::

    ros2 launch robot_parameter_identification dashboard.launch.py \\
        follow_joint_trajectory_action:=/left_arm_controller/follow_joint_trajectory \\
        port:=8301 obstacle_file:=/home/me/left_arm_obstacles.json

Types matter here. A launch substitution is a string, and rclpy refuses a
string where the node declared a double, so numeric arguments are wrapped in
ParameterValue with an explicit value_type rather than passed raw.
"""

from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


TEXT_ARGUMENTS = (
    ("profile_path", "", "robot profile YAML; blank derives one from the URDF"),
    ("output_directory", "identification_results", "where results are written"),
    ("obstacle_file", "", "where the drawn obstacle scene is kept between "
                          "sessions; blank keeps it in memory only"),
    ("joint_state_topic", "/joint_states", "sensor_msgs/JointState source"),
    ("dynamic_joint_state_topic", "/dynamic_joint_states",
     "control_msgs/DynamicJointState source; blank falls back to joint_states"),
    ("robot_description_topic", "/robot_description", "URDF source"),
    ("follow_joint_trajectory_action",
     "/joint_trajectory_controller/follow_joint_trajectory",
     "the only path used to command motion"),
    ("effort_source", "current",
     "which channel the identification regresses against: current or torque. "
     "The unit of every identified parameter follows from this."),
    ("signal.position", "position", "interface carrying joint position"),
    ("signal.velocity", "velocity",
     "interface carrying joint velocity; blank differentiates position"),
    ("signal.current", "current",
     "interface carrying motor current; blank if the drive has none"),
    ("signal.torque", "",
     "interface carrying joint torque; blank if the drive has none"),
    ("signal.temperature", "temperature",
     "interface carrying joint temperature; blank disables the thermal guard"),
    ("signal.enabled", "enabled",
     "interface carrying the drive-enabled flag; blank disables that guard"),
    ("signal.fault_code", "fault_code",
     "interface carrying the drive fault word; blank disables that guard"),
    ("signal.voltage", "",
     "interface carrying bus voltage. Mapping it only enables the guard when "
     "a written profile supplies the window, because a derived profile's "
     "window is a default rather than a measurement."),
)

NUMERIC_ARGUMENTS = (
    ("port", "8300", int, "web port"),
    ("telemetry_stale_s", "0.5", float,
     "how old a telemetry frame may be before it counts as lost"),
    ("maximum_speed_deg_s", "0.0", float,
     "top sweep speed; 0 keeps the conservative derived default. Viscous "
     "friction is invisible at crawling speeds, so a full-size arm needs this "
     "raised. Capped at half of what the URDF rates each joint for."),
    ("workspace_limit_deg", "[0.0]", List[float],
     "cap on how far each joint may swing, in degrees; one value or one per "
     "joint, 0 for no cap. The URDF describes the arm, not the stand it is "
     "bolted to, so set this whenever the surroundings are not modelled."),
)


def generate_launch_description() -> LaunchDescription:
    declarations = [
        DeclareLaunchArgument(name, default_value=default, description=text)
        for name, default, text in TEXT_ARGUMENTS
    ] + [
        DeclareLaunchArgument(name, default_value=default, description=text)
        for name, default, _type, text in NUMERIC_ARGUMENTS
    ]

    parameters = {name: LaunchConfiguration(name)
                  for name, _default, _text in TEXT_ARGUMENTS}
    parameters.update({
        name: ParameterValue(LaunchConfiguration(name), value_type=value_type)
        for name, _default, value_type, _text in NUMERIC_ARGUMENTS
    })

    return LaunchDescription(declarations + [
        Node(
            package="robot_parameter_identification",
            executable="dashboard",
            name="robot_parameter_identification",
            output="screen",
            parameters=[parameters],
        ),
    ])
