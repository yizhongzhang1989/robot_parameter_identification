"""Shared fixtures for the generic tests.

The package ships no robot, so the tests build the arm they need. The workspace
model is used only because it is the one available here - nothing in the
package knows about it.
"""

import unittest
from pathlib import Path

from robot_parameter_identification import profile as profile_module

JOINT_COUNT = 7
PREFIX = "right_arm_"

# Plausible but arbitrary: the simulated plant needs *some* gains, and these are
# exactly the quantities a real calibration would go on to measure.
GAINS = (0.42, 0.37, 0.96, 1.05, 1.06, 0.90, 1.20)
COULOMB = (0.05, 0.20, 0.04, 0.15, 0.03, 0.03, 0.02)
VISCOUS = (0.004, 0.006, 0.003, 0.005, 0.002, 0.002, 0.001)

_CACHE = {}


def workspace_file(relative: str) -> Path:
    candidates = (
        Path(relative),
        Path(__file__).resolve().parents[3] / relative,
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise unittest.SkipTest(f"{relative} is not available")
    return path


def synthetic_urdf(joints: int = JOINT_COUNT, prefix: str = PREFIX,
                   collision: bool = True) -> str:
    """A serial arm written here, so the tests need no robot installed.

    Deliberately *not* a uniform arm. Identical links make the regressor
    columns near-degenerate, which lets pivoted column selection swap ties and
    makes conditioning look order-dependent; a real arm is not like that, and a
    fixture that is would test the wrong thing. Link inertias, offsets and axes
    all differ, and the base carries a rotation so base-alignment is testable.
    """
    tilt = "0.05 -0.03 0.02"
    parts = [f'<?xml version="1.0"?><robot name="synthetic">'
             f'<link name="{prefix}base_link"/>']
    # Shoulder-elbow-wrist, alternating perpendicular axes: the layout of a real
    # 7-DOF arm. A chain stacked purely along z is degenerate and would make the
    # conditioning tests pass or fail for reasons no real robot has.
    axes = ("0 0 1", "0 1 0", "0 0 1", "0 1 0", "0 0 1", "0 1 0", "0 0 1")
    laterals = ((0.0, 0.0), (0.03, 0.0), (0.0, 0.04), (-0.035, 0.0),
                (0.0, -0.03), (0.028, 0.0), (0.0, 0.02))
    for index in range(1, joints + 1):
        parent = f"{prefix}base_link" if index == 1 else f"{prefix}link{index - 1}"
        axis = axes[(index - 1) % len(axes)]
        offset_x, offset_y = laterals[(index - 1) % len(laterals)]
        mass = 3.1 - 0.32 * index
        ixx = 0.031 + 0.004 * index
        iyy = 0.024 - 0.002 * index
        izz = 0.017 + 0.003 * index
        offset = 0.10 if index == 1 else 0.18 + 0.02 * (index % 3)
        length = 0.16 + 0.02 * (index % 4)
        origin = (f'<origin xyz="{offset_x} {offset_y} {offset}" rpy="{tilt}"/>'
                  if index == 1
                  else f'<origin xyz="{offset_x} {offset_y} {offset}"/>')
        shape = ""
        if collision:
            shape = (f'<collision><origin xyz="0 0 {length / 2:.3f}"/>'
                     f'<geometry><box size="0.07 0.07 {length:.3f}"/></geometry>'
                     '</collision>')
        parts.append(f"""
        <link name="{prefix}link{index}">
        <inertial><origin xyz="0.01 {0.004 * index:.3f} {length / 2:.3f}"/>
        <mass value="{mass:.3f}"/>
        <inertia ixx="{ixx:.4f}" ixy="0.001" ixz="0.0005"
                 iyy="{iyy:.4f}" iyz="0.0008" izz="{izz:.4f}"/>
        </inertial>{shape}</link>
        <joint name="{prefix}joint{index}" type="revolute">
        <parent link="{parent}"/><child link="{prefix}link{index}"/>
        {origin}<axis xyz="{axis}"/>
        <limit lower="-2.6" upper="2.6" effort="60" velocity="3"/></joint>""")
    return "".join(parts) + "</robot>"


def scene_path() -> str:
    return str(workspace_file("src/robot_description/mujoco/mjcf/scene.xml"))


def xacro_path() -> Path:
    return workspace_file(
        "install/robot_description/share/robot_description/urdf/robot.urdf.xacro")


def test_profile() -> profile_module.RobotProfile:
    """A profile for the workspace arm, built here rather than shipped."""
    if "profile" not in _CACHE:
        _CACHE["profile"] = profile_module.RobotProfile.from_dict({
            "schema_version": 1,
            "name": "test-7dof",
            "joints": {
                "prefix": PREFIX,
                "names": [f"{PREFIX}joint{i}" for i in range(1, JOINT_COUNT + 1)],
            },
            "limits": {
                "position_deg": [177.6, 129.9, 177.6, 134.9, 177.6, 127.9, 359.8],
                "continuous_current_a": [3.0, 4.1, 3.0, 3.1, 1.1, 1.15, 0.6],
                "peak_current_a": [4.0, 5.0, 4.0, 4.0, 1.5, 1.5, 0.8],
            },
            "envelope": {"temperature_c": 45.0, "sustained_speed_deg_s": 15.0},
        }, source="<test fixture>")
    return _CACHE["profile"]


# The moved tests refer to this by its former name.
rm75_profile = test_profile
