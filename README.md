# robot_parameter_identification

Dynamic parameter identification for robot arms, as a module you can point at a
robot you did not build.

Identifies rigid-body inertial parameters and a joint friction model by
commanding a designed excitation campaign and regressing measured effort against
the rigid-body regressor. Ships a web dashboard with a live 3D view and an
editable obstacle scene that feeds collision screening.

## The contract with your robot

The module never talks to a driver, an SDK or a vendor protocol. It speaks only
standard ROS, and it is explicit about what it needs.

| Purpose | Interface | Required |
|---|---|---|
| Command motion | `control_msgs/action/FollowJointTrajectory` | yes |
| Kinematics and inertia priors | `/robot_description` (`std_msgs/String`) | yes |
| Joint telemetry | `sensor_msgs/JointState` **or** `control_msgs/DynamicJointState` | yes |

From telemetry it needs two signals per joint and can use five more:

| Signal | Required | Without it |
|---|---|---|
| `position` | yes | — |
| `effort` (motor current or joint torque) | yes | — |
| `velocity` | no | differentiated from position |
| `temperature` | no | no thermal ceiling guard |
| `enabled` | no | no drive-enabled guard |
| `fault_code` | no | no fault guard |
| `voltage` | no | no bus-voltage guard |

**If your robot does not publish something, republish it yourself.** Write a
small node that turns whatever your driver gives you into a `JointState` or a
`DynamicJointState` interface, and point the module at it. That boundary is
deliberate: it keeps every robot-specific line of code outside this repo.

Names are configuration, not code. A robot that calls its current
`motor_current` and its temperature `temp` is described like this:

```yaml
telemetry:
  dynamic_joint_state_topic: /dynamic_joint_states
  signals:
    position: position
    velocity: velocity
    effort: motor_current
    temperature: temp
    enabled: null          # not published; the enable guard is reported off
    fault_code: null
    effort_unit: ampere    # or newton_metre
```

The dashboard states which guards are dark because a signal is unmapped, rather
than skipping them silently.

## What it does not command

Only positions, through the trajectory action. Effort is the quantity being
measured, so commanding it would beg the question. The module never switches
controllers on its own.

## Obstacles and collision screening

The campaign proposes poses the arm has never held, so something has to veto the
ones that would hit the bench. That check runs on the model, using Pinocchio's
collision backend — no dependency beyond the one the dynamics already needs.

Obstacles are boxes **bound to a frame**. A box on the base frame is a bench; a
box on a distal link is a tool or a payload shroud and travels with the arm. You
place them by dragging in the 3D view; the pose is stored relative to the parent
frame, so the kinematics do the rest.

## Status

Working and tested:

- identification core — regression, excitation design, four-phase campaign,
  physical-consistency checks. Carried over from a stack that has completed six
  hardware runs on a 7-DOF arm.
- the ROS contract above (`interfaces.py`)
- obstacle scene and collision screening (`obstacles.py`)
- a dependency-light rehearsal plant for dry runs (`plants/analytic.py`)

Not yet built:

- the dashboard: 3D view, obstacle editing UI, parameter visualisation
- the ROS node entry point and launch files

`107 passed, 3 skipped`. The skips need a robot or a physics engine present.

### A caveat worth knowing

The rehearsal plant shares its rigid-body model with the identifier, so a clean
rehearsal proves the *pipeline*, not the *physics*. It cannot catch an error
that lives in the model itself. When a second engine is installed, cross-check
against it; `plants.independent_engine_available()` says whether one is.

## Running the tests

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD:$PWD/test:$PYTHONPATH" python3 -m pytest -q test
```

The suite builds its own 7-DOF arm, so no robot description need be installed.

## License

Apache-2.0.