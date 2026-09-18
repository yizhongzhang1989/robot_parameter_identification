"""Pure loading and rigid-only prediction of top-level ampere-domain fits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np


class InvalidCurrentModel(ValueError):
    """The selected fit has invalid structure, identity, units or numbers."""


def _joint_names(value, label="joint_names"):
    if (not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or
            not value or any(not isinstance(name, str) or not name for name in value) or
            len(set(value)) != len(value)):
        raise InvalidCurrentModel(f"{label} must be nonempty ordered unique joint names")
    return list(value)


def _finite_number(value, label):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise InvalidCurrentModel(f"{label} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise InvalidCurrentModel(f"{label} must be a finite number")


def _fit(columns, parameters):
    if not isinstance(columns, (list, tuple)) or not isinstance(parameters, (list, tuple)):
        raise InvalidCurrentModel("model columns and parameters must be arrays")
    if len(columns) != len(parameters):
        raise InvalidCurrentModel("model column/parameter count mismatch")
    for column in columns:
        if not isinstance(column, Integral) or isinstance(column, bool):
            raise InvalidCurrentModel("model columns must be integer indices")
    for value in parameters:
        _finite_number(value, "model parameter")


def _component_values(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InvalidCurrentModel("component keys must be strings")
            _component_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _component_values(item)
    elif value is not None and not isinstance(value, bool):
        _finite_number(value, "component value")


def load_identification(folder, prefix: str | None = None, *,
                        joint_names: Sequence[str] | None = None,
                        require_complete_pass: bool = False) -> dict:
    """Read result.json without consulting exported models or other instances.

    Prefixes are literal, so both ``right_`` and ``right_arm_`` select names
    such as ``right_arm_joint1``. Explicit joint_names enforce exact ordering
    and may be used without a prefix. At least one selector is required.
    Ampere units and finite structured fits are mandatory; complete/pass is
    opt-in so offline callers can inspect intermediate fits. Joint count and
    hardware capability are the caller's contract, not model restrictions.

    Returns fresh folder, joint_names, friction, components, parameters,
    columns and validation_rms_a fields. InvalidCurrentModel denotes schema
    or selection errors; filesystem and JSON decoding errors pass through.
    """
    if prefix is not None and (not isinstance(prefix, str) or not prefix):
        raise InvalidCurrentModel("prefix must be a nonempty string")
    expected = None if joint_names is None else _joint_names(joint_names, "expected joints")
    if prefix is None and expected is None:
        raise InvalidCurrentModel("select a model prefix or explicit ordered joint names")
    payload = json.loads((Path(folder) / "result.json").read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise InvalidCurrentModel("result must be an object")
    names = _joint_names(payload.get("joint_names"))
    if prefix is not None and not all(name.startswith(prefix) for name in names):
        raise InvalidCurrentModel(
            f"run describes {names[:1]}..., not an arm named {prefix!r}")
    if expected is not None and names != expected:
        raise InvalidCurrentModel("model must describe the selected ordered joints")
    if payload.get("effort_unit") != "ampere":
        raise InvalidCurrentModel(
            f"expected an ampere model, found {payload.get('effort_unit')!r}")
    verdict = payload.get("verdict")
    if verdict is not None and not isinstance(verdict, Mapping):
        raise InvalidCurrentModel("model verdict must be an object")
    if require_complete_pass and (
            payload.get("complete") is not True or (verdict or {}).get("state") != "pass"):
        raise InvalidCurrentModel("model must be complete with a passing verdict")
    joints = payload.get("joints")
    if not isinstance(joints, list) or len(joints) != len(names):
        raise InvalidCurrentModel("result has fewer fits than joints or a mismatched fit array")
    model = {"folder": str(folder), "joint_names": names, "friction": [],
             "components": [], "parameters": [], "columns": [], "validation_rms_a": []}
    for entry in joints:
        if not isinstance(entry, Mapping):
            raise InvalidCurrentModel("each joint fit must be an object")
        columns, parameters = entry.get("columns"), entry.get("parameters")
        _fit(columns, parameters)
        friction = entry.get("friction")
        components = entry.get("components")
        friction = {} if friction is None else friction
        components = {} if components is None else components
        if not isinstance(friction, Mapping) or not isinstance(components, Mapping):
            raise InvalidCurrentModel("friction and components must be objects")
        for key, value in friction.items():
            if not isinstance(key, str):
                raise InvalidCurrentModel("friction keys must be strings")
            if value is not None:
                _finite_number(value, f"friction {key}")
        _component_values(components)
        for key in ("coulomb_transition_deg_s", "stribeck_speed_deg_s"):
            if components.get(key) is not None:
                _finite_number(components[key], key)
        validation = entry.get("validation_rms_a")
        if validation is not None:
            _finite_number(validation, "validation_rms_a")
            if validation < 0:
                raise InvalidCurrentModel("validation_rms_a must be nonnegative")
        model["columns"].append(list(columns))
        model["parameters"].append(list(parameters))
        model["friction"].append(copy.deepcopy(dict(friction)))
        model["components"].append(copy.deepcopy(dict(components)))
        model["validation_rms_a"].append(validation)
    return model


def gravity_current(model: dict, arm, pose_deg) -> np.ndarray:
    """Sum rigid columns in stored order, excluding offsets and friction.

    The supplied arm owns its regressor and rigid parameter count. Its joint
    names, when exposed, must match the fit exactly. Inputs are not modified.
    """
    if not isinstance(model, Mapping):
        raise InvalidCurrentModel("model must be an object")
    names = _joint_names(model.get("joint_names"))
    if hasattr(arm, "joint_names") and list(arm.joint_names) != names:
        raise InvalidCurrentModel("kinematic model does not match the selected ordered joints")
    columns, parameters = model.get("columns"), model.get("parameters")
    if (not isinstance(columns, (list, tuple)) or
            not isinstance(parameters, (list, tuple)) or
            len(columns) != len(names) or len(parameters) != len(names)):
        raise InvalidCurrentModel("model must contain one column/parameter array per joint")
    for joint_columns, joint_parameters in zip(columns, parameters):
        _fit(joint_columns, joint_parameters)
    rigid_columns = arm.parameter_count
    if (not isinstance(rigid_columns, Integral) or isinstance(rigid_columns, bool) or
            rigid_columns < 0):
        raise InvalidCurrentModel("rigid parameter count must be a nonnegative integer")
    try:
        pose = np.array(pose_deg, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidCurrentModel("pose must contain one finite value per joint") from error
    if pose.shape != (len(names),) or not np.isfinite(pose).all():
        raise InvalidCurrentModel("pose must contain one finite value per joint")
    regressor = arm.torque_regressor(pose)
    try:
        regressor = np.asarray(regressor, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidCurrentModel("regressor must contain finite rigid rows") from error
    if (regressor.ndim != 2 or regressor.shape[0] != len(names) or
            regressor.shape[1] < rigid_columns or not np.isfinite(regressor).all()):
        raise InvalidCurrentModel("regressor must contain one finite rigid row per joint")
    drawn = np.zeros(len(names))
    with np.errstate(over="ignore", invalid="ignore"):
        for index, (joint_columns, joint_parameters) in enumerate(zip(columns, parameters)):
            for column, value in zip(joint_columns, joint_parameters):
                if 0 <= int(column) < rigid_columns:
                    drawn[index] += float(regressor[index][int(column)]) * float(value)
    if not np.isfinite(drawn).all():
        raise InvalidCurrentModel("predicted gravity current must be finite")
    return drawn