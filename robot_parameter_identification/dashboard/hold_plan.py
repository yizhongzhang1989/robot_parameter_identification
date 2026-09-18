"""Offline gravity-hold previews and execution of their frozen targets."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

from robot_parameter_identification import gravity_current_model, hold_candidates
from robot_parameter_identification.arm_identity import (
    ArmIdentity, DEFAULT_MAXIMUM_COMMAND_A, DEFAULT_PEAK_CURRENT_A,
)
from robot_parameter_identification.identification import ArmModel
from robot_parameter_identification.system_config import (
    checked_value, system_defaults, write_system_config_snapshot,
)

RIGHT_JOINTS = list(ArmIdentity("right").joint_names)
_OFFLINE_MODEL_LOCK = threading.RLock()


def _workspace_root() -> Path:
    candidates = (*Path.cwd().parents, *Path(__file__).resolve().parents)
    for candidate in (Path.cwd(), *candidates):
        if (candidate / "src" / "robot_description" / "urdf" / "robot.urdf.xacro").is_file():
            return candidate
    return Path.cwd()


def _offline_arm_model(prefix="right_"):
    """Render workspace geometry only when an offline caller supplies no live URDF."""
    import xacro

    with _OFFLINE_MODEL_LOCK:
        mappings = {
            "use_mock_hardware": "true",
            "right_arm_xyz": "0 -0.2 0.4", "right_arm_rpy": "0 0 0",
            "left_arm_xyz": "0 0.2 0.4", "left_arm_rpy": "0 0 0",
        }
        try:
            import yaml
            from common.workspace_utils import get_config_dir

            with (Path(get_config_dir()) / "robot_mounts.yaml").open(encoding="utf-8") as handle:
                mounts = yaml.safe_load(handle) or {}
            for arm in ("right_arm", "left_arm"):
                section = mounts.get(arm) or {}
                for key in ("xyz", "rpy"):
                    if key in section:
                        mappings[f"{arm}_{key}"] = " ".join(
                            str(float(value)) for value in section[key])
        except Exception as error:
            import warnings

            warnings.warn(f"offline hold model using default mounts ({error})", stacklevel=2)
        path = _workspace_root() / "src" / "robot_description" / "urdf" / "robot.urdf.xacro"
        urdf = xacro.process_file(str(path), mappings=mappings).toxml()
    return ArmModel.from_urdf_text(urdf, prefix)


def _load_backend():
    """Expose the legacy planning surface using only local shared functions."""
    source = (_workspace_root() / "identification_results" /
              "optimal_excitation_regime_separated-20260825-232541")
    return SimpleNamespace(
        campaign=SimpleNamespace(
            _executed_poses=hold_candidates.executed_poses,
            admissible=hold_candidates.admissible,
            random_then_order=hold_candidates.random_then_order,
            spread_then_order=hold_candidates.spread_then_order,
            transit_clear=hold_candidates.transit_clear),
        identified=SimpleNamespace(
            load_identification=gravity_current_model.load_identification,
            gravity_current=gravity_current_model.gravity_current,
            CONTINUOUS_CURRENT_A=np.asarray(DEFAULT_MAXIMUM_COMMAND_A),
            PEAK_CURRENT_A=np.asarray(DEFAULT_PEAK_CURRENT_A)),
        current=SimpleNamespace(
            DEFAULT_SOURCE=source, ACKNOWLEDGEMENT="I_AM_HOLDING_ARM_AND_ESTOP_READY",
            DEFAULT_CORRIDOR_DEG=5.0, MAXIMUM_TEMPERATURE_C=40.0),
        arm_model=_offline_arm_model)


def _source_path(source):
    path = Path(source).expanduser().resolve()
    return path if path.name == "result.json" else path / "result.json"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_is_current(plan) -> bool:
    """Whether the exact previewed result still exists and has the same bytes."""
    try:
        return _digest(_source_path(plan["source"])) == plan["source_digest"]
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _vector(values, label):
    array = np.asarray(values, dtype=float)
    if array.shape != (7,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain seven finite values")
    return array


def _command_limits(values):
    limits = _vector(values, "maximum command current").copy()
    if np.any(limits <= 0) or np.any(limits > DEFAULT_MAXIMUM_COMMAND_A):
        raise ValueError("maximum command current exceeds the supported command envelope")
    return limits


def _validate_source(payload, names):
    if not isinstance(payload, dict):
        raise ValueError("source must be a model object")
    ArmIdentity.from_joint_names(names)
    if payload.get("joint_names") != names:
        raise ValueError("source must describe the selected arm's seven ordered joints")
    verdict = payload.get("verdict")
    if (payload.get("complete") is not True or
            not isinstance(verdict, dict) or verdict.get("state") != "pass" or
            payload.get("effort_unit") != "ampere"):
        raise ValueError("source must be a complete, passing ampere model")
    joints = payload.get("joints")
    if not isinstance(joints, list) or len(joints) != 7:
        raise ValueError("source requires seven complete fits")
    for joint in joints:
        if not isinstance(joint, dict):
            raise ValueError("source requires a fit object for every joint")
        columns = joint.get("columns")
        parameters = joint.get("parameters")
        if (not isinstance(columns, list) or not columns or
                not isinstance(parameters, list) or len(columns) != len(parameters) or
                any(type(column) is not int or column < 0 for column in columns) or
                len(set(columns)) != len(columns) or
                np.asarray(parameters, dtype=float).shape != (len(columns),) or
                not np.isfinite(np.asarray(parameters, dtype=float)).all()):
            raise ValueError("source contains an incomplete or non-finite fit")


def build_hold_plan(source: str, joint_names: list, start_deg: list, count: int,
                    collision_free, *, backend=None, system_config=None,
                    urdf_text: str | None = None, exclude_poses_deg=None,
                    maximum_command_a=DEFAULT_MAXIMUM_COMMAND_A) -> dict:
    """Select a fixed preview; collision_free must screen the fresh scene in degrees.

    Invalid sources, insufficient admissible poses, or blocked transits raise
    ValueError. The caller owns scene freshness and must replan after changes.
    """
    settings = system_defaults() if system_config is None else system_config
    if type(count) is not int:
        raise ValueError("count must be an integer")
    checked_value(count, settings, "hold_test.poses", integer=True)
    if not callable(collision_free):
        raise ValueError("a fresh scene collision guard is required")
    command_limits = _command_limits(maximum_command_a)
    excluded = [] if exclude_poses_deg is None else exclude_poses_deg
    if not isinstance(excluded, list) or len(excluded) > 1000:
        raise ValueError("excluded poses must be a list of at most 1000 joint vectors")
    excluded_keys = {tuple(round(float(value), 3) for value in _vector(pose, "excluded pose"))
                     for pose in excluded}
    names = list(joint_names)
    identity = ArmIdentity.from_joint_names(names)
    if not source.strip() and identity.name != "right":
        raise ValueError("an explicit gravity source is required for the selected arm")
    start = _vector(start_deg, "start_deg").copy()
    backend = backend if backend is not None else _load_backend()
    path = _source_path(source.strip() or backend.current.DEFAULT_SOURCE)
    digest = _digest(path)
    _validate_source(json.loads(path.read_text("utf-8")), names)
    model = backend.identified.load_identification(path.parent, identity.prefix)
    arm = (ArmModel.from_urdf_text(urdf_text, identity.model_prefix)
           if urdf_text is not None else backend.arm_model(identity.prefix))
    if model["joint_names"] != names:
        raise ValueError("loaded model joint order differs from preview")
    if list(arm.joint_names) != names:
        raise ValueError("URDF joint order differs from the selected calibration")
    if any(not any(column < arm.parameter_count for column in columns)
           for columns in model["columns"]):
        raise ValueError("each joint requires gravity regressor columns")
    candidates = np.asarray(backend.campaign._executed_poses(path.parent, names), dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 7:
        raise ValueError("candidate poses must have seven joints")
    candidates = candidates[np.isfinite(candidates).all(axis=1)]
    if excluded_keys:
        keep = [tuple(round(float(value), 3) for value in pose) not in excluded_keys
                for pose in candidates]
        candidates = candidates[np.asarray(keep, dtype=bool)]
    gravity = np.asarray([
        _vector(backend.identified.gravity_current(model, arm, pose), "predicted current")
        for pose in candidates], dtype=float).reshape((-1, 7))
    keep = backend.campaign.admissible(
        gravity, np.zeros(len(candidates)), command_limits, 0.0)
    candidates, gravity = candidates[keep], gravity[keep]
    if len(candidates) < count:
        raise ValueError("insufficient finite poses inside the commissioned current envelope")
    seed = secrets.randbits(63)
    order = backend.campaign.random_then_order(
        candidates, np.abs(gravity).max(axis=1), count, seed=seed)
    if len(order) != count or len(set(order)) != count:
        raise ValueError("selection did not produce the requested distinct poses")
    poses = candidates[order]
    cursor = start
    for index, pose in enumerate(poses, 1):
        if not backend.campaign.transit_clear(
                cursor, pose, lambda sample: not collision_free(sample.copy())):
            raise ValueError(f"pose {index} has a blocked target or transit")
        cursor = pose
    plan = {"source": str(path.parent), "source_digest": digest,
            "joint_names": names, "start_deg": start.tolist(),
            "poses_deg": poses.tolist(), "predicted_current_a": gravity[order].tolist(),
            "maximum_command_a": command_limits.tolist(), "count": count, "seed": seed}
    if excluded_keys:
        plan["excluded_poses_deg"] = [list(pose) for pose in sorted(excluded_keys)]
    if not source_is_current(plan):
        raise ValueError("source changed during planning")
    return plan


def execute_hold_plan(plan, seconds, output_directory, abort, on_progress,
                      run_child, *, move, backend=None, system_config=None) -> dict:
    """Execute copied targets through service-owned moves and hold children.

    abort is a zero-argument predicate (or threading.Event). on_progress takes
    (phase, detail). move takes a copied pose list in degrees and owns motion
    and telemetry validation; it returns a dict with ok exactly True on success
    and optional reason, ros_deg, and other evidence. run_child runs holds only.
    Only a PASS hold with verified recovery completes a pose.
    No stop is labelled STOPPED here; uncertain or interrupted runs fail closed.
    """
    summary = {"result": "FAIL", "records": [], "reason": "",
               "restore_errors": [], "stop_verified": False}
    try:
        settings = system_defaults() if system_config is None else system_config
        plan = json.loads(json.dumps(plan, allow_nan=False))
        if not source_is_current(plan):
            raise ValueError("source changed or is unavailable; rebuild the preview")
        checked_value(seconds, settings, "hold_test.seconds")
        checked_value(plan["count"], settings, "hold_test.poses", integer=True)
        identity = ArmIdentity.from_joint_names(plan["joint_names"])
        _validate_source(json.loads(_source_path(plan["source"]).read_text("utf-8")),
                 list(identity.joint_names))
        if (type(plan["count"]) is not int or
            len(plan["poses_deg"]) != plan["count"] or
                len(plan["predicted_current_a"]) != plan["count"]):
            raise ValueError("invalid hold plan structure")
        _vector(plan["start_deg"], "start_deg")
        poses = [_vector(pose, "pose").tolist() for pose in plan["poses_deg"]]
        command_limits = _command_limits(plan.get("maximum_command_a", DEFAULT_MAXIMUM_COMMAND_A))
        for current in plan["predicted_current_a"]:
            if np.any(np.abs(_vector(current, "predicted current")) > command_limits):
                raise ValueError("predicted current exceeds the frozen instance command limit")
        backend = backend if backend is not None else _load_backend()
        defaults = settings["dashboard"]["hold_test"]
        corridor = checked_value(defaults["corridor_deg"], settings, "hold_test.corridor_deg")
        temperature = checked_value(defaults["temperature_c"], settings, "hold_test.temperature_c")
        output = Path(output_directory).expanduser().resolve()
        reports = [output / f"hold_{index:02d}.json" for index in range(1, len(poses) + 1)]
        if any(report.exists() for report in reports):
            raise ValueError("hold report paths already exist; use a fresh output directory")
        output.mkdir(parents=True, exist_ok=True)
        snapshot = (write_system_config_snapshot(output, settings)
                    if system_config is not None else None)
        if snapshot is not None:
            summary["system_config"] = str(snapshot)

        def check_abort():
            if (abort() if callable(abort) else abort.is_set()):
                raise RuntimeError("operator stop requested; completion not verified")

        def progress(detail):
            on_progress("hold_set", detail)

        for index, (pose, report_path) in enumerate(zip(poses, reports), 1):
            check_abort()
            if not source_is_current(plan):
                raise ValueError("source changed before move")
            record = {"index": index, "pose_deg": pose.copy(), "outcome": "FAIL"}
            summary["records"].append(record)
            summary["stop_verified"] = False
            progress({"target_pose": index, "pose_deg": pose.copy(), "stage": "moving"})
            check_abort()
            verdict = move(pose.copy())
            record["move"] = verdict
            check_abort()
            if not isinstance(verdict, dict):
                raise ValueError(f"pose {index} move returned no verdict dict")
            if verdict.get("ok") is not True:
                raise RuntimeError(f"pose {index} move failed: {verdict.get('reason')}")
            check_abort()
            if not source_is_current(plan):
                raise ValueError("source changed before hold")
            progress({"target_pose": index, "pose_deg": pose.copy(), "stage": "holding"})
            check_abort()
            command = [
                "ros2", "run", "rm_control", "hold_check",
                "--output", str(report_path), "--seconds", str(seconds),
                "--corridor-deg", str(corridor), "--temperature-c", str(temperature),
                "--source", plan["source"], "--arm", identity.name,
                "--ack", backend.current.ACKNOWLEDGEMENT]
            if snapshot is not None:
                command.extend(["--system-config", str(snapshot)])
            held = run_child(command, seconds + defaults["child_timeout_margin_s"])
            record["hold_returncode"] = held.returncode
            stopped = abort() if callable(abort) else abort.is_set()
            report = json.loads(report_path.read_text("utf-8"))
            record["hold"] = report
            errors = report.get("restore_errors")
            if not isinstance(errors, list):
                errors = ["hold report omitted valid recovery errors"]
            summary["restore_errors"].extend(str(error) for error in errors)
            summary["stop_verified"] = (
                report.get("stop_verified") is True and not errors)
            if (held.returncode != 0 or report.get("result") != "PASS" or
                    report.get("session_failure") or
                    not summary["stop_verified"]):
                raise RuntimeError(
                    f"pose {index} hold not verified: "
                    f"{report.get('reason', report.get('result'))}")
            if stopped:
                raise RuntimeError("operator stop requested; run interrupted")
            check_abort()
            record["outcome"] = "PASS"
            progress({"target_pose": None, "pose_deg": None,
                      "completed_pose": index,
                      "completed_pose_indices": list(range(1, index + 1)),
                      "stage": "completed"})
        summary.update(result="PASS", reason=f"{len(poses)} holds verified")
    except Exception as error:
        summary["reason"] = str(error)
    return summary
