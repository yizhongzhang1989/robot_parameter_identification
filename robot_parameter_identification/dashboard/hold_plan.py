"""Offline gravity-hold previews and execution of their frozen targets."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from pathlib import Path
import threading

import numpy as np


RIGHT_JOINTS = [f"right_arm_joint{index}" for index in range(1, 8)]
_IMPORT_LOCK = threading.RLock()


def _load_backend():
    """Load installed flat legacy modules without changing global import bindings."""
    import builtins
    import importlib.util
    import sys
    from types import SimpleNamespace

    from ament_index_python.packages import get_package_prefix

    directory = Path(get_package_prefix("rm_control")) / "lib" / "rm_control"
    modules = {}

    def render_xacro(command, **_options):
        import xacro

        if (len(command) < 2 or command[0] != "xacro" or
                any(":=" not in value for value in command[2:])):
            raise ValueError("offline model loading only permits in-process xacro")
        mappings = dict(value.split(":=", 1) for value in command[2:])
        return SimpleNamespace(
            stdout=xacro.process_file(command[1], mappings=mappings).toxml(),
            stderr="", returncode=0)

    def load(name):
        if name in modules:
            return modules[name]
        spec = importlib.util.spec_from_file_location(
            f"{__name__}._legacy_{name}", directory / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"installed rm_control helper unavailable: {name}")
        module = importlib.util.module_from_spec(spec)

        def import_local(import_name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and import_name == "subprocess" and name == "identified_zero_force_drag":
                return SimpleNamespace(run=render_xacro)
            if level == 0 and (directory / f"{import_name}.py").is_file():
                return load(import_name)
            return builtins.__import__(import_name, globals, locals, fromlist, level)

        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=import_local)
        modules[name] = module
        previous = sys.modules.get(spec.name)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            if previous is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous
        return module

    with _IMPORT_LOCK:
        original_path = sys.path[:]
        try:
            campaign = load("identified_static_hold_campaign")
            identified = load("identified_zero_force_drag")
            current = load("forward_current_controller_test")
            if current.WORKSPACE_ROOT is not None:
                identified.REPO = current.WORKSPACE_ROOT
        finally:
            sys.path[:] = original_path

    def arm_model():
        with _IMPORT_LOCK:
            original_path = sys.path[:]
            try:
                return identified._arm_model("right_")[0]
            finally:
                sys.path[:] = original_path

    return SimpleNamespace(campaign=campaign, identified=identified,
                           current=current, arm_model=arm_model)


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


def _validate_source(payload, names):
    if not isinstance(payload, dict):
        raise ValueError("source must be a model object")
    if names != RIGHT_JOINTS or payload.get("joint_names") != names:
        raise ValueError("source must describe the seven ordered right-arm joints")
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
                    collision_free, *, backend=None) -> dict:
    """Select a fixed preview; collision_free must screen the fresh scene in degrees.

    Invalid sources, insufficient admissible poses, or blocked transits raise
    ValueError. The caller owns scene freshness and must replan after changes.
    """
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("count must be an integer in [1, 20]")
    if not callable(collision_free):
        raise ValueError("a fresh scene collision guard is required")
    names = list(joint_names)
    start = _vector(start_deg, "start_deg").copy()
    backend = backend if backend is not None else _load_backend()
    path = _source_path(source.strip() or backend.current.DEFAULT_SOURCE)
    digest = _digest(path)
    _validate_source(json.loads(path.read_text("utf-8")), names)
    model = backend.identified.load_identification(path.parent, "right_")
    arm = backend.arm_model()
    if model["joint_names"] != names:
        raise ValueError("loaded model joint order differs from preview")
    if any(not any(column < arm.parameter_count for column in columns)
           for columns in model["columns"]):
        raise ValueError("each joint requires gravity regressor columns")
    candidates = np.asarray(backend.campaign._executed_poses(path.parent, names), dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 7:
        raise ValueError("candidate poses must have seven joints")
    candidates = candidates[np.isfinite(candidates).all(axis=1)]
    gravity = np.asarray([
        _vector(backend.identified.gravity_current(model, arm, pose), "predicted current")
        for pose in candidates], dtype=float).reshape((-1, 7))
    keep = backend.campaign.admissible(
        gravity, np.zeros(len(candidates)), backend.identified.CONTINUOUS_CURRENT_A, 0.0)
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
            "count": count, "seed": seed}
    if not source_is_current(plan):
        raise ValueError("source changed during planning")
    return plan


def execute_hold_plan(plan, seconds, output_directory, abort, on_progress,
                      run_child, *, move, backend=None) -> dict:
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
        plan = json.loads(json.dumps(plan, allow_nan=False))
        if not source_is_current(plan):
            raise ValueError("source changed or is unavailable; rebuild the preview")
        if not math.isfinite(seconds) or not 0.5 <= seconds <= 10.0:
            raise ValueError("seconds must be in [0.5, 10]")
        if (plan["joint_names"] != RIGHT_JOINTS or type(plan["count"]) is not int or
                not 1 <= plan["count"] <= 20 or len(plan["poses_deg"]) != plan["count"] or
                len(plan["predicted_current_a"]) != plan["count"]):
            raise ValueError("invalid hold plan structure")
        _vector(plan["start_deg"], "start_deg")
        poses = [_vector(pose, "pose").tolist() for pose in plan["poses_deg"]]
        for current in plan["predicted_current_a"]:
            _vector(current, "predicted current")
        backend = backend if backend is not None else _load_backend()
        corridor = 5.0
        temperature = min(40.0, backend.current.MAXIMUM_TEMPERATURE_C)
        output = Path(output_directory).expanduser().resolve()
        reports = [output / f"hold_{index:02d}.json" for index in range(1, len(poses) + 1)]
        if any(report.exists() for report in reports):
            raise ValueError("hold report paths already exist; use a fresh output directory")
        output.mkdir(parents=True, exist_ok=True)

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
            held = run_child([
                "ros2", "run", "rm_control", "hold_check",
                "--output", str(report_path), "--seconds", str(seconds),
                "--corridor-deg", str(corridor), "--temperature-c", str(temperature),
                "--source", plan["source"], "--arm", "right",
                "--ack", backend.current.ACKNOWLEDGEMENT], seconds + 300.0)
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
