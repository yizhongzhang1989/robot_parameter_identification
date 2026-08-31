"""Turn a load sweep into numbers, and the numbers into a page.

The computation lives here rather than in ``tools`` because the sweep writes
its own report when it finishes, and the runner cannot import a script that is
not installed with the package. The command line tool is a thin shell over the
same functions, so what the page says and what the terminal says cannot drift
apart.

One thing is measured here that the sweep itself does not measure. Splitting a
pair of opposed passes gives friction in the half-difference and gravity in the
half-sum, and the half-sum is a free calibration: it is the current the drive
needed to hold a known gravity torque, so regressing it against that torque
gives amperes per newton metre for that joint. Joint one comes out at 0.4186
with a correlation of -0.99996. Every joint has its own, they differ by a factor
of three across the arm, and using one joint's figure for all seven would put
the torque numbers out by that much.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import html
import json
import time

import numpy as np

REPORT_NAME = "report.html"


# -- reading --------------------------------------------------------------


def read(folder):
    """Records and manifest. A torn last line is the run still writing."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
    rows = []
    with (folder / "records.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows, manifest


def split(rows, arm=None):
    """Half-difference and half-sum, per joint, level and speed rung.

    Anything that reverses with direction is friction; anything that does not
    is gravity and the error in the estimate of it.
    """
    grouped = defaultdict(lambda: {"+": [], "-": []})
    for row in rows:
        grouped[(row["joint"], row["level"], row["speed_deg_s"])][
            row["direction"]].append(row)
    table = []
    for (joint, level, speed), sides in sorted(grouped.items()):
        if not sides["+"] or not sides["-"]:
            continue
        forward = float(np.mean([r["current_a"] for r in sides["+"]]))
        back = float(np.mean([r["current_a"] for r in sides["-"]]))
        every = sides["+"] + sides["-"]
        signed_values = [
          r["load"].get("axial_signed_nm", np.nan) for r in every]
        if arm is not None and not np.isfinite(signed_values).all():
          signed_values = [
            float(arm.inverse_dynamics(r["pose_deg"])[joint])
            for r in every]
        table.append({
            "joint": joint, "level": level, "speed": speed,
            "friction": (forward - back) / 2.0,
            "gravity": (forward + back) / 2.0,
            "load": float(np.mean([r["load"]["axial_nm"] for r in every])),
            "signed": float(np.mean(signed_values)),
            "temperature": float(np.mean([r["temperature_c"] for r in every])),
            "passes": len(every),
        })
    return table


def torque_calibration(table, joint: int) -> dict:
    """Signed effort gain, zero offset and how well they are determined.

    From the half-sum, which is the current spent holding gravity rather than
    fighting friction. Free, because the sweep already drove both directions at
    every posture in order to separate the two.

    Against the *signed* torque where the records carry it. The current spent
    holding gravity reverses when the torque does, and the sweep searches the
    whole workspace, so it finds postures either side of zero; regressing a
    signed current on an unsigned torque gave joint three a correlation of
    -0.01 and a constant of -0.02 A per Nm, which then reported friction as
    1370 per cent of the load. Older runs have only the magnitude, and matching
    magnitude against magnitude recovers the same figure to within a few per
    cent, so that is the fallback rather than a refusal.
    """
    mine = [r for r in table if r["joint"] == joint]
    if len(mine) < 4:
        return {"gain": float("nan"), "offset": float("nan"),
                "fitness": float("nan"), "signed": False}
    hold = np.array([r["gravity"] for r in mine])
    signed = np.array([r.get("signed", np.nan) for r in mine])
    if np.isfinite(signed).all() and signed.std() > 1e-6:
        torque = signed
        has_sign = True
    else:
        torque, hold = np.array([r["load"] for r in mine]), np.abs(hold)
        has_sign = False
    if torque.std() < 1e-4 or hold.std() < 1e-6:
        return {"gain": float("nan"), "offset": float("nan"),
                "fitness": float("nan"), "signed": has_sign}
    fitness = float(np.corrcoef(torque, hold)[0, 1])
    # A joint whose gravity current does not track its gravity torque has not
    # been calibrated by this, whatever number least squares returns.
    if not np.isfinite(fitness) or abs(fitness) < 0.95:
        return {"gain": float("nan"), "offset": float("nan"),
                "fitness": fitness, "signed": has_sign}
    gain, offset = np.polyfit(torque, hold, 1)
    return {"gain": float(gain), "offset": float(offset),
            "fitness": fitness, "signed": has_sign}


def torque_constant(table, joint: int) -> tuple[float, float]:
    """Absolute amperes per newton metre, retained for report compatibility."""
    calibration = torque_calibration(table, joint)
    return abs(calibration["gain"]), calibration["fitness"]


def grid(table, joint: int):
    """Speeds, one friction curve per level, and the load each level held."""
    mine = [r for r in table if r["joint"] == joint]
    if not mine:
        return None
    levels = sorted({r["level"] for r in mine})
    speeds = sorted({r["speed"] for r in mine})
    stack, loads, kept = [], [], []
    for level in levels:
        row = {r["speed"]: r for r in mine if r["level"] == level}
        if len(row) < len(speeds):
            continue
        stack.append([row[s]["friction"] for s in speeds])
        loads.append(float(np.mean([row[s]["load"] for s in speeds])))
        kept.append(level)
    # Eight parameters against three points is not a fit but an interpolation
    # that reports zero residual and means nothing.
    if not stack or len(speeds) < 6:
        return None
    return {"speeds": np.array(speeds, dtype=float),
            "stack": np.array(stack, dtype=float),
            "loads": np.array(loads, dtype=float), "levels": kept}


# -- fitting --------------------------------------------------------------


def fit(speeds, stack, loads, load_terms="full"):
    """One law for the joint, load as a parameter where the levels allow one.

    ``load_terms`` is "full" with enough levels to see the shape change with
    load, "coulomb" with only two, where the offset between them is all that
    can honestly be claimed, and "none" when the load could not be varied.
    Width and viscous slope are shared across loads because they belong to the
    joint rather than to what it is carrying; fitting a width per level lets it
    collapse and hand the low-speed shape to the Stribeck term, which then
    reports a load dependence that is an artefact of that trade.
    """
    from scipy.optimize import least_squares  # noqa: PLC0415

    def model(theta):
        width, viscous, c0, ck, d0, dk, vs0, vsk = theta
        return np.array([
            (c0 + ck * load) * np.tanh(speeds / max(width, 1e-3))
            + (d0 + dk * load) * np.exp(
                -speeds / max(vs0 + vsk * load, 1e-3))
            + viscous * speeds
            for load in loads])

    high = [20.0, 1.0, 5.0, 5.0, 5.0, 5.0, 200.0, 50.0]
    start = [0.8, 0.004, 0.24, 0.12, 0.04, 0.19, 0.5, 0.25]
    if load_terms != "full":
        high[5] = high[7] = 1e-9
        start[5] = start[7] = 0.0
    if load_terms == "none":
        high[3] = 1e-9
        start[3] = 0.0
    found = least_squares(
        lambda t: (model(t) - stack).ravel(), start,
        bounds=([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0], high))
    width, viscous, c0, ck, d0, dk, vs0, vsk = found.x
    return {"width": width, "viscous": viscous, "c0": c0, "ck": ck,
            "d0": d0, "dk": dk, "vs0": vs0, "vsk": vsk,
            "load_terms": load_terms,
            "rms": float(np.sqrt(np.mean(found.fun ** 2)))}


def curve(model, speed, load):
    speed = np.asarray(speed, dtype=float)
    return ((model["c0"] + model["ck"] * load)
            * np.tanh(speed / model["width"])
            + (model["d0"] + model["dk"] * load)
            * np.exp(-np.abs(speed) / (model["vs0"] + model["vsk"] * load))
            * np.sign(speed)
            + model["viscous"] * speed)


TERMS = ("axial_nm", "radial_n", "thrust_n", "tilt_nm")


def load_regression(rows, table, joint: int):
    """Friction against every load term that moves, and whether that is honest.

    Gravity presses a joint four ways at once and a posture that raises one
    tends to raise the others. The condition number is the whole point of this:
    it says whether the split between the terms is a measurement or arithmetic.
    """
    by_level = defaultdict(list)
    for row in rows:
        if row["joint"] == joint:
            by_level[row["level"]].append(row)
    levels = sorted(by_level)
    if len(levels) < 4:
        return None
    matrix = np.array([[float(np.mean([r["load"][t] for r in by_level[l]]))
                        for t in TERMS] for l in levels])
    friction = np.array([float(np.mean(
        [abs(r["friction"]) for r in table
         if r["joint"] == joint and r["level"] == l])) for l in levels])
    varying = [k for k in range(4) if matrix[:, k].std() > 1e-6]
    design = np.column_stack([np.ones(len(levels))]
                             + [matrix[:, k] for k in varying])
    coefficients, *_ = np.linalg.lstsq(design, friction, rcond=None)
    condition = float(np.linalg.cond(design))
    predicted = design @ coefficients
    with np.errstate(invalid="ignore"):
        correlation = {
            TERMS[k]: (None if matrix[:, k].std() < 1e-9
                       else float(np.corrcoef(matrix[:, 0], matrix[:, k])[0, 1]))
            for k in range(1, 4)}
    return {
        "levels": len(levels),
        "condition": condition,
        "offset": float(coefficients[0]),
        "coefficients": {TERMS[k]: float(coefficients[i + 1])
                         for i, k in enumerate(varying)},
        "correlation": correlation,
        "rms": float(np.sqrt(np.mean((friction - predicted) ** 2))),
        "trustworthy": condition < 10.0,
    }


# -- the whole story ------------------------------------------------------


def summarise(folder, arm=None) -> dict:
    """Everything the page shows, computed once."""
    rows, manifest = read(folder)
    table = split(rows, arm=arm)
    repeats = int(manifest["plan"].get("repeats", 1))
    joints, temperatures = [], [r["temperature_c"] for r in rows]
    # Every joint's own constant first, so a joint that could not be
    # calibrated falls back to this arm rather than to a number measured on
    # one joint of it. They run 0.42 to 1.55 across these seven.
    calibrations = {
        entry["joint"]: torque_calibration(table, entry["joint"])
        for entry in manifest["joints"]}
    resolved = [abs(value["gain"]) for value in calibrations.values()
                if not np.isnan(value["gain"])]
    fallback = float(np.median(resolved)) if resolved else 0.4186
    for entry in manifest["joints"]:
        number = entry["joint"]
        done = len({r["key"] for r in rows if r["joint"] == number})
        wanted = len(entry["passes"]) * repeats * 2
        calibration = calibrations[number]
        gain = calibration["gain"]
        fitness = calibration["fitness"]
        borrowed = bool(np.isnan(gain))
        data = grid(table, number)
        item = {
            "joint": number,
            "name": entry["name"],
            "levels": entry["levels"],
            "span_nm": entry["span_nm"],
            "reachable_nm": entry["reachable_nm"],
            "loadable": entry["loadable"],
            "note": entry.get("note", ""),
            "arc_room_deg": entry.get("arc_room_deg", 0.0),
            "refused": len(entry.get("refused", [])),
            "measured": done, "designed": wanted,
            "amps_per_nm": None if borrowed else round(abs(gain), 4),
            "amps_per_nm_fit": None if np.isnan(fitness) else round(fitness, 5),
            "amps_per_nm_borrowed": borrowed,
            "amps_per_nm_used": round(fallback if borrowed else abs(gain), 4),
            "signed_amps_per_nm": (
                round(gain, 6)
                if not borrowed and calibration["signed"] else None),
            "effort_offset_a": (
                round(calibration["offset"], 6)
                if not borrowed and calibration["signed"] else None),
        }
        if data is not None:
            enough = entry["loadable"] and len(data["levels"]) >= 2
            terms = ("none" if not enough
                     else "full" if len(data["levels"]) >= 3 else "coulomb")
            model = fit(data["speeds"], data["stack"], data["loads"], terms)
            mean = float(np.mean(np.abs(data["stack"])))
            item["fit"] = {k: (round(v, 6) if isinstance(v, float) else v)
                           for k, v in model.items()}
            # A joint measured as frictionless is a fault in the measurement,
            # not a residual of zero per cent of nothing.
            item["fit"]["share_of_mean"] = (round(100 * model["rms"] / mean, 2)
                                            if mean > 1e-9 else None)
            item["speeds"] = [round(float(v), 3) for v in data["speeds"]]
            item["loads"] = [round(float(v), 4) for v in data["loads"]]
            item["measured_curves"] = [[round(float(v), 5) for v in curve_]
                                       for curve_ in data["stack"]]
            item["fitted_curves"] = [
                [round(float(v), 5)
                 for v in curve(model, data["speeds"], load)]
                for load in data["loads"]]
            scale = fallback if borrowed else abs(gain)
            worth = []
            for which, index in (("light", 0),
                                 ("heavy", len(data["levels"]) - 1)):
                measured = data["stack"][index]
                residual = measured - curve(model, data["speeds"],
                                            data["loads"][index])
                friction = float(np.mean(np.abs(measured))) / scale
                error = float(np.sqrt(np.mean(residual ** 2))) / scale
                worth.append({
                    "which": which,
                    "load_nm": round(float(data["loads"][index]), 3),
                    "friction_nm": round(friction, 4),
                    "residual_nm": round(error, 4),
                    "left": (round(100 * error / friction, 2)
                             if friction > 1e-9 else None),
                    "gain": (round(friction / error, 1)
                             if error > 1e-6 and friction > 1e-9 else None),
                })
            item["worth"] = worth
            if terms != "none":
                item["per_nm"] = round(model["ck"], 5)
                item["share_of_load"] = round(100 * model["ck"] / scale, 1)
        item["regression"] = load_regression(rows, table, number)
        joints.append(item)
    return {
        "kind": "load_sweep",
        "created": manifest.get("created"),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "joint_names": manifest.get("joint_names", []),
        "plan": manifest.get("plan", {}),
        "speeds_deg_s": manifest.get("speeds_deg_s", []),
        "records": len(rows),
        "pairs": len(table),
        "measured": sum(j["measured"] for j in joints),
        "designed": sum(j["designed"] for j in joints),
        "temperature": {
            "low": round(min(temperatures), 1) if temperatures else None,
            "high": round(max(temperatures), 1) if temperatures else None},
        "joints": joints,
    }


def score_validation(arm, observations, summary: dict) -> dict:
    """Score a saved sweep model on observations it never trained on.

    The baseline is the URDF inverse dynamics converted with the sweep's
    signed effort gain, plus the sweep's fitted friction law at the current
    speed and gravity load. This puts it on the same per-sample current metric
    as the optimal-excitation model without refitting either side.
    """
    observations = list(observations)
    if not observations:
        return {"available": False, "reason": "no validation observations"}
    joints = sorted(summary.get("joints") or [], key=lambda item: item["joint"])
    if len(joints) != arm.joint_count:
        return {"available": False,
                "reason": "load sweep joint count does not match this arm"}
    missing = [item["name"] for item in joints if not item.get("fit")]
    if missing:
        return {"available": False,
                "reason": "load sweep has no friction fit for "
                          + ", ".join(missing)}

    squared = [[] for _ in range(arm.joint_count)]
    for record in observations:
        position = np.asarray(record.position_deg, dtype=float)
        velocity = np.asarray(record.velocity_deg_s, dtype=float)
        acceleration = np.asarray(record.acceleration_deg_s2, dtype=float)
        measured = np.asarray(record.current_a, dtype=float)
        torque = arm.inverse_dynamics(position, velocity, acceleration)
        loads = arm.joint_loads(position)[:, 0]
        for joint, model in enumerate(joints):
            signed_gain = model.get("signed_amps_per_nm")
            gain = (float(signed_gain) if signed_gain is not None
                    else float(model["amps_per_nm_used"]))
            offset = float(model.get("effort_offset_a") or 0.0)
            prediction = (gain * float(torque[joint]) + offset
                          + float(curve(model["fit"], velocity[joint],
                                        loads[joint])))
            squared[joint].append((prediction - measured[joint]) ** 2)

    errors = [float(np.sqrt(np.mean(values))) for values in squared]
    return {
        "available": True,
        "method": "urdf_inverse_dynamics_plus_load_sweep_friction",
        "validation_samples": len(observations),
        "validation_rms_a": errors,
        "mean_validation_rms_a": float(np.mean(errors)),
        "worst_validation_rms_a": float(np.max(errors)),
        "joints": [
            {"joint": index, "name": joints[index]["name"],
             "validation_rms_a": errors[index]}
            for index in range(len(errors))
        ],
    }


def score_saved_sweep(folder, arm, observations,
                      expected_joint_names=None) -> dict:
    """Load one sweep and score it, refusing a result from another arm."""
    summary = summarise(folder, arm=arm)
    expected = list(expected_joint_names or arm.joint_names)
    recorded = list(summary.get("joint_names") or [])
    if recorded and recorded != expected:
        return {"available": False,
                "reason": "load sweep joint names do not match this arm"}
    result = score_validation(arm, observations, summary)
    result["source"] = str(Path(folder))
    return result


def write_report(folder) -> Path:
    folder = Path(folder)
    page = render(summarise(folder))
    path = folder / REPORT_NAME
    path.write_text(page, encoding="utf-8")
    return path


def render(document: dict) -> str:
    blob = json.dumps(document, ensure_ascii=False,
                      default=lambda v: getattr(v, "tolist", lambda: str(v))())
    blob = blob.replace("</", "<\\/")
    strings = json.dumps(TEXT, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"load sweep {document.get('created', '')}")
    return _TEMPLATE.replace("__TITLE__", title) \
                    .replace("__STRINGS__", strings) \
                    .replace("__DATA__", blob)


TEXT = {
    "title": {"en": "Load sweep", "zh": "负载扫掠报告"},
    "subtitle": {
        "en": "What each joint's friction does with the load it carries, "
              "measured one joint at a time across every load gravity can be "
              "made to put on it.",
        "zh": "逐个关节测量：摩擦如何随其所承受的负载变化。负载由重力施加，"
              "覆盖每个关节在工作空间内可达到的全部范围。",
    },
    "lang": {"en": "中文", "zh": "English"},

    "run.head": {"en": "The run", "zh": "本次运行"},
    "run.records": {"en": "records", "zh": "记录条数"},
    "run.passes": {"en": "passes measured", "zh": "已测量通过数"},
    "run.pairs": {"en": "level-speed pairs", "zh": "负载-速度组合"},
    "run.temperature": {"en": "joint temperature", "zh": "关节温度"},
    "run.speeds": {"en": "speed ladder", "zh": "速度阶梯"},
    "run.created": {"en": "started", "zh": "开始时间"},

    "design.head": {"en": "What was measured, and why that much",
                    "zh": "测量了什么，以及为何是这个量"},
    "design.say": {
        "en": "The number of load levels is not a setting. It follows from the "
              "span the search actually found, divided by the smallest gap this "
              "measurement can resolve. Gravity loads the shoulder over nearly "
              "five newton metres and the wrist over half a milli-newton-metre, "
              "so asking for ten levels everywhere would return ten copies of "
              "one measurement on the joints that cannot be loaded.",
        "zh": "负载等级数不是设定值，而是由搜索实际找到的跨度除以本测量能分辨的最小间隔"
              "得出。重力能给肩部关节约 5 N·m 的负载，而给腕部关节只有 0.0005 N·m，"
              "因此对无法加载的关节强行取十个等级，只会得到同一次测量的十份副本。",
    },
    "design.joint": {"en": "joint", "zh": "关节"},
    "design.levels": {"en": "levels", "zh": "等级"},
    "design.span": {"en": "load span N·m", "zh": "负载跨度 N·m"},
    "design.reach": {"en": "reachable N·m", "zh": "可达范围 N·m"},
    "design.measured": {"en": "measured", "zh": "已测量"},
    "design.arc": {"en": "pass arc °", "zh": "扫掠弧度 °"},
    "design.refused": {"en": "refused", "zh": "被拒绝"},
    "design.kt": {"en": "A per N·m", "zh": "安培/N·m"},
    "design.ktfit": {"en": "r", "zh": "相关系数 r"},
    "design.ktsay": {
        "en": "Amperes per newton metre is measured per joint from the "
              "half-sum of each opposed pair, which is the current spent "
              "holding gravity rather than fighting friction. It converts "
              "every current below into joint torque, and it runs from 0.42 "
              "to 1.55 across this arm, so one joint's figure cannot stand "
              "for the rest. A value marked * was borrowed from the median: "
              "that joint's gravity current did not track its gravity torque "
              "closely enough to calibrate anything, which is what happens "
              "when the load cannot be varied.",
        "zh": "安培/N·m 由每对反向通过的半和逐关节测得——那是用于抵抗重力而非克服摩擦的"
              "电流。下文所有电流值都用它换算成关节力矩。该系数在本机械臂上从 0.42 "
              "到 1.55，相差近四倍，因此不能用某一个关节的值代表其余关节。"
              "带 * 的数值借用了各关节中位数：该关节的重力电流与其重力力矩相关性不足，"
              "无法据此标定——负载无法改变时就会这样。",
    },

    "law.head": {"en": "The law fitted to each joint", "zh": "各关节拟合出的摩擦律"},
    "law.say": {
        "en": "Coulomb friction that rises with load, a Stribeck term that "
              "makes the peak just above standstill, and a viscous slope. The "
              "width and the viscous slope are shared across loads because "
              "they belong to the joint, not to what it is carrying.",
        "zh": "随负载上升的库仑摩擦、造成低速峰值的 Stribeck 项，以及粘滞斜率。"
              "过渡宽度与粘滞斜率在各负载间共享，因为它们属于关节本身，"
              "而非它当前承载的东西。",
    },
    "law.coulomb": {"en": "coulomb A", "zh": "库仑 A"},
    "law.pernm": {"en": "per N·m", "zh": "每 N·m"},
    "law.dip": {"en": "stribeck A", "zh": "Stribeck A"},
    "law.width": {"en": "width °/s", "zh": "宽度 °/s"},
    "law.viscous": {"en": "viscous", "zh": "粘滞"},
    "law.rms": {"en": "residual A", "zh": "残差 A"},
    "law.share": {"en": "of mean", "zh": "占均值"},
    "law.terms": {"en": "load terms", "zh": "负载项"},
    "law.full": {"en": "full", "zh": "完整"},
    "law.coulombonly": {"en": "offset only", "zh": "仅偏移"},
    "law.none": {"en": "none", "zh": "无"},

    "curve.head": {"en": "Friction against speed, one line per load",
                   "zh": "摩擦-速度曲线，每条对应一个负载"},
    "curve.say": {
        "en": "Dots are measured, lines are the fitted law. Dark is light "
              "load, bright is heavy. The speed axis is logarithmic because "
              "the ladder is, and because everything worth seeing at the "
              "bottom happens below two degrees per second.",
        "zh": "圆点为实测，曲线为拟合。深色为轻载，亮色为重载。"
              "速度轴取对数，因为阶梯本身是对数分布的，而低速端值得看的现象"
              "都发生在 2 °/s 以下。",
    },
    "curve.speed": {"en": "speed °/s", "zh": "速度 °/s"},
    "curve.friction": {"en": "friction A", "zh": "摩擦 A"},
    "curve.pick": {"en": "joint", "zh": "关节"},

    "worth.head": {"en": "What compensation would be worth",
                   "zh": "摩擦补偿的价值"},
    "worth.say": {
        "en": "Friction as felt at the joint, and what is left after the law "
              "above is subtracted. The gain is how much lighter the joint "
              "would feel if the model were used to cancel it.",
        "zh": "关节处实际感受到的摩擦，以及减去上述摩擦律之后的残余。"
              "增益表示若用该模型进行补偿，关节会变轻多少倍。",
    },
    "worth.friction": {"en": "friction N·m", "zh": "摩擦 N·m"},
    "worth.residual": {"en": "left N·m", "zh": "残余 N·m"},
    "worth.left": {"en": "left", "zh": "剩余比例"},
    "worth.gain": {"en": "gain", "zh": "增益"},
    "worth.at": {"en": "at", "zh": "负载"},
    "worth.light": {"en": "lightest", "zh": "最轻"},
    "worth.heavy": {"en": "heaviest", "zh": "最重"},

    "rise.head": {"en": "Does friction rise with load", "zh": "摩擦是否随负载上升"},
    "rise.say": {
        "en": "It does, on every joint that can be loaded at all. The share is "
              "the fraction of the torque a joint holds that it spends on "
              "friction. Read the caveat below before treating any of these as "
              "a physical coefficient.",
        "zh": "凡是能够加载的关节，答案都是肯定的。"
              "占比指关节所承载力矩中被摩擦消耗的比例。"
              "在把这些数值当作物理系数使用之前，请先阅读下方的重要说明。",
    },
    "rise.pernm": {"en": "A per N·m", "zh": "安培/N·m"},
    "rise.share": {"en": "share of the load carried", "zh": "占所承载负载"},
    "rise.span": {"en": "over the span", "zh": "全跨度累计"},
    "rise.pinned": {"en": "how well pinned down", "zh": "确定程度"},
    "rise.levels": {"en": "levels", "zh": "个等级"},
    "rise.two": {"en": "two levels: an offset, not a shape",
                 "zh": "仅两个等级：只能给出偏移，给不出形状"},
    "rise.novary": {"en": "load could not be varied", "zh": "负载无法改变"},

    "caveat.head": {"en": "The caveat that matters most", "zh": "最重要的一条说明"},
    "caveat.say": {
        "en": "Gravity presses a joint four ways at once: a torque about its "
              "own axis, a radial force, a thrust along the axis, and a "
              "tilting moment. A posture that raises one tends to raise the "
              "others, so this sweep measures friction against a bundle and "
              "credits the whole bundle to whichever term it is plotted "
              "against. The condition number below says whether the split "
              "between the terms is a measurement or arithmetic: under about "
              "ten it can be believed, and none of these are.",
        "zh": "重力同时以四种方式压迫一个关节：绕其自身轴的力矩、径向力、沿轴推力，"
              "以及倾覆力矩。使其中一项增大的位形往往也使其余各项增大，"
              "因此本次扫掠测得的是这一“负载包”的整体效应，"
              "并把整包效应记在了被用作横轴的那一项头上。"
              "下表的条件数说明各项之间的拆分究竟是测量还是算术："
              "低于约 10 才可信，而这里没有一个达标。",
    },
    "caveat.why": {
        "en": "It is not a search failure. A posture only measures one load if "
              "the load holds still while the joint sweeps, which means "
              "sitting at an extremum of the moment about the axis — and there "
              "the axial and tilting components are locked to each other. "
              "Screening poses for that condition raises the correlation "
              "between them from 0.48 to 0.90 on joint four. The constraint "
              "that makes the measurement clean is the one that confounds it. "
              "Separating these terms needs an external payload, or a design "
              "that sweeps through a load excursion on purpose and uses the "
              "per-sample load recorded with every row.",
        "zh": "这不是搜索没做好。一个位形只有在关节扫掠期间负载保持不变时，"
              "才代表单一负载，而这要求停在力矩对该轴的极值点上——"
              "在极值点处，轴向分量与倾覆分量是彼此锁死的。"
              "仅仅筛选出满足该条件的位形，就使关节四上两者的相关性从 0.48 升到 0.90。"
              "让测量变干净的约束，正是造成混淆的约束。"
              "要真正拆开这些项，需要外部配重，"
              "或者故意扫过一段负载变化并利用每行已记录的瞬时负载。",
    },
    "caveat.cond": {"en": "condition", "zh": "条件数"},
    "caveat.corr": {"en": "correlation with the axial torque",
                    "zh": "与轴向力矩的相关性"},
    "caveat.radial": {"en": "radial force", "zh": "径向力"},
    "caveat.thrust": {"en": "thrust", "zh": "轴向推力"},
    "caveat.tilt": {"en": "tilting moment", "zh": "倾覆力矩"},
    "caveat.const": {"en": "constant", "zh": "恒定"},
    "caveat.verdict": {"en": "verdict", "zh": "判定"},
    "caveat.trust": {"en": "separable", "zh": "可分离"},
    "caveat.notrust": {"en": "confounded", "zh": "混淆"},

    "levels.head": {"en": "Every load level, and the posture that held it",
                    "zh": "全部负载等级及其对应位形"},
    "levels.say": {
        "en": "The axis-to-gravity angle is recorded because a joint carrying a "
              "heavy arm feels nothing if its own axis points along gravity. "
              "Without it a heavily laden posture and a merely well aligned one "
              "cannot be told apart.",
        "zh": "记录轴线与重力夹角，是因为一个关节即便承载着沉重的手臂，"
              "只要其自身轴线指向重力方向，它也感受不到任何负载。"
              "没有这一列，就无法区分“真正重载”与“恰好对齐”。",
    },
    "levels.level": {"en": "level", "zh": "等级"},
    "levels.axial": {"en": "axial N·m", "zh": "轴向 N·m"},
    "levels.radial": {"en": "radial N", "zh": "径向 N"},
    "levels.thrust": {"en": "thrust N", "zh": "推力 N"},
    "levels.tilt": {"en": "tilt N·m", "zh": "倾覆 N·m"},
    "levels.angle": {"en": "axis to gravity °", "zh": "轴线-重力夹角 °"},
    "levels.drift": {"en": "load drift N·m", "zh": "负载漂移 N·m"},
    "levels.pose": {"en": "posture °", "zh": "位形 °"},
    "levels.none": {"en": "no load axis: gravity cannot move this joint's load",
                    "zh": "无负载轴：重力无法改变该关节的负载"},
}


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
:root{--bg:#12151a;--card:#191d24;--line:#2a3038;--ink:#e8ecf2;--muted:#8b96a5;
--ok:#4ec98a;--warn:#e0b341;--bad:#e2565a;--accent:#4da3ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.6 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:16px;
padding:14px 28px;background:#12151aee;border-bottom:1px solid var(--line);
backdrop-filter:blur(6px)}
h1{font-size:18px;margin:0;font-weight:600}
h2{font-size:16px;margin:0 0 6px;font-weight:600}
main{max-width:1120px;margin:0 auto;padding:24px 28px 80px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:0 0 18px}
.say{color:var(--muted);margin:0 0 14px;max-width:78ch}
.spacer{flex:1}
button{background:#222833;color:var(--ink);border:1px solid var(--line);
border-radius:7px;padding:6px 14px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--accent)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
td.n{text-align:right}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px}
.pass{background:#1d3b2c;color:var(--ok)}
.warn{background:#3d3520;color:var(--warn)}
.fail{background:#3d2224;color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
gap:12px}
.kv{background:#1e232b;border-radius:8px;padding:10px 14px}
.kv .k{color:var(--muted);font-size:12px}
.kv .v{font-size:16px;font-variant-numeric:tabular-nums}
canvas{width:100%;background:#151920;border-radius:8px;border:1px solid var(--line)}
select{background:#222833;color:var(--ink);border:1px solid var(--line);
border-radius:6px;padding:5px 10px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.law{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
background:#1e232b;border-radius:8px;padding:12px 14px;margin:10px 0 0;
white-space:pre-wrap;line-height:1.7}
.note{color:var(--warn);font-size:12.5px}
.legend{display:flex;gap:8px;align-items:center;color:var(--muted);
font-size:12px;margin:8px 0 0;flex-wrap:wrap}
.swatch{display:inline-block;width:14px;height:10px;border-radius:2px}
</style>
</head>
<body>
<header>
  <h1 data-i18n="title"></h1>
  <span class="spacer"></span>
  <button id="lang" data-i18n="lang"></button>
</header>
<main id="root"></main>

<script id="strings" type="application/json">__STRINGS__</script>
<script id="data" type="application/json">__DATA__</script>
<script>
const TEXT = JSON.parse(document.getElementById('strings').textContent);
const D = JSON.parse(document.getElementById('data').textContent);
const JOINTS = D.joints || [];
let lang = (navigator.language || 'en').toLowerCase().startsWith('zh') ? 'zh' : 'en';
let picked = (JOINTS.find((j) => j.measured_curves) || JOINTS[0] || {}).joint || 0;

const t = (k) => (TEXT[k] ? (TEXT[k][lang] ?? TEXT[k].en) : k);
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = (v, d) => (v === null || v === undefined || Number.isNaN(v))
  ? '\\u2014' : Number(v).toFixed(d === undefined ? 3 : d);

/* Dark to bright with the load, so a glance at the chart says which curve is
 * carrying what without reading the legend. */
function shade(fraction) {
  const a = [40, 70, 120], b = [120, 220, 255];
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * fraction));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function head() {
  const temp = D.temperature || {};
  const cells = [
    [t('run.records'), (D.records || 0).toLocaleString()],
    [t('run.passes'), `${D.measured}/${D.designed}`],
    [t('run.pairs'), D.pairs],
    [t('run.temperature'), `${num(temp.low, 1)}\\u2013${num(temp.high, 1)} \\u00b0C`],
    [t('run.created'), esc(D.created || '')],
    [t('run.speeds'), `${(D.speeds_deg_s || []).length} \\u00d7 `
      + `${num((D.speeds_deg_s || [])[0], 2)}\\u2013`
      + `${num((D.speeds_deg_s || []).slice(-1)[0], 0)} \\u00b0/s`],
  ];
  return `<section><h2>${t('run.head')}</h2><p class="say">${t('subtitle')}</p>
    <div class="grid">${cells.map(([k, v]) =>
      `<div class="kv"><div class="k">${k}</div><div class="v">${v}</div></div>`
    ).join('')}</div></section>`;
}

function design() {
  const rows = JOINTS.map((j) => `<tr>
    <td>${esc(j.name)}</td>
    <td class="n">${j.levels.length}</td>
    <td class="n">${num(j.span_nm, 3)}</td>
    <td class="n">${num(j.reachable_nm[0], 2)}\\u2013${num(j.reachable_nm[1], 2)}</td>
    <td class="n">${j.measured}/${j.designed}</td>
    <td class="n">${num(j.arc_room_deg, 0)}</td>
    <td class="n">${j.refused}</td>
    <td class="n">${j.amps_per_nm === null ? num(j.amps_per_nm_used, 4) + '*' : num(j.amps_per_nm, 4)}</td>
    <td class="n">${j.amps_per_nm_fit === null ? '\\u2014' : num(j.amps_per_nm_fit, 5)}</td>
    </tr>${j.note ? `<tr><td colspan="9" class="note">${esc(j.note)}</td></tr>` : ''}`
  ).join('');
  return `<section><h2>${t('design.head')}</h2>
    <p class="say">${t('design.say')}</p>
    <table><thead><tr>
      <th>${t('design.joint')}</th><th>${t('design.levels')}</th>
      <th>${t('design.span')}</th><th>${t('design.reach')}</th>
      <th>${t('design.measured')}</th><th>${t('design.arc')}</th>
      <th>${t('design.refused')}</th><th>${t('design.kt')}</th>
      <th>${t('design.ktfit')}</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <p class="say" style="margin:12px 0 0">${t('design.ktsay')}</p></section>`;
}

function lawName(kind) {
  return kind === 'full' ? t('law.full')
    : kind === 'coulomb' ? t('law.coulombonly') : t('law.none');
}

function law() {
  const rows = JOINTS.filter((j) => j.fit).map((j) => {
    const f = j.fit;
    return `<tr><td>${esc(j.name)}</td>
      <td class="n">${num(f.c0, 4)}</td><td class="n">${num(f.ck, 4)}</td>
      <td class="n">${num(f.d0, 4)}</td><td class="n">${num(f.dk, 4)}</td>
      <td class="n">${num(f.width, 3)}</td><td class="n">${num(f.viscous, 5)}</td>
      <td class="n">${num(f.rms, 4)}</td>
      <td class="n">${f.share_of_mean === null ? '\\u2014' : num(f.share_of_mean, 1) + '%'}</td>
      <td>${lawName(f.load_terms)}</td></tr>`;
  }).join('');
  const written = JOINTS.filter((j) => j.fit).map((j) => {
    const f = j.fit;
    const vs = f.vsk > 1e-8
      ? `(${num(f.vs0, 3)} + ${num(f.vsk, 3)} L)` : num(f.vs0, 3);
    return `${esc(j.name)}
  f(v,L) = (${num(f.c0, 4)} + ${num(f.ck, 4)} L) tanh(v / ${num(f.width, 3)})`
      + `\\n         + (${num(f.d0, 4)} + ${num(f.dk, 4)} L) exp(-|v| / ${vs}) sign(v)`
      + `\\n         + ${num(f.viscous, 5)} v`;
  }).join('\\n\\n');
  return `<section><h2>${t('law.head')}</h2><p class="say">${t('law.say')}</p>
    <table><thead><tr>
      <th>${t('design.joint')}</th><th>${t('law.coulomb')}</th>
      <th>${t('law.pernm')}</th><th>${t('law.dip')}</th><th>${t('law.pernm')}</th>
      <th>${t('law.width')}</th><th>${t('law.viscous')}</th>
      <th>${t('law.rms')}</th><th>${t('law.share')}</th><th>${t('law.terms')}</th>
    </tr></thead><tbody>${rows}</tbody></table>
    <div class="law">${esc(written)}</div>
    <p class="say" style="margin:10px 0 0">v in \\u00b0/s, L in N\\u00b7m,
      f in amperes.</p></section>`;
}

function charts() {
  const options = JOINTS.filter((j) => j.measured_curves).map((j) =>
    `<option value="${j.joint}"${j.joint === picked ? ' selected' : ''}>`
    + `${esc(j.name)}</option>`).join('');
  return `<section><h2>${t('curve.head')}</h2>
    <p class="say">${t('curve.say')}</p>
    <p><label>${t('curve.pick')} <select id="pick">${options}</select></label></p>
    <canvas id="chart" width="1040" height="440"></canvas>
    <div class="legend" id="legend"></div></section>`;
}

function draw() {
  const joint = JOINTS.find((j) => j.joint === picked);
  const canvas = document.getElementById('chart');
  if (!joint || !canvas || !joint.measured_curves) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const L = 62, R = 16, T = 16, B = 44;
  ctx.clearRect(0, 0, W, H);

  const speeds = joint.speeds;
  const all = joint.measured_curves.flat().concat(joint.fitted_curves.flat());
  const yLo = Math.min(0, Math.min(...all)), yHi = Math.max(...all) * 1.05;
  const xLo = Math.log10(speeds[0]), xHi = Math.log10(speeds[speeds.length - 1]);
  const X = (v) => L + (Math.log10(v) - xLo) / (xHi - xLo) * (W - L - R);
  const Y = (v) => H - B - (v - yLo) / (yHi - yLo) * (H - T - B);

  ctx.strokeStyle = '#2a3038'; ctx.fillStyle = '#8b96a5';
  ctx.font = '11px system-ui'; ctx.lineWidth = 1;
  for (let k = 0; k <= 5; k++) {
    const v = yLo + (yHi - yLo) * k / 5, y = Y(v);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(v.toFixed(2), L - 8, y + 4);
  }
  ctx.textAlign = 'center';
  for (const v of [0.5, 1, 2, 5, 10, 20, 60]) {
    if (v < speeds[0] || v > speeds[speeds.length - 1]) continue;
    const x = X(v);
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke();
    ctx.fillText(String(v), x, H - B + 18);
  }
  ctx.fillText(t('curve.speed'), (L + W - R) / 2, H - 8);
  ctx.save(); ctx.translate(14, (T + H - B) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(t('curve.friction'), 0, 0); ctx.restore();

  const loads = joint.loads;
  const lo = Math.min(...loads), hi = Math.max(...loads);
  loads.forEach((load, i) => {
    const colour = shade(hi > lo ? (load - lo) / (hi - lo) : 1);
    ctx.strokeStyle = '#0b0d11'; ctx.lineWidth = 3.4;
    ctx.beginPath();
    joint.fitted_curves[i].forEach((v, k) =>
      k ? ctx.lineTo(X(speeds[k]), Y(v)) : ctx.moveTo(X(speeds[k]), Y(v)));
    ctx.stroke();
    ctx.strokeStyle = colour; ctx.lineWidth = 1.6;
    ctx.beginPath();
    joint.fitted_curves[i].forEach((v, k) =>
      k ? ctx.lineTo(X(speeds[k]), Y(v)) : ctx.moveTo(X(speeds[k]), Y(v)));
    ctx.stroke();
    ctx.fillStyle = colour;
    joint.measured_curves[i].forEach((v, k) => {
      ctx.beginPath(); ctx.arc(X(speeds[k]), Y(v), 2.6, 0, 7); ctx.fill();
    });
  });
  document.getElementById('legend').innerHTML = loads.map((load, i) =>
    `<span><i class="swatch" style="background:${shade(
      hi > lo ? (load - lo) / (hi - lo) : 1)}"></i> ${num(load, 2)} N\\u00b7m</span>`
  ).join('');
}

function worth() {
  const rows = JOINTS.filter((j) => j.worth).map((j) => j.worth.map((w, i) =>
    `<tr><td>${i ? '' : esc(j.name)}</td>
     <td class="n">${num(w.friction_nm, 3)}</td>
     <td class="n">${num(w.residual_nm, 4)}</td>
     <td class="n">${w.left === null ? '\\u2014' : num(w.left, 1) + '%'}</td>
     <td class="n">${w.gain === null ? '\\u2014' : num(w.gain, 1) + '\\u00d7'}</td>
     <td>${w.which === 'light' ? t('worth.light') : t('worth.heavy')},
         ${num(w.load_nm, 2)} N\\u00b7m</td></tr>`).join('')).join('');
  return `<section><h2>${t('worth.head')}</h2><p class="say">${t('worth.say')}</p>
    <table><thead><tr><th>${t('design.joint')}</th>
      <th>${t('worth.friction')}</th><th>${t('worth.residual')}</th>
      <th>${t('worth.left')}</th><th>${t('worth.gain')}</th>
      <th>${t('worth.at')}</th></tr></thead>
      <tbody>${rows}</tbody></table></section>`;
}

function rise() {
  const rows = JOINTS.map((j) => {
    if (j.per_nm === undefined || j.per_nm === null) {
      return `<tr><td>${esc(j.name)}</td><td class="n">\\u2014</td>
        <td class="n">\\u2014</td><td class="n">\\u2014</td>
        <td>${t('rise.novary')}</td></tr>`;
    }
    const two = j.fit.load_terms === 'coulomb';
    const span = j.loads[j.loads.length - 1] - j.loads[0];
    const kt = j.amps_per_nm_used || 0.4186;
    return `<tr><td>${esc(j.name)}</td>
      <td class="n">${num(j.per_nm, 4)}</td>
      <td class="n">${num(j.share_of_load, 1)}%</td>
      <td class="n">${num(j.per_nm * span / kt, 3)} N\\u00b7m</td>
      <td>${two ? t('rise.two') : j.levels.length + ' ' + t('rise.levels')}</td>
      </tr>`;
  }).join('');
  return `<section><h2>${t('rise.head')}</h2><p class="say">${t('rise.say')}</p>
    <table><thead><tr><th>${t('design.joint')}</th><th>${t('rise.pernm')}</th>
      <th>${t('rise.share')}</th><th>${t('rise.span')}</th>
      <th>${t('rise.pinned')}</th></tr></thead>
      <tbody>${rows}</tbody></table></section>`;
}

function caveat() {
  const cell = (v) => v === null || v === undefined
    ? `<td class="n">${t('caveat.const')}</td>` : `<td class="n">${num(v, 3)}</td>`;
  const rows = JOINTS.filter((j) => j.regression).map((j) => {
    const g = j.regression;
    const ok = g.trustworthy;
    return `<tr><td>${esc(j.name)}</td>
      <td class="n">${g.levels}</td>
      ${cell(g.correlation.radial_n)}${cell(g.correlation.thrust_n)}
      ${cell(g.correlation.tilt_nm)}
      <td class="n">${num(g.condition, 1)}</td>
      <td><span class="pill ${ok ? 'pass' : 'fail'}">${
        ok ? t('caveat.trust') : t('caveat.notrust')}</span></td></tr>`;
  }).join('');
  return `<section><h2>${t('caveat.head')}</h2>
    <p class="say">${t('caveat.say')}</p>
    <table><thead><tr><th>${t('design.joint')}</th><th>${t('rise.levels')}</th>
      <th>${t('caveat.radial')}</th><th>${t('caveat.thrust')}</th>
      <th>${t('caveat.tilt')}</th><th>${t('caveat.cond')}</th>
      <th>${t('caveat.verdict')}</th></tr></thead>
      <tbody>${rows}</tbody></table>
    <p class="say" style="margin:14px 0 0">${t('caveat.why')}</p></section>`;
}

function levels() {
  const blocks = JOINTS.map((j) => {
    if (!j.levels.length) return '';
    const rows = j.levels.map((v) => `<tr>
      <td class="n">${v.index}</td>
      <td class="n">${num(v.terms.axial_nm, 3)}</td>
      <td class="n">${num(v.terms.radial_n, 2)}</td>
      <td class="n">${num(v.terms.thrust_n, 2)}</td>
      <td class="n">${num(v.terms.tilt_nm, 3)}</td>
      <td class="n">${num(v.axis_gravity_deg, 1)}</td>
      <td class="n">${num(v.load_drift_nm, 4)}</td>
      <td class="mono">${v.pose_deg.map((p) => p.toFixed(0)).join(', ')}</td>
      </tr>`).join('');
    return `<h2 style="margin-top:18px">${esc(j.name)}</h2>
      ${j.loadable ? '' : `<p class="note">${t('levels.none')}</p>`}
      <table><thead><tr><th>${t('levels.level')}</th>
        <th>${t('levels.axial')}</th><th>${t('levels.radial')}</th>
        <th>${t('levels.thrust')}</th><th>${t('levels.tilt')}</th>
        <th>${t('levels.angle')}</th><th>${t('levels.drift')}</th>
        <th>${t('levels.pose')}</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
  }).join('');
  return `<section><h2>${t('levels.head')}</h2>
    <p class="say">${t('levels.say')}</p>${blocks}</section>`;
}

function paint() {
  document.documentElement.lang = lang;
  document.querySelectorAll('[data-i18n]').forEach(
    (n) => { n.textContent = t(n.dataset.i18n); });
  document.getElementById('root').innerHTML =
    head() + design() + law() + charts() + worth() + rise() + caveat() + levels();
  const pick = document.getElementById('pick');
  if (pick) pick.addEventListener('change', (e) => {
    picked = Number(e.target.value); draw();
  });
  draw();
}

document.getElementById('lang').addEventListener('click', () => {
  lang = lang === 'en' ? 'zh' : 'en';
  paint();
});
paint();
</script>
</body>
</html>
"""
