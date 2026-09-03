"""Write one campaign to a folder: raw data, the fitted result, and a report.

A single JSON file was enough to reload a run but not to understand one. What
goes out now is a directory: every observation as CSV, the machine-readable
result beside it, and a self-contained HTML page that reads the two back to a
person. The page carries its own data and its own script, so it survives being
copied off the robot, emailed, or opened with no network.
"""

from __future__ import annotations

from pathlib import Path
import csv
import html
import json
import time

RESULT_NAME = "result.json"
OBSERVATIONS_NAME = "observations.csv"
RAW_FRAMES_NAME = "raw_frames.csv"
REPORT_NAME = "report.html"
GRAVITY_MODEL_NAME = "gravity_model.json"
MODEL_URDF_NAME = "robot_description.urdf"

# Column order per joint in the CSV. Kept explicit so the header and the rows
# cannot drift apart.
JOINT_COLUMNS = ("position_deg", "velocity_deg_s", "acceleration_deg_s2",
                 "effort", "temperature_c")
# Every frame behind a fitted observation, at the publisher's full rate.
RAW_CHANNELS = (
  ("position_deg", "position_deg", 5),
  ("velocity_deg_s", "speed_deg_s", 5),
  ("effort", "current_a", 6),
  ("temperature_c", "temperature_c", 2),
)
OPTIONAL_RAW_CHANNELS = (
  ("voltage_v", "voltage_v", 3),
  ("enabled", "enabled", None),
  ("fault_code", "fault_code", None),
  ("drive_current_a", "drive_current_a", 6),
  ("joint_torque_nm", "joint_torque_nm", 6),
)


def write_run(directory, payload: dict, observations=None,
              stamp: str | None = None, raw_frames=None,
              model_urdf: str = "") -> Path:
    """Create one folder for this run and fill it. Returns the folder."""
    mode = str(payload.get("mode") or "run")
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    folder = Path(directory) / f"{mode}-{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / RESULT_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if payload.get("gravity_model"):
      (folder / GRAVITY_MODEL_NAME).write_text(
        json.dumps(payload["gravity_model"], indent=2,
               ensure_ascii=False), encoding="utf-8")
      if model_urdf:
        (folder / MODEL_URDF_NAME).write_text(
          str(model_urdf), encoding="utf-8")
    names = joint_names(payload)
    rows = write_observations(folder / OBSERVATIONS_NAME, observations or [],
                              names)
    frames = write_raw_frames(folder / RAW_FRAMES_NAME, raw_frames or [], names)
    (folder / REPORT_NAME).write_text(
        render_report(payload, observation_rows=rows, stamp=stamp,
                      raw_rows=frames),
        encoding="utf-8")
    return folder


def joint_names(payload: dict) -> list[str]:
    """URDF joint names, falling back to positional labels."""
    names = [str(name) for name in (payload.get("joint_names") or [])]
    count = len(payload.get("joints") or [])
    if len(names) >= count:
        return names[:count]
    return names + [f"joint{index + 1}" for index in range(len(names), count)]


def write_observations(path, observations, names: list[str]) -> int:
    """Every sample the campaign kept, one row each. Returns the row count."""
    header = ["phase", "time_s", "motion", "window_frames",
              "window_fit_rms_deg"]
    for name in names:
        header += [f"{name}.{column}" for column in JOINT_COLUMNS]
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for record in observations:
            row = [getattr(record, "phase", ""),
                   _round(getattr(record, "time_s", 0.0), 4),
                   getattr(record, "motion", ""),
                   getattr(record, "window_frames", 0),
                   _round(getattr(record, "window_fit_rms_deg", 0.0), 6)]
            for index in range(len(names)):
                row += [_round(_at(record, "position_deg", index), 5),
                        _round(_at(record, "velocity_deg_s", index), 5),
                        _round(_at(record, "acceleration_deg_s2", index), 5),
                        _round(_at(record, "current_a", index), 6),
                        _round(_at(record, "temperature_c", index), 2)]
            writer.writerow(row)
            written += 1
    return written


def write_raw_frames(path, frames, names: list[str]) -> int:
    """Every frame the plant saw, at the rate the driver published it.

    The fit sees the fitted windows, not these. They are written so the
    collapse from a burst of frames to one sample can be checked rather than
    taken on trust.
    """
    frames = list(frames)
    channels = list(RAW_CHANNELS)
    channels.extend(
      channel for channel in OPTIONAL_RAW_CHANNELS
      if any(frame.get(channel[1]) for frame in frames))
    header = ["phase", "motion", "frame", "stamp_s"]
    for name in names:
      header += [f"{name}.{column}" for column, _key, _digits in channels]
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame in frames:
            row = [frame.get("phase", ""), frame.get("motion", ""),
                   frame.get("frame", ""), _round(frame.get("stamp_s"), 6)]
            for index in range(len(names)):
              for _column, key, digits in channels:
                value = _item(frame, key, index)
                row.append(
                  int(value) if digits is None and value is not None
                  else _round(value, digits or 0))
            writer.writerow(row)
            written += 1
    return written


def _item(frame: dict, key: str, index: int):
    values = frame.get(key)
    try:
        return values[index]
    except (TypeError, IndexError, KeyError):
        return None


def _at(record, field: str, index: int):
    values = getattr(record, field, None)
    try:
        return values[index]
    except (TypeError, IndexError, KeyError):
        return None


def _round(value, digits: int):
    if value is None:
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def render_report(payload: dict, observation_rows: int = 0,
                  stamp: str | None = None, raw_rows: int = 0) -> str:
    """A standalone HTML page. No network, no build step, both languages."""
    document = {
        "payload": payload,
        "names": joint_names(payload),
        "rows": observation_rows,
        "rawRows": raw_rows,
        "stamp": stamp or time.strftime("%Y%m%d-%H%M%S"),
        "files": {"result": RESULT_NAME, "observations": OBSERVATIONS_NAME,
                  "raw": RAW_FRAMES_NAME,
                  "gravityModel": GRAVITY_MODEL_NAME,
                  "modelUrdf": MODEL_URDF_NAME},
    }
    blob = json.dumps(document, ensure_ascii=False, default=_plain)
    # A literal </script> inside the data would end the block early.
    blob = blob.replace("</", "<\\/")
    strings = json.dumps(TEXT, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"{payload.get('mode', 'run')} {document['stamp']}")
    return _TEMPLATE.replace("__TITLE__", title) \
                    .replace("__STRINGS__", strings) \
                    .replace("__DATA__", blob)


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return str(value)


# Both languages live together so a term cannot be updated in one and left
# stale in the other.
TEXT = {
    "title": {"en": "Identification report", "zh": "参数辨识报告"},
  "title.gravity": {"en": "Gravity effort model report",
            "zh": "重力驱动量模型报告"},
  "title.gravity_rehearsal": {"en": "Analytic gravity rehearsal report",
                "zh": "解析重力预演报告"},
    "subtitle": {
        "en": "What was measured, what was fitted, and how far to trust it.",
        "zh": "本次测量了什么、拟合出了什么，以及结果可信到什么程度。",
    },
  "subtitle.gravity": {
    "en": "Hardware provenance, the pair-averaged gravity predictor, "
        "independent validation, and the boundary of what may be deployed.",
    "zh": "真机数据溯源、成对平均重力预测器、独立验证，以及可部署范围的明确边界。",
  },
    "lang": {"en": "中文", "zh": "English"},

    "verdict.head": {"en": "Verdict", "zh": "总体判定"},
    "verdict.pass": {"en": "pass", "zh": "通过"},
    "verdict.warn": {"en": "warn", "zh": "警告"},
    "verdict.fail": {"en": "fail", "zh": "失败"},
    "verdict.unknown": {"en": "unknown", "zh": "未知"},
    "verdict.pass.say": {
        "en": "Every joint met its limits. The parameters may be used.",
        "zh": "所有关节均满足判定标准，辨识参数可以使用。",
    },
    "verdict.gravity.pass.say": {
      "en": "Independent validation passed for this empirical effort-domain "
          "gravity predictor. This does not validate physical link masses "
          "or install the model in a runtime controller.",
      "zh": "该电流/驱动量域经验重力预测器已通过独立位姿验证。"
          "这不代表连杆质量与质心已被物理辨识，也不代表模型已装入运行时控制器。",
    },
    "verdict.gravity_rehearsal.pass.say": {
      "en": "The analytic rehearsal recovered its planted model. It validates "
          "the calculation path, not hardware measurements.",
      "zh": "解析预演已复现预设模型；它验证的是计算链路，不是真机测量。",
    },
    "verdict.warn.say": {
        "en": "The run completed, but at least one joint is weakly determined. "
              "Read the per-joint table before using those values.",
        "zh": "运行已完成，但至少有一个关节的参数确定性较弱。使用前请先查看逐关节表格。",
    },
    "verdict.fail.say": {
        "en": "At least one joint failed, or the run was cut short. These "
              "parameters should not be used as they stand.",
        "zh": "至少有一个关节未通过，或运行被中断。当前参数不应直接使用。",
    },
    "verdict.unknown.say": {
        "en": "No verdict was recorded, so nothing here has been checked.",
        "zh": "本次运行没有记录判定结果，因此以下内容均未经检验。",
    },
    "verdict.aborted": {"en": "Run stopped early", "zh": "运行被提前中止"},
    "verdict.skipped": {"en": "Motions the arm refused, and the run went on "
                              "without them",
                        "zh": "机械臂拒绝执行、运行跳过后继续的动作"},

    "summary.head": {"en": "This run", "zh": "本次运行"},
    "summary.mode": {"en": "mode", "zh": "运行方式"},
    "summary.mode.hardware": {"en": "hardware", "zh": "真机"},
    "summary.mode.rehearsal": {"en": "rehearsal", "zh": "预演"},
    "summary.mode.gravity": {"en": "gravity hardware calibration",
                 "zh": "重力真机标定"},
    "summary.mode.gravity_rehearsal": {"en": "analytic gravity rehearsal",
                       "zh": "解析重力预演"},
    "summary.mode.optimal_excitation": {
      "en": "optimal excitation", "zh": "最优激励轨迹辨识"},
    "summary.when": {"en": "recorded", "zh": "记录时间"},
    "summary.joints": {"en": "joints", "zh": "关节数"},
    "summary.quantity": {"en": "identified quantity", "zh": "辨识量"},
    "summary.action": {"en": "trajectory action", "zh": "轨迹动作服务"},
    "summary.samples": {"en": "fitted samples", "zh": "拟合样本数"},
    "summary.raw": {"en": "raw frames behind them", "zh": "其背后的原始帧数"},
    "summary.validation": {"en": "validation samples", "zh": "验证样本数"},
    "quantity.current": {"en": "motor current", "zh": "电机电流"},
    "quantity.torque": {"en": "joint torque", "zh": "关节扭矩"},

        "provenance.head": {"en": "Data source and provenance", "zh": "数据来源与溯源"},
        "provenance.hardware": {"en": "real hardware", "zh": "真机采集"},
        "provenance.rehearsal": {"en": "analytic rehearsal", "zh": "解析预演"},
        "provenance.other": {"en": "offline or simulated", "zh": "离线或仿真"},
        "provenance.hardware.say": {
        "en": "This result contains timestamped raw frames published while the "
          "real trajectory controller moved the arm.",
        "zh": "本结果包含真机轨迹控制器驱动机械臂期间发布的带时间戳原始帧。",
        },
        "provenance.rehearsal.say": {
        "en": "This result came from the analytic plant. Zero raw hardware "
          "frames is expected and must not be read as a real acquisition.",
        "zh": "本结果来自解析被控对象。原始真机帧为零是预期行为，不能当作真机采集。",
        },
        "provenance.other.say": {
        "en": "The artifact does not establish real-hardware provenance.",
        "zh": "该产物不能证明来自真机采集。",
        },
        "provenance.source": {"en": "source", "zh": "来源"},
        "provenance.raw": {"en": "raw frames", "zh": "原始帧"},
        "provenance.observations": {"en": "fitted observations", "zh": "拟合观测"},
        "provenance.duration": {"en": "publisher duration", "zh": "发布时长"},
        "provenance.rate": {"en": "approximate raw rate", "zh": "原始帧约计频率"},
        "provenance.topic": {"en": "telemetry topic", "zh": "遥测话题"},
        "provenance.groups": {"en": "raw motion groups", "zh": "原始运动组"},
        "provenance.phasecheck": {"en": "raw phase/tag mismatches",
                      "zh": "原始 phase/tag 不一致"},
        "provenance.software": {"en": "software", "zh": "软件环境"},
        "provenance.profile": {"en": "profile source", "zh": "Profile 来源"},
        "provenance.margin": {"en": "collision safety margin", "zh": "碰撞安全余量"},

        "gravity.head": {"en": "Gravity compensation model", "zh": "重力补偿模型"},
        "gravity.say": {
        "en": "At each pose, positive and negative crossings are averaged at "
          "every probe speed before fitting. Odd friction cancels before "
          "the static Pinocchio regressor plus offset is solved.",
        "zh": "每个位姿先在每档探针速度上平均正反向穿越，再进行拟合。"
          "奇对称摩擦在进入静态 Pinocchio 回归量加偏置求解之前已经抵消。",
        },
        "gravity.empirical": {"en": "empirical effort predictor",
              "zh": "经验驱动量预测器"},
        "gravity.physical.missing": {
        "en": "Physical link mass, center of mass, and rotational inertia are "
          "not available. Current-domain fits are independent per output "
          "joint and are not one shared SI rigid-body model.",
        "zh": "没有得到物理连杆质量、质心与转动惯量。电流域拟合按输出关节独立进行，"
          "并不是一套共享的 SI 刚体模型。",
        },
        "gravity.runtime.missing": {
        "en": "Runtime controller loader: not integrated. The normal impedance "
          "controller still computes gravity from the URDF.",
        "zh": "运行时控制器加载器：尚未集成。常规阻抗控制器仍从 URDF 计算重力。",
        },
        "gravity.pairs.ok": {"en": "all direction/speed pairs complete",
             "zh": "全部方向/速度配对完整"},
        "gravity.pairs.bad": {"en": "incomplete direction/speed pairs",
              "zh": "存在不完整方向/速度配对"},
        "gravity.model": {"en": "model type", "zh": "模型类型"},
        "gravity.hash": {"en": "URDF SHA-256", "zh": "URDF 哈希（SHA-256）"},
        "gravity.trainposes": {"en": "training poses", "zh": "训练位姿"},
        "gravity.validposes": {"en": "independent validation poses",
               "zh": "独立验证位姿"},
        "gravity.mean": {"en": "mean validation RMS", "zh": "平均验证均方根"},
        "gravity.worst": {"en": "worst validation RMS", "zh": "最差验证均方根"},
        "gravity.columns": {"en": "named retained columns and coefficients",
            "zh": "保留列名称与系数"},
        "gravity.speedcheck": {"en": "speed-pair consistency RMS",
               "zh": "速度对一致性均方根"},

    "joints.head": {"en": "Per joint", "zh": "逐关节结果"},
    "joints.head.gravity": {"en": "Directional friction nuisance fit",
                "zh": "方向性摩擦干扰拟合"},
    "joints.say": {
        "en": "Coulomb friction is the constant part that opposes motion; "
          "viscous friction grows with speed. The load ratio multiplies "
          "the absolute rigid-body effort predicted in the same units, so "
          "it is dimensionless (A/A or N·m/N·m), not a direct payload "
          "measurement. Validation error is measured on motion the fit "
          "never saw, so it is the honest one.",
        "zh": "库仑摩擦是与运动方向相反的恒定部分，粘滞摩擦随速度增大。"
          "载荷比例乘以同单位的刚体驱动量绝对值，因此它是无量纲比例"
          "（A/A 或 N·m/N·m），并非直接测得的负载质量或力矩。"
          "验证误差在拟合从未见过的运动上测得，因此最能反映真实水平。",
    },
        "joints.say.gravity": {
        "en": "This separate fit explains directional probe differences so "
          "friction can be audited. It is not the gravity model above. "
          "Only two speed magnitudes were measured, so retained Stribeck "
          "or load terms are empirical in-domain features, not identified physics.",
        "zh": "这套独立拟合用于解释探针正反向差异，以便审计摩擦；它不是上方的重力模型。"
          "实验只测了两档速度，因此保留的 Stribeck 或载荷项只是域内经验特征，"
          "不能解释为已辨识的物理规律。",
        },
    "col.joint": {"en": "joint", "zh": "关节"},
    "col.coulomb": {"en": "Coulomb", "zh": "库仑摩擦"},
    "col.load": {"en": "load ratio", "zh": "载荷比例"},
    "col.kept": {"en": "extra columns kept", "zh": "保留的附加项"},
    "col.viscous": {"en": "viscous", "zh": "粘滞摩擦"},
    "col.offset": {"en": "offset", "zh": "偏置"},
    "col.train": {"en": "training RMS", "zh": "训练均方根"},
    "col.holdout": {"en": "holdout RMS", "zh": "留出均方根"},
    "col.valid": {"en": "validation RMS", "zh": "验证均方根"},
    "col.condition": {"en": "condition", "zh": "条件数"},
    "col.rank": {"en": "rank", "zh": "有效秩"},
    "col.samples": {"en": "samples", "zh": "样本数"},
    "col.state": {"en": "state", "zh": "判定"},
    "col.reason": {"en": "reason", "zh": "原因"},

    "formula.head": {"en": "Fitted friction formulas", "zh": "拟合摩擦公式"},
    "formula.head.gravity": {"en": "Directional friction diagnostic formulas",
                 "zh": "方向性摩擦诊断公式"},
    "formula.say": {
      "en": "Numeric formulas used by the predictor. Here v is joint speed "
          "in degrees per second, L = |I_rigid| is the absolute rigid-body "
          "effort in the identified unit, and I_f is friction effort. The "
          "friction chart excludes rigid-body gravity and inertia, so its "
          "peak is smaller than measured total effort.",
      "zh": "预测器实际使用的数值公式。其中 v 为关节速度（度/秒），"
          "L = |I_rigid| 为辨识单位下的刚体驱动量绝对值，I_f 为摩擦驱动量。"
          "摩擦图已扣除刚体重力与惯性，因此其峰值小于实测总驱动峰值。",
    },
        "formula.say.gravity": {
      "en": "These formulas belong to the nuisance fit, not the pair-averaged "
        "gravity predictor. They are shown to audit what was cancelled; "
        "do not extrapolate the two measured speeds into a physical friction law.",
      "zh": "这些公式属于干扰项拟合，不属于成对平均重力预测器。"
        "它们用于审计被抵消的成分；不要把两档实测速度外推为物理摩擦定律。",
        },
        "formula.stribeck.on": {"en": "Stribeck retained", "zh": "保留 Stribeck"},
        "formula.stribeck.off": {"en": "Stribeck rejected", "zh": "拒绝 Stribeck"},
        "formula.stribeck.peak": {"en": "local low-speed peak", "zh": "存在低速局部尖峰"},
        "formula.stribeck.nopeak": {"en": "no local peak", "zh": "无局部尖峰"},
        "formula.totalpeak": {"en": "measured total peak", "zh": "实测总驱动峰值"},
        "formula.frictionpeak": {"en": "plotted friction peak", "zh": "图中摩擦峰值"},

        "steady.head": {"en": "Controlled low-speed friction audit",
                        "zh": "受控低速摩擦审计"},
        "steady.say": {
        "en": "Phase B is grouped by the same load posture, commanded speed, "
          "direction and repeat after subtracting the frozen rigid-body model. "
          "These rows do not refit the dynamic predictor. A classical peak is "
          "reported only when friction at 0.5 deg/s or below exceeds the "
          "highest-speed rung by more than 0.02 A; visual peaks in the mixed "
          "trajectory scatter do not count.",
        "zh": "B 阶段在扣除冻结的刚体模型后，按相同负载位形、指令速度、方向和重复次数分组。"
          "这些数据不参与动态预测器的重新拟合。只有当 0.5 度/秒及以下的摩擦比最高速度档"
          "高出 0.02 A 以上时，才判定为经典低速尖峰；动态轨迹混合散点中的视觉尖点不计。",
        },
        "steady.levels": {"en": "load levels", "zh": "负载层数"},
        "steady.lowpeak": {"en": "low-speed peak minus top rung",
                           "zh": "低速峰减最高速度档"},
        "steady.interior": {"en": "largest interior hump",
                            "zh": "最大内部隆起"},
        "steady.peak": {"en": "classical peak", "zh": "经典尖峰"},
        "steady.nopeak": {"en": "no classical peak", "zh": "无经典尖峰"},

        "compare.head": {"en": "Optimal excitation vs load sweep",
             "zh": "最优激励与负载扫掠对比"},
        "compare.say": {
        "en": "Both models are scored on the same multi-joint Fourier "
          "validation trajectories. Neither model trained on these rows; "
          "a lower RMS is better.",
        "zh": "两个模型均在同一组多关节傅里叶验证轨迹上评分，双方都未使用这些数据训练；"
          "均方根误差越低越好。",
        },
        "compare.met": {"en": "target met", "zh": "目标达成"},
        "compare.missed": {"en": "target not met", "zh": "目标未达成"},
        "compare.unavailable": {"en": "Comparison unavailable",
                "zh": "无法进行对比"},
        "compare.mean": {"en": "mean RMS", "zh": "平均均方根"},
        "compare.worst": {"en": "worst RMS", "zh": "最差均方根"},
        "compare.optimal": {"en": "optimal excitation", "zh": "最优激励"},
        "compare.sweep": {"en": "load sweep", "zh": "负载扫掠"},
        "compare.improvement": {"en": "improvement", "zh": "改善幅度"},
        "compare.source": {"en": "sweep source", "zh": "扫掠来源"},

    "charts.head": {"en": "Per-joint diagnostics", "zh": "逐关节诊断图"},
    "charts.pick": {"en": "joint", "zh": "关节"},
    "charts.lock": {"en": "same scale for every joint",
                    "zh": "所有关节共用同一刻度"},
    "charts.lock.say": {
        "en": "Each chart is scaled to its own joint by default, so a small "
              "joint and a large one fill the frame identically and look the "
              "same. Tick this to put them all on one scale and compare them.",
        "zh": "默认每张图按各自关节的数据缩放，于是量级小的关节和量级大的关节"
              "同样填满画框，看上去完全一样。勾选此项可将所有关节置于同一刻度以便比较。",
    },
    "charts.friction": {"en": "Friction curve and its samples",
                        "zh": "摩擦曲线与样本"},
    "charts.friction.say": {
        "en": "The vertical axis is not raw current. Each point is the measured "
              "effort minus what the fitted rigid-body model predicts for that "
              "pose and motion, so gravity and inertia have already been taken "
              "out. What is left is the part friction has to account for.",
        "zh": "纵轴不是原始电流。每个点是实测驱动量减去刚体模型对该位姿和运动的预测，"
              "重力与惯性已被扣除，剩下的就是摩擦需要解释的部分。",
    },
    "charts.friction.green": {
        "en": "from phase B, and only where this joint was the one being "
              "swept. It moves back and forth about a single nominal pose "
              "across the whole speed range, so with the pose barely changing, "
              "a difference between two outlined points is a difference in speed. "
          "Sweep samples keep their load color and use a white outline.",
        "zh": "来自 B 摩擦阶段，且只包含本关节自己被扫掠的样本。此时它绕同一标称位姿"
            "往复运动、覆盖整个速度区间，位姿几乎不变，因此两个描边点之间的差异就是"
          "速度造成的差异。扫掠样本保留所属负载层颜色，并加白色描边。",
    },
    "charts.friction.blue": {
        "en": "all other training samples. In an optimal-excitation run these "
              "are the multi-joint Fourier trajectories. Each point uses the "
              "color of its nearest load-quantile curve, from blue at light "
              "load to red at heavy load.",
        "zh": "其余所有训练样本。在最优激励运行中，它们就是多关节傅里叶轨迹。"
              "每个点使用最近负载分位曲线的颜色，由轻载蓝色过渡到重载红色。",
    },
    "charts.friction.curve": {
        "en": "the fitted model: six colored curves at load quantiles from "
              "light to heavy. Points and curves with the same color share a "
              "load cluster; the outer curves bound the shaded 5–95% range.",
        "zh": "拟合模型按由轻到重的负载分位数画六条彩色曲线。同色点与曲线属于"
              "同一负载层，最外两条曲线界定阴影中的 5–95% 范围。",
    },
    "charts.friction.steady": {
        "en": "the controlled steady passes, drawn as circles joined by a "
              "dashed line at positive speed, one series per load level. "
              "Each circle is the median of a whole window held at one "
              "commanded speed with the pose fixed, so it is friction at a "
              "speed in a way no trajectory point is. A Stribeck peak would "
              "show here as a hump above the line that follows it.",
        "zh": "受控恒速段，在正速度一侧以圆点加虚线绘制，每个负载层一条。"
              "每个圆点是某一指令速度下整个窗口的中位数，位姿保持不变，"
              "因此它才是真正意义上“某一速度下的摩擦”，而轨迹点不是。"
              "若存在 Stribeck 尖峰，它会在此处表现为高出后续曲线的隆起。",
    },
    "charts.friction.still": {
        "en": "Rows where this joint was standing still (|v| below {v} °/s) "
              "are withheld from the scatter: {n} of them here. At rest "
              "friction has no determined sign, so such a row records only "
              "where the position servo settled inside the stiction band. "
              "Every joint but the swept one stands still through a sweep, "
              "so those rows are the majority, and drawn against speed they "
              "stack into a vertical band at zero that reads as a peak no "
              "speed curve can pass through.",
        "zh": "本关节处于静止（|v| 小于 {v} °/s）的样本已从散点中剔除，此处共 "
              "{n} 个。静止时摩擦没有确定符号，这类样本只记录了位置伺服停在"
              "静摩擦带内的哪个位置。扫掠某一关节时其余关节都静止，因此这类"
              "样本占多数；把它们按速度画出来，就会在零速处堆成一条竖直亮带，"
              "看起来像尖峰，而任何速度曲线都不可能穿过它。",
    },
    "charts.friction.warn": {
        "en": "When outlined sweep points are present, sweep and trajectory "
              "groups differ in pose as well as speed, so a step between them "
              "is not by itself a speed effect. Match points to same-color "
              "curves; use the residual chart below for point-by-point error.",
        "zh": "若存在描边扫掠点，扫掠组与轨迹组不仅速度不同，位姿也不同，因此两组"
              "落差本身不能证明速度效应。应将测量点与同色曲线比较，逐点模型误差"
              "则查看下方残差图。",
    },
    "charts.steadylow": {"en": "Low speed, measured against fitted",
                          "zh": "低速段：实测与拟合对比"},
    "charts.steadylow.say": {
        "en": "The chart above runs to tens of degrees per second, so the "
              "whole steady range is a few pixels wide in it. Here the axis "
              "stops at the fastest controlled pass. Circles joined by a "
              "solid line are the measured medians; the dashed line is the "
              "fitted curve evaluated at the same load. A Stribeck peak "
              "means friction falls as speed rises out of zero: it would "
              "appear as a circle high on the left with lower circles to its "
              "right. A line that only climbs has no such peak to fit.",
        "zh": "上方图的横轴直到每秒几十度，整个恒速区间在其中只占几个像素。"
              "此处横轴只到最快的受控恒速段。实线连接的圆点是实测中位数，"
              "虚线是同一负载下的拟合曲线。Stribeck 尖峰的含义是速度从零升高时"
              "摩擦反而下降：它会表现为左侧圆点偏高、右侧圆点更低。一条只升不降的"
              "折线里没有可供拟合的尖峰。",
    },
    "charts.residual": {"en": "Residual against speed", "zh": "残差-速度关系"},
    "charts.residual.say": {
        "en": "The vertical axis is what the whole model still gets wrong: "
              "measured effort minus the full prediction, rigid body and "
              "friction together. Zero would be a perfect prediction.",
        "zh": "纵轴是整个模型仍未预测对的部分：实测驱动量减去完整预测（刚体加摩擦）。"
              "零表示预测完全正确。",
    },
    "charts.residual.red": {
        "en": "all non-sweep training samples: the same group drawn in blue "
          "on the chart above.",
        "zh": "所有非扫掠训练样本，即上图中以蓝色绘制的同一组数据。",
    },
    "charts.residual.green": {
        "en": "phase B, and only where this joint was the one being swept: "
              "one joint, one pose, the full speed range.",
        "zh": "B 阶段中本关节自己被扫掠的样本：单个关节、单一位姿、覆盖整个速度区间。",
    },
    "charts.residual.read": {
        "en": "A shapeless band about zero is measurement noise, and is what a "
              "sound model looks like. A residual that tilts or curves with "
              "speed, or that sits to one side of zero, is physics the model "
              "does not contain rather than noise.",
        "zh": "围绕零线、没有形状的带状散布是测量噪声，模型健康时就应如此。"
              "若残差随速度倾斜或弯曲，或整体偏向零线一侧，那是模型未包含的物理效应，"
              "而非噪声。",
    },
    "charts.residual.caveat": {
        "en": "Validation samples are not drawn here. These are residuals on "
              "the data the model was fitted to, so they flatter it; the "
              "validation column in the table above is the honest measure.",
        "zh": "此处不含验证阶段的样本。这些是模型拟合所用数据上的残差，因而偏乐观；"
              "上方表格中的验证均方根才是诚实的衡量。",
    },
    "charts.error": {"en": "Error per joint", "zh": "各关节误差"},
    "charts.condition": {"en": "Condition number per joint", "zh": "各关节条件数"},
    "charts.condition.say": {
        "en": "How well the experiment separated one parameter from another. "
              "Tall bars mean the motion barely excited that joint's "
              "parameters, so its numbers carry little evidence.",
        "zh": "反映实验对各参数的区分能力。柱越高说明运动对该关节参数的激励越弱，"
              "其数值所依据的证据越少。",
    },
    "charts.speed": {"en": "speed (°/s)", "zh": "速度（°/秒）"},
    # The colour is named in the text as well as shown, so the legend still
    # works for a reader who cannot separate the two hues.
    "charts.sweep": {"en": "Outlined squares — sweep samples",
             "zh": "描边方点 —— 扫掠样本"},
    "charts.other": {"en": "Colors — training samples grouped by load",
             "zh": "彩色 —— 按负载分组的训练样本"},
    "charts.curve": {"en": "Lines — fitted curve cluster at the same loads",
             "zh": "曲线 —— 相同负载层的拟合曲线簇"},
    "charts.steady": {"en": "Dashed circles — measured steady curve",
             "zh": "虚线圆点 —— 实测恒速曲线"},
    "charts.red": {"en": "Red — other training samples",
             "zh": "红色 —— 其余训练样本"},
    "charts.green": {"en": "Green — sweep samples", "zh": "绿色 —— 扫掠样本"},

    "phases.head": {"en": "Phases", "zh": "各阶段"},
    "phases.say": {
        "en": "A gravity: hold still at many poses. B friction: sweep one joint "
              "at a time. C inertia: move everything on a smooth trajectory. "
              "D validation: fresh motion, used only for scoring.",
        "zh": "A 重力：在多个位姿静止保持。B 摩擦：逐个关节做往复扫掠。"
              "C 惯性：沿平滑轨迹整体运动。D 验证：全新运动，仅用于评分。",
    },
        "phases.gravity.say": {
        "en": "A gravity: bidirectional crossings at designed training poses. "
          "D validation: the same measurement at separately seeded poses "
          "that never enter the fit.",
        "zh": "A 重力：在设计的训练位姿做双向穿越。D 验证：在独立种子生成、"
          "从未进入拟合的位姿执行同样测量。",
        },
    "col.phase": {"en": "phase", "zh": "阶段"},
    "col.observations": {"en": "observations", "zh": "观测数"},
    "col.duration": {"en": "duration", "zh": "时长"},
    "col.peaktemp": {"en": "peak temperature", "zh": "最高温度"},
    "col.peakspeed": {"en": "peak speed", "zh": "最高速度"},
    "col.peakeffort": {"en": "peak effort", "zh": "最大驱动量"},
    "col.aborted": {"en": "stopped by", "zh": "中止原因"},
    "phase.A_gravity": {"en": "A gravity", "zh": "A 重力"},
    "phase.B_friction": {"en": "B friction", "zh": "B 摩擦"},
    "phase.C_inertia": {"en": "C inertia", "zh": "C 惯性"},
    "phase.D_validation": {"en": "D validation", "zh": "D 验证"},

    "rehearsal.head": {"en": "Rehearsal self-check", "zh": "预演自检"},
    "rehearsal.say": {
        "en": "A rehearsal plants known friction in a simulated arm and the fit "
              "must find it again. Completing proves the code runs; recovering "
              "the planted value proves it computes.",
        "zh": "预演会在仿真手臂中预设已知摩擦，拟合必须重新找回该值。"
              "跑完只能证明代码能运行，找回预设值才能证明计算正确。",
    },
    "rehearsal.ok": {"en": "recovered", "zh": "已复现"},
    "rehearsal.bad": {"en": "not recovered", "zh": "未能复现"},
    "col.expected": {"en": "planted", "zh": "预设值"},
    "col.recovered": {"en": "recovered", "zh": "复现值"},
    "col.error": {"en": "error", "zh": "误差"},
    "rehearsal.worst": {"en": "worst error", "zh": "最大误差"},
    "rehearsal.tolerance": {"en": "tolerance", "zh": "容差"},

    "plan.head": {"en": "Settings used", "zh": "所用设置"},
    "plan.say": {
        "en": "The plan as it ran, after any clamping to the robot's envelope.",
        "zh": "实际执行的方案参数，已按机器人安全包络做过限幅。",
    },

    "files.head": {"en": "Files in this folder", "zh": "本文件夹内容"},
    "files.result": {
        "en": "The fitted parameters and every summary number on this page, "
              "machine readable.",
        "zh": "拟合参数及本页所有汇总数值，机器可读格式。",
    },
        "files.result.gravity": {
        "en": "The self-describing pair-averaged gravity predictor, named "
          "regressor columns, coefficients, model hash, provenance, "
          "validation evidence, and explicit deployment limitations.",
        "zh": "自描述的成对平均重力预测器、命名回归列、系数、模型哈希、数据溯源、"
          "验证证据及明确的部署限制。",
        },
    "files.observations": {
        "en": "One row per fitted sample: the phase, the motion it came from, "
              "how many raw frames the window held and how well they fitted a "
              "straight motion, then per joint the position, speed, "
              "acceleration, effort and temperature. This is what the fit saw. "
              "Columns are named after the URDF joints.",
        "zh": "每个拟合样本一行：阶段、来源运动、窗口内的原始帧数及其拟合残差，"
              "随后是每个关节的位置、速度、加速度、驱动量和温度。这是拟合实际看到的数据。"
              "列名与 URDF 关节名对应。",
    },
    "files.gravity_model": {
      "en": "The gravity predictor alone: named columns, coefficients, model "
          "identity, validation, and deployment status for a future loader.",
      "zh": "独立重力预测器：命名列、系数、模型身份、验证与部署状态，供后续加载器使用。",
    },
    "files.model_urdf": {
      "en": "The exact robot description whose hash and Pinocchio regressor "
          "define this model.",
      "zh": "定义本模型哈希与 Pinocchio 回归量的精确机器人描述。",
    },
    "files.raw": {
        "en": "Every frame the driver published during each motion, at its full "
          "rate, including available voltage, enabled/fault state and "
          "alternate current/torque channels. The fit does not use these; "
          "they are here so the collapse from a burst into one sample and "
          "the active safety evidence can be checked.",
        "zh": "每次运动中驱动器发布的每一帧，按其原始速率记录。拟合不使用这些数据，"
          "并保留可用的电压、使能/故障状态及另一电流/力矩通道。提供它们是为了核查"
          "由一批帧塌缩成一个样本的过程，以及当时实际生效的安全证据。",
    },
              "files.sources": {
              "en": "Paths to the original steady and dynamic observation/raw-frame "
                "files. They are referenced instead of duplicated in this combined run.",
              "zh": "原始稳态与动态观测/原始帧文件的路径。组合结果引用这些文件，"
                "而不重复复制大体量原始数据。",
              },
    "files.report": {"en": "This page.", "zh": "本页面。"},

    "glossary.head": {"en": "Terms", "zh": "术语说明"},
    "g.rms": {
        "en": "RMS error — the typical size of the gap between measured and "
              "predicted. Same units as the identified quantity.",
        "zh": "均方根误差 —— 实测值与预测值之间差距的典型大小，单位与辨识量相同。",
    },
    "g.holdout": {
        "en": "Holdout — samples set aside from the fit and scored afterwards.",
        "zh": "留出集 —— 从拟合中预留、事后用于评分的样本。",
    },
    "g.condition": {
        "en": "Condition number — how much the answer moves when the data "
              "wobbles. Small is good; above a thousand, treat with care.",
        "zh": "条件数 —— 数据轻微扰动时结果的变动程度。越小越好；超过一千应谨慎对待。",
    },
    "g.rank": {
        "en": "Effective rank — how many parameters the data could actually "
              "pin down, as opposed to how many were asked for.",
        "zh": "有效秩 —— 数据实际能确定的参数个数，未必等于所要求的个数。",
    },
    "g.coulomb": {
        "en": "Coulomb friction — the part that does not depend on speed, only "
              "on direction of travel.",
        "zh": "库仑摩擦 —— 与速度大小无关、只与运动方向有关的摩擦分量。",
    },
    "g.viscous": {
        "en": "Viscous friction — the part proportional to speed.",
        "zh": "粘滞摩擦 —— 与速度成正比的摩擦分量。",
    },
    "g.load": {
      "en": "Load ratio — friction added per unit absolute rigid-body "
          "effort. It is dimensionless and uses the fitted current/torque "
          "model as a load proxy; it does not separately identify radial "
          "force, thrust and tilting moment.",
      "zh": "载荷比例 —— 每单位刚体驱动量绝对值增加的摩擦。它是无量纲量，"
          "以拟合出的电流/力矩模型作为载荷代理；不会分别辨识径向力、轴向推力和"
          "倾覆力矩。",
    },
    "g.unphysical": {
        "en": "A negative Coulomb or viscous term is unphysical: friction "
              "cannot drive a joint. It means the fit absorbed some other "
              "error into that term.",
        "zh": "库仑或粘滞项为负是非物理的：摩擦不可能驱动关节，"
              "这说明拟合把其他误差吸收进了该项。",
    },
    "none": {"en": "none", "zh": "无"},
    "unit.deg": {"en": "°", "zh": "°"},
}


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<style>
:root{--bg:#12151a;--card:#191d24;--line:#2a3038;--ink:#e8ecf2;--muted:#8b96a5;
--ok:#4ec98a;--warn:#e0b341;--bad:#e2565a;--accent:#4da3ff;--sweep:#78dc96}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.6 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:16px;
padding:14px 28px;background:#12151aee;border-bottom:1px solid var(--line);
backdrop-filter:blur(6px)}
h1{font-size:18px;margin:0;font-weight:600}
h2{font-size:16px;margin:0 0 6px;font-weight:600}
main{max-width:1080px;margin:0 auto;padding:24px 28px 80px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:0 0 18px;overflow-x:auto}
.say{color:var(--muted);margin:0 0 14px;max-width:75ch}
.spacer{flex:1}
button{background:#222833;color:var(--ink);border:1px solid var(--line);
border-radius:7px;padding:6px 14px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--accent)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
td.n{text-align:right}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:12px}
.pass{background:#1d3b2c;color:var(--ok)}
.warn{background:#3d3520;color:var(--warn)}
.fail{background:#3d2224;color:var(--bad)}
.unknown{background:#2a3038;color:var(--muted)}
.big{font-size:26px;font-weight:600;margin:0 0 4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
gap:12px}
.kv{background:#1e232b;border-radius:8px;padding:10px 14px}
.kv .k{color:var(--muted);font-size:12px}
.kv .v{font-size:16px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
canvas{width:100%;background:#151920;border-radius:8px;border:1px solid var(--line)}
select{background:#222833;color:var(--ink);border:1px solid var(--line);
border-radius:6px;padding:5px 10px}
.legend{display:flex;gap:18px;color:var(--muted);font-size:12px;margin:8px 0 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;
margin-right:6px;vertical-align:-1px}
.key{list-style:none;margin:10px 0 0;padding:0;max-width:80ch}
.key li{display:flex;gap:10px;align-items:flex-start;margin:0 0 7px;
color:var(--muted);font-size:13px;line-height:1.5}
.key i{flex:none;width:11px;height:11px;border-radius:2px;margin-top:5px}
.key b{color:var(--ink);font-weight:600}
.load-clusters{display:flex;flex-wrap:wrap;gap:8px 16px;margin:10px 0 2px;
color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.load-clusters span{display:inline-flex;align-items:center;gap:6px}
.load-clusters i{display:inline-block;width:18px;height:4px;border-radius:1px}
.formula-list{display:grid;gap:0}
.formula-row{display:grid;grid-template-columns:minmax(150px,220px) 1fr;gap:14px;
padding:9px 0;border-bottom:1px solid var(--line);align-items:start}
.formula-row:last-child{border-bottom:0}
.formula-row code{white-space:normal;overflow-wrap:anywhere;line-height:1.7}
.formula-note{margin-top:5px;color:var(--muted);font-size:12px;line-height:1.5}
.model-columns{min-width:280px;max-width:580px;white-space:normal;
overflow-wrap:anywhere;line-height:1.5}
code{background:#222833;padding:1px 6px;border-radius:4px;font-size:12px}
dl{margin:0}dt{margin-top:10px;font-weight:600}dd{margin:2px 0 0;color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
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
const DOC = JSON.parse(document.getElementById('data').textContent);
const P = DOC.payload || {};
const JOINTS = P.joints || [];
const NAMES = DOC.names || [];
let lang = (navigator.language || 'en').toLowerCase().startsWith('zh') ? 'zh' : 'en';
// A language switch rebuilds the DOM, so what the reader had selected has to
// live outside it.
let locked = false;
let picked = 0;

const t = (key) => (TEXT[key] ? (TEXT[key][lang] ?? TEXT[key].en) : key);
/* t() echoes the key when there is no entry, which is the wanted behaviour for
 * a missing label but not for data that may hold anything. */
const tr = (key, fallback) => (TEXT[key] ? t(key) : fallback);
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const num = (v, d) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : Number(v).toFixed(d === undefined ? 3 : d);

const UNIT = P.effort_unit === 'newton_metre' ? 'N·m' : 'A';
const SOURCE = P.effort_source === 'torque' ? 'quantity.torque' : 'quantity.current';

/* ---------------- sections ---------------- */

function verdictSection() {
  const v = P.verdict || {};
  const state = P.aborted ? 'fail' : (v.state || 'unknown');
  const rows = (v.joints || []).map((j, i) => `<tr>
    <td>${esc(NAMES[i] || j.joint)}</td>
    <td><span class="pill ${j.state}">${t('verdict.' + j.state)}</span></td>
    <td>${esc(j.reason || '')}</td></tr>`).join('');
  const skipped = P.skipped || [];
  const gravity = P.gravity_model?.available && state === 'pass';
  const verdictSay = gravity
    ? t(P.mode === 'gravity_rehearsal'
      ? 'verdict.gravity_rehearsal.pass.say' : 'verdict.gravity.pass.say')
    : t('verdict.' + state + '.say');
  const gaps = skipped.length ? `<p class="say" style="color:var(--warn)">
      ${t('verdict.skipped')}: ${skipped.length} &mdash;
      ${esc(skipped.slice(0, 4).map(s => s.motion).join(', '))}${
        skipped.length > 4 ? ' &hellip;' : ''}</p>` : '';
  return `<section>
    <h2 data-i18n="verdict.head"></h2>
    <p class="big"><span class="pill ${state}">${t('verdict.' + state)}</span></p>
    <p class="say">${verdictSay}</p>
    ${P.aborted ? `<p class="say" style="color:var(--bad)">
      ${t('verdict.aborted')}: ${esc(P.aborted)}</p>` : ''}
    ${gaps}
    <table><thead><tr>
      <th data-i18n="col.joint"></th><th data-i18n="col.state"></th>
      <th data-i18n="col.reason"></th></tr></thead><tbody>${rows}</tbody></table>
  </section>`;
}

function provenanceSection() {
  const p = P.provenance;
  if (!p) return '';
  const hardware = p.source === 'real_hardware';
  const rehearsal = p.source === 'analytic_rehearsal';
  const state = p.hardware_evidence ? 'pass' : hardware ? 'fail'
    : rehearsal ? 'warn' : 'unknown';
  const sourceKey = hardware ? 'provenance.hardware'
    : rehearsal ? 'provenance.rehearsal' : 'provenance.other';
  const sayKey = hardware ? 'provenance.hardware.say'
    : rehearsal ? 'provenance.rehearsal.say' : 'provenance.other.say';
  const software = p.software || {};
  const configuration = p.configuration || {};
  const profile = configuration.profile || {};
  const cells = [
    ['provenance.source', t(sourceKey)],
    ['provenance.raw', p.raw_frame_count ?? 0],
    ['provenance.observations', p.fitted_observation_count ?? 0],
    ['provenance.duration', p.publisher_duration_s == null
      ? '—' : `${num(p.publisher_duration_s, 1)} s`],
    ['provenance.rate', p.approximate_raw_rate_hz == null
      ? '—' : `${num(p.approximate_raw_rate_hz, 1)} Hz`],
    ['provenance.topic', p.telemetry_topic || '—'],
    ['provenance.groups', p.raw_motion_groups ?? 0],
    ['provenance.phasecheck', p.raw_phase_tag_mismatches ?? '—'],
    ['provenance.software', [software.package_version,
      `Python ${software.python_version || '?'}`,
      `NumPy ${software.numpy_version || '?'}`,
      `Pinocchio ${software.pinocchio_version || '?'}`].filter(Boolean).join(' · ')],
    ['provenance.profile', profile.source || p.profile_source || '—'],
    ['provenance.margin', configuration.collision_safety_margin_m == null
      ? '—' : `${num(configuration.collision_safety_margin_m * 1000, 0)} mm`],
  ];
  return `<section><h2>${t('provenance.head')}</h2>
    <p class="big"><span class="pill ${state}">${t(sourceKey)}</span></p>
    <p class="say">${t(sayKey)}</p><div class="grid">
    ${cells.map(([key, value]) => `<div class="kv"><div class="k">${t(key)}</div>
      <div class="v mono">${esc(value)}</div></div>`).join('')}
    </div></section>`;
}

function gravityModelSection() {
  const model = P.gravity_model;
  if (!model) return '';
  if (!model.available) return `<section><h2>${t('gravity.head')}</h2>
    <p class="big"><span class="pill fail">${t('verdict.fail')}</span></p>
    <p class="say">${esc(model.reason || 'unavailable')}</p></section>`;
  const pairing = model.pairing_audit || {};
  const incomplete = [
    ...(pairing.incomplete_training_poses || []),
    ...(pairing.incomplete_validation_poses || []),
    ...(pairing.malformed_training_tags || []),
    ...(pairing.malformed_validation_tags || []),
  ];
  const validation = (model.external_validation_rms || []).map(Number)
    .filter(Number.isFinite);
  const mean = validation.length
    ? validation.reduce((sum, value) => sum + value, 0) / validation.length : null;
  const worst = validation.length ? Math.max(...validation) : null;
  const cells = [
    ['gravity.model', model.model_type || '—'],
    ['gravity.trainposes', model.training_poses ?? 0],
    ['gravity.validposes', model.external_validation_poses ?? 0],
    ['gravity.mean', mean == null ? '—' : `${num(mean, 5)} ${UNIT}`],
    ['gravity.worst', worst == null ? '—' : `${num(worst, 5)} ${UNIT}`],
    ['gravity.hash', model.urdf_sha256 || '—'],
  ];
  const rows = (model.joints || []).map((joint, index) => {
    const columns = joint.retained_columns || [];
    const expression = columns.map((column) =>
      `${column.feature}=${num(column.coefficient, 8)}`).join(' · ');
    return `<tr><td>${esc(joint.output_joint || NAMES[index] || index + 1)}</td>
      <td class="mono model-columns">${esc(expression || '—')}</td>
      <td class="n">${num(joint.speed_pair_consistency_rms, 5)} ${UNIT}</td>
      <td class="n">${num(joint.external_validation_rms, 5)} ${UNIT}</td></tr>`;
  }).join('');
  return `<section><h2>${t('gravity.head')}</h2>
    <p class="big"><span class="pill pass">${t('gravity.empirical')}</span></p>
    <p class="say">${t('gravity.say')}</p>
    <p class="say"><code>${esc(model.predictor_equation || '')}</code></p>
    <p class="say" style="color:var(--warn)">${t('gravity.physical.missing')}</p>
    <p class="say" style="color:var(--warn)">${t('gravity.runtime.missing')}</p>
    <p class="say"><span class="pill ${incomplete.length ? 'fail' : 'pass'}">
      ${t(incomplete.length ? 'gravity.pairs.bad' : 'gravity.pairs.ok')}</span></p>
    <div class="grid">${cells.map(([key, value]) =>
      `<div class="kv"><div class="k">${t(key)}</div>
       <div class="v mono">${esc(value)}</div></div>`).join('')}</div>
    <h2 style="margin-top:20px">${t('gravity.columns')}</h2>
    <table><thead><tr><th>${t('col.joint')}</th><th>${t('gravity.columns')}</th>
      <th class="n">${t('gravity.speedcheck')}</th>
      <th class="n">${t('col.valid')}</th></tr></thead><tbody>${rows}</tbody></table>
  </section>`;
}

function summarySection() {
  const cells = [
    ['summary.mode', tr('summary.mode.' + (P.mode || ''), P.mode || '—')],
    ['summary.when', DOC.stamp],
    ['summary.joints', JOINTS.length],
    ['summary.quantity', t(SOURCE) + ' (' + UNIT + ')'],
    ['summary.samples', DOC.rows],
    ['summary.raw', DOC.rawRows || 0],
    ['summary.validation', P.validation_samples ?? 0],
    ['summary.action', P.action || '—'],
  ];
  return `<section><h2 data-i18n="summary.head"></h2><div class="grid">
    ${cells.map(([k, v]) => `<div class="kv"><div class="k">${t(k)}</div>
      <div class="v mono">${esc(v)}</div></div>`).join('')}
  </div></section>`;
}

function jointSection() {
  const rows = JOINTS.map((entry, i) => {
    const f = entry.friction || {};
    const parts = entry.components || {};
    const bad = (f.coulomb ?? 0) < 0 || (f.viscous ?? 0) < 0;
    // Two joints with the same Coulomb figure are not the same model if one of
    // them also carries a load term, so the columns it kept are named here.
    const kept = [parts.load_friction ? 'load' : null,
                  parts.stribeck ? 'stribeck' : null,
                  parts.actuator_inertia ? 'rotor' : null]
                 .filter(Boolean).join(' + ') || t('none');
    return `<tr>
      <td>${esc(NAMES[i] || i + 1)}</td>
      <td class="n"${bad ? ' style="color:var(--bad)"' : ''}>${num(f.coulomb)}</td>
      <td class="n">${parts.load_friction ? num(f.load_friction) : '—'}</td>
      <td class="n"${bad ? ' style="color:var(--bad)"' : ''}>${num(f.viscous, 5)}</td>
      <td class="n">${num(f.offset)}</td>
      <td>${esc(kept)}</td>
      <td class="n">${num(entry.residual_rms_a, 4)}</td>
      <td class="n">${num(entry.holdout_rms_a, 4)}</td>
      <td class="n">${num(entry.validation_rms_a, 4)}</td>
      <td class="n">${num(entry.condition_number, 0)}</td>
      <td class="n">${entry.effective_rank ?? '—'}</td>
      <td class="n">${entry.samples ?? '—'}</td></tr>`;
  }).join('');
  const gravity = !!P.gravity_model;
  return `<section><h2>${t(gravity ? 'joints.head.gravity' : 'joints.head')}</h2>
    <p class="say">${t(gravity ? 'joints.say.gravity' : 'joints.say')}</p>
    <table><thead><tr>
      <th data-i18n="col.joint"></th>
      <th class="n">${t('col.coulomb')} (${UNIT})</th>
      <th class="n">${t('col.load')} (${UNIT}/${UNIT})</th>
      <th class="n">${t('col.viscous')} (${UNIT}/(°/s))</th>
      <th class="n">${t('col.offset')} (${UNIT})</th>
      <th data-i18n="col.kept"></th>
      <th class="n" data-i18n="col.train"></th>
      <th class="n" data-i18n="col.holdout"></th>
      <th class="n" data-i18n="col.valid"></th>
      <th class="n" data-i18n="col.condition"></th>
      <th class="n" data-i18n="col.rank"></th>
      <th class="n" data-i18n="col.samples"></th>
    </tr></thead><tbody>${rows}</tbody></table></section>`;
}

function formulaTerm(value, expression, digits) {
  const coefficient = Number(value || 0);
  return {negative: coefficient < 0,
          text: `${Math.abs(coefficient).toFixed(digits)}${expression ? ` ${expression}` : ''}`};
}

function frictionFormula(entry) {
  const friction = entry.friction || {};
  const components = entry.components || {};
  const stribeckSpeed = Number(components.stribeck_speed_deg_s || 1);
  const terms = [];
  if (components.friction !== false) {
    const width = Number(components.coulomb_transition_deg_s || 0);
    const shape = width > 0 ? `tanh(v / ${width.toFixed(4)})` : 'sgn(v)';
    terms.push(formulaTerm(friction.coulomb, shape, 4));
  }
  if (components.load_friction) {
    terms.push(formulaTerm(friction.load_friction, 'L sgn(v)', 4));
  }
  if (components.stribeck) {
    terms.push(formulaTerm(
      friction.stribeck,
      `sgn(v) exp(-|v| / ${stribeckSpeed.toFixed(4)})`, 4));
  }
  if (components.load_stribeck) {
    terms.push(formulaTerm(
      friction.load_stribeck,
      `L sgn(v) exp(-|v| / ${stribeckSpeed.toFixed(4)})`, 4));
  }
  if (components.friction !== false) {
    terms.push(formulaTerm(friction.viscous, 'v', 6));
  }
  if (components.actuator_inertia) {
    terms.push(formulaTerm(friction.actuator_inertia, 'a', 6));
  }
  if (components.offset !== false || friction.offset !== undefined) {
    terms.push(formulaTerm(friction.offset, '', 5));
  }
  if (!terms.length) return 'I_f(v,L) = 0';
  const body = terms.map((term, index) => {
    if (index === 0) return `${term.negative ? '-' : ''}${term.text}`;
    return `${term.negative ? ' - ' : ' + '}${term.text}`;
  }).join('');
  return `I_f(v,L) = ${body}`;
}

function hasStribeckPeak(entry, load) {
  const friction = entry.friction || {};
  const components = entry.components || {};
  if (!(components.stribeck || components.load_stribeck)) return false;
  const coulomb = Number(friction.coulomb || 0);
  const stribeck = Number(friction.stribeck || 0)
    + Number(friction.load_stribeck || 0) * Math.abs(load || 0);
  if (!(stribeck > 1e-9)) return false;
  const viscous = Number(friction.viscous || 0);
  const transition = Number(components.coulomb_transition_deg_s || 0);
  const decay = Number(components.stribeck_speed_deg_s || 1);
  const derivative = (velocity) => {
    const ratio = transition > 0 ? velocity / transition : 0;
    const sech2 = transition > 0 ? 1 / Math.cosh(ratio) ** 2 : 0;
    return (transition > 0 ? coulomb / transition * sech2 : 0)
      - stribeck / decay * Math.exp(-velocity / decay) + viscous;
  };
  let previous = derivative(1e-6);
  if (previous < 0) return true;
  const ceiling = Math.max(8, 5 * transition, 5 * decay);
  for (let index = 1; index <= 1000; index += 1) {
    const current = derivative(ceiling * index / 1000);
    if (previous > 0 && current < 0) return true;
    previous = current;
  }
  return false;
}

function formulaDiagnostic(entry, index) {
  const friction = entry.friction || {};
  const components = entry.components || {};
  const points = (P.friction_samples || [])[index] || [];
  const load = points.length
    ? Math.max(...points.map((point) => Math.abs(point.load || 0))) : 0;
  const active = (Number(friction.stribeck || 0)
    + Number(friction.load_stribeck || 0) * load) > 1e-9;
  const status = t(active ? 'formula.stribeck.on' : 'formula.stribeck.off');
  const peak = active
    ? t(hasStribeckPeak(entry, load)
      ? 'formula.stribeck.peak' : 'formula.stribeck.nopeak') : '';
  const scale = Number(components.stribeck_speed_deg_s || 0);
  const details = active
    ? `${status}: Fs0=${num(friction.stribeck, 4)} ${UNIT}, `
      + `FsL=${num(friction.load_stribeck, 4)}, `
      + `vs=${num(scale, 3)} °/s, ${peak}`
    : status;
  const frictionPeak = points.length
    ? Math.max(...points.map((point) => Math.abs(point.effort || 0))) : null;
  const totalPeak = entry.peak_measured_effort;
  return `${details} · ${t('formula.totalpeak')}=${num(totalPeak, 4)} ${UNIT}`
    + ` · ${t('formula.frictionpeak')}=${num(frictionPeak, 4)} ${UNIT}`;
}

function formulaSection() {
  const rows = JOINTS.map((entry, index) => `<div class="formula-row">
    <strong>${esc(NAMES[index] || index + 1)}</strong>
    <div><code>${esc(frictionFormula(entry))}</code>
      <div class="formula-note">${esc(formulaDiagnostic(entry, index))}</div></div>
    </div>`).join('');
  const gravity = !!P.gravity_model;
  return `<section><h2>${t(gravity ? 'formula.head.gravity' : 'formula.head')}</h2>
    <p class="say">${t(gravity ? 'formula.say.gravity' : 'formula.say')}</p>
    <div class="formula-list">${rows}</div></section>`;
}

function steadyFrictionSection() {
  const audit = P.steady_friction_audit || {};
  if (!audit.available) return '';
  const rows = (audit.joints || []).map((entry, index) => {
    const levels = entry.load_levels || [];
    const status = entry.classical_low_speed_peak
      ? 'steady.peak' : 'steady.nopeak';
    return `<tr>
      <td>${esc(NAMES[index] || index + 1)}</td>
      <td class="n">${entry.observations ?? 0}</td>
      <td class="n">${levels.length}</td>
      <td class="n">${num(entry.maximum_low_speed_peak_a, 4)} ${UNIT}</td>
      <td class="n">${num(entry.maximum_interior_peak_a, 4)} ${UNIT}</td>
      <td style="color:var(--${entry.classical_low_speed_peak ? 'bad' : 'ok'})">
        ${t(status)}</td></tr>`;
  }).join('');
  return `<section><h2>${t('steady.head')}</h2>
    <p class="say">${t('steady.say')}</p>
    <table><thead><tr>
      <th>${t('col.joint')}</th><th class="n">${t('col.observations')}</th>
      <th class="n">${t('steady.levels')}</th>
      <th class="n">${t('steady.lowpeak')}</th>
      <th class="n">${t('steady.interior')}</th>
      <th>${t('col.state')}</th>
    </tr></thead><tbody>${rows}</tbody></table></section>`;
}

function comparisonSection() {
  const c = P.comparison || {};
  if (!Object.keys(c).length) return '';
  if (!c.available) return `<section><h2>${t('compare.head')}</h2>
    <p class="say">${t('compare.unavailable')}: ${esc(c.reason || '')}</p></section>`;
  const signed = (v) => (v === null || v === undefined) ? '—'
    : `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(1)}%`;
  const rows = (c.joints || []).map((j, i) => `<tr>
    <td>${esc(NAMES[i] || j.name || i + 1)}</td>
    <td class="n">${num(j.optimal_validation_rms_a, 4)} ${UNIT}</td>
    <td class="n">${num(j.sweep_validation_rms_a, 4)} ${UNIT}</td>
    <td class="n" style="color:var(--${j.optimal_better ? 'ok' : 'bad'})">
      ${signed(j.improvement_percent)}</td></tr>`).join('');
  const state = c.target_met ? 'pass' : 'fail';
  return `<section><h2>${t('compare.head')}</h2><p class="say">${t('compare.say')}</p>
    <p class="big"><span class="pill ${state}">${t(c.target_met ? 'compare.met' : 'compare.missed')}</span></p>
    <div class="grid">
      <div class="kv"><div class="k">${t('compare.mean')}</div><div class="v">
        ${num(c.optimal_mean_validation_rms_a, 4)} / ${num(c.sweep_mean_validation_rms_a, 4)} ${UNIT}
        (${signed(c.mean_improvement_percent)})</div></div>
      <div class="kv"><div class="k">${t('compare.worst')}</div><div class="v">
        ${num(c.optimal_worst_validation_rms_a, 4)} / ${num(c.sweep_worst_validation_rms_a, 4)} ${UNIT}
        (${signed(c.worst_improvement_percent)})</div></div>
      <div class="kv"><div class="k">${t('compare.source')}</div><div class="v mono">
        ${esc(c.source || '—')}</div></div>
    </div>
    <table><thead><tr><th>${t('col.joint')}</th>
      <th class="n">${t('compare.optimal')}</th><th class="n">${t('compare.sweep')}</th>
      <th class="n">${t('compare.improvement')}</th></tr></thead>
      <tbody>${rows}</tbody></table></section>`;
}

function key(colour, term, detail) {
  return `<li><i style="background:${colour}"></i><span><b>${term}</b>
    &nbsp;${detail}</span></li>`;
}

function chartSection() {
  const options = JOINTS.map((_, i) =>
    `<option value="${i}">${esc(NAMES[i] || i + 1)}</option>`).join('');
  return `<section><h2 data-i18n="charts.head"></h2>
    <label>${t('charts.pick')}
      <select id="pick">${options}</select></label>
    <label style="margin-left:16px"><input type="checkbox" id="lock"/>
      ${t('charts.lock')}</label>
    <p class="say" style="margin-top:8px" data-i18n="charts.lock.say"></p>
    <h2 style="margin-top:18px" data-i18n="charts.friction"></h2>
    <p class="say" data-i18n="charts.friction.say"></p>
    <canvas id="c-friction" height="300"></canvas>
    <div id="load-legend" class="load-clusters"></div>
    <ul class="key">
      ${key('#f8fafc', t('charts.sweep'), t('charts.friction.green'))}
      ${key('linear-gradient(90deg,#2563eb,#06b6d4,#22c55e,#facc15,#f97316,#ef4444)',
        t('charts.other'), t('charts.friction.blue'))}
      ${key('linear-gradient(90deg,#2563eb,#06b6d4,#22c55e,#facc15,#f97316,#ef4444)',
        t('charts.curve'), t('charts.friction.curve'))}
      ${key('repeating-linear-gradient(90deg,#f8fafc 0 4px,transparent 4px 8px)',
        t('charts.steady'), t('charts.friction.steady'))}
    </ul>
    <p class="say" id="still-note"></p>
    <p class="say" data-i18n="charts.friction.warn"></p>
    <h2 style="margin-top:22px" data-i18n="charts.steadylow"></h2>
    <p class="say" data-i18n="charts.steadylow.say"></p>
    <canvas id="c-steady" height="260"></canvas>
    <h2 style="margin-top:22px" data-i18n="charts.residual"></h2>
    <p class="say" data-i18n="charts.residual.say"></p>
    <canvas id="c-residual" height="220"></canvas>
    <ul class="key">
      ${key('var(--bad)', t('charts.red'), t('charts.residual.red'))}
      ${key('var(--sweep)', t('charts.green'), t('charts.residual.green'))}
    </ul>
    <p class="say" data-i18n="charts.residual.read"></p>
    <p class="say" data-i18n="charts.residual.caveat"></p>
    <h2 style="margin-top:22px" data-i18n="charts.error"></h2>
    <canvas id="c-error" height="200"></canvas>
    <h2 style="margin-top:22px" data-i18n="charts.condition"></h2>
    <p class="say" data-i18n="charts.condition.say"></p>
    <canvas id="c-condition" height="200"></canvas>
  </section>`;
}

function phaseSection() {
  const rows = (P.phases || []).map((p) => `<tr>
    <td>${esc(tr('phase.' + p.phase, p.phase))}</td>
    <td class="n">${p.observations ?? '—'}</td>
    <td class="n">${num(p.duration_s, 1)} s</td>
    <td class="n">${num(p.peak_temperature_c, 1)} °C</td>
    <td class="n">${num(p.peak_speed_deg_s, 1)} °/s</td>
    <td class="n">${num(p.peak_current_a, 3)} ${UNIT}</td>
    <td>${p.aborted ? esc(p.aborted) : t('none')}</td></tr>`).join('');
  return `<section><h2 data-i18n="phases.head"></h2>
    <p class="say">${t(P.gravity_model ? 'phases.gravity.say' : 'phases.say')}</p>
    <table><thead><tr>
      <th data-i18n="col.phase"></th><th class="n" data-i18n="col.observations"></th>
      <th class="n" data-i18n="col.duration"></th>
      <th class="n" data-i18n="col.peaktemp"></th>
      <th class="n" data-i18n="col.peakspeed"></th>
      <th class="n" data-i18n="col.peakeffort"></th>
      <th data-i18n="col.aborted"></th>
    </tr></thead><tbody>${rows}</tbody></table></section>`;
}

function rehearsalSection() {
  const r = P.rehearsal_check;
  if (!r || !r.available) return '';
  const rows = (r.joints || []).map((j, i) => `<tr>
    <td>${esc(NAMES[i] || j.joint)}</td>
    <td class="n">${num(j.expected, 4)}</td>
    <td class="n">${num(j.recovered, 4)}</td>
    <td class="n">${num(j.error, 4)}</td></tr>`).join('');
  return `<section><h2 data-i18n="rehearsal.head"></h2>
    <p class="say" data-i18n="rehearsal.say"></p>
    <p class="big"><span class="pill ${r.passed ? 'pass' : 'fail'}">
      ${t(r.passed ? 'rehearsal.ok' : 'rehearsal.bad')}</span></p>
    <p class="say">${t('rehearsal.worst')} ${num(r.worst_coulomb_error, 4)}
      ${UNIT} · ${t('rehearsal.tolerance')} ${num(r.tolerance, 4)} ${UNIT}</p>
    <table><thead><tr>
      <th data-i18n="col.joint"></th><th class="n" data-i18n="col.expected"></th>
      <th class="n" data-i18n="col.recovered"></th>
      <th class="n" data-i18n="col.error"></th>
    </tr></thead><tbody>${rows}</tbody></table></section>`;
}

function planSection() {
  const plan = P.plan || {};
  const gravityKeys = [
    'static_poses', 'static_candidates', 'gravity_validation_poses',
    'gravity_probe_deg', 'gravity_probe_speed_deg_s',
    'gravity_probe_speeds_deg_s', 'maximum_speed_deg_s',
    'position_margin_deg', 'temperature_ceiling_c', 'workspace_limit_deg',
    'workspace_range_deg', 'start_deg', 'seed',
  ];
  const keys = P.gravity_model
    ? gravityKeys.filter((key) => Object.hasOwn(plan, key))
    : Object.keys(plan).sort();
  const rows = keys.map((key) => `<tr>
    <td class="mono">${esc(key)}</td>
    <td class="n mono">${esc(JSON.stringify(plan[key]))}</td></tr>`).join('');
  return `<section><h2 data-i18n="plan.head"></h2>
    <p class="say" data-i18n="plan.say"></p>
    <table><tbody>${rows}</tbody></table></section>`;
}

function fileSection() {
  const items = [
    [DOC.files.result, P.gravity_model ? 'files.result.gravity' : 'files.result'],
    ...(P.gravity_model ? [[DOC.files.gravityModel, 'files.gravity_model']] : []),
    ...(P.gravity_model ? [[DOC.files.modelUrdf, 'files.model_urdf']] : []),
    [DOC.files.observations, 'files.observations'],
    ...(DOC.rawRows ? [[DOC.files.raw, 'files.raw']] : []),
    ...(P.data_sources ? [['raw_frame_sources.json', 'files.sources']] : []),
    ['report.html', 'files.report'],
  ];
  return `<section><h2 data-i18n="files.head"></h2><dl>
    ${items.map(([name, key]) => `<dt><a href="${esc(name)}"><code>${esc(name)}</code></a></dt>
      <dd>${t(key)}</dd>`).join('')}</dl></section>`;
}

function glossarySection() {
  const keys = ['g.rms', 'g.holdout', 'g.condition', 'g.rank', 'g.coulomb',
                'g.viscous', 'g.load', 'g.unphysical'];
  return `<section><h2 data-i18n="glossary.head"></h2><dl>
    ${keys.map((k) => `<dd style="margin:8px 0">${t(k)}</dd>`).join('')}
  </dl></section>`;
}

/* ---------------- charts ---------------- */

function frame(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  // Stashed on first use. Reading the logical height back from the height
  // attribute would be self-feeding: scaling the backing store writes that
  // same attribute, so every redraw would scale the chart again.
  if (!canvas.dataset.logicalHeight) {
    canvas.dataset.logicalHeight = String(+canvas.getAttribute('height') || 220);
  }
  const height = +canvas.dataset.logicalHeight;
  const width = Math.max(240, Math.round(canvas.clientWidth || 900));
  canvas.style.height = height + 'px';
  const backingWidth = Math.round(width * ratio);
  const backingHeight = Math.round(height * ratio);
  if (canvas.width !== backingWidth) canvas.width = backingWidth;
  if (canvas.height !== backingHeight) canvas.height = backingHeight;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '11px system-ui, sans-serif';
  return { ctx, width, height };
}

function axes(ctx, box) {
  ctx.strokeStyle = '#2a3038';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(box.left, box.top);
  ctx.lineTo(box.left, box.bottom);
  ctx.lineTo(box.right, box.bottom);
  ctx.stroke();
}

function empty(ctx, width, height) {
  ctx.fillStyle = '#8b96a5';
  ctx.textAlign = 'center';
  ctx.fillText('—', width / 2, height / 2);
  ctx.textAlign = 'left';
}

/* Ticks rather than a lone min and max in the corner: when the scale is the
 * only thing that changes between two joints, one small number is not enough
 * to notice it has changed. */
function yTicks(ctx, box, low, high, unit) {
  const span = (high - low) || 1;
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i += 1) {
    const value = low + (span * i) / 4;
    const y = box.bottom - ((value - low) / span) * (box.bottom - box.top);
    ctx.strokeStyle = i === 0 || i === 4 ? '#2a3038' : '#20252d';
    ctx.beginPath();
    ctx.moveTo(box.left, y);
    ctx.lineTo(box.right, y);
    ctx.stroke();
    ctx.fillStyle = '#8b96a5';
    ctx.fillText(value.toFixed(2) + (i === 4 ? ' ' + unit : ''),
                 box.left - 6, y + 3);
  }
  ctx.textAlign = 'left';
}

function caption(ctx, box, text) {
  ctx.fillStyle = '#c6cedb';
  ctx.fillText(text, box.left + 8, box.top + 13);
}

/* The speed axis carried a name but no numbers, so a cluster could be seen
 * without being placed. */
function xTicks(ctx, box, maxSpeed, sx) {
  const step = niceStep(2 * maxSpeed / 8);
  ctx.textAlign = 'center';
  for (let v = -Math.floor(maxSpeed / step) * step; v <= maxSpeed; v += step) {
    const x = sx(v);
    if (x < box.left - 1 || x > box.right + 1) continue;
    ctx.strokeStyle = v === 0 ? '#2a3038' : '#20252d';
    ctx.beginPath();
    ctx.moveTo(x, box.top);
    ctx.lineTo(x, box.bottom);
    ctx.stroke();
    ctx.fillStyle = '#8b96a5';
    ctx.fillText(formatTick(v, step), x, box.bottom + 14);
  }
  ctx.textAlign = 'left';
}

function niceStep(raw) {
  const power = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))));
  for (const factor of [1, 2, 2.5, 5, 10]) {
    if (factor * power >= raw) return factor * power;
  }
  return 10 * power;
}

function formatTick(value, step) {
  const digits = step >= 1 ? 0 : (step >= 0.1 ? 1 : 2);
  return value.toFixed(digits);
}

function frictionValue(entry, velocity, load) {
  const f = entry.friction || {};
  const components = entry.components || {};
  const transition = components.coulomb_transition_deg_s || 0;
  const stribeckSpeed = components.stribeck_speed_deg_s || 1;
  const sign = Math.sign(velocity);
  const reversal = transition > 0 ? Math.tanh(velocity / transition) : sign;
  const decay = sign * Math.exp(-Math.abs(velocity) / stribeckSpeed);
  return (f.coulomb || 0) * reversal
    + (f.load_friction || 0) * Math.abs(load || 0) * sign
    + (f.stribeck || 0) * decay
    + (f.load_stribeck || 0) * Math.abs(load || 0) * decay
    + (f.viscous || 0) * velocity + (f.offset || 0);
}

function frictionCurve(entry, maxSpeed, load) {
  const curve = [];
  for (let i = 0; i <= 160; i += 1) {
    const v = -maxSpeed + (2 * maxSpeed * i) / 160;
    curve.push([v, frictionValue(entry, v, load)]);
  }
  return curve;
}

const LOAD_LEVEL_COUNT = 6;
const LOAD_COLORS = ['#2563eb', '#06b6d4', '#22c55e',
                     '#facc15', '#f97316', '#ef4444'];

// Representative 5%-95% quantiles. Equal-count bins keep every color visible
// even when the load distribution is concentrated near one end.
function frictionLoads(entry, points) {
  if (!((entry.components || {}).load_friction
        || (entry.components || {}).load_stribeck)) return [0];
  const loads = points.map((s) => Math.abs(s.load || 0)).sort((a, b) => a - b);
  if (!loads.length) return [0];
  const levels = [];
  for (let index = 0; index < LOAD_LEVEL_COUNT; index += 1) {
    const quantile = 0.05 + 0.90 * index / (LOAD_LEVEL_COUNT - 1);
    const value = loads[Math.round((loads.length - 1) * quantile)];
    if (!levels.length || Math.abs(value - levels[levels.length - 1]) > 1e-9) {
      levels.push(value);
    }
  }
  return levels;
}

function loadClusterIndex(load, levels) {
  const value = Math.abs(load || 0);
  let best = 0;
  for (let index = 1; index < levels.length; index += 1) {
    if (Math.abs(value - levels[index]) < Math.abs(value - levels[best])) {
      best = index;
    }
  }
  return best;
}

function loadColor(index) {
  return LOAD_COLORS[index % LOAD_COLORS.length];
}

// The controlled steady passes hold one joint at one commanded speed and
// average a whole window, so their medians are friction at a speed in a way
// no trajectory point is. Drawn on the same axes they settle by eye whether
// the measurement rises to a low-speed peak or climbs monotonically, which
// the fitted curve alone can only assert.
function steadyOverlay(index) {
  const audit = P.steady_friction_audit || {};
  if (!audit.available) return [];
  const entry = (audit.joints || [])[index];
  if (!entry) return [];
  return (entry.load_levels || []).map((level) => ({
    load: level.load_median_a || 0,
    points: (level.curve || [])
      .filter((p) => Number.isFinite(p.speed_deg_s)
                     && Number.isFinite(p.friction_a))
      .map((p) => [p.speed_deg_s, p.friction_a]),
  })).filter((series) => series.points.length > 1);
}

function updateLoadLegend(levels) {
  const legend = document.getElementById('load-legend');
  if (!legend) return;
  legend.innerHTML = levels.map((load, index) =>
    `<span><i style="background:${loadColor(index)}"></i>`
    + `L${index + 1} = ${num(load, 3)} ${UNIT}</span>`).join('');
}

function frictionSpan(index) {
  const entry = JOINTS[index] || {};
  const points = (P.friction_samples || [])[index] || [];
  const maxSpeed = Math.max(10, ...points.map((s) => Math.abs(s.speed)));
  let values = points.map((s) => s.effort);
  frictionLoads(entry, points).forEach((load) => {
    values = values.concat(frictionCurve(entry, maxSpeed, load).map((p) => p[1]));
  });
  steadyOverlay(index).forEach((series) => {
    values = values.concat(series.points.map((p) => p[1]));
  });
  return { low: Math.min(...values), high: Math.max(...values) };
}

function drawFriction(canvas, entry, samples, label, forced, steady) {
  const { ctx, width, height } = frame(canvas);
  if (!entry) return empty(ctx, width, height);
  const points = samples || [];
  const series = steady || [];
  const speeds = points.map((s) => s.speed);
  const maxSpeed = Math.max(10, ...speeds.map(Math.abs));
  const loads = frictionLoads(entry, points);
  const curves = loads.map((load) => frictionCurve(entry, maxSpeed, load));
  updateLoadLegend(loads);
  let values = points.map((s) => s.effort);
  curves.forEach((curve) => { values = values.concat(curve.map((p) => p[1])); });
  series.forEach((s) => { values = values.concat(s.points.map((p) => p[1])); });
  const low = forced ? forced.low : Math.min(...values);
  const high = forced ? forced.high : Math.max(...values);
  const span = (high - low) || 1;
  const box = { left: 62, right: width - 12, top: 12, bottom: height - 44 };
  const sx = (v) => box.left + ((v + maxSpeed) / (2 * maxSpeed)) * (box.right - box.left);
  const sy = (e) => box.bottom - ((e - low) / span) * (box.bottom - box.top);
  axes(ctx, box);
  yTicks(ctx, box, low, high, UNIT);
  xTicks(ctx, box, maxSpeed, sx);
  points.forEach((s) => {
    if (s.sweep) return;
    ctx.fillStyle = loadColor(loadClusterIndex(s.load, loads));
    ctx.fillRect(sx(s.speed) - 1.5, sy(s.effort) - 1.5, 3, 3);
  });
  points.forEach((s) => {
    if (!s.sweep) return;
    ctx.fillStyle = loadColor(loadClusterIndex(s.load, loads));
    ctx.fillRect(sx(s.speed) - 2, sy(s.effort) - 2, 4, 4);
    ctx.strokeStyle = '#f8fafc';
    ctx.strokeRect(sx(s.speed) - 2.5, sy(s.effort) - 2.5, 5, 5);
  });
  if (curves.length > 1) {
    ctx.fillStyle = 'rgba(255,255,255,.055)';
    ctx.beginPath();
    curves[0].forEach(([v, e], i) => {
      const x = sx(v), y = sy(e); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    [...curves[curves.length - 1]].reverse().forEach(
      ([v, e]) => ctx.lineTo(sx(v), sy(e)));
    ctx.closePath();
    ctx.fill();
  }
  curves.forEach((curve, index) => {
    ctx.strokeStyle = loadColor(index);
    ctx.lineWidth = (index === 0 || index === curves.length - 1) ? 2.2 : 1.6;
    ctx.beginPath();
    curve.forEach(([v, e], i) => { const x = sx(v), y = sy(e); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  });
  series.forEach((entry) => {
    const color = loadColor(loadClusterIndex(entry.load, loads));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    entry.points.forEach(([v, e], i) => {
      const x = sx(v), y = sy(e); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    entry.points.forEach(([v, e]) => {
      ctx.beginPath();
      ctx.arc(sx(v), sy(e), 3, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#f8fafc';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  });
  const f = entry.friction || {};
  caption(ctx, box, `${label}   c ${(f.coulomb || 0).toFixed(3)} ${UNIT}`
    + `${entry.components?.load_friction ? `   μ ${(f.load_friction || 0).toFixed(3)}` : ''}`
    + `${entry.components?.load_stribeck ? `   FsL ${(f.load_stribeck || 0).toFixed(3)}` : ''}`
    + `   v ${(f.viscous || 0).toFixed(4)} ${UNIT}/(°/s)`
    + `${loads.length > 1 ? `   |load| ${loads[0].toFixed(2)}–${loads[loads.length - 1].toFixed(2)} ${UNIT}` : ''}`);
  ctx.fillStyle = '#8b96a5';
  ctx.textAlign = 'center';
  ctx.fillText(t('charts.speed'), (box.left + box.right) / 2, height - 8);
  ctx.textAlign = 'left';
}

// The steady range is a sliver of the main chart's axis. Given its own frame,
// the measured medians and the fitted curve can be read against each other at
// the speeds a Stribeck peak would live in.
function drawSteady(canvas, entry, series, loads, label) {
  const { ctx, width, height } = frame(canvas);
  if (!entry || !series || !series.length) return empty(ctx, width, height);
  const maxSpeed = Math.max(...series.map(
    (s) => Math.max(...s.points.map((p) => p[0]))));
  if (!(maxSpeed > 0)) return empty(ctx, width, height);
  const models = series.map((s) => {
    const curve = [];
    for (let i = 0; i <= 80; i += 1) {
      const v = Math.max(1e-3, (maxSpeed * i) / 80);
      curve.push([v, frictionValue(entry, v, s.load)]);
    }
    return curve;
  });
  let values = series.flatMap((s) => s.points.map((p) => p[1]));
  models.forEach((curve) => { values = values.concat(curve.map((p) => p[1])); });
  const pad = (Math.max(...values) - Math.min(...values)) * 0.12 || 0.01;
  const low = Math.min(...values) - pad;
  const high = Math.max(...values) + pad;
  const span = (high - low) || 1;
  const box = { left: 62, right: width - 12, top: 12, bottom: height - 44 };
  const sx = (v) => box.left + (v / maxSpeed) * (box.right - box.left);
  const sy = (e) => box.bottom - ((e - low) / span) * (box.bottom - box.top);
  axes(ctx, box);
  yTicks(ctx, box, low, high, UNIT);
  ctx.textAlign = 'center';
  series[0].points.forEach(([v]) => {
    ctx.strokeStyle = '#20252d';
    ctx.beginPath();
    ctx.moveTo(sx(v), box.top);
    ctx.lineTo(sx(v), box.bottom);
    ctx.stroke();
    ctx.fillStyle = '#8b96a5';
    ctx.fillText(String(v), sx(v), box.bottom + 14);
  });
  ctx.textAlign = 'left';
  series.forEach((s, index) => {
    const color = loadColor(loadClusterIndex(s.load, loads));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    models[index].forEach(([v, e], i) => {
      const x = sx(v), y = sy(e); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.points.forEach(([v, e], i) => {
      const x = sx(v), y = sy(e); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    s.points.forEach(([v, e]) => {
      ctx.beginPath();
      ctx.arc(sx(v), sy(e), 3.2, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#f8fafc';
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  });
  if (label) caption(ctx, box, label);
  ctx.fillStyle = '#8b96a5';
  ctx.textAlign = 'center';
  ctx.fillText(t('charts.speed'), (box.left + box.right) / 2, height - 8);
  ctx.textAlign = 'left';
}

function drawResidual(canvas, points, label, forced) {
  const { ctx, width, height } = frame(canvas);
  if (!points || !points.length) return empty(ctx, width, height);
  const box = { left: 62, right: width - 12, top: 12, bottom: height - 44 };
  const maxSpeed = Math.max(1, ...points.map((p) => Math.abs(p.speed)));
  const maxRes = forced || Math.max(1e-6, ...points.map((p) => Math.abs(p.residual)));
  const sx = (v) => box.left + ((v + maxSpeed) / (2 * maxSpeed)) * (box.right - box.left);
  const sy = (r) => (box.top + box.bottom) / 2 - (r / maxRes) * ((box.bottom - box.top) / 2);
  axes(ctx, box);
  yTicks(ctx, box, -maxRes, maxRes, UNIT);
  xTicks(ctx, box, maxSpeed, sx);
  ctx.fillStyle = 'rgba(226,86,90,.5)';
  points.forEach((p) => { if (!p.sweep) ctx.fillRect(sx(p.speed) - 1, sy(p.residual) - 1, 2, 2); });
  ctx.fillStyle = 'rgba(120,220,150,.9)';
  points.forEach((p) => { if (p.sweep) ctx.fillRect(sx(p.speed) - 1.5, sy(p.residual) - 1.5, 3, 3); });
  if (label) caption(ctx, box, label);
  ctx.fillStyle = '#8b96a5';
  ctx.textAlign = 'center';
  ctx.fillText(t('charts.speed'), (box.left + box.right) / 2, height - 8);
  ctx.textAlign = 'left';
}

function drawBars(canvas, values, format, ceiling) {
  const { ctx, width, height } = frame(canvas);
  if (!values.length) return empty(ctx, width, height);
  const box = { left: 54, right: width - 12, top: 12, bottom: height - 34 };
  const top = Math.max(ceiling || 0, ...values.map((v) => v.value || 0)) || 1;
  const slot = (box.right - box.left) / values.length;
  axes(ctx, box);
  values.forEach((entry, i) => {
    const h = ((entry.value || 0) / top) * (box.bottom - box.top);
    const x = box.left + slot * i + slot * 0.2;
    ctx.fillStyle = entry.bad ? '#e2565a' : '#4da3ff';
    ctx.fillRect(x, box.bottom - h, slot * 0.6, h);
    ctx.fillStyle = '#8b96a5';
    ctx.textAlign = 'center';
    ctx.fillText(entry.label, x + slot * 0.3, height - 18);
    ctx.fillText(format(entry.value), x + slot * 0.3, height - 6);
  });
  ctx.textAlign = 'left';
  ctx.fillStyle = '#8b96a5';
  ctx.fillText(format(top), 4, box.top + 8);
}

function shortName(name, index) {
  const text = String(name || index + 1);
  return text.length > 10 ? text.slice(-9) : text;
}

function redrawCharts() {
  const index = Math.min(picked, Math.max(0, JOINTS.length - 1));
  const name = NAMES[index] || String(index + 1);
  const shared = !!(document.getElementById('lock') || {}).checked;
  let span = null;
  let residualCap = null;
  if (shared) {
    const all = JOINTS.map((_, i) => frictionSpan(i));
    span = { low: Math.min(...all.map((s) => s.low)),
             high: Math.max(...all.map((s) => s.high)) };
    residualCap = Math.max(1e-6, ...(P.residual_samples || []).flat()
      .map((p) => Math.abs(p.residual)));
  }
  drawFriction(document.getElementById('c-friction'), JOINTS[index],
               (P.friction_samples || [])[index], name, span,
               steadyOverlay(index));
  drawSteady(document.getElementById('c-steady'), JOINTS[index],
             steadyOverlay(index),
             frictionLoads(JOINTS[index] || {},
                           (P.friction_samples || [])[index] || []),
             name);
  const still = document.getElementById('still-note');
  if (still) {
    still.textContent = t('charts.friction.still')
      .replace('{v}', num(P.friction_standstill_speed_deg_s || 0.05, 2))
      .replace('{n}', String((P.friction_standstill_excluded || [])[index] || 0));
  }
  drawResidual(document.getElementById('c-residual'),
               (P.residual_samples || [])[index], name, residualCap);
  drawBars(document.getElementById('c-error'), JOINTS.map((e, i) => ({
    label: shortName(NAMES[i], i),
    value: e.validation_rms_a ?? e.holdout_rms_a ?? e.residual_rms_a ?? 0,
  })), (v) => Number(v).toFixed(3));
  drawBars(document.getElementById('c-condition'), JOINTS.map((e, i) => ({
    label: shortName(NAMES[i], i),
    value: e.condition_number || 0,
    bad: (e.condition_number || 0) > 1000,
  })), (v) => Math.round(v).toString(), 1000);
}

/* ---------------- render ---------------- */

function render() {
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  const gravityTitle = P.gravity_model
    ? (P.mode === 'gravity_rehearsal' ? 'title.gravity_rehearsal' : 'title.gravity')
    : 'title';
  document.querySelector('h1').setAttribute('data-i18n', gravityTitle);
  document.getElementById('root').innerHTML = [
    `<p class="say" data-i18n="${P.gravity_model ? 'subtitle.gravity' : 'subtitle'}"></p>`,
    provenanceSection(), verdictSection(), gravityModelSection(),
    summarySection(), jointSection(), formulaSection(),
    steadyFrictionSection(), comparisonSection(), chartSection(),
    phaseSection(), rehearsalSection(), planSection(), fileSection(),
    glossarySection(),
  ].join('');
  for (const node of document.querySelectorAll('[data-i18n]')) {
    node.textContent = t(node.getAttribute('data-i18n'));
  }
  const pick = document.getElementById('pick');
  if (pick) {
    pick.value = String(picked);
    pick.addEventListener('change', () => {
      picked = parseInt(pick.value || '0', 10);
      redrawCharts();
    });
  }
  const lock = document.getElementById('lock');
  if (lock) {
    lock.checked = locked;
    lock.addEventListener('change', () => { locked = lock.checked; redrawCharts(); });
  }
  redrawCharts();
}

document.getElementById('lang').addEventListener('click', () => {
  lang = lang === 'zh' ? 'en' : 'zh';
  render();
});
window.addEventListener('resize', redrawCharts);
render();
</script>
</body>
</html>
"""
