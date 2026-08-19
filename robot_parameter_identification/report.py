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

# Column order per joint in the CSV. Kept explicit so the header and the rows
# cannot drift apart.
JOINT_COLUMNS = ("position_deg", "velocity_deg_s", "acceleration_deg_s2",
                 "effort", "temperature_c")
# Every frame behind a fitted observation, at the publisher's full rate.
RAW_COLUMNS = ("position_deg", "velocity_deg_s", "effort", "temperature_c")


def write_run(directory, payload: dict, observations=None,
              stamp: str | None = None, raw_frames=None) -> Path:
    """Create one folder for this run and fill it. Returns the folder."""
    mode = str(payload.get("mode") or "run")
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    folder = Path(directory) / f"{mode}-{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / RESULT_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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
    header = ["phase", "motion", "frame", "stamp_s"]
    for name in names:
        header += [f"{name}.{column}" for column in RAW_COLUMNS]
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame in frames:
            row = [frame.get("phase", ""), frame.get("motion", ""),
                   frame.get("frame", ""), _round(frame.get("stamp_s"), 6)]
            for index in range(len(names)):
                row += [_round(_item(frame, "position_deg", index), 5),
                        _round(_item(frame, "speed_deg_s", index), 5),
                        _round(_item(frame, "current_a", index), 6),
                        _round(_item(frame, "temperature_c", index), 2)]
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
                  "raw": RAW_FRAMES_NAME},
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
    "subtitle": {
        "en": "What was measured, what was fitted, and how far to trust it.",
        "zh": "本次测量了什么、拟合出了什么，以及结果可信到什么程度。",
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

    "summary.head": {"en": "This run", "zh": "本次运行"},
    "summary.mode": {"en": "mode", "zh": "运行方式"},
    "summary.mode.hardware": {"en": "hardware", "zh": "真机"},
    "summary.mode.rehearsal": {"en": "rehearsal", "zh": "预演"},
    "summary.when": {"en": "recorded", "zh": "记录时间"},
    "summary.joints": {"en": "joints", "zh": "关节数"},
    "summary.quantity": {"en": "identified quantity", "zh": "辨识量"},
    "summary.action": {"en": "trajectory action", "zh": "轨迹动作服务"},
    "summary.samples": {"en": "fitted samples", "zh": "拟合样本数"},
    "summary.raw": {"en": "raw frames behind them", "zh": "其背后的原始帧数"},
    "summary.validation": {"en": "validation samples", "zh": "验证样本数"},
    "quantity.current": {"en": "motor current", "zh": "电机电流"},
    "quantity.torque": {"en": "joint torque", "zh": "关节扭矩"},

    "joints.head": {"en": "Per joint", "zh": "逐关节结果"},
    "joints.say": {
        "en": "Coulomb friction is the constant part that opposes motion; "
              "viscous friction grows with speed. Validation error is measured "
              "on motion the fit never saw, so it is the honest one.",
        "zh": "库仑摩擦是与运动方向相反的恒定部分，粘滞摩擦随速度增大。"
              "验证误差在拟合从未见过的运动上测得，因此最能反映真实水平。",
    },
    "col.joint": {"en": "joint", "zh": "关节"},
    "col.coulomb": {"en": "Coulomb", "zh": "库仑摩擦"},
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
        "en": "from phase B, the friction sweeps. One joint moves back and "
              "forth about a single nominal pose across the whole speed range. "
              "Because the pose barely changes, a difference between two green "
              "points is a difference in speed.",
        "zh": "来自 B 摩擦阶段的扫掠。此时只有一个关节绕同一标称位姿往复运动，"
              "覆盖整个速度区间。由于位姿几乎不变，两个绿点之间的差异就是速度造成的差异。",
    },
    "charts.friction.blue": {
        "en": "from phase A, held still at many poses, and phase C, all joints "
              "on a smooth trajectory. Together they cover many poses but "
              "almost only low speed.",
        "zh": "来自 A 重力阶段（在多个位姿静止保持）与 C 惯性阶段（所有关节沿平滑轨迹运动）。"
              "两者覆盖了很多位姿，但速度几乎都很低。",
    },
    "charts.friction.curve": {
        "en": "the fitted model, coulomb x tanh(speed / transition) + viscous x "
              "speed + offset. A good fit runs through the middle of the cloud "
              "at every speed, not just on average.",
        "zh": "拟合模型：库仑 x tanh(速度 / 过渡宽度) + 粘滞 x 速度 + 偏置。"
              "拟合良好时，曲线在每个速度处都穿过点云中部，而不只是总体居中。",
    },
    "charts.friction.warn": {
        "en": "The two groups differ in pose as well as in speed, so a step "
              "between blue and green is not on its own evidence about speed. "
              "Read the speed dependence from the green points; use the blue "
              "ones to see how the curve behaves near rest.",
        "zh": "两组数据不只是速度不同，位姿也不同，因此蓝色与绿色之间的落差本身"
              "并不能作为速度效应的证据。速度依赖性应依据绿点判断，蓝点用来观察"
              "曲线在近零速处的表现。",
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
        "en": "phases A and C: the same low-speed, many-pose group drawn in "
              "blue on the chart above.",
        "zh": "A 与 C 阶段：即上图中以蓝色绘制的那组低速、多位姿数据。",
    },
    "charts.residual.green": {
        "en": "phase B, the sweeps: one joint, one pose, the full speed range.",
        "zh": "B 阶段的扫掠：单个关节、单一位姿、覆盖整个速度区间。",
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
    "charts.sweep": {"en": "Green — sweep samples", "zh": "绿色 —— 扫掠样本"},
    "charts.other": {"en": "Blue — static and trajectory phases",
                     "zh": "蓝色 —— 静态与轨迹阶段"},
    "charts.curve": {"en": "Yellow — fitted curve", "zh": "黄色 —— 拟合曲线"},
    "charts.red": {"en": "Red — static and trajectory phases",
                   "zh": "红色 —— 静态与轨迹阶段"},
    "charts.green": {"en": "Green — sweep samples", "zh": "绿色 —— 扫掠样本"},

    "phases.head": {"en": "Phases", "zh": "各阶段"},
    "phases.say": {
        "en": "A gravity: hold still at many poses. B friction: sweep one joint "
              "at a time. C inertia: move everything on a smooth trajectory. "
              "D validation: fresh motion, used only for scoring.",
        "zh": "A 重力：在多个位姿静止保持。B 摩擦：逐个关节做往复扫掠。"
              "C 惯性：沿平滑轨迹整体运动。D 验证：全新运动，仅用于评分。",
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
    "files.raw": {
        "en": "Every frame the driver published during each motion, at its full "
              "rate. The fit does not use these; they are here so the collapse "
              "from a burst of frames into one sample can be checked.",
        "zh": "每次运动中驱动器发布的每一帧，按其原始速率记录。拟合不使用这些数据，"
              "提供它们是为了让由一批帧塌缩成一个样本的这一步可被核查。",
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
padding:18px 20px;margin:0 0 18px}
.say{color:var(--muted);margin:0 0 14px;max-width:75ch}
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
.unknown{background:#2a3038;color:var(--muted)}
.big{font-size:26px;font-weight:600;margin:0 0 4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
gap:12px}
.kv{background:#1e232b;border-radius:8px;padding:10px 14px}
.kv .k{color:var(--muted);font-size:12px}
.kv .v{font-size:16px;font-variant-numeric:tabular-nums}
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
  return `<section>
    <h2 data-i18n="verdict.head"></h2>
    <p class="big"><span class="pill ${state}">${t('verdict.' + state)}</span></p>
    <p class="say">${t('verdict.' + state + '.say')}</p>
    ${P.aborted ? `<p class="say" style="color:var(--bad)">
      ${t('verdict.aborted')}: ${esc(P.aborted)}</p>` : ''}
    <table><thead><tr>
      <th data-i18n="col.joint"></th><th data-i18n="col.state"></th>
      <th data-i18n="col.reason"></th></tr></thead><tbody>${rows}</tbody></table>
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
    const bad = (f.coulomb ?? 0) < 0 || (f.viscous ?? 0) < 0;
    return `<tr>
      <td>${esc(NAMES[i] || i + 1)}</td>
      <td class="n"${bad ? ' style="color:var(--bad)"' : ''}>${num(f.coulomb)}</td>
      <td class="n"${bad ? ' style="color:var(--bad)"' : ''}>${num(f.viscous, 5)}</td>
      <td class="n">${num(f.offset)}</td>
      <td class="n">${num(entry.residual_rms_a, 4)}</td>
      <td class="n">${num(entry.holdout_rms_a, 4)}</td>
      <td class="n">${num(entry.validation_rms_a, 4)}</td>
      <td class="n">${num(entry.condition_number, 0)}</td>
      <td class="n">${entry.effective_rank ?? '—'}</td>
      <td class="n">${entry.samples ?? '—'}</td></tr>`;
  }).join('');
  return `<section><h2 data-i18n="joints.head"></h2>
    <p class="say" data-i18n="joints.say"></p>
    <table><thead><tr>
      <th data-i18n="col.joint"></th>
      <th class="n">${t('col.coulomb')} (${UNIT})</th>
      <th class="n">${t('col.viscous')} (${UNIT}/(°/s))</th>
      <th class="n">${t('col.offset')} (${UNIT})</th>
      <th class="n" data-i18n="col.train"></th>
      <th class="n" data-i18n="col.holdout"></th>
      <th class="n" data-i18n="col.valid"></th>
      <th class="n" data-i18n="col.condition"></th>
      <th class="n" data-i18n="col.rank"></th>
      <th class="n" data-i18n="col.samples"></th>
    </tr></thead><tbody>${rows}</tbody></table></section>`;
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
    <canvas id="c-friction" height="260"></canvas>
    <ul class="key">
      ${key('var(--sweep)', t('charts.sweep'), t('charts.friction.green'))}
      ${key('var(--accent)', t('charts.other'), t('charts.friction.blue'))}
      ${key('var(--warn)', t('charts.curve'), t('charts.friction.curve'))}
    </ul>
    <p class="say" data-i18n="charts.friction.warn"></p>
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
    <p class="say" data-i18n="phases.say"></p>
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
  const rows = Object.keys(plan).sort().map((key) => `<tr>
    <td class="mono">${esc(key)}</td>
    <td class="n mono">${esc(JSON.stringify(plan[key]))}</td></tr>`).join('');
  return `<section><h2 data-i18n="plan.head"></h2>
    <p class="say" data-i18n="plan.say"></p>
    <table><tbody>${rows}</tbody></table></section>`;
}

function fileSection() {
  const items = [
    [DOC.files.result, 'files.result'],
    [DOC.files.observations, 'files.observations'],
    [DOC.files.raw, 'files.raw'],
    ['report.html', 'files.report'],
  ];
  return `<section><h2 data-i18n="files.head"></h2><dl>
    ${items.map(([name, key]) => `<dt><code>${esc(name)}</code></dt>
      <dd>${t(key)}</dd>`).join('')}</dl></section>`;
}

function glossarySection() {
  const keys = ['g.rms', 'g.holdout', 'g.condition', 'g.rank', 'g.coulomb',
                'g.viscous', 'g.unphysical'];
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

function frictionCurve(entry, maxSpeed) {
  const f = entry.friction || {};
  const transition = (entry.components || {}).coulomb_transition_deg_s || 0;
  const curve = [];
  for (let i = 0; i <= 160; i += 1) {
    const v = -maxSpeed + (2 * maxSpeed * i) / 160;
    const rev = transition > 0 ? Math.tanh(v / transition) : Math.sign(v);
    curve.push([v, (f.coulomb || 0) * rev + (f.viscous || 0) * v + (f.offset || 0)]);
  }
  return curve;
}

function frictionSpan(index) {
  const entry = JOINTS[index] || {};
  const points = (P.friction_samples || [])[index] || [];
  const maxSpeed = Math.max(10, ...points.map((s) => Math.abs(s.speed)));
  const values = frictionCurve(entry, maxSpeed).map((p) => p[1])
    .concat(points.map((s) => s.effort));
  return { low: Math.min(...values), high: Math.max(...values) };
}

function drawFriction(canvas, entry, samples, label, forced) {
  const { ctx, width, height } = frame(canvas);
  if (!entry) return empty(ctx, width, height);
  const points = samples || [];
  const speeds = points.map((s) => s.speed);
  const maxSpeed = Math.max(10, ...speeds.map(Math.abs));
  const curve = frictionCurve(entry, maxSpeed);
  const values = curve.map((p) => p[1]).concat(points.map((s) => s.effort));
  const low = forced ? forced.low : Math.min(...values);
  const high = forced ? forced.high : Math.max(...values);
  const span = (high - low) || 1;
  const box = { left: 62, right: width - 12, top: 12, bottom: height - 30 };
  const sx = (v) => box.left + ((v + maxSpeed) / (2 * maxSpeed)) * (box.right - box.left);
  const sy = (e) => box.bottom - ((e - low) / span) * (box.bottom - box.top);
  axes(ctx, box);
  yTicks(ctx, box, low, high, UNIT);
  ctx.strokeStyle = '#2a3038';
  ctx.beginPath(); ctx.moveTo(sx(0), box.top); ctx.lineTo(sx(0), box.bottom); ctx.stroke();
  ctx.fillStyle = 'rgba(77,163,255,.45)';
  points.forEach((s) => { if (!s.sweep) ctx.fillRect(sx(s.speed) - 1, sy(s.effort) - 1, 2, 2); });
  ctx.fillStyle = 'rgba(120,220,150,.9)';
  points.forEach((s) => { if (s.sweep) ctx.fillRect(sx(s.speed) - 1.5, sy(s.effort) - 1.5, 3, 3); });
  ctx.strokeStyle = '#e0b341'; ctx.lineWidth = 1.8; ctx.beginPath();
  curve.forEach(([v, e], i) => { const x = sx(v), y = sy(e); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  const f = entry.friction || {};
  caption(ctx, box, `${label}   c ${(f.coulomb || 0).toFixed(3)} ${UNIT}`
    + `   v ${(f.viscous || 0).toFixed(4)} ${UNIT}/(°/s)`);
  ctx.fillStyle = '#8b96a5';
  ctx.textAlign = 'center';
  ctx.fillText(t('charts.speed'), (box.left + box.right) / 2, height - 8);
  ctx.textAlign = 'left';
}

function drawResidual(canvas, points, label, forced) {
  const { ctx, width, height } = frame(canvas);
  if (!points || !points.length) return empty(ctx, width, height);
  const box = { left: 62, right: width - 12, top: 12, bottom: height - 30 };
  const maxSpeed = Math.max(1, ...points.map((p) => Math.abs(p.speed)));
  const maxRes = forced || Math.max(1e-6, ...points.map((p) => Math.abs(p.residual)));
  const sx = (v) => box.left + ((v + maxSpeed) / (2 * maxSpeed)) * (box.right - box.left);
  const sy = (r) => (box.top + box.bottom) / 2 - (r / maxRes) * ((box.bottom - box.top) / 2);
  axes(ctx, box);
  yTicks(ctx, box, -maxRes, maxRes, UNIT);
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
               (P.friction_samples || [])[index], name, span);
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
  document.getElementById('root').innerHTML = [
    `<p class="say" data-i18n="subtitle"></p>`,
    verdictSection(), summarySection(), jointSection(), chartSection(),
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
