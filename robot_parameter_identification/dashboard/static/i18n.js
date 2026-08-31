/* Translations for the panel.
 *
 * Keys are dotted and flat. Static markup carries data-i18n; anything built
 * in JS calls t(). The language sticks in localStorage so a reload does not
 * throw the operator back into English.
 */

export const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'zh', label: '中文' },
];

const STORE_KEY = 'rpi-lang';

const DICT = {
  /* ---- shell ---- */
  'app.title': { en: 'Parameter identification', zh: '参数辨识' },
  'pill.model': { en: 'model', zh: '模型' },
  'pill.telemetry': { en: 'telemetry', zh: '遥测' },
  'pill.action': { en: 'action', zh: '动作服务' },
  'pill.clearance': { en: 'clearance', zh: '间隙' },
  'pill.clear': { en: 'clear', zh: '无碰撞' },
  'pill.contact': { en: 'contact', zh: '已接触' },
  'pill.offline': { en: 'offline', zh: '离线' },
  'state.idle': { en: 'idle', zh: '空闲' },
  'state.running': { en: 'running', zh: '运行中' },

  /* ---- 3D overlays ---- */
  'view.meshes': { en: 'meshes', zh: '网格模型' },
  'view.frames': { en: 'frames', zh: '坐标系' },
  'view.labels': { en: 'labels', zh: '标签' },
  'view.ghost': { en: 'planned pose', zh: '规划位姿' },

  'obst.title': { en: 'Obstacles', zh: '障碍物' },
  'obst.add': { en: '+ box', zh: '+ 方块' },
  'obst.delete': { en: 'delete', zh: '删除' },
  'obst.clear': { en: 'clear', zh: '清空' },
  'obst.move': { en: 'move', zh: '平移' },
  'obst.rotate': { en: 'rotate', zh: '旋转' },
  'obst.scale': { en: 'size', zh: '缩放' },
  'obst.parent': { en: 'bolted to', zh: '固定于' },
  'obst.size': { en: 'size x', zh: '尺寸 x' },
  'obst.pos': { en: 'pos x', zh: '位置 x' },
  'obst.roll': { en: 'roll', zh: '横滚' },
  'obst.pitch': { en: 'pitch', zh: '俯仰' },
  'obst.yaw': { en: 'yaw', zh: '偏航' },
  'obst.hint': {
    en: 'A box on the base is a bench. A box on a link travels with the arm.',
    zh: '固定在基座上的方块代表工作台；固定在连杆上的方块会随手臂一起运动。',
  },
  'obst.nomodel': { en: 'no model yet', zh: '尚未载入模型' },

  /* ---- run ---- */
  'run.title': { en: 'Run', zh: '运行' },
  'run.rehearse': { en: 'Rehearse', zh: '预演' },
  'run.hardware': { en: 'Run on hardware', zh: '真机运行' },
  'run.home': { en: 'Home', zh: '回零位' },
  'run.stop': { en: 'Stop', zh: '停止' },
  'run.armed': {
    en: 'Rehearsal passed: the hardware run is armed.',
    zh: '预演已通过：真机运行已解锁。',
  },
  'run.locked': {
    en: 'A rehearsal must pass before a campaign may move the arm. '
      + 'Homing is always available.',
    zh: '必须先通过预演，标定流程才能驱动手臂。回零位不受此限制。',
  },
  'run.notstarted': { en: 'not started', zh: '未开始' },
  'run.samples': { en: 'samples', zh: '样本' },
  'run.pose': { en: 'pose', zh: '位姿' },
  'run.worst': { en: 'worst {v}° from zero', zh: '距零位最大 {v}°' },
  'run.failed': { en: 'failed', zh: '失败' },
  'run.stopped': { en: 'stopped', zh: '已停止' },

  /* ---- optimal excitation ---- */
  'optimal.title': { en: 'Optimal excitation identification', zh: '最优激励轨迹辨识' },
  'optimal.hint': {
    en: 'Load-conditioned 0.1–2°/s constant-speed subtrajectories identify steady friction; distinct multi-joint Fourier trajectories selected against the cumulative regressor identify rigid dynamics. Separate Fourier trajectories are used only for validation and comparison.',
    zh: '在不同负载下以 0.1–2°/s 运行恒速子轨迹，用于辨识稳态摩擦；按累计回归量选择彼此不同的多关节傅里叶轨迹，用于辨识刚体动力学。另设独立傅里叶轨迹，仅用于验证和对比。',
  },
  'optimal.training': { en: 'training trajectories', zh: '训练轨迹数' },
  'optimal.validation': { en: 'validation trajectories', zh: '验证轨迹数' },
  'optimal.duration': { en: 'seconds each', zh: '每条时长（秒）' },
  'optimal.postures': { en: 'low-speed load postures', zh: '低速负载位形数' },
  'optimal.repeats': { en: 'passes per direction', zh: '每方向重复次数' },
  'optimal.frequency': { en: 'Fourier base frequency (Hz)', zh: '傅里叶基频（Hz）' },
  'optimal.reuse': {
    en: 'reuse latest completed low-speed phase',
    zh: '复用最近一次已完成的低速阶段',
  },
  'optimal.start': { en: 'Run optimal excitation', zh: '开始最优激励辨识' },
  'optimal.trajectory': { en: 'trajectory', zh: '轨迹' },

  /* ---- load sweep ---- */
  'sweep.title': { en: 'Load sweep', zh: '负载扫掠' },
  'sweep.hint': {
    en: 'Drives a speed ladder at a series of gravity loads, one joint at a '
      + 'time. The levels are searched for, not set: how hard gravity can load '
      + 'a joint depends on where every other joint stands, and on how far its '
      + 'own axis is from vertical. Joints it cannot load are swept once and '
      + 'said so.',
    zh: '逐个关节，在一系列重力负载下跑一遍速度阶梯。负载等级是搜索出来的，不是设定的：'
      + '重力能给一个关节多大负载，取决于其余每个关节停在哪里，也取决于它自己的轴离竖直'
      + '方向有多远。无法加载的关节只扫一次，并如实说明。',
  },
  'sweep.start': { en: 'Sweep loads', zh: '开始扫掠' },
  'sweep.resume': { en: 'resume the last one', zh: '续跑上一次' },
  'sweep.designing': {
    en: 'searching postures for joint {j}…',
    zh: '正在为关节 {j} 搜索位形…',
  },
  'sweep.running': {
    en: '{j}, level {l}/{L} at {v}°/s — {n}/{t} passes ({p}%), {m} min',
    zh: '{j}，第 {l}/{L} 级，{v}°/s — 已驱动 {n}/{t} 次（{p}%），{m} 分钟',
  },
  'sweep.done': {
    en: 'finished: {n} passes driven, {s} skipped',
    zh: '完成：驱动 {n} 次，跳过 {s} 次',
  },
  'sweep.failed': { en: 'failed: {v}', zh: '失败：{v}' },

  'phase.A_gravity': { en: 'gravity', zh: '重力' },
  'phase.B_friction': { en: 'friction', zh: '摩擦' },
  'phase.C_inertia': { en: 'inertia', zh: '惯性' },
  'phase.D_validation': { en: 'validation', zh: '验证' },

  /* ---- connection ---- */
  'conn.title': { en: 'Connection', zh: '连接' },
  'conn.transport': { en: 'transport', zh: '传输方式' },
  'conn.topic': { en: 'topic', zh: '话题' },
  'conn.action': { en: 'action', zh: '动作服务' },
  'conn.effort_unit': { en: 'identified quantity', zh: '辨识量' },
  'conn.sample_age': { en: 'data age', zh: '数据时延' },
  'conn.driven': { en: 'joints driven', zh: '受控关节数' },
  'conn.profile': { en: 'profile', zh: '机器人档案' },
  'conn.shapes': { en: 'robot shapes', zh: '机器人碰撞体' },
  'conn.obstacles': { en: 'obstacles', zh: '障碍物' },
  'conn.guards_all': { en: 'All guards active.', zh: '全部保护均已生效。' },
  'conn.guards_off': {
    en: 'Guards off because the value is not available: {v}.',
    zh: '以下保护因缺少对应信号而未生效：{v}。',
  },
  'conn.noprofile': {
    en: 'No profile: waiting for the controller to name the joints it drives. '
      + 'Nothing can run until then.',
    zh: '尚无机器人档案：正在等待控制器上报它所驱动的关节名。在此之前无法运行。',
  },

  /* ---- live telemetry ---- */
  'live.title': { en: 'Live telemetry', zh: '实时数据' },
  'live.hint': {
    en: 'Straight from the state topic, one row per joint. A dash means the '
      + 'robot does not publish that quantity.',
    zh: '直接来自状态话题，每个关节一行。短横线表示该机器人不发布此量。',
  },
  'live.joint': { en: 'joint', zh: '关节' },
  'live.position': { en: 'position', zh: '位置' },
  'live.speed': { en: 'speed', zh: '速度' },
  'live.current': { en: 'current', zh: '电流' },
  'live.torque': { en: 'torque', zh: '扭矩' },
  'live.temperature': { en: 'temperature', zh: '温度' },
  'live.voltage': { en: 'voltage', zh: '电压' },
  'live.fault': { en: 'fault', zh: '故障' },
  'live.disabled': { en: 'off', zh: '未使能' },
  'live.waiting': { en: 'waiting for telemetry', zh: '等待遥测数据' },
  'live.fitted': { en: 'fitted', zh: '用于辨识' },

  /* ---- results ---- */
  'fit.title': { en: 'Fit quality', zh: '拟合质量' },
  'fit.hint': {
    en: 'Residual per joint. Validation is the number that counts: training '
      + 'error can always be made small.',
    zh: '每个关节的残差。真正说明问题的是验证误差：训练误差总是可以做小。',
  },
  'compare.title': { en: 'Optimal excitation vs load sweep', zh: '最优激励与负载扫掠对比' },
  'compare.hint': {
    en: 'Both models are scored on the same Fourier validation trajectories, which neither model trained on.',
    zh: '两个模型均在同一组傅里叶验证轨迹上评分，双方都未使用这些数据训练。',
  },
  'compare.met': { en: 'target met', zh: '目标达成' },
  'compare.missed': { en: 'target not met', zh: '目标未达成' },
  'compare.unavailable': { en: 'comparison unavailable', zh: '无法进行对比' },
  'compare.mean': { en: 'mean optimal / sweep', zh: '平均误差（最优激励 / 扫掠）' },
  'compare.worst': { en: 'worst optimal / sweep', zh: '最差误差（最优激励 / 扫掠）' },
  'compare.optimal': { en: 'optimal RMS', zh: '最优激励 RMS' },
  'compare.sweep': { en: 'sweep RMS', zh: '扫掠 RMS' },
  'compare.improvement': { en: 'improvement', zh: '改善幅度' },
  'friction.title': { en: 'Friction curve', zh: '摩擦曲线' },
  'friction.hint': {
    en: 'Fitted curve against the samples it was fitted to. Sweep samples are '
      + 'drawn apart. For trajectory samples, dark-to-bright blue indicates '
      + 'light-to-heavy rigid-body effort; the yellow band is the fitted load range.',
    zh: '拟合曲线与其所用样本的对比。扫掠样本单独着色。轨迹样本由深蓝到亮蓝表示'
      + '刚体驱动量由轻到重，黄色阴影为拟合载荷范围。',
  },
  'friction.joint': { en: 'joint', zh: '关节' },
  'residual.title': { en: 'Residual against speed', zh: '残差-速度关系' },
  'residual.hint': {
    en: 'Structure here is unmodelled physics, not noise.',
    zh: '此处若出现规律性结构，说明存在未建模的物理效应，而非噪声。',
  },
  'excite.title': { en: 'Excitation', zh: '激励充分性' },
  'excite.hint': {
    en: 'Condition number per joint. Large values mean the experiment barely '
      + 'moved some parameter, so trust it less.',
    zh: '每个关节的条件数。数值越大说明实验对某个参数的激励越弱，其结果越不可信。',
  },
  'params.title': { en: 'Parameters', zh: '辨识参数' },
  'params.none': { en: 'no result yet', zh: '尚无结果' },
  'params.physical': { en: 'physical', zh: '物理合理' },
  'params.unphysical': { en: 'unphysical', zh: '非物理' },
  'log.title': { en: 'Log', zh: '日志' },

  'verdict.pass': { en: 'pass', zh: '通过' },
  'verdict.warn': { en: 'warn', zh: '警告' },
  'verdict.fail': { en: 'fail', zh: '失败' },
  'verdict.unknown': { en: 'unknown', zh: '未知' },

  'recovery.ok': {
    en: 'Rehearsal recovered the planted friction to within {v} '
      + '(tolerance {t}).',
    zh: '预演成功复现了预设摩擦，误差 {v}（容差 {t}）。',
  },
  'recovery.bad': {
    en: 'Rehearsal ran but did NOT recover the planted friction: worst error '
      + '{v} exceeds {t}. The hardware button stays locked.',
    zh: '预演已运行，但未能复现预设摩擦：最大误差 {v} 超过 {t}。真机按钮保持锁定。',
  },

  /* ---- results folder ---- */
  'out.title': { en: 'Saved runs', zh: '已保存结果' },
  'out.hint': {
    en: 'Each run writes a folder holding the full raw data and a report you '
      + 'can open in a browser.',
    zh: '每次运行都会写入一个文件夹，内含完整原始数据和可直接用浏览器打开的报告。',
  },
  'out.report': { en: 'open report', zh: '打开报告' },
  'out.none': { en: 'nothing saved yet', zh: '尚未保存任何结果' },
};

let current = 'en';
try {
  const stored = localStorage.getItem(STORE_KEY);
  if (stored && LANGUAGES.some((entry) => entry.code === stored)) current = stored;
} catch (error) { /* private mode: English is a fine default */ }

export function getLang() { return current; }

export function setLang(code) {
  if (!LANGUAGES.some((entry) => entry.code === code)) return;
  current = code;
  try { localStorage.setItem(STORE_KEY, code); } catch (error) { /* ignore */ }
  document.documentElement.lang = code === 'zh' ? 'zh-CN' : 'en';
  applyStatic(document);
  for (const fn of listeners) fn(code);
}

const listeners = [];
export function onLangChange(fn) { listeners.push(fn); }

/** Translate. Unknown keys return the key, which makes a gap obvious. */
export function t(key, vars) {
  const entry = DICT[key];
  let text = entry ? (entry[current] ?? entry.en) : key;
  if (vars) {
    for (const [name, value] of Object.entries(vars)) {
      text = text.split(`{${name}}`).join(String(value));
    }
  }
  return text;
}

/** Fill every [data-i18n] element under root. */
export function applyStatic(root) {
  for (const node of root.querySelectorAll('[data-i18n]')) {
    node.textContent = t(node.getAttribute('data-i18n'));
  }
  for (const node of root.querySelectorAll('[data-i18n-title]')) {
    node.title = t(node.getAttribute('data-i18n-title'));
  }
}
