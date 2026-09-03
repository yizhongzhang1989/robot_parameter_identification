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
  'view.ghost_hint': {
    en: 'The poses the last run designed, as skeletons: blue for training, '
      + 'green for the held-out check, amber for friction sweep postures. The '
      + 'pose being executed is drawn solid.',
    zh: '上一次运行设计出的位形骨架：蓝色为训练位形，'
      + '绿色为留出验证，橙色为摩擦扫掠位形。正在执行的位形以实线突出。',
  },
  'view.gravity': { en: 'mass and centre of mass', zh: '质量与质心' },
  'view.gravity_hint': {
    en: 'The URDF\'s own gravity terms: one ball per link, sized by mass and '
      + 'placed at the centre of mass it declares, with the lever back to the '
      + 'link frame. The cyan ball is the whole robot\'s centre of mass and '
      + 'the arrow drops from it to the floor.',
    zh: '直接来自 URDF 的重力相关参数：每个连杆一个小球，大小按质量，'
      + '位置在它声明的质心处，连线是相对连杆坐标系的力臂。'
      + '青色球是整机质心，箭头从它垂直指向地面。',
  },
  'view.gravity_total': { en: 'whole robot', zh: '整机' },

  'signals.title': { en: 'Live signals', zh: '实时信号' },
  'signals.plot': { en: 'plot', zh: '曲线' },
  'signals.off': { en: 'off', zh: '收起' },
  'signals.normal': { en: 'default', zh: '默认' },
  'signals.big': { en: 'expand', zh: '展开' },
  'signals.pick': {
    en: 'Click a joint to keep it out of the plot.',
    zh: '点关节可把它从曲线里去掉。',
  },
  'signals.nodata': { en: 'no data yet', zh: '暂无数据' },
  'signals.waiting': {
    en: 'Waiting for telemetry.',
    zh: '等待遥测数据。',
  },
  'signals.lost': {
    en: 'Telemetry stopped. The last frame is not left on screen, because a '
      + 'frozen number reads exactly like a live one.',
    zh: '遥测已中断。最后一帧不会继续留在屏上——冻住的数字看起来和实时的一模一样。',
  },

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
  'obst.save': { en: 'save as', zh: '另存为' },
  'obst.saved': { en: 'configuration written to {v}', zh: '配置已写入 {v}' },
  'obst.where': {
    en: 'this scene, the planner envelope and the gravity settings are kept in '
      + '{v}; type a name to save a copy, then launch the next arm with '
      + 'config_file_path:= pointing at it',
    zh: '障碍物场景、规划包络和重力标定参数都会自动保存到 {v}；'
      + '填个名字可另存一份，下次启动另一台机器人时用 config_file_path:= 指过去',
  },
  'obst.nowhere': {
    en: 'this dashboard was launched without config_file_path:=, so nothing '
      + 'was loaded and every edit -- obstacles, planner envelope, gravity '
      + 'settings -- is lost on restart. Type a name and save, then relaunch '
      + 'with config_file_path:= pointing at it.',
    zh: '本次启动没有指定 config_file_path:=，所以没有载入任何配置，'
      + '所做的编辑（障碍物、规划包络、重力标定参数）重启后都会丢失。'
      + '填个名字保存，下次启动时用 config_file_path:= 指过去。',
  },

  /* ---- panel groups ---- */
  'group.control': { en: 'Overall control', zh: '总体控制' },
  'group.gravity': { en: 'Gravity compensation', zh: '重力补偿标定' },
  'group.rest': { en: 'Remaining calibration', zh: '其余标定' },
  'group.space': { en: 'Planner envelope', zh: '规划包络' },

  /* ---- gravity identification ---- */
  'grav.title': { en: 'Gravity identification', zh: '重力参数辨识' },
  'grav.hint': {
    en: 'Every pose is crossed both ways at two speeds. Standing still leaves '
      + 'static friction free to take any value in its band, so the pair mean '
      + 'is gravity and the half difference is friction. No acceleration here, '
      + 'so this model holds the arm up; it does not move it fast.',
    zh: '每个位形都以两种速度双向穿越。静止时静摩擦可在摩擦带内取任意值，'
      + '所以成对均值是重力，半差是摩擦。这里没有加速度，'
      + '所以得到的模型只能把手臂托住，不能支持快速运动。',
  },
  'grav.poses': { en: 'training poses', zh: '训练位形数' },
  'grav.check': { en: 'held-out poses', zh: '留出验证位形数' },
  'grav.arc': { en: 'crossing arc (°)', zh: '穿越幅度（°）' },
  'grav.slow': { en: 'slow probe (°/s)', zh: '慢速探针（°/s）' },
  'grav.fast': { en: 'fast probe (°/s)', zh: '快速探针（°/s）' },
  'grav.rehearse': { en: 'Rehearse gravity', zh: '重力预演' },
  'grav.plan': { en: 'Plan poses', zh: '只规划' },

  /* ---- planner envelope + pose review ---- */
  'space.title': { en: 'Planner envelope', zh: '规划包络' },
  'space.hint': {
    en: 'Where the planner may take each joint. Left unset it uses the arm\'s '
      + 'own range, which describes the arm and not the cell it stands in. A '
      + 'cell is rarely symmetric: an arm mounted at an angle meets the bench '
      + 'swinging one way and nothing the other. Changing this clears any '
      + 'armed run, because the poses that were rehearsed are not the poses '
      + 'this will design.',
    zh: '规划器可以把每个关节带到哪里。不设时用机械臂自身的行程，'
      + '而那描述的是手臂本身，不是它所在的工作单元。工作单元很少是对称的：'
      + '斜装的手臂往一边摆会碰到台子，往另一边则什么都碰不到。'
      + '改动后已解锁的运行会被清除，因为预演过的位形不是新包络会规划出的位形。',
  },
  'space.low': { en: 'low (°)', zh: '下限（°）' },
  'space.high': { en: 'high (°)', zh: '上限（°）' },
  'space.arm': { en: "arm's range", zh: '机械臂行程' },
  'space.fill': { en: 'set all', zh: '全部填入' },
  'space.apply': { en: 'Apply envelope', zh: '应用包络' },
  'space.reset': { en: "Use the arm's range", zh: '恢复为机械臂行程' },
  'space.applied': { en: 'envelope applied', zh: '包络已应用' },
  'space.wasreset': { en: "envelope reset to the arm's range",
                      zh: '包络已恢复为机械臂行程' },
  'space.set': { en: 'Set here.', zh: '已在此设定。' },
  'space.pending': {
    en: 'Edited but not applied. The numbers in red are not what the planner '
      + 'is using; press Apply envelope.',
    zh: '已修改但尚未应用。红色的数字不是规划器正在用的值，请点“应用包络”。',
  },
  'space.unset': {
    en: 'Not set, so the planner uses the whole arm, out to ±{v}°. On a '
      + 'dual-arm robot that reaches the other arm.',
    zh: '未设定，规划器使用整个行程，最远到 ±{v}°。'
      + '双臂机器人上这个范围能够到另一条手臂。',
  },
  'inspect.title': { en: 'Review the planned poses', zh: '人工检查规划位形' },
  'inspect.hint': {
    en: 'Step through them in the 3D view before anything moves. The selected '
      + 'pose is drawn solid; the margin is the widest clearance it was '
      + 're-screened against and still passed, so the smallest number is the '
      + 'pose to look at.',
    zh: '在任何东西动起来之前，先在 3D 视图里逐个看。选中的位形以实线绘出；'
      + '“余量”是该位形重新筛查仍能通过的最大间隙，所以数字最小的那个最值得看。',
  },
  'inspect.margin': { en: 'margin', zh: '余量' },
  'inspect.worst': { en: 'tightest', zh: '最紧的' },
  'inspect.fly': { en: 'fly the tour', zh: '虚拟走一遍' },
  'inspect.land': { en: 'stop', zh: '停下' },
  'inspect.none': { en: 'nothing planned yet', zh: '尚未规划' },
  'grav.run': { en: 'Run on hardware', zh: '真机标定' },
  'grav.planning': {
    en: 'Designing poses and screening every one of them against the scene. '
      + 'This takes a few seconds and moves nothing.',
    zh: '正在设计位形，并逐个做碰撞筛查。需要几秒钟，不会有任何运动。',
  },
  'grav.visiting': {
    en: 'Running: pose {v}. The orange arm is flying the same tour in the '
      + 'view, far faster than the arm itself moves.',
    zh: '进行中：第 {v} 个位形。左侧橙色的那条臂在虚拟走同一路径，'
      + '速度远快于机械臂本身。',
  },
  'grav.astray': {
    en: 'The collision screen holds every joint this dashboard does not drive '
      + 'where it was when the screen was built, and these have moved since: '
      + '{v}. Rebuild the screen and plan again, or put them back.',
    zh: '碰撞筛查把本面板驱动不了的关节固定在建立筛查时的位置，而它们已经动了：{v}。'
      + '重建筛查并重新规划，或者把它们摆回去。',
  },
  'grav.rescreen': { en: 'Rebuild the screen', zh: '重建碰撞筛查' },
  'grav.armed': {
    en: 'The dry run recovered what it planted. These numbers are armed.',
    zh: '预演成功复现了预设摩擦，当前参数已解锁。',
  },
  'grav.locked': {
    en: 'Rehearse first. The dry run must recover the friction it plants '
      + 'before the arm is allowed to move.',
    zh: '请先预演。预演必须复现它自己预设的摩擦，才能驱动手臂。',
  },
  'grav.stale': {
    en: 'The settings changed since the dry run that passed. Rehearse again '
      + 'with these numbers, then run.',
    zh: '参数在通过的预演之后改动过。请用当前参数重新预演，再执行。',
  },
  'grav.passes': { en: 'crossings planned', zh: '计划穿越次数' },
  'grav.toofew': {
    en: 'Too few poses: joint 1 carries {v} gravity terms and each pose gives '
      + 'one row, so the fit would be underdetermined rather than merely noisy.',
    zh: '位形太少：关节 1 共有 {v} 个重力项，而每个位形只提供一行，'
      + '拟合会欠定，而不只是噪声大。',
  },
  'grav.holdout': { en: 'held-out error', zh: '留出集误差' },
  'grav.worst': { en: 'worst joint', zh: '最差关节' },
  'grav.preview': { en: 'planned poses drawn', zh: '已绘制位形数' },

  'run.activity': { en: 'Activity', zh: '当前任务' },

  /* ---- run ---- */
  'run.title': { en: 'Full campaign', zh: '完整标定流程' },
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
  'conn.extra_topics': { en: 'extra topics', zh: '附加话题' },
  'conn.action': { en: 'action', zh: '动作服务' },
  'conn.effort_unit': { en: 'identified quantity', zh: '辨识量' },
  'conn.sample_age': { en: 'data age', zh: '数据时延' },
  'conn.driven': { en: 'joints driven', zh: '受控关节数' },
  'conn.profile': { en: 'profile', zh: '机器人档案' },
  'conn.shapes': { en: 'robot shapes', zh: '机器人碰撞体' },
  'conn.margin': { en: 'clearance margin', zh: '安全间隙' },
  'conn.envelope': { en: 'planner envelope', zh: '规划范围' },
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

  /* ---- robot envelope editor ---- */
  'profile.open': { en: 'Edit envelope', zh: '编辑机器人档案' },
  'profile.title': { en: 'Robot envelope', zh: '机器人档案' },
  'profile.close': { en: 'close', zh: '关闭' },
  'profile.hint': {
    en: 'Every number that belongs to this arm rather than to the method. '
      + 'There is one to edit before any file exists: without a file the '
      + 'module derives it from the URDF and the joints the controller drives.',
    zh: '所有属于这条臂、而非属于方法的数字。没有配置文件时也照样可以改：'
      + '模块会从 URDF 和控制器所驱动的关节推导出一份，这里编辑的就是它。',
  },
  'profile.nomodel': {
    en: 'Nothing to edit yet: waiting for /robot_description and for the '
      + 'controller to name the joints it drives.',
    zh: '暂无可编辑内容：正在等待 /robot_description 以及控制器上报受控关节名。',
  },
  'profile.busy': {
    en: 'A run is going. The envelope is shown but cannot change until it ends.',
    zh: '正在运行中。档案只读，运行结束后才能修改。',
  },
  'profile.guard_on': {
    en: 'Current ceilings are set, so the current guard is armed.',
    zh: '电流上限已设定，电流保护已生效。',
  },
  'profile.guard_off': {
    en: 'No current ceiling is set, so the current guard is off. That is the '
      + 'honest state until somebody measures one.',
    zh: '未设定电流上限，因此电流保护未生效。在有人实测出来之前，这就是如实的状态。',
  },
  'profile.name': { en: 'name', zh: '名称' },
  'profile.limits': { en: 'Per joint', zh: '逐关节' },
  'profile.envelope': { en: 'Envelope', zh: '包络' },
  'profile.joint': { en: 'joint', zh: '关节' },
  'profile.position': { en: 'reach', zh: '机械限位' },
  'profile.workspace': { en: 'workspace cap', zh: '工作空间限位' },
  'profile.continuous': { en: 'continuous', zh: '连续电流' },
  'profile.peak': { en: 'peak', zh: '峰值电流' },
  'profile.current_hint': {
    en: 'Leave a current ceiling blank when nobody has measured it. Blank '
      + 'means no limit, and the current guard stays off rather than trip on '
      + 'a guess.',
    zh: '没实测过的电流上限就留空。留空表示无上限，电流保护宁可不生效，'
      + '也不拿一个猜出来的阈值去跳闸。',
  },
  'profile.temperature_c': { en: 'temperature ceiling', zh: '温度上限' },
  'profile.sustained_speed_deg_s': { en: 'campaign speed ceiling', zh: '实验速度上限' },
  'profile.peak_speed_deg_s': { en: 'overspeed trip', zh: '超速跳闸' },
  'profile.position_margin_deg': { en: 'limit margin', zh: '限位余量' },
  'profile.minimum_voltage_v': { en: 'minimum bus voltage', zh: '母线电压下限' },
  'profile.maximum_voltage_v': { en: 'maximum bus voltage', zh: '母线电压上限' },
  'profile.sustained_current_window_s': { en: 'continuous-current window', zh: '连续电流窗口' },
  'profile.sustained_speed_window_s': { en: 'sustained-speed window', zh: '持续速度窗口' },
  'profile.current_slew_a_s': { en: 'current slew', zh: '电流变化率' },
  'profile.sender_gap_s': { en: 'sender gap', zh: '发送间隔' },
  'profile.telemetry_stale_s': { en: 'telemetry stale after', zh: '遥测过期时间' },
  'profile.probe_current_fraction': { en: 'probe current fraction', zh: '试探电流比例' },
  'profile.apply': { en: 'Apply', zh: '应用' },
  'profile.reset': { en: 'Reset', zh: '还原' },
  'profile.save': { en: 'Save file', zh: '保存为文件' },
  'profile.filename': { en: 'save as', zh: '文件名' },
  'profile.applied': { en: 'envelope applied', zh: '档案已应用' },
  'profile.from_panel': { en: 'edited here', zh: '面板内已修改' },
  'profile.was_reset': { en: 'envelope reset', zh: '档案已还原' },
  'profile.saved': { en: 'written to {v}', zh: '已写入 {v}' },

  /* ---- signal names, shared by the live overlay and the tables ---- */
  'live.joint': { en: 'joint', zh: '关节' },
  'live.position': { en: 'position', zh: '位置' },
  'live.speed': { en: 'speed', zh: '速度' },
  'live.current': { en: 'current', zh: '电流' },
  'live.torque': { en: 'torque', zh: '扭矩' },
  'live.temperature': { en: 'temperature', zh: '温度' },
  'live.voltage': { en: 'voltage', zh: '电压' },

  /* ---- jogging ---- */
  'jog.title': { en: 'Jog', zh: '点动' },
  'jog.hint': {
    en: 'Drives the same trajectory controller the campaign uses. Travel is '
      + 'the identification envelope, not the URDF\u2019s, and every pose is '
      + 'screened against the obstacle scene before the arm is asked to go '
      + 'there.',
    zh: '走的是辨识时同一个轨迹控制器。行程取辨识包络而非 URDF 的极限，'
      + '并且每个目标位姿在下发前都会先做一次障碍物碰撞筛查。',
  },
  'jog.enable': { en: 'Enable jogging', zh: '启用点动' },
  'jog.disable': { en: 'Disable jogging', zh: '停用点动' },
  'jog.zero': { en: 'All to zero', zh: '全部归零' },
  'jog.nomodel': {
    en: 'Waiting for a model and a motion plan.',
    zh: '等待模型与运动规划就绪。',
  },
  'phase.jogging': { en: 'jogging', zh: '点动中' },

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
  'recovery.gravity_bad': {
    en: 'The planted friction came back, but gravity did not: worst held-out '
      + 'error {v} exceeds {t}. Friction and gravity are separated by the '
      + 'crossing pair, so one can be exact while the other is '
      + 'underdetermined. Add poses and rehearse again.',
    zh: '预设摩擦复现了，但重力没有：留出集最大误差 {v} 超过 {t}。'
      + '双向穿越把摩擦和重力分开了，所以一者准确不代表另一者可辨。'
      + '请增加位形数重新预演。',
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
