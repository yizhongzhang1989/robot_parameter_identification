# robot_parameter_identification

机械臂动力学参数辨识模块——可以直接指向一台并非你自己搭建的机器人。

它下发一段设计好的激励轨迹，把实测力矩（或电流）对刚体回归矩阵做回归，从而辨识刚体惯性参数与关节摩擦模型。模块自带一个 Web 面板，包含实时 3D 视图，以及一个可编辑的障碍物场景，供碰撞筛查使用。

## 系统默认配置

系统默认值集中在 [system_config.yaml](robot_parameter_identification/config/system_config.yaml)，
包含 ROS 连接与输出路径、dashboard 操作默认值、完整辨识/负载扫描计划、预演参数、
刷新频率和视图选项。修改选中的配置文件后重启面板即可生效，不需要修改 Python 或 JavaScript。

默认启动读取 `$XDG_CONFIG_HOME/robot_parameter_identification/system_config.yaml`；
未设置 `XDG_CONFIG_HOME` 时使用 `~/.config/robot_parameter_identification/system_config.yaml`。
首次启动缺少该文件时，自动创建父目录并从包内完整模板生成它。已有文件不会被覆盖；
格式、类型、未知字段或安全范围错误会阻止启动，不会静默恢复为出厂值。

切换不同场合的配置：

```bash
ros2 launch robot_parameter_identification dashboard.launch.py \
  system_config:=/absolute/path/to/system_config.yaml \
  controller:=right_arm_joint_trajectory_controller
```

指定的文件不存在时，同样自动生成。直接运行节点也支持：

```bash
ros2 run robot_parameter_identification dashboard --ros-args \
  -p system_config:=/absolute/path/to/system_config.yaml
```

例如，以下部分配置覆盖默认输出目录和拖动超限速度，未列出的字段继承包内模板：

```yaml
schema_version: 1
ros:
  controller: right_arm_joint_trajectory_controller
  output_directory: /absolute/path/to/results
  config_file_path: /absolute/path/to/cell.json
dashboard:
  drag_test:
    maximum_speed_deg_s: 80.0
```

- 启动参数优先级：显式 ROS/launch 参数 > 选中的 `system_config` > 包内模板。
- 面板操作参数优先级：本次输入 > `config_file_path` 中已保存的实验设置 > 系统默认值。
- `system_config` 是只读的系统默认配置；`config_file_path` 仍是可自动保存的场景、工作空间和重力实验 JSON，二者不能指向同一文件。
- 模板里的 `ros.config_file_path` 默认仍为空，不会改变原有实验设置的保存行为。
- 所有相对路径仍按启动目录解析，不相对于 YAML 文件；不同机器部署建议使用绝对路径。
- `controls` 的 `source` 引用实际配置值，浏览器和后端从同一来源取值。已打开页面中的后续用户输入不会被轮询覆盖。
- `campaign.derive_speed_ladders: true` 保留原有自动速度阶梯；设为 `false` 才直接采用配置中的摩擦/验证速度列表，仍受机器人速度上限约束。
- 软件允许范围统一由 `ranges` 提供，文件值优先于原代码中的范围；机器人 profile/URDF 与底层硬件保护独立生效。
- `GET /api/system-config` 可只读检查选中的路径及启动覆盖后的值；运行报告的 provenance 会记录配置快照。

### 配置参数范围

修改 `ranges` 下的 `min`/`max` 即可同时改变面板允许输入和后端的软件校验范围。
`max: null` 表示不增加软件上限，不表示绕过机器人限制；数值仍须有限、类型正确。
例如下列配置替换原有 60 度/秒、20 个姿态和 10 秒的软件范围：

```yaml
ranges:
  motion:
    transit_speed_deg_s: {min: 0.1, max: 80.0}
  hold_test:
    poses: {min: 1, max: 24}
    seconds: {min: 0.5, max: 15.0}
  campaign:
    static_poses: {min: 4, max: 80}
profile_derivation:
  maximum_default_speed_deg_s: 60.0
  default_speed_fraction: 0.5
```

这只是配置方法示例，不代表当前硬件已验证适合这些数值。调整范围后，相关默认值也应落在范围内。
配置读入时会拒绝逆序范围、非有限端点、非法整数范围及与面板默认值冲突的设置。

| 位置 | 作用 |
|---|---|
| `ranges.motion` | 回零、点动及姿态间转移速度 |
| `ranges.gravity` | 独立重力标定的探测速度 |
| `ranges.hold_test` | 保持姿态数、时长、偏移、温度及保持超速阈值 |
| `ranges.drag_test` | 拖动超速及温度停止阈值 |
| `ranges.campaign` | 姿态数、候选数、采样率、傅里叶参数、摩擦等全部数值实验计划范围 |
| `ranges.load_sweep` | 负载档位、速度、重复数、搜索数量、重试等扫描参数范围 |
| `profile_derivation` | 自动档案中原来的 20 度/秒封顶、额定速度比例和位置比例 |
| `planning.acceleration_per_speed_s_inv` | 从机器人持续速度推导加速度上限的系数 |
| `dashboard.hold_test` | 默认保持参数、到位容差、规划后漂移容差、等待超时等 |
| `runtime`、`rehearsal` | 场景漂移、采样/显示过滤和预演验收等已有可配置阈值 |

`controls.*.range` 引用上述范围，`source` 只引用默认值，不再重复定义 `min/max`。
旧文件的 `controls.*.min/max` 会在读取时迁移到相应范围，不改写原文件；显式 `ranges` 优先。
多个旧控件对同一范围给出不同的非默认端点时会报错，需在 `ranges` 中明确唯一值。

转移速度的有效上限为 `min(ranges.motion.transit_speed_deg_s.max, profile.sustained_speed_deg_s)`。
因此只修改面板默认速度不会修改允许范围；若上限仍是自动档案的 20，需调整 `profile_derivation`。
派生比例不得超过 URDF 给出的额定能力，显式机器人档案则继续作为独立约束。
`GET /api/system-config` 的 `controls` 返回配置范围，`control_ranges` 返回结合档案后的有效范围，
`constraints` 说明档案来源；主状态轮询也同步有效范围，后续用户输入不会被自动改值。

保持/拖动的每次运行目录会保存独立的 `system_config.yaml` 快照，并通过 `--system-config` 传给
`rm_control hold_check` / `manual_drag`，两端使用同一组范围。子程序缺少指定快照时会拒绝启动。
共享 CLI 不传该参数时使用自身默认值。已安装的 `gravity_compensation_test`、
`forward_current_controller_test`、`identified_static_hold_campaign` 旧入口现已 fail closed，
只提示转向 Dashboard 多位姿保持或共享 `hold_check` / `manual_drag`；保留的纯函数仅供离线使用。
不要继续把旧入口作为硬件执行路径。部署需同时更新辨识包与 `rm_control`，不能混用新旧入口。

本改动不修改硬件插件限流、实际限速、停流确认或急停约束。例如当前共享硬件接口的电流模式
仍有独立的 120 度/秒限速；更改软件阈值不等于提高了硬件允许值。端口范围、关节数量、单位、
有限数值及正时长/正尺寸等协议或数学约束也不会因配置范围变化而被跳过。

## 快速开始

以本机 RM75 双臂为例，按**第一次接一台臂**来写：此刻没有任何配置文件，也不需要有。

### 1. 启动机器人

本模块不参与这一步，用你平时的方式启动即可：

```bash
cd ~/Documents/RobotControl
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch robot_bringup real.launch.py
```

它只要三样东西，本机上分别是轨迹控制器 `right_arm_joint_trajectory_controller`、`/robot_description`、`/dynamic_joint_states`。

若要使用面板中的“开始保持检查”或“开始手动拖动”，启动时改用同一完整系统的电流能力入口：

```bash
ros2 launch robot_bringup direct_current.launch.py \
  direct_current_ack:=I_ACCEPT_DIRECT_CURRENT_CONTROLLER_RISK
```

该入口仍然启动正常的双臂 `robot_state_publisher`、唯一一个 `controller_manager`、两个 JTC、
关节与 F/T broadcaster；只是额外为右臂预加载 inactive 的 A 域 controller 和独立停流 guard。
不要在已经运行的 `real.launch.py` 上叠加执行它。部署后只需这一次显式重启，之后每次保持/拖动
都只切换 controller，不退出或替换任何 ROS node。

### 2. 启动面板（另开终端）

```bash
cd ~/Documents/RobotControl
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch robot_parameter_identification dashboard.launch.py \
    controller:=right_arm_joint_trajectory_controller
```

打开 <http://localhost:8300>。只有 `controller` 必须给：action 和 `controller_state` 话题都由它推出，受控关节由控制器自己宣告，其余全走默认值。左臂把它换成 `left_arm_joint_trajectory_controller` 并加 `port:=8301`。

**在面板里改过任何东西之后，这条命令就不够了。** `config_file_path` 默认为空，为空即「所有设置只在内存里」：上次画好的盒子、设好的规划包络、调好的重力标定参数都不会回来，这次改的也不会留下。第一次可以不给，之后每次都要给：

```bash
ros2 launch robot_parameter_identification dashboard.launch.py \
    controller:=right_arm_joint_trajectory_controller \
    config_file_path:=config/rm75_cell.json
```

相对路径按**启动面板时所在的目录**解析，因此上面这条要在 `~/Documents/RobotControl` 下敲。**指一个还不存在的文件是对的用法**：它就是这次的保存目标，第一次编辑会把它连同父目录一起建出来。面板的笔记区会写明这一次到底发生了什么——载入了几个障碍物、包络恢复成了什么、路径上还没有文件（附绝对路径）、还是压根没给这个参数——空配置的三种成因不会长得一模一样。

### 3. 在面板里把这条臂描述清楚

「连接」卡片 →「编辑机器人档案」。此刻的档案是从 URDF 推导来的：位置和速度限位已经有了。所有电流阈值和电流窗口都不属于 `RobotProfile`，面板也不提供这些输入。

- 所有 JTC 路径（含标定、回零、点动和负载扫描）只记录实测电流，不实施软件电流限制；真正的电流驱动保护由 `rm_control` 独立负责
- 收紧工作空间限位：URDF 描述的是臂，不是它旁边的工作台
- 温度上限、母线电压窗口照驱动器手册填
- **应用**立即生效，**保存为文件**存成 YAML；下次用 `profile_path:=` 指过去就不必再填

本机 RM75 右臂已经有一份填好的：`config/rm75_parameter_identification.yaml`。

### 4. 跑起来

1. 先在左侧 3D 视图里把工作台、夹具画成盒子——激励会提出机械臂从未到过的位姿，碰撞筛查靠的就是它们。**想让它们下次还在，启动时必须带 `config_file_path:=`**
2. **Rehearse（预演）** 必须先通过。它植入已知摩擦并要求辨识器找回来，正是这道闸门抓到过拟合悄悄返回全零
3. 通过后 **Run on hardware** / **Run optimal excitation** / **Sweep loads** 才解锁
4. **Home** 不设闸门：它不做辨识，存在的意义就是把一台已被拒绝使能的臂救回来

结果写在启动面板时所在目录下的 `identification_results/`。

### 之后可能要加的参数

| 参数 | 何时需要 |
|---|---|
| `profile_path:=...` | 已经存好一份档案，不想每次重填 |
| `gravity_test_source:=...` | 指定完整、通过、安培域且关节名匹配的辨识结果；支持 `{arm}` 实例名模板；仅右臂允许留空使用固定已投用模型 |
| `maximum_speed_deg_s:=60.0` | 计划速度默认封顶 10 °/s；黏滞摩擦在爬行速度下根本看不出来 |
| `signal.voltage:=voltage` | 驱动器有 `voltage` 接口、档案又给了电压窗口，母线电压保护才真正生效 |
| `config_file_path:=config/rm75_cell.json` | 让障碍物场景、规划包络和重力标定参数在两次会话之间留存；不给它，它们既不会读回也不会保存 |

## 与机器人之间的约定

模块从不与驱动、SDK 或厂商私有协议打交道。它只讲标准 ROS，并且明确列出自己需要什么。

| 用途 | 接口 | 是否必需 |
|---|---|---|
| 下发运动 | `control_msgs/action/FollowJointTrajectory` | 是 |
| 运动学与惯性先验 | `/robot_description`（`std_msgs/String`） | 是 |
| 关节遥测 | `sensor_msgs/JointState` **或** `control_msgs/DynamicJointState` | 是 |

遥测中每个关节有两路信号是必需的，另有五路可选：

| 信号 | 是否必需 | 缺失时的后果 |
|---|---|---|
| `position` | 是 | — |
| `effort`（电机电流或关节力矩） | 是 | — |
| `velocity` | 否 | 由位置差分得到 |
| `temperature` | 否 | 没有温度上限保护 |
| `enabled` | 否 | 没有驱动使能保护 |
| `fault_code` | 否 | 没有故障码保护 |
| `voltage` | 否 | 没有母线电压保护 |

**如果你的机器人不发布某个量，就自己把它转发出来。** 写一个小节点，把驱动给你的任何数据转成 `JointState` 或 `DynamicJointState` 接口，然后把模块指向它。这个边界是刻意划出来的：它把所有与具体机器人相关的代码都挡在本仓库之外。

命名属于配置，而非代码。一台把电流叫做 `motor_current`、把温度叫做 `temp` 的机器人，这样描述即可：

```yaml
telemetry:
  dynamic_joint_state_topic: /dynamic_joint_states
  signals:
    position: position
    velocity: velocity      # 置 null 则改为由位置差分得到
    current: motor_current  # 驱动不上报电流时置 null
    torque: effort          # 默认值；驱动不上报力矩时置 null
    effort_source: current  # 两路都有时优先回归哪一路
    temperature: temp
    enabled: null           # 未发布；面板会标明该保护处于关闭状态
    fault_code: null
```

**两路默认都会去找，因此通常什么都不用配。** `effort` 是 Humble 上 ros2_control 唯一标准化的力矩类接口名，`current` 是后续版本对另一路的命名，所以这两个默认值合起来覆盖了绝大多数机器人：只发 `current` 的（如 RM75）、只发 `effort` 的（如 UR、Franka、KUKA），都能直接读到。

`effort_source` 是**偏好而非要求**。驱动只发另一路时，它有权否决这个偏好：模块会自动改用实际到达的那一路，并在日志和面板笔记里写明改用了什么、单位是什么。一个从不到达的通道不是"在两个量之间做选择"，而是一次没有报错的停摆。

每个辨识参数的单位都由 `effort_source` 推导而来，因此不可能与实际读取的通道相矛盾：`current` 得到安培，`torque` 得到牛·米。**若驱动两者都上报，两路都会记录，并在面板上各占一列。**

两路都找不到时，模块会节流地打印一条日志，列出它要的接口名和话题上实际有的接口名——这样接一台新机器人时，第一眼就知道该把哪个 `signal.*` 指到哪。

面板会明确指出哪些保护因为信号未映射而处于关闭状态，而不是悄悄跳过它们。

### 某个量不在主状态话题上

有些量只通过厂商自己的节点进入 ROS，不在 `joint_states` / `dynamic_joint_states` 里。这种情况不作特例处理：把它转发成一个 `control_msgs/DynamicJointState` 话题，然后在启动时列出来即可。附加话题按关节名合并，并且优先于主状态话题——你既然专门指名了它，那就是在声明这个量从哪来。

```bash
extra_telemetry_topics:="['/right_arm/motor_currents']" signal.current:=motor_current
```

## 它不会下发什么

辨识、标定、归零与点动只通过轨迹 action 下发位置，不指令被辨识的力矩或电流。
另行确认的保持与手动拖动使用共享 `rm_control` runtime 切换控制器并发送电流；
Dashboard 本身不创建第二个电流发送器，也不自动启流或自动重试失败任务。

## 机器人档案

档案里装的是属于**这条臂**的位置控制约束：机械限位、工作空间限位、温度上限、超速跳闸、母线电压窗口、限位余量。方法本身对这些一无所知，换一台机器人就是换一份档案。

**没有档案也能跑。** 第一次接一台从没标定过的臂时，本来就不该先手写一份 YAML 才允许开机：模块会从 `/robot_description` 和控制器上报的受控关节列表推导一份，位置与速度限位取自 URDF。

**所有电流阈值和电流窗口均不属于 `RobotProfile`。** 连续/峰值电流、持续电流窗口、电流变化率和试探电流比例不再由档案保存、推导或编辑，也不以空值或无穷大表示。直接电流驱动的限值与保护由 `rm_control` 独立配置，面板接口中的 `hardware_current_limits` 仍提供该驱动的限值。

面板「连接」卡片里的**编辑机器人档案**打开一个弹窗，逐关节位置限位用表格，其他包络用输入框，改完点应用即刻生效。所有 JTC 位置控制路径，包括重力、完整和最优激励标定、回零（home）、点动（jog）及负载扫描（loadsweep），电流都只是被测量：只记录，不实施软件电流限制，不按电流停机或回退激励幅度。电流保持与手动拖动不属于这些 JTC 路径，继续使用独立的 `rm_control` 电流驱动保护。

标定的 `result.json` 保存 `current_measurements`，HTML 报告展示「标定实测电流」：逐关节有效/无效帧数、最小/最大电流、绝对峰值和采样 RMS；JSON 另保存峰值的发布者时间戳。不保存参考限值或超过参考值的帧数。统计覆盖监控器启用后的全部有效遥测，包括转场与被舍弃的采集窗口，不重复计入拟合平均样本。`raw_frames.csv` 仍只保存采集窗口。异常退出也保存已测统计；未测得的数据标为空而非零。历史结果文件不改写。

温度、速度、碰撞、限位、驱动使能/故障、已配置的电压窗口与操作员停止保护不变。实测电流统计不能直接作为电流驱动安全限值，后续仍需独立验证与配置；此修改不改变 `rm_control` 的电流驱动保护，也不改变标定轨迹的速度或加速度设置。

面板里应用的数字和手写文件里的数字享受同等待遇——两者都是操作员给的。「连接」卡片会把当前档案标为 `面板内已修改`，好让人区分「正在生效的」和「存在盘上的」。

**保存**把当前档案写成 YAML，下次用 `profile_path:=` 指过去即可。写入位置只能是启动时的结果目录，或启动时 `profile_path` 指定的那个文件——面板监听在所有网卡上，采信请求里的路径就等于开放任意文件写。

## 障碍物与碰撞筛查

激励实验会提出机械臂从未到过的位姿，因此必须有人否决那些会撞上工作台的位姿。该检查在模型上运行，使用 Pinocchio 的碰撞后端——不引入动力学之外的任何额外依赖。

障碍物是**绑定到某个坐标系**的长方体。绑在 base 坐标系上的盒子就是工作台；绑在末端连杆上的盒子则是工具或负载护罩，会随臂一起运动。你在 3D 视图里拖拽即可放置，位姿按相对父坐标系存储，其余交给运动学。

### 配置文件

用 `config_file_path:=<路径>` 启动，面板里改过的东西就有了归宿：**每次编辑后立即写盘，启动时自动读回**，不需要手动保存。写入采用先写临时文件再改名的方式，中途断电不会留下半份配置。相对路径按启动面板时所在的目录解析。

**一个文件装的是整套「这个工作单元」的描述**，不只是盒子：

| 段 | 内容 | 为什么不能从别处推出来 |
|---|---|---|
| `obstacles` | 绑定到坐标系的长方体 | 工作台不在 URDF 里 |
| `workspace_range_deg` | 逐关节的规划上下界 | URDF 描述的是这条臂，不是它旁边那条 |
| `gravity` | 重力标定卡片的位姿数、验证位姿数、探针幅度与速度 | 是针对这台臂调出来的实验设置 |

分成三个文件是把「同一个工作单元」拆散：换一台机器人要一起换，忘了换其中一个就是一次撞机。所以是一个文件、一个参数。写盘时也整份写：某一段改了，其余段原样带过去，不会被清空——障碍物在模型到达之前还没法放置，这时改包络也不会把文件里已有的盒子抹掉。

**路径上还没有文件，不是错误，而是新建一份配置的正确开头。** 读不到就从空配置开始，但这个路径依旧是保存目标：第一次编辑就把文件连同父目录一起建出来。把「已存在」当成写盘的前提，第一份配置就永远无从诞生。

**不给这个参数，就没有任何东西被自动保存**——设置只存在于内存里，进程一停就没了，只有「另存为」写过的文件还在。面板不会假装相反：没给 `config_file_path:=` 时，障碍物卡片直接说明本次启动不会留存，而不是指着一个从没被写过的路径说「编辑会自动保存到这里」。

换一台机器人时文件名当然要换，所以障碍物面板里有一个**另存为**：填个文件名再点它，整份配置会写进结果目录，下次启动那台臂时用 `config_file_path:=` 指过去即可。留空则覆盖启动时指定的那个文件。

只接受文件名，不接受路径——面板监听在所有网卡上，采信请求里的路径就等于开放任意文件写。`../escape.json`、`/etc/passwd`、`sub/dir.json`、不以 `.json` 结尾的，一律拒绝并说明理由。

文件格式是带版本号的 JSON，每个障碍物都自报形状：

```json
{
  "schema_version": 2,
  "obstacles": [
    {"shape": "box", "parent_frame": "universe", "name": "bench",
     "size_m": [0.4, 0.6, 0.05], "xyz_m": [0.3, 0.0, -0.05],
     "rpy_deg": [0.0, 0.0, 0.0], "enabled": true, "id": "0fa5f899"}
  ],
  "workspace_range_deg": [[-30.0, 95.0], [-120.0, 10.0]],
  "gravity": {"static_poses": 24, "gravity_validation_poses": 8,
              "gravity_probe_deg": 5.0,
              "gravity_probe_speeds_deg_s": [1.0, 3.0]}
}
```

**这个格式是为了以后加形状而设计的**，扩展点有三处：

- `shape` 字段人人都有。今天只认 `box`；将来加圆柱，就是在 `obstacles.py` 的 `SHAPES` 和 `_geometry_for` 里各加一处。
- **不认识的形状会被跳过并说明原因，而不是被当成盒子读进来**，也不会连累文件里其余的障碍物。今天的版本读到一个 `cylinder` 会记一条"obstacle dropped: unknown obstacle shape 'cylinder'"，其余照常载入。
- `schema_version` 只在旧版本读不动新文件时才会拒绝：版本号比本 build 高就报错，而不是猜。版本 1 是「只有障碍物」的旧格式，照常读得进来，下一次写盘时升为 2。

**包络坏了就整段丢掉，不半段生效。** 少了几个关节的包络，等于那几个关节可以随便出圈，比没有包络更糟；因此某一项不合法（上界不高于下界、非有限数）时整段作废并在笔记区说明，而不是逐关节修补。

同一份文件因此可以在不同机型、不同版本之间共享——绑在本机器人没有的坐标系上的盒子同样是跳过而非致命。

## 手动点动

面板的运行卡片里，每个关节各有一根滑块，松手时下发一次目标位姿。

它走的是**辨识时同一个 `FollowJointTrajectory` action**，因此没有第二条命令通路需要单独保证安全。三重约束：

- 行程取**辨识包络**（`reach_deg`）而非 URDF 极限——URDF 描述的是这条臂，不是它被螺栓固定在什么上面，而滑块是把臂开进周围环境最容易的方式。超出范围的请求被**夹紧**，不是拒绝。
- 每个目标位姿在下发前先过一次**障碍物碰撞筛查**，撞了就拒绝并说明撞到什么。
- 速度上限固定在 10 °/s，远低于辨识用的任何速度：操作员看的是机械臂，不是曲线，而且没有撤销。

点动会**独占**轨迹控制器，所以启用期间辨识按钮全部禁用，反之亦然。夹紧和筛查都在服务端做——请求不一定来自这个面板。

## 面板

启动方式见上面的[快速开始](#快速开始)。要点只有一条：面板与机器人分两步起，它只连接一台已经在跑的机器人，绝不代你启动任何东西。

`controller` 就是全部：轨迹 action 和 `controller_state` 话题都由它推导出来，受控关节列表则由控制器自己宣告，不用你来告诉它。若某个控制器的 action 不在自己的名字下，可以用 `follow_joint_trajectory_action` 给出完整路径，它优先生效。

其余每一个与具体机器人有关的量都是启动参数：遥测话题、各信号的接口名、附加话题、工作空间限位、最高扫掠速度、障碍物文件、端口。换一条臂——双臂中的另一条，或者另一台机器人——改的是参数，不是代码。

左半屏是机械臂，由正运动学绘制，用的正是碰撞检查和回归所用的同一套模型，因此画面不会与数学脱节。点击盒子可选中，拖拽 gizmo 可平移、旋转或改变尺寸，并可从下拉框中选择它固连的坐标系。

左下角是**实时信号**：位置、速度、电流（或力矩）、温度一并列出，只列这台机器人真的在发布的那几路——没人发布的量宁可不出现，也不画成一条零线冒充读数。只有位置带指示条，条的满量程就是这个关节自己的行程，所以它同时告诉你"还剩多少余量"，这是单个数字给不出的。曲线有三档：**收起**只留数值，**默认**在数值右侧并排，**展开**让曲线占满整块面板。画哪一路自己点，关节用颜色区分，点关节名即可把它从曲线里去掉。

这块面板不走状态轮询，而是从 `/api/telemetry` 按游标取增量：状态轮询要捎带碰撞检查和目录扫描，快不起来；而机器人以上百赫兹发布，每秒采它四次得到的不是一个更慢的信号，而是另一个被削掉了峰值的信号。桥接层留一小段帧缓存，面板按游标把没看过的全部取走，标题上那个 Hz 就是它实际收到的速率。

右半屏围绕真正能判断拟合好坏的那几张图组织：

- **拟合质量** —— 每个关节的训练、留出与验证误差并排显示。训练误差总能做小；只有第三根柱子才算数。
- **摩擦曲线** —— 拟合曲线叠加在用于拟合的样本点上。一条曲线单独看总是很有说服力，无论它错得多离谱；正是在这张图里，低速段失准的换向模型才会暴露出来。
- **残差-转速图** —— 这里出现的结构是未建模的物理，不是噪声。
- **激励充分性** —— 每个关节的条件数与其上限对照，让某个实验几乎没激励到的参数变得可见，而不只是被顺带报告一下。
- **参数** —— 按关节列出，并标注物理可行 / 不可行。

硬件运行只设一道闸门：必须先通过一次预演（rehearsal）。这道闸门不是走过场——预演会植入已知摩擦并要求把它重新辨识出来，正是它抓到了拟合悄悄返回全零的问题。归零（homing）不设闸门，因为它不做任何辨识，其全部意义就在于把一台已被拒绝使能的臂救回来。连接面板会点名任何因机器人未发布对应信号而关闭的保护。

### 两条臂都从当前位姿出发

**不要求任何一条臂先回零位。** 多臂工作单元里零位本身可能就是碰撞构型，「先归零再说」既不总是可行，也是一次没被筛查过的运动。

- **本面板驱动的那条臂**：巡回路径以它**当前所在位姿**为起点设计（`CampaignPlan.start_deg`），第一段转移——也是最长的一段——因此是被真实筛查过的。运行开始时硬件 plant 校验的是「臂在设计所假设的那个起点 ±1°」，而不是「臂在零位」。
- **本面板驱动不了的那条臂**：碰撞模型以它**当前的关节角**做归约（`buildReducedModel(full, lock, q_live)`），而不是零位。被锁的关节不是被删掉，它们的连杆仍在模型里，只是停在给定构型上——所以「给定哪个构型」直接决定筛查查的是不是这台机器人。实测：按零位钉住时左腕与它真实位置差 **618.6 mm**；按实测角钉住时差 **0.0 mm**。

代价是筛查只在「另一条臂没动」的前提下成立，所以有三重保障：

1. `screen_drift()` 持续比对实时读数与建模时的参考构型，超过 2° 就在重力卡片红字列出，并锁住「真机标定」；
2. 卡片上出现**重建碰撞筛查**按钮，点它按当前构型重建，并作废已有的预演解锁——之前放行的位形是按旧构型放行的；
3. 运行期间每到一个位形都复查一次，另一条臂中途被动了就中止。

起点位姿会量化到 0.5° 再进入设计和解锁签名，这样伺服静止时的编码器噪声不会反复作废解锁；而 plant 校验用的容差是 1°，比量化粗一档。

### JTC 移动速度

总体控制中的回零、点动，以及重力标定和多位姿保持检查，分别提供速度输入框，单位为
度/秒。默认值依次为 10、10、10、5；后端允许范围为 0.1 到当前安全配置的持续速度上限，
且不超过 60。设置在任务启动时固定，运行或规划期间不能编辑；点动速度需停止点动后再改。
这些输入仅控制 JTC 去往目标位姿的转场，不改变标定探针速度、电流保持参数或手动拖动的
超速停止阈值。实际移动时长仍受最短轨迹时长和控制器约束影响。

点到点 `HardwarePlant.move_to()` 在发送轨迹前，用新收到的位置确认至少 0.2 秒内
所有关节的跨度不超过 0.01°，样本接收年龄不超过 50 ms；两秒内无法确认则不发出目标。
确认后显式发送 `t=0` 的实测位置和零速度，再发送目标位置和零终点速度，时长由同一个
实测起点计算。这样 JTC 不会把静止时带量化噪声的速度读数作为样条初始速度，导致未要求
移动的关节也出现参考位移。已有停止、暂停和遥测保护在等待期间仍生效；动作重试必须重新
确认起点。恒速扫描和其他已明确给出速度的轨迹段不经过此改写，控制器全局参数及电流限值不变。

重力转场速度随标定设置保存，并参与预演授权签名，修改后必须重新预演才能真机标定。
保持检查的转场速度在执行时选择，不改变已规划的目标点；实际速度记录在该次执行摘要中。
回零、点动和保持速度不持久化，页面重新加载后恢复默认值。

### 面板统一状态接口

每次点击重力标定或多位姿保持的“规划”都会使用新的随机种子重新选点。重力规划的种子
在当前会话中固定给后续预演和真机执行，重新规划会取消上一轮预演授权，需重新预演。
保持计划记录随机种子和精确目标，执行不再抽样；若重规划抽到相同点集，会有限重试，
仍无不同结果时明确拒绝，不能把相同点集当成新的规划。

面板不再由各标定模式各写一套状态文字。服务中的 `publish_event()` 是唯一的操作员事件入口，原有 `note()` 只是它的兼容名称；`activity_payload()` 把当前任务、通用进度字典和最近事件合在一起，同时由 `/api/activity` 和 `/api/state.activity_feed` 提供。右侧 panel 分成两行：上半部分独立滚动，底部信息区不参与折叠并保留至少 160 px 高度。信息区第一行显示当前模式，第二行以 13 px 正文完整换行显示当前动态，内容较长时信息区继续自动增高；点击重力预演后立即显示已启动，运行时按后端最近一次回调依次显示设计位姿、采样、位姿完成和保存结果，而不是从累计字段猜测当前步骤，空闲后保持最后一条实际输出，不提供完整日志。状态轮询不会并发，旧响应不能倒退覆盖新进度。重力“已解锁”只作为被动摘要，在没有任何输出可显示时兜底，不会覆盖上一条运行结果。重力卡片原先显示的规划中、碰撞构型漂移、已失效/未解锁、位形数不足和执行位形进度全部由这一信息区显示，卡片内不再保留第二份状态文字。

3D 视图同样只认一个 `scene_activity`：其中包含模式、阶段、当前目标位姿、各阶段已完成位姿，以及是否有可自动快放的关节位姿序列。重力运行会在开始测试一个点之前发布 `target_pose`，因此琥珀色高亮始终指向当前目标，而不是刚完成的上一个点；完成全部探针数据的位姿改为灰青色，尚未执行的位姿仍保留阶段颜色。重力预演、完整预演和后续新增的轨迹类任务使用同一个协议；模式特有的代码只负责发布数据，不负责控制 canvas。

规划骨架由所选 controller 的关节集合及其在 URDF 中的父子关系确定，不依赖左右臂名称、
关节数量或全模型最后一个坐标帧。`ArmModel.skeleton_paths()` 输出每条受控链及其固定工具
末端；分叉 controller 的多条路径分别绘制，不在无父子关系的关节之间连线。
`/api/preview` 的每个位姿带有 `paths`；单路径仍提供兼容的 `points`，多路径时该字段为空。
多个路径共用一个位姿索引、高亮和播放步骤。切换 controller 会清空旧预览并更新 token；
配置模型与 controller 关节不一致时不生成预览。

真机重力标定支持暂停与恢复。`/api/pause` 先锁住后续 JTC goal；已经发出的 goal 不会被半途切断，而是在成功结束后的安全边界进入暂停。一个重力位姿是事务边界：该点所有速度、两个方向的探针都完成后，拟合样本和原始帧才算提交。若暂停落在点位中间，该点已经产生的 observation、phase 计数、峰值和 raw frame 会一起回滚；`/api/resume` 从同一位姿索引重新完整测试。暂停期间归零、点动和其他 campaign 都保持禁用，仍可用“停止”结束并保存此前完整提交的点位。

每次运行仍会写入独立目录中的 `report.html`。`/api/reports` 按运行模式给出最新报告的安全 `/runs/...` URL，重力标定卡片直接显示最新真机重力报告；路径最终仍由结果目录边界检查保护，前端不能指定任意文件。

### 独立重力补偿验证

重力卡片底部有两项**有人值守的验证**。它们不重新拟合模型，也不把 dashboard 变成另一个
电流发送器；dashboard 负责规划、互斥、进度和证据展示，位置运动复用标定的
`HardwarePlant`，电流保持仍由 `rm_control` 中的独立程序负责。

- **多位姿保持检查**：先点击“规划”，从已投用模型的历史观测中选择已知位姿，筛查指令电流硬限、
  当前场景、关节范围和逐段路径。规划只生成数据，不发出运动命令。可使用同一套 3D 位姿
  检查和动画控件查看。执行必须提供该计划的 ID；修改点数、模型、场景或起点后需重新规划。
  位置控制器按冻结目标逐点到位，再由 `hold_check` 保持并验证恢复。
  不再以“实测电流超出拟合摩擦检查带”为由拒绝启流。保持期间以启流前的固定基线位姿为参考，
  任意关节在任一方向的累计漂移 **大于 5°** 就停止并判为 FAIL；等于 5° 不触发漂移错误。
  指令电流硬限、实测峰值电流、速度、温度、电压、遥测新鲜度和停流确认等保护仍然有效。
  每点报告记录 `drift_limit_deg` 和各关节最大位移；恢复成功不会把漂移失败改成 PASS。
  到位确认使用 action 完成后新接收的遥测，最多等待 1 秒，仍保留原有 1° 到位容差，
  不再以完成前的缓存样本判定未到位，也不额外重发轨迹。
  保持子进程经本机 `/api/controller-state` 复用 Dashboard 的长期 ROS 列表连接；每次响应
  必须来自请求之后发出的新查询，并保留完整接口校验，不能用旧清单代替恢复证据。
  启流前还须确认控制器的命令接收代数实际增加至少 20 次，而非只统计本地发送次数。
  每次移动前通过 `scene_activity` 发布下一目标，完成后更新已完成索引；不新增 canvas 绘制分支。
  真机活动期间禁用预览巡游与手动选点，避免动画覆盖实际目标。
- **任意位置手动拖动**：原子切换到同一个 forward current controller，连续发送辨识出的
  重力补偿电流；会话不设固定时长，操作员点击停止后由共享 runtime 验证停流并恢复 JTC。

两个按钮都会先要求操作员确认已经扶住机械臂、工作区无障碍且急停可用；只有确认后前端才会
提交精确的 `I_AM_HOLDING_ARM_AND_ESTOP_READY`，服务端还会独立复核。两项验证与标定、归零、
点动共用同一个 activity 槽，不能并行。规划期间也禁止启动运动。顶部 **停止** 和验证区的
**停止** 使用同一路径：位置运动请求取消并确认 action 的终态；电流保持向所属进程组发送
`SIGINT`，由执行程序完成停流和恢复。若取消或停流恢复证据缺失，任务保留故障占用，
但历史错误本身不禁用按钮：检查失败且停流恢复已确认时，直接回到空闲，保留 FAIL 结果。
保持与手动拖动共用 `_finish_current_activity()` 收尾；停流未确认或 `restore_errors` 非空
都会保留故障占用。状态轮询中的 `_reconcile_hold_recovery()` 对两者被动复核，
无需为了清除旧错误再次重启 Dashboard。
复核要求控制器清单新鲜、所选臂 JTC 持有位置接口且 active、对应电流控制器 inactive，七关节
已使能、无故障、电流模式退出且停机确认；硬件遥测序号必须持续推进并新鲜，至少连续
1 秒速度不超过 1 度/秒且各关节累计移动不超过 0.5 度。已观察到的非终态 JTC 目标会
阻止释放；若此前位置运动未确认结束，还必须取得明确的目标终态证据。robot 重启后未
发布过目标状态时，只有全部历史 JTC 移动都已确认完成的保持任务可使用硬件状态复核。
复核通过只释放旧占用、保留失败记录并作废旧计划和预演授权，不自动续跑、切控制器或发运动命令。
关闭 dashboard 节点同样会请求停止并有界等待 worker 退出。

#### Explicit Position Recovery

When passive checks cannot prove position ownership, the operator can request
guarded recovery for the original selected arm after supporting it and checking
E-stop readiness. The [HTTP route](robot_parameter_identification/dashboard/http_server.py)
accepts this exact body, not an `ack` field or a model source:

```http
POST /api/recover-position
Content-Type: application/json

{"acknowledgement": "I_AM_HOLDING_ARM_AND_ESTOP_READY"}
```

`GET /api/state` exposes `current_recovery`: `required`, `available` and `running`
are booleans; `reason` is a string; `arm` is the original instance name when its
binding can be resolved, and is omitted otherwise. Availability requires the
previous owner to have exited, the original selection/live binding, fresh
telemetry and controller inventory, and exclusive inactive current ownership.
The request starts the installed `rm_control recover_position`; it needs no
calibration source, loads no gravity model, publishes no current and never
activates the current controller. See the [runtime usage](../../src/rm_control/runtime/README.md).

An accepted request is not proof of recovery. Its report is separate from the
original failed hold/drag result, and even a successful recovery child leaves the
activity pending until fresh passive proof passes. Reconciliation releases the
lock but never rewrites the failed result or resumes the old task.

直接电流期间，所选臂 JTC 暂时 inactive，但 `ros2_control_node`、`RMSystemHardware`、其他臂 JTC、
joint/F/T broadcaster、TF 和其它 ROS node 全部保持运行。Dashboard 的 sample/history 只有 ROS
topic callback 可以写入，不再接受 subprocess stdout 遥测；`/dynamic_joint_states` 因此持续提供
关节角度、电流、速度、温度、电压、使能和故障码。不得并行启动第二套 ros2_control。

每次验证写入 `identification_results/gravity_hold_test-.../` 或
`gravity_drag_test-.../`。保持检查另存 `plan.json`，记录精确目标、模型摘要及场景签名，
每点的 `hold_XX.json` 保留底层证据；拖动的 `status.json` 是运行中的原子状态，
`gravity_test_summary.json` 是最终 `PASS`、`FAIL` 或 `STOPPED` 结论。卡片会显示实际使用的
辨识模型 source、原因、退出码和输出路径，并通过受结果目录约束的 `/runs/...` 链接打开 summary。

**本次共享重构的完整真机验收：PENDING。** 两条已安装手臂都需要各自独立完成新的
24 个训练位姿 + 8 个验证位姿标定，以及各 50 个不同位姿的保持验收（10 组，每组 5 点）。
本次文档更新未运行程序或硬件测试；下面的旧报告保留其历史结论，不代表重构后的新一轮通过。

2026-09-16 的漂移标准真机验收已完成一轮连续 10 组 × 5 个位姿，每点保持 3 秒，
转场设置在 5–20°/秒范围随机；50 点均通过且无服务重试或恢复错误，最大关节漂移约 0.160°。
证据位于工作区 `test_data/hold_acceptance_20260916/attempt07/campaign.json`；此前失败轮次
单独保留，不拼接为成功，也不将旧模型检查带标准的结果当作本轮验收。
同日清理无效试验代码并重新部署后，再次完整执行 10 组 × 5 个不同位姿，首轮全部通过。
每点仍保持 3 秒，转场设置为 5.2–18.9°/秒；最大漂移约 0.148°，最大到位误差约 0.418°，
无服务重试或恢复错误。本轮独立证据位于工作区 `test_data/hold_reacceptance_20260916/REPORT.md`。
#### Per-instance calibration sources

Gravity validation uses `ArmIdentity.from_joint_names` to require exactly seven
ordered joints from one instance: `right_arm_joint1..7`, `left_arm_joint1..7`,
or, for example, `station_3_arm_joint1..7`. The existing capability line shows
the selected instance and any refusal reason. Ordinary identification remains generic.

Set `ros.gravity_test_source` in the separately managed system configuration, or
pass the existing `gravity_test_source` launch parameter. For example:

```yaml
ros:
  gravity_test_source: "/absolute/calibrations/{arm}/accepted-run"
```

Selecting `left` resolves only `/absolute/calibrations/left/accepted-run/result.json`;
`station_3` resolves only its own directory. An explicit literal directory or
`result.json` path remains literal. There is no newest-result discovery, right-arm
data copying, or cross-arm fallback. An empty source is allowed only for `right`,
using the same fixed commissioned backend default; the dashboard resolves,
validates and forwards that exact absolute source too.

Before enabling validation or launching a current child, the dashboard checks
that `result.json` is complete, has verdict `pass`, uses `ampere`, contains seven
finite fits, and matches the selected ordered joint names exactly. A matching
source alone does not enable the controls: the live description must declare
one owning seven-joint `RMSystemHardware` with acknowledged direct current,
`read_only=false`, position/current interfaces, and valid independent endpoint
and guard bindings. Existing ACKs and all safety thresholds remain in force.

#### Shared Core And Hardware Boundary

| Layer | Shared owner | Contract |
| --- | --- | --- |
| Gravity-current model | [gravity_current_model.py](robot_parameter_identification/gravity_current_model.py) | `load_identification()` and `gravity_current()` accept arbitrary joint counts with exact ordered names. |
| Hold candidates | [hold_candidates.py](robot_parameter_identification/hold_candidates.py) | Measured-pose loading, selection, current admissibility and callback-only transit screening. |
| RM instance binding | [arm_identity.py](robot_parameter_identification/arm_identity.py) | `ArmIdentity` names seven axes; `ArmBinding.from_description()` validates their exclusive RM hardware owner, interfaces, endpoint, guard and limits. |
| Current lifecycle | [current_session.py](../../src/rm_control/runtime/current_session.py) | One implementation for hold, manual drag and recovery-only execution. |
| Dashboard lifecycle | [service.py](robot_parameter_identification/dashboard/service.py) | Shared finalization and passive reconciliation for both hold and manual drag. |

The model and candidate helpers do not create ROS nodes, processes or hardware
connections. The loader reads top-level joint fits from `result.json`; prediction
sums only rigid columns in their stored order, preserving the existing numerical
calculation exactly. It adds no offset or friction and does not substitute the
paired predictor in `gravity_model.json`. Complete/pass validation is optional
for pure offline loading and mandatory at the production hardware boundary.

Hold planning builds `ArmModel` from the live URDF using the instance's
`<name>_arm_` prefix and checks its joint order against the calibration. It does
not reuse stored-source geometry across mounts. The service passes the live
`ArmBinding.maximum_command_a` into `build_hold_plan(..., maximum_command_a=...)`;
the planner copies that ordered vector into the frozen plan and its digest, and
execution checks predictions against it again. Measured continuous/peak limits
are not command caps. Source, description (including limits), selection and scene
changes require replanning. Existing cap values and their commissioning basis are
unchanged. Changing the selection discards sequence, position and timing evidence for
recovery. A failed hold remains tied to its original joints and trajectory action;
healthy evidence from another arm cannot release it.

Generic numeric helpers do not make arbitrary hardware plug-compatible. A new
instance needs its own explicit kinematics/mounts, hardware interfaces, exclusive
joints, endpoint, guard, profiles and accepted source; see the
[bringup extension process](../../src/robot_bringup/README.md). The current RM
runtime remains seven-axis and the public launch/URDF topology remains dual RM75.

Live acceptance requires the coordinated `rm_control` current-session version
that accepts the selected `--arm`, independently validates the explicit source,
and binds its model and endpoints to live `/robot_description`. These dashboard
changes do not commission another arm or establish a new full physical PASS.
Do not deploy the dashboard alone against a right-only current runtime.

正常的位置控制器与电流控制器切换不退出 ROS node。电流接管前必须预热新命令；退出时确认
电流已关闭、取得停流后的新 UDP 反馈，再用当前角度初始化位置控制，避免跳回旧目标。
命令超时或非法输入会撤销本次电流会话，保留首次故障原因，不以 200 Hz 重复刷屏；新命令
不能自动恢复故障会话。必须显式退出电流控制器，确认停流、反馈健康且手臂静止后，才能
重新切换。如果停流无法确认，则保持锁定，不允许用位置控制接管掩盖故障。

保持与拖动检查全程核对 `direct_current_active`、`direct_current_fault_latched`、
`direct_current_stop_confirmed`、控制器命令租约和实际 UDP 序号，并周期性读取机器人电流
使能状态。ROS 回调与同步读回分开运行；中途停流、冻结反馈或恢复失败都不能产生 `PASS`。
这些检查需要同时部署新版硬件插件、机器人描述和 runner；缺少状态接口时会在接管前拒绝。
手动拖动的超速停止默认 **120°/s**，范围 **1–120°/s**，与集成硬件的 **120°/s** 上限一致。
这是速度停止阈值，不是速度指令，也不提供摩擦补偿或碰撞自由的保证。

### 重力结果、原始证据与部署边界

重力真机运行的主结果不是方向性摩擦混合拟合，而是
`pair_averaged_empirical_gravity_effort_regressor`。每个位姿必须在每档探针速度下各有
一条正向和负向 observation；先分别平均正反向实测电流与其**实际位姿**对应的静态
Pinocchio 回归量，再跨速度平均，最后只用 `A_gravity` 位姿拟合静态刚体列加常数偏置，
并只在独立种子的 `D_validation` 位姿评分。这样奇对称摩擦在进入重力拟合之前就被抵消。
任何缺失、重复、异常或未配置的方向/速度标签都会使 `gravity_model.available=false`，
而不是静默丢弃后继续给出通过判定。

每次新重力运行保存：

- `raw_frames.csv`：真机 JTC 运动期间驱动器发布的全速原始帧；解析预演为 0 行。
- `observations.csv`：窗口拟合后实际进入计算的观测；训练和验证 phase 独立标注。
- `gravity_model.json`：命名的 Pinocchio 刚体列、每关节系数、偏置、外部验证误差、
  配对审计、URDF 哈希、驱动关节和被锁定另一臂构型。
- `robot_description.urdf`：生成该回归量的精确机器人描述，哈希必须与模型一致。
- `result.json` 与 `report.html`：完整结果以及面向人的双语审计报告。

报告首先标明 `real_hardware` 或 `analytic_rehearsal`，并给出 raw frame 数、发布时长、
约计频率、遥测话题、profile/障碍物/安全余量快照、Python/NumPy/Pinocchio 版本和核心
实现文件 SHA-256。原来的重力+摩擦逐行拟合仍保留，但明确降级为“方向性摩擦干扰诊断”；
仅有 1/3 °/s 两档速度时，Stribeck/load 项不能解释为已辨识的物理摩擦定律。

当前主产物是**按输出关节独立的电流（或力矩）域经验重力预测器**，不是共享的 SI 连杆
模型。重力专用数据不能恢复转动惯量；电流域又混入未知的逐关节电流-力矩比例，因此不能
从这里诚实地写出物理质量和质心，也不能直接改写 URDF。常规 `rm_impedance_control`
仍从 URDF 计算重力，尚无 `gravity_model.json` 加载器；报告中的“通过”只表示该经验预测器
通过独立位姿验证，不表示已经部署到控制器。

## 历史状态

以下是早期功能与测试计数记录，不是本次重构的测试结果或真机验收结论。

已工作并经过测试：

- 辨识核心 —— 回归、激励设计、四阶段实验流程、物理一致性检查。承接自一套已在 7 自由度臂上完成六次硬件运行的实现。
- 上文所述的 ROS 约定（`interfaces.py`）
- 障碍物场景与碰撞筛查（`obstacles.py`）
- 用于干跑的轻依赖预演被控对象（`plants/analytic.py`）
- 面板：节点、HTTP API、3D 视图、障碍物编辑、图表

`133 passed, 3 skipped`。

已在一台真实 7 自由度臂上验证：全部静态资源正常服务，36 个连杆变换与 16 个网格经由 mesh 代理渲染，障碍物可通过 HTTP 添加并参与碰撞筛查，遥测与 action server 均被正确检测到。

尚待完成：

- 仍有三个测试模块引用旧的 MuJoCo 被控对象因而被跳过，需要迁移到解析被控对象
- 还没有任何一次完整实验流程通过本面板端到端跑完

### 一个值得知道的告诫

预演被控对象与辨识器共用同一套刚体模型，因此一次干净的预演证明的是**流程**，而不是**物理**。它无法发现模型自身内部的错误。等第二套引擎装上之后，应当与之交叉验证；`plants.independent_engine_available()` 会告诉你是否已有可用的独立引擎。

## 运行测试

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD:$PWD/test:$PYTHONPATH" python3 -m pytest -q test
```

测试套件会自行构建一台 7 自由度臂，因此无需安装任何机器人描述文件。

## 许可证

Apache-2.0。