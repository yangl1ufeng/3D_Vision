# 整臂避障与路径规划

2026-11-11：UR5＋社区 2F-140 新增独立规划场景、整臂/自身碰撞检查、被抓物体几何检查、RRT-Connect 路径搜索与执行复核。无需新增软件；继续使用 D 盘 `.venv-sim`。

## 启动

双击 `start_planning_simulation.cmd`：红色障碍位于抓取区与绿色放置区之间，机械臂抓起方块后绕开障碍完成放置。原有 Panda 和 UR5 无规划入口保留。

```powershell
Set-Location 'D:\新建文件夹\3D_Vision-main\projects\graspgen_feasibility'
.\.venv-sim\Scripts\python.exe run_simulation.py --robot ur5_robotiq140 --planning --scenario barrier --gui --keep-open
```

可选场景：

| 参数 | 期望行为 |
|---|---|
| `--scenario free` | 无额外障碍，检查并执行抓放 |
| `--scenario barrier` | 直接路径不满足要求，搜索绕行路径并执行 |
| `--scenario blocked_goal` | 预抓取位置被占据，在执行规划轨迹前拒绝 |
| `--scenario unreachable` | 目标位置超出工作空间，IK 验证失败后拒绝 |

拒绝场景退出码为 1，是预期测试结果；可以同时加 `--gui --keep-open` 查看。默认障碍场景使用边长 50 mm、质量 0.1 kg 的方块，其他物体尺寸与场景组合尚未做此规划阶段验收。

## 实现与数据边界

- `simulation/planning.py`：独立 DIRECT 客户端中的模型、碰撞查询、IK、双向 RRT-Connect 和路径复核。
- `simulation/planned_demo.py`：先规划后执行，保留物理接触抓取，监测实际执行。
- `verify_motion_planning.py`：四个场景验收，拒绝场景必须没有启动任何规划运动段。
- `recheck_motion_plans.py`：从保存的关节/物体快照恢复规划条件，进行更密采样复核。
- `tests/test_motion_planning.py`：整臂、目标接触、自碰撞、搬运物体、不可达和场景隔离测试。

规划查询只重置独立副本的关节和物体位姿，不改变执行世界。每段规划使用当时的机械臂关节角、手指姿态和物体位姿；原有夹爪几何与动力学假设见 [UR5_SIMULATION.md](UR5_SIMULATION.md)。

自碰撞检查覆盖非邻接连杆，明确排除同一刚体组、直接相邻关节组，以及夹爪同侧四连杆闭环的设计接触对。后者为内指节与外指节/内指组件的连接，并非屏蔽整个夹爪。底座与安装台的固定接触允许；其他连杆与安装台仍检查。

目标物体始终保留在碰撞场景中。闭爪及抓取后只允许两片指垫接触目标，不能通过移除整个目标绕过碰撞。夹爪预开口根据本例已知方块宽度增加 20 mm，而不是全张开；这仍是人工已知几何，不是视觉估计。

预抓取和搬运段先检查关节空间直连，不通过时搜索 RRT-Connect。接近、抬起、下放和撤离保持笛卡尔直线并逐点求 IK，不会为了避障擅自改变这些动作的形状；无法通过检查即拒绝。IK 检查末端位置和朝向，以及关节范围与分支跳变。目前使用单一 IK 种子，不穷举全部解。

抓起后的规划副本根据实测 TCP→物体变换移动物体碰撞几何，检查其与环境及机器人非指垫部位的干涉。**只有副本使用此几何关系，物理场景没有固定物体到夹爪的约束。** 下放高度依据实际夹持偏移及物体最低点计算，先停在桌面上方约 2 mm，再释放；允许起落阶段的桌面支撑接触，不允许穿透。

## 检查精度与失败分类

| 项目 | 设置 |
|---|---|
| 规划净空 | 4 mm |
| 搜索边采样 | 最大单关节增量 0.04 rad |
| 执行前复核 | 最大单关节增量 0.005 rad |
| 保存结果加密复核 | 0.0025 rad，同一模型与碰撞引擎 |
| 物理执行检查 | 每 4 个 240 Hz 步进，即约 16.7 ms |
| 执行拒绝阈值 | 检测到超过 1 mm 的非允许穿透即中止 |
| 搜索 | 随机种子 7，最多 2000 轮，名义搜索预算 15 秒 |

这些是离散采样，不是连续碰撞检测或对任意薄障碍的安全证明。执行监测是在采样点发现异常后停止仿真，也不构成实机安全控制。搜索超出预算只表示这次没有找到路径，不能推断数学上不存在路径。

日志区分 `invalid_start`、`invalid_goal`、`ik_unreachable`、`ik_branch_discontinuity`、`search_budget_exhausted`、`fine_validation_rejected`、`execution_collision` 等原因。初始环境和夹爪预整形先完成，再进入受检的规划运动。

## 验收记录

四场景汇总：`runs/planning_validation/20261111_172044_222047/summary.json`。

| 场景 | 结果 |
|---|---|
| 无障碍 | 完整抓放通过；1113 次执行几何检查，无禁止碰撞 |
| 障碍 | 两段改用 RRT-Connect，完整抓放通过；1249 次执行检查，无禁止碰撞 |
| 目标被占据 | `invalid_goal`；没有启动规划轨迹 |
| 超出工作范围 | `ik_unreachable`；没有启动规划轨迹 |

障碍场景预抓取搜索 4 轮、搬运搜索 8 轮。首次直连拒绝点的距离小于 4 mm，因此日志中的“拒绝”不必都代表已经穿透。加密复核确认：预抓取直连违反净空要求；搬运直连确实与障碍相交。绕行后六段路径均通过 0.0025 rad 复核，结果在该场景 `dense_recheck.json` 中。这是同一碰撞模型上的加密复核，不是独立物理引擎验证。

窗口演示通过：`runs/simulation/20261111_172153_085609/`，结果与无窗口障碍场景一致。最终方块中心约为 `(0.51706, 0.22113, 0.02499)` m，位于绿色区域。

新增模型测试 **7 项通过**；旧三种尺寸 UR5 抓放、夹空、Panda 共五场景回归全部通过，记录在 `runs/simulation_validation/20261111_172202_862089/`。

调试期间的拒绝结果仍保留，包括原固定下放高度使物体几何穿入桌面、全张开夹爪接近障碍时净空不足等；没有修改碰撞阈值来放行这些路径。

```powershell
.\.venv-sim\Scripts\python.exe -m unittest tests.test_motion_planning -v
.\.venv-sim\Scripts\python.exe verify_motion_planning.py
.\.venv-sim\Scripts\python.exe recheck_motion_plans.py --run runs\planning_validation\20261111_172044_222047\barrier
```

每次运行保存 `planning.json`（阶段、拒绝原因、关节路径、规划快照、被抓物体变换、参数和检查数量）、`result.json`、`robot_model.json`、轨迹 CSV 和 PNG。汇总保存源码哈希。当前帧为静态人工构建场景，不支持执行途中环境移动后实时重规划。

## 下一阶段

对接 GraspGen 夹爪坐标、接近轴及实际开口，增加多候选姿态的 IK/碰撞筛选；随后接入虚拟 RGB-D 与 Gemini 2 采集。本阶段没有重新运行或恢复视觉模型环境，也没有验证真实相机输入。
