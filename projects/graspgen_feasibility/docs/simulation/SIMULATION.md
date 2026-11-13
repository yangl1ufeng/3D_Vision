# 仿真起步：接触驱动的单物体抓放

**当前最新入口：** `start_planning_simulation.cmd` 展示 UR5 抓放与障碍绕行，验收和限制见 [整臂规划](MOTION_PLANNING.md)。

**后续进展：UR5＋2F-140 已接入并通过三种尺寸方块抓放。** 新入口为 `start_ur5_simulation.cmd`，模型修正、实际开口、TCP 和验收记录见 [UR5 仿真说明](UR5_SIMULATION.md)。本页继续保留 Panda 起步基线的操作与历史记录。

本阶段建立独立 PyBullet 环境，验证物理抓放、图像和日志输出。使用 PyBullet 随包提供的 Franka Panda 及其夹爪，**仅作为环境基线**。这不是 UR5 / Robotiq 2F-140 适配，不调用 SAM2、GraspGen 或真实相机，不改变现有视觉核心与历史实验。

## 本机目录与启动

所有新增安装、缓存和编译临时文件位于 D 盘当前仓库内：

- `.downloads/pip/`：下载与构建缓存。
- `.downloads/tmp/`：安装临时文件。
- `projects/graspgen_feasibility/.venv-sim/`：独立仿真环境（复用已安装 Python 3.10 的基础解释器）。
- `projects/graspgen_feasibility/runs/simulation/`：每次运行单独保存结果。

双击本目录 `start_simulation.cmd`，或在 PowerShell 执行：

```powershell
Set-Location 'D:\新建文件夹\3D_Vision-main\projects\graspgen_feasibility'
.\.venv-sim\Scripts\python.exe run_simulation.py --gui --keep-open
```

无窗口验证与反向验证：

```powershell
.\.venv-sim\Scripts\python.exe run_simulation.py
.\.venv-sim\Scripts\python.exe run_simulation.py --no-grasp
```

第一条成功时退出码为 0；第二条故意保持夹爪张开，预期在 hold 阶段失败，退出码为 1。每次输出新目录，显式 `--output` 也不允许覆盖已有目录。关闭窗口或 Ctrl+C 退出保留的 GUI。

## 场景和判据

- 单位：米、千克、秒、弧度；桌面高度为世界坐标 z=0。
- 方块边长 0.05 m、质量 0.1 kg；初始中心 (0.5, 0, 0.025)。
- 放置区中心 (0.5, 0.22)，绿色方形为无碰撞标记。
- 240 Hz 物理步进，150 次求解器迭代；物体摩擦系数 0.8，手指摩擦系数 1.0，单指力上限 20 N。
- 预抓取、接近、闭爪、抬起、保持、搬运、下放、松爪、撤离逐段执行。
- 抓取成功：整个 3 秒保持段抬升至少 0.1 m，且至少 95% 步进存在双指有效接触。
- 完整任务成功：抓取成功、放置后物体中心落在目标中心 ±0.04 m 内，高度与桌面接触相符，速度足够低，且未检测到超过 1 mm 的机器人活动连杆与桌面穿透。
- 物体仅在初始化时设置位姿；没有把物体固定到手上的约束，没有抓住后直接改物体位置。

输出包括 `result.json`、每 12 个物理步记录一次的 `trajectory.csv`，以及初始、保持、成功放置的 PNG。失败可能没有全部画面，须以 JSON 状态为准。反向验证不能计入正常抓取成功率。

## 本机验证记录（2026-11-11）

Python 3.10.11 / PyBullet 3.2.7 已安装，`pip check` 通过。官方源码 SHA-256 已核对，本机编译的 Windows wheel 保存在 D 盘下载缓存中。GUI 已实际启动并执行，识别到本机 RTX 4060 Laptop GPU；旧视觉文档中的硬件环境不代表本次环境。

| 验证 | 结果目录（runs/simulation 下） | 结果 |
|---|---|---|
| 无窗口完整抓放 | `20261111_163653_617884` | 成功；最低抬升 0.17199 m，保持 3 秒，双指接触覆盖 100% |
| 张开夹爪反向验证 | `20261111_163535_475524` | 如期在 hold 阶段失败；物体未抬起、双指接触为 0 |
| 窗口完整抓放 | `20261111_163716_214978` | 成功；最终物体位置、保持高度与无窗口结果一致 |
| 早期窗口失败（保留） | `20261111_163613_007155` | lift 阶段 TCP 偏差 0.4328 m，已判失败 |

早期窗口失败后增加显式关闭实时步进、禁用鼠标物理拖拽及键盘物理快捷键；随后窗口运行通过。未单独隔离早期失败根因，不能据此断定一定由鼠标输入导致。运行演示时可调整视角，但不能拖动物体；GUI 中的 3 秒保持指仿真时间，低帧率或系统调度可能使实际等待更长。

正常演示最终物体中心约为 (0.49997, 0.23407, 0.02499) m，落在绿色区域内。以上均是固定方块场景的功能验证，不作为跨物体抓取成功率。

## 当前边界

人工给定位置和方向，使用逆运动学跟踪笛卡尔路点；没有障碍搜索、完整自碰撞验收或真实传感器。后续 UR5 适配已为两个模型增加每段末端位置和姿态误差检查。桌面接触监测不能等同于整臂路径规划。图像为 TinyRenderer 渲染，GUI 使用 OpenGL。

Panda 的抓放结果不能代表 Robotiq 2F-140 或 GraspGen 的效果。当前接触参数仅是基线配置，没有真实物体参数标定，重复同一确定性场景也不是泛化成功率。

## 下一步

1. 验证并接入机械臂＋2F-140 的完整模型、连接件、手指联动、碰撞几何和 TCP。保持当前基线可回放。
2. 增加候选姿态到机器人 TCP 的转换、关节限位、自碰撞及整臂避障规划。
3. 用虚拟相机生成带内外参的 RGB-D，适配既有 `scene.json` 输入；恢复独立视觉环境并接入 GraspGen。不要把 Panda 抓取直接当成 2F-140 候选执行。
4. 接 Gemini 2、桌面标定板、简单物体尺寸/位姿拟合；最后加入多视角和障碍实验。

## 重新安装（已有 Python 3.10 与 MSVC C++ 工具）

在仓库根目录执行。PyBullet 3.2.7 的此 Windows/Python 组合需要从官方源码构建；下载、编译可能需要几分钟。无需安装付费仿真软件。

```powershell
New-Item -ItemType Directory -Force .downloads\pip,.downloads\tmp | Out-Null
$env:TEMP = Join-Path $PWD '.downloads\tmp'
$env:TMP = $env:TEMP
python -m venv projects\graspgen_feasibility\.venv-sim
& projects\graspgen_feasibility\.venv-sim\Scripts\python.exe -m pip install --cache-dir .downloads\pip -r projects\graspgen_feasibility\requirements-sim.txt
```

资源来源：[PyBullet 官方发行包](https://pypi.org/project/pybullet/3.2.7/)、[Bullet 官方仓库](https://github.com/bulletphysics/bullet3)、[随包 Panda 模型](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/franka_panda)。第三方模型保留包内许可证。
