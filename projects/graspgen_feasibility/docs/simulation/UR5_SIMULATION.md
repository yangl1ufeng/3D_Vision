# UR5＋Robotiq 2F-140 仿真适配

**后续进展：** 已加入可选的整臂避障规划模式，见 [路径规划说明](MOTION_PLANNING.md)，入口为 `start_planning_simulation.cmd`。本页记录不带规划的模型与抓放基线。

2026-11-11：完成组合模型加载、夹爪联动、实际开口标定、显式 TCP 和人工指定姿态的物理抓放。所有下载与新增依赖仍位于 D 盘；沿用 `.venv-sim`，不需要安装新软件。

## 启动与复现

双击 `start_ur5_simulation.cmd`。原 `start_simulation.cmd` 仍运行 Panda 基线。

```powershell
Set-Location 'D:\新建文件夹\3D_Vision-main\projects\graspgen_feasibility'
.\.venv-sim\Scripts\python.exe run_simulation.py --robot ur5_robotiq140 --gui --keep-open

# 无窗口；90 mm 方块
.\.venv-sim\Scripts\python.exe run_simulation.py --robot ur5_robotiq140 --cube-size 0.09

# 故意保持夹爪张开，预期失败、退出码 1
.\.venv-sim\Scripts\python.exe run_simulation.py --robot ur5_robotiq140 --no-grasp

# 模型测试与五场景验收
.\.venv-sim\Scripts\python.exe -m unittest tests.test_ur5_simulation -v
.\.venv-sim\Scripts\python.exe verify_simulation_baselines.py
```

模型未下载时先执行 `powershell -ExecutionPolicy Bypass -File .\download_simulation_models.ps1`。该脚本只为本次进程设置执行策略，不修改系统策略；下载固定提交的源码 ZIP，保留 LICENSE 和 README。

当前模型来源：[ElectronicElephant/pybullet_ur5_robotiq](https://github.com/ElectronicElephant/pybullet_ur5_robotiq)，固定提交 `37f7d66d3196490618bf89ffe2b49dcb7e07aec3`。控制器为本项目实现，不运行下载仓库的 Python 示例。上游代码许可与模型来源声明保留在下载目录；模型引用链见上游 README，不能把仓库代码许可直接解释为所有 CAD 模型的统一许可。

## 模型问题与修正

上游控制代码将 2F-140 沿用了 2F-85 的开口换算，因此本项目不使用该公式。模型原始文件保留，`simulation/ur5_robotiq.py` 从原文件生成 `ur5_robotiq_140_project.urdf`：

| 项目 | 处理 |
|---|---|
| 六个机械臂关节 | 按名称解析，使用 URDF 力矩上限和关节范围 |
| 五个联动关节 | 从 URDF mimic 元素读取正负倍率，显式创建齿轮联动约束 |
| 两个附加指垫 | 原始质量各 1 kg、惯量张量退化；改为各 0.01 kg 和均匀盒体惯量，保持碰撞尺寸 |
| `ee_link` | 作为坐标标记显式设为零质量，避免 PyBullet 缺省赋予 1 kg |
| 工具中心 | 新增 `project_grasp_tcp`，相对夹爪基座沿 +z 偏移 0.2 m，方向一致 |
| 实际开口 | 对主关节 0～0.7 rad 采样 141 个位姿，从指垫内表面距离建立宽度→角度映射 |

**测得附加指垫模型最大净开口为 0.1285346 m。** 接口拒绝 0.14 m 的请求，不把名义 140 mm 当作已验证开口。指垫质量、摩擦系数和驱动力矩是仿真假设，没有真实硬件标定。

主驱动关节力矩上限 2 N·m；齿轮约束上限 50；每个指垫摩擦系数 1.0。这不是 Panda 的单指 20 N 力控制，也不是对真实夹持力的推算。闭合与搬运不修改物体位姿，不创建夹爪到物体的固定约束。

每次 UR5 运行生成 `robot_model.json`，记录来源提交、原始 URDF/网格/许可文件哈希、派生 URDF 哈希、模型改动、实际开口与 TCP 变换。下载目录被 Git 忽略，可通过固定提交重新取得。

## 坐标与运动边界

世界坐标桌面 z=0；夹爪基座 +z 指向指尖，TCP 使用相同轴向，位于基座前方 0.2 m。它是固定任务参考点，**不是随开合变化的指垫中心**。本例 50 mm 方块采用 TCP 高度 45 mm、朝向 RPY=(π,0,π)，避免指垫碰桌和手腕关节范围边界。其他边长使用 TCP 高度“边长/2＋20 mm”。

GraspGen 的 `graspgen_gripper` 与该 TCP 的固定变换尚未验证；`robot_model.json` 和结果明确记录 `graspgen_frame_mapping_verified=false`。不能直接把 GraspGen 输出矩阵作为本 TCP 目标。

控制器在关节范围内选取最接近当前角度的等价解；若单次指令与当前位置差超过 0.25 rad，则停止并要求重新规划，避免在 ±π 边界直接取模导致大幅转动。每段运动检查末端位置误差 ≤15 mm 和姿态误差 ≤3°。搬运中物体抬升低于 80 mm 会判定掉落。

当前是笛卡尔路点＋逆运动学，没有实现障碍路径搜索、整臂自碰撞验收或被抓物体的规划碰撞几何。桌面穿透监测、关节范围与跳变检查不等于完整路径规划。

## 已完成验收

模型测试 **6 项通过**：实际开口与非法输入、动态开合达到目标宽度、TCP 固定变换、指垫正惯量、±π 跳变拒绝、等价关节角转换且不改输入。

最终五场景结果位于 `runs/simulation_validation/20261111_165927_590028/summary.json`，之前一轮 `20261111_165455_918831` 也保留：

| 场景 | 结果 | 保持段最低抬升 |
|---|---|---|
| UR5，30 mm 方块 | 抓起、保持 3 秒、放置成功 | 162.1 mm |
| UR5，50 mm 方块 | 抓起、保持 3 秒、放置成功 | 153.4 mm |
| UR5，90 mm 方块 | 抓起、保持 3 秒、放置成功 | 135.0 mm |
| UR5，夹爪不闭合 | 如期在 hold 阶段判失败 | 约 0 |
| Panda 旧基线 | 回归通过 | 172.0 mm |

上述三个 UR5 正常场景保持期间双指接触覆盖率均为 100%，未检测到机器人与桌面超过 1 mm 的穿透。每组包含 `result.json`、`robot_model.json`、`trajectory.csv` 和 PNG。汇总保存测试时的源码哈希；后续说明文字调整可能改变源码文件哈希，不修改历史结果。

窗口完整抓放已通过，记录位于 `runs/simulation/20261111_165855_679320/`，最终位置与 50 mm 无窗口场景一致。GUI 按累计仿真时间控制播放，避免 Windows 逐步休眠误差累计；此次完整运行约 25 秒。此前窗口在 settle 阶段断连的记录 `20261111_165537_787164` 保留且判为失败，断连原因未单独确认。后续若窗口中途关闭，会保存明确的中断原因，不把未验收的任务记为成功。

早期失败保留在 `runs/simulation/20261111_165201_028980/`：能抬起，但手腕跨越 ±π 时大幅转动，搬运掉落。后续加入跳变拒绝，选择避开边界的对称抓取朝向后通过。该失败不能计作完整抓放成功。

这些是固定合成场景的功能验收，不是真实抓取成功率或随机场景泛化能力。

## 接下来

1. 增加完整机械臂的自碰撞、环境碰撞检查和关节空间避障规划，单独验证“末端可达但整臂路径不可行”的拒绝案例。
2. 确认 GraspGen 的夹爪网格、接近轴与 TCP 变换，匹配实际开口与允许接触区域。
3. 接入虚拟 RGB-D 以及现有点云输入格式，再接 Gemini 2 的真实采集与桌面坐标标定。
