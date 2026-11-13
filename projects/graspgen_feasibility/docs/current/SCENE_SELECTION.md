# 选择场景与抓取目标

统一入口现在读取场景配置，导出所选物体点云、生成候选、检查路径并执行仿真。目标名称不再限于 `cup`；可以使用几何文件中的任意物体 ID。当前支持方盒 `box` 和实心圆柱近似 `solid_cylinder_proxy`，其他形状会明确拒绝，需要另外增加对应的碰撞体支持。

## 使用

在本项目目录运行。默认配置是 [measured_table.json](../../configs/measured_table.json)，包含已采集的纸盒和水杯。

```powershell
# 查看配置中有哪些目标
.\.venv-sim\Scripts\python.exe run_measured_grasp.py --list-targets

# 相同场景，分别选择水杯或纸盒
.\.venv-sim\Scripts\python.exe run_measured_grasp.py --target cup
.\.venv-sim\Scripts\python.exe run_measured_grasp.py --target box

# 换成另一份场景配置；只准备数据时可加 --prepare-only
.\.venv-sim\Scripts\python.exe run_measured_grasp.py --scene configs/my_scene.json --target my_object
```

双击 `start_measured_grasp.cmd` 默认运行配置中指定的目标；该启动文件也支持传入相同参数。每次默认创建带微秒时间戳的新目录 `runs/measured_grasp/<时间>/`。手动指定 `--output` 时要求目录不存在，以免覆盖历史结果。

## 场景配置

复制 `configs/measured_table.json` 后修改以下内容。文件路径相对于配置文件所在目录解析，不依赖启动时所在文件夹。

| 字段 | 作用 |
|---|---|
| `geometry_path` | 重建几何 JSON，包括 `objects` 中每个物体的形状、尺寸、桌面坐标中心和方向 |
| `pointcloud_path` | 对应的实测彩色 PLY，使用当前重建导出的二进制格式；坐标需与几何文件一致 |
| `target` | 默认抓取物体 ID，可用 `--target` 覆盖 |
| `table_to_robot_translation_m` | 整个重建场景在虚拟机器人坐标里的平移，单位米 |
| `drop_xy_robot_m` | 机器人坐标下的放置区中心 |
| `table`、`pedestal` | 仿真桌面、机座支撑的中心与半尺寸；规划与执行使用同一份参数 |
| `target_physics` | 所选目标的仿真假设，包括质量、摩擦 |

任意数量的非目标物体都会作为静态障碍物加入场景，所选目标为动态刚体。纸盒可成为目标，水杯可成为障碍物。实际颜色和点云外观仍来自 PLY；有转角的方盒会先转换到自身局部坐标，再随刚体运动，避免目标切换后点云旋转错位。

此阶段场景配置只支持桌面坐标到机器人坐标的平移；物体自身支持绕竖直轴的转角。不支持任意相机外参或自动目标识别。场景几何和点云需要先由采集、配准、建模流程产生，不能只替换照片就直接运行。桌面及物体尺寸应与实际拟合结果相符。

## 结果与输入校验

每次结果目录包括：

- `run_status.json`：所选场景和目标、已完成阶段、运行状态或失败原因。
- `input/scene_snapshot.json`：本次使用的完整场景参数与几何快照。
- `input/target_robot_points.npy` 和 `input_provenance.json`：目标测量点、点数、数据来源哈希。
- `candidates/`：真实模型推理的全部候选与评分。
- `planning/screening.json`：每个候选的规划判定与排除原因。
- `execution/task.json`：自动尝试顺序与总体结果；每个尝试目录内另存 `result.json` 和 `pick_place.gif`。

场景、目标、点云内容、虚拟摆放和物理参数共同生成本次输入标识，贯穿推理、筛选和执行。目标或配置变化后误用旧候选会在动作开始前被拒绝。执行使用保存的快照；原配置后来修改不会静默替换这次已检查的几何。

`success` 表示完整抓放通过；`no_valid_candidate` 表示本次候选均未通过接近规划；`attempts_exhausted` 表示通过筛选的候选在后续步骤中未完成任务；`failed` 表示某个软件处理阶段出错。找不到抓法时仍保存记录，不能解释为已证明这个物体永远不可抓。

## 检查

```powershell
.\.venv-sim\Scripts\python.exe -m unittest tests.test_scene_selection tests.test_graspgen_bridge tests.test_motion_planning -v
```

测试覆盖改名物体、旋转目标的点云变换、纸盒成为动态目标、其余物体成为障碍物、规划与执行桌面一致、错误目标/旧场景候选拒绝，以及无可用候选时仍保存报告。

## 本次实际验证

15 项相关测试通过，并从统一入口分别完成两次真实模型推理和仿真流程。各阶段的目标与场景标识已交叉核对。

| 目标 | 提取点数 | 模型候选 | 通过接近筛选 | 实际尝试 | 结果 |
|---|---:|---:|---:|---:|---|
| 水杯 | 8,595 | 64 | 9 | 3 | 候选 33 完整抓放成功，保持抬升约 14 厘米 |
| 纸盒 | 16,355 | 64 | 1 | 1 | 候选 9 未夹起，保持阶段双指接触比例为 0，保留失败记录 |

[水杯任务记录](../../runs/measured_grasp/generalized_cup_validation/execution/task.json) · [纸盒任务记录](../../runs/measured_grasp/generalized_box_validation/execution/task.json) · [核对汇总](../../runs/measured_grasp/scene_selection_validation.json)

两次运行使用同一份真实采集数据，验证了目标切换和原水杯流程回归；改名物体与更换坐标平移在测试中验证。尚未采集新的桌面布局，因此不能把本次结果称为已通过多场景真实数据验证。纸盒抓取失败是实际保存的结果，本步没有为追求成功改动其测量尺寸或候选姿态。
