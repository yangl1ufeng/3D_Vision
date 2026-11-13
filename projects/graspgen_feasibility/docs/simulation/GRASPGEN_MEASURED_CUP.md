# 实测水杯点云 → GraspGen → UR5 仿真抓放

入口已升级为可选择场景和目标，最新配置说明见 [场景与目标选择](../current/SCENE_SELECTION.md)。以下最初的单场景结果保留；当前分阶段命令已同步更新。

2026-11-12 已在保存的 Gemini 2 双视角桌面数据上接通模型候选与机械臂执行。抓取位置和方向来自 GraspGen，执行前只做坐标转换，没有把候选改回手工俯抓姿态。

## 运行结果

- 从原始彩色点云提取 8,595 个杯子测量点，固定种子 1000 采样 8,192 点输入模型；生成 64 个候选。
- 9 个候选通过当前 IK、整臂碰撞及接近路径检查；55 个被拒绝。IK 求解失败表示此求解尝试失败，不是物理上绝对不可达的证明。
- 按模型评分依次执行，第一次完整成功后停止。候选 6 夹起成功、下降放置时因手腕距纸盒不足 4 毫米而中止；候选 45 在动态到达预抓位置后，重新求解接近姿态失败；候选 33 完成抓放。
- 这轮尝试共 3 次：2 次通过夹起保持检查，1 次完成整套抓放。这是单场景演示，不是统计抓取成功率。

[完整任务记录](../../runs/reconstructed_ur5/graspgen_cup_verified/automatic_run/task.json) · [全部候选的筛选记录](../../runs/reconstructed_ur5/graspgen_cup_verified/screening.json) · [自动抓取动画](../../runs/reconstructed_ur5/graspgen_cup_verified/automatic_run/candidate_33/cup_pick_place.gif)

动画保留纸盒和水杯的实测彩色点云外观，碰撞仍使用隐藏的方盒和实心圆柱。执行中的水杯受重力、接触和摩擦影响，没有固定到夹爪上；规划器中的附着几何仅用于预判搬运碰撞。

## 坐标对应依据

官方 GraspGen 2F-140 碰撞网格和配置以夹爪基座为原点，沿 x 轴闭合、沿 +z 接近。当前仿真模型沿 y 轴闭合，TCP 在基座 +z 方向 0.2 米处。因此使用：

```text
T_world_tcp = T_world_graspgen_base @ T_graspgen_base_sim_tcp

T_graspgen_base_sim_tcp =
[[ 0,  1, 0, 0   ],
 [-1,  0, 0, 0   ],
 [ 0,  0, 1, 0.2 ],
 [ 0,  0, 0, 1   ]]
```

程序直接读取官方 OBJ 中两个指垫组件，与仿真正运动学计算出的张开指垫中心对比，误差约 0.054 毫米；另检查实际 TCP 的基座偏移。报告包含官方碰撞网格哈希。这验证坐标与张开指垫地标，不代表两模型所有关节状态和物理参数完全相同。旋转矩阵转换为四元数也有极端角度回归测试。

## 一键复跑

双击 `start_measured_grasp.cmd`，或在本项目目录运行：

```powershell
.\.venv-sim\Scripts\python.exe run_measured_grasp.py
```

入口依次导出带来源哈希的点云、在 `.venv-vision` 中真实推理、在 `.venv-sim` 中筛选并按评分执行，结果进入新的 `runs/measured_grasp/<时间>/`。原来的失败尝试不会被删除。各阶段也可独立运行：

```powershell
.\.venv-sim\Scripts\python.exe prepare_measured_grasps.py --scene configs/measured_table.json --target cup --output runs/my_grasp/input
.\.venv-vision\Scripts\python.exe smoke_graspgen.py --points runs/my_grasp/input/target_robot_points.npy --provenance runs/my_grasp/input/input_provenance.json --candidates 64 --output runs/my_grasp/candidates
.\.venv-sim\Scripts\python.exe -m simulation.graspgen_bridge runs/my_grasp/candidates/candidates.npz --scene runs/my_grasp/input/scene_snapshot.json --output runs/my_grasp/planning
.\.venv-sim\Scripts\python.exe -m simulation.graspgen_task runs/my_grasp/planning/screening.json --output runs/my_grasp/execution
```

已有真实推理来自 `runs/graspgen_restore/measured_cup_64_run`，权重 SHA-256 验证通过；仅推理耗时约 2.93 秒，不含加载和仿真。输入导出记录位于 `runs/reconstructed_ur5/graspgen_cup_verified/input`，其 NPY 哈希可与推理报告核对。

## 当前边界

目标是已选定的水杯，先前的图像区域由人工选择，当前通过几何范围提取点云；尚未实现任意场景自动识别目标。桌面到机器人底座仍采用虚拟摆放平移，未做实机标定。杯子质量 0.09 千克、摩擦等参数属于仿真假设，杯口和背面未完整重建。每个候选在复位后的独立仿真中尝试，失败后切换候选不等于具备现实机器人故障恢复能力。模型评分只用于排序。
