# Gemini 2 重建场景接入 UR5 仿真

## 已完成

读取两视角实测深度拟合的 `geometry.json`，把桌面、纸盒、水杯碰撞体放入 PyBullet 中，与 UR5＋Robotiq 2F-140 同场景显示。几何来自 [多视角重建](../camera/MULTIVIEW_CUP_BOX.md)，桌面上看不到的表面由简化几何补齐，并非完整实物扫描。

本次选择水杯作为目标，纸盒作为障碍物。机械臂和相机之间没有实测外参，因此采用可复现的**虚拟摆放**：桌面坐标整体平移 `(0.65, -0.40, 0)` 米到机器人底座坐标。这个数值只用于仿真布局，不代表真实相机位置或真实机器人安装位置。

计算结果：水杯估计直径约 65.4 毫米，预留 20 毫米开爪余量后约 85.4 毫米，低于本仿真夹爪实测最大开口 128.5 毫米。纸盒估计短边约 127.4 毫米，加同样余量后约 147.4 毫米，因此不作为第一次夹取目标。

从 UR5 初始姿态到水杯上方的路径通过独立碰撞场景的规划与细采样验证；再向下接近杯身的采样路径也通过。规划器检查了整臂自碰撞以及桌面、纸盒、水杯碰撞。此结果说明**当前虚拟摆放和假定抓取方向下可接近**；尚未证明杯子能被稳定夹住或抬起。水杯目前是实心圆柱碰撞近似，未表示杯口、内腔、把手。TCP 朝向为预设的俯抓姿态，尚未与 GraspGen 输出坐标完成对接。

## 查看和复跑

从本项目目录运行：

```powershell
.\.venv-sim\Scripts\python.exe -m simulation.reconstructed_scene --output runs/reconstructed_ur5/cup_box_first_plan
```

输出位于 [cup_box_first_plan](../../runs/reconstructed_ur5/cup_box_first_plan/result.json)：`result.json` 保存输入、虚拟坐标变换、夹爪宽度检查、关节路径、停止原因和限制；同目录的 `scene_initial.png`、`scene_pregrasp.png`、`scene_approach.png` 依次显示初始场景、到达水杯上方和接近杯身时的姿态。此次运行仅生成路径与预览图，不驱动动态物理抓取。

回归检查：

```powershell
.\.venv-sim\Scripts\python.exe -m unittest tests.test_reconstructed_scene tests.test_motion_planning -v
```

## 完整抓放动画

基于同一份重建几何，已完成一次**仿真中的接触驱动抓放**：移动到杯子上方、接近、闭爪、抬升、保持、搬运、下降、开爪、退出和静置。水杯是有质量和摩擦的动态圆柱，运行中没有把杯子绑定到夹爪或直接移动杯子。

[观看保留实测点云外观的 GIF 动画](../../runs/reconstructed_ur5/cup_pick_place_measured/cup_pick_place.gif)；[读取逐阶段结果](../../runs/reconstructed_ur5/cup_pick_place_measured/result.json)。本次水杯最低抬升约 0.120 米，保持阶段双指接触比例为 1.0，最终落在绿色放置区。动画中的绿区是预设仿真放置目标，不是相机观测到的物体。

### 点云外观与物理碰撞

原始重建成果是 [Gemini 2 彩色点云](../../runs/gemini_multiview/20261112_170625_237128/desk_box_cup_measured.ply)，共 123,935 个实测点，而不是封闭的三角网格。新版动画从中选取杯子约 8,595 点、纸盒约 16,355 点：纸盒点保持原位，杯子点随其真实仿真刚体姿态一起移动。画面隐藏了之前的蓝色圆柱与橙色方盒，桌面仍用平面显示以避开原始桌面深度边缘的噪点。

PyBullet 的抓取、接触和碰撞仍使用由同一深度数据拟合出的隐藏圆柱和方盒。点云只覆盖相机看见的表面，直接用它作动态刚体会留下洞和不稳定接触；动画没有把未拍到的杯背面、杯内腔或纸盒背面补成“真实表面”。需要更接近实物的碰撞体时，应先补拍视角，再生成可检验的封闭网格并重新验证抓放。

复跑：

```powershell
.\.venv-sim\Scripts\python.exe -m simulation.reconstructed_animation --output runs/reconstructed_ur5/my_animation
```

若只想查看以前的简化几何外观，可加 `--visual-mode proxy`；默认显示实测点云外观。

动画演示的是重建几何的物理近似；杯子的内腔、材质和把手都没有被真实测量。这里的俯抓姿态经过仿真试验确定，尚不是 GraspGen 输出。若沿用 GraspGen 姿态，还需明确其夹爪坐标到本模型 TCP 的变换。
