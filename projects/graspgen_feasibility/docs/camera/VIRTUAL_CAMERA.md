# 虚拟 RGB-D 输入：已完成与待完成

2026-11-12：`capture_simulation.py` 渲染 RGB、深度，输出原有 `offline_core.scene_io.load_scene` 可读取的场景。两个场景均已完成真实 GraspGen 推理，各生成 16 个有效候选；尚未实现候选驱动机械臂的视觉抓取闭环。

```powershell
Set-Location 'D:\新建文件夹\3D_Vision-main\projects\graspgen_feasibility'
.\.venv-sim\Scripts\python.exe capture_simulation.py
.\.venv-sim\Scripts\python.exe capture_simulation.py --cube-xy 0.6 -0.1
```

每次输出独立的 `runs/virtual_camera/<时间>/`：RGB PNG、米制深度 NPY、有效像素掩码、目标掩码、scene.json 和 capture_validation.json。深度是光轴 z；OpenGL 深度缓冲经过非线性转换。内参包含像素中心约定，外参显式转换到 OpenCV 相机坐标（右、下、前）。无返回像素设为无效，不把远裁面当作观测表面。

目标掩码来自仿真器物体 ID，属于真值分割，仅用来隔离相机和抓取接口问题，不能宣传为 SAM2 或自动分割结果。单视角点云也不是物体的完整模型。物体位姿只在采集场景初始化时设定，没有执行抓取。

## 验证

- `20261111_183228_541935`：物体位于约 (0.5, 0)，765 个目标像素。
- `20261111_183317_409604`：物体位于约 (0.6, -0.1)，911 个目标像素。
- 两场景均用现有 `load_scene` 成功加载 480×640 RGB 与标定深度。
- 桌面 z 的 P95 误差约 1.10 mm；目标表面 P95 误差分别约 1.25 和 1.14 mm，均通过 3 mm 阈值。
- 误差与独立已知桌面平面和方块 AABB 比较，而不是仅验证投影/反投影公式互逆。当前只有简单方块和桌面验收，不能推断任意表面精度。

## GraspGen 恢复现状

官方源码已下载到 D 盘 `third_party/GraspGen-2dd8852e1be60f5f9d277fafcc621835cdf59110/`；提交号保存在仓库根 `.downloads/graspgen-inspection/commit.txt`。兼容修改应用于工作副本 `third_party/GraspGen`，保留原始源码。

视觉环境 `.venv-vision` 已准备好：复用系统 PyTorch 2.12.1+cu126，使用 D 盘 CUDA 12.6 编译工具成功安装 PointNet2。独立仿真环境仍使用 `.venv-sim`。本机编译脚本为 `build_pointnet_current.cmd`，须经英文路径联接执行，不能直接套用旧机器的 `build_pointnet.cmd`。

两份 2F-140 权重共 510,864,342 字节及 YAML 已下载，权重 SHA-256 验证通过。官方场景和本页两个虚拟场景均实际运行成功；虚拟场景模型推理约 0.62 秒。详见 [当前环境、实测记录、复跑入口与限制](../setup/GRASPGEN_RESTORED.md)。

官方夹爪配置沿 x 方向闭合、z 方向接近；当前仿真夹爪在其基座坐标内沿 y 方向闭合，TCP 沿 z 偏移 0.2 m。需要进一步用官方碰撞网格与仿真模型验证旋转、偏移、指垫和开口差异，暂不写入未经验证的姿态转换。

下一顺序：验证夹爪变换和几何 → 筛选虚拟 RGB-D 生成的候选 → 将可达且无碰撞的候选交给规划器 → 验证物理抓取。当前相机数据已接到抓取生成器，规划演示仍为独立链路。
