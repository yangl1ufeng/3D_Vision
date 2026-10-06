# 三维桌面抓取：离线核心、避碰与多视角准备实验

已将 **SAM2 分割 → 三维点云处理 → 官方 GraspGen 抓取生成 → 场景碰撞检查** 拆成模块。一个命令分析一个目标并保存全部候选，包含被排除的候选。

2026-10-05新增默认启用的**连续直线接近检查**，并完成碰撞采样可靠性对照。结果、局限与复跑步骤见 [COLLISION_EXPERIMENTS.md](COLLISION_EXPERIMENTS.md)。阶段1报告保留为历史终点筛选基线。

随后已完成**遮挡/未知空间处理及多视角融合预实验**，详见 [VISIBILITY_FUSION_RESULTS.md](VISIBILITY_FUSION_RESULTS.md)。默认新增深度观测门控：需带内外参的深度图，观测不足会拒绝并提示补拍。官方XYZ样例缺少该标定，若要复现此前几何筛选请加 `--no-observation`。

**准备实验5、6也已完成并核验**：159组噪声/缺点/掩码/真实SAM2框提示试验，以及48组融合点云压力测试，最大原始点数2,517,731。51次预处理中止和6次细障碍采样丢失均保留。50项单元测试通过。结果、显存/耗时表和失败示例见 [ROBUSTNESS_STRESS_RESULTS.md](ROBUSTNESS_STRESS_RESULTS.md)；正式入口尚未自动解决分割选错目标的问题。

## 一条命令运行

在已有环境的本机 PowerShell 中，从任意目录运行：

```powershell
& "D:\3D_Vision\projects\graspgen_feasibility\.venv\Scripts\python.exe" "D:\3D_Vision\projects\graspgen_feasibility\analyze_scene.py" --official-sample --no-observation
```

控制台打印输出目录。默认输出到项目的 `runs/offline/时间戳/`。上述官方XYZ几何演示显式关闭缺少标定依据的观测门控；真实深度输入默认开启。首次运行把已下载的官方场景转换为规范输入，随后复用转换结果；不会自动联网下载。默认生成 100 个候选，评分阈值 0.8。

从项目目录运行的其他例子：

```powershell
.venv\Scripts\python.exe analyze_scene.py --official-sample --candidates 200 --seed 1000
.venv\Scripts\python.exe analyze_scene.py --official-sample --no-approach
.venv\Scripts\python.exe analyze_scene.py --input D:\my_scene\scene.json --output runs\offline\my_scene
```

`--output` 必须指向不存在或空的目录，防止覆盖历史实验。失败时退出码为 2，并在输出目录保存 `failure.json`。参数解析或输出路径错误会直接报错。

## 模块职责与接口

| 模块 | 输入 | 输出 |
|---|---|---|
| `scene_io.py` / `geometry.py` | 场景 JSON、RGB、深度或 XYZ、坐标变换 | `Scene`：逐像素世界 XYZ（米）、有效像素、目标提示 |
| `segmentation.py` | `Scene` 的 RGB＋目标框，或显式目标掩码 | `Segmentation`：二值掩码、模型质量分、后端信息 |
| `pointcloud.py` | `Scene`＋`Segmentation` | `Clouds`：目标原始/过滤点云、完整/抽样背景、离群点 |
| `grasp.py` | 过滤后的目标点云、候选数、种子 | `Grasps`：N×4×4 姿态、N 个评分、模型信息 |
| `collision.py` | 背景点云、夹爪表面、候选姿态 | 每个候选的检查状态、保留标记、排除原因 |
| `approach.py` | 完整观测背景/目标点云、夹爪凸部件、候选姿态 | 连续直线接近状态、预抓取姿态、首次相交进度及来源 |
| `observation.py` | 标定深度视角、连续接近几何 | 空闲/表面/未知/冲突，补拍提示和整段包络可观测性 |
| `fusion.py` / `fuse_scene.py` | 同一静态目标的多份深度与掩码、相机初始位姿 | 配准与质量拒绝、融合点云及逐点来源、可用深度证据 |
| `recheck_fused_scene.py` | 已有候选运行目录、通过检查的融合结果 | 保留候选身份/评分，应用来源相机修正，复查完整接近与多视角观测 |
| `pipeline.py` / `profiling.py` / `export.py` | 配置与上述模块 | 一次分析的所有结果、来源和性能记录 |

数据结构定义在 `offline_core/types.py`，入口为 `analyze_scene.py`，默认配置在 `configs/offline_default.json`。配置中的模型/代码路径相对于本项目目录；输入素材路径相对于场景 JSON 的目录。`resolved_config.json` 会保存解析后的绝对模型路径和实际参数。

## 坐标系和单位

- 算法内部点云、夹爪几何、姿态平移及距离阈值全部以 **米** 为单位。
- 输入单位必须声明为 `m`、`mm`，或 `raw`＋正的 `meters_per_unit`。声明矛盾会报错。
- 相机坐标采用 OpenCV：x 向右、y 向下、z 向前；深度为光轴 z 深度。
- `T_world_camera` 为合法的 4×4 刚体变换，平移以米计，使用齐次列向量：`p_world = T_world_camera @ p_camera`。
- 输出命名为 `T_world_graspgen_gripper`：把官方夹爪模型局部坐标变换到场景世界坐标。世界坐标名称由 `frames.world` 指定，可以把首帧相机作为世界坐标，此时需显式给出单位矩阵。
- GraspGen 已把目标中心加回生成姿态，本适配器不重复加中心；按官方推理入口补齐齐次矩阵的右下角为 1。
- 输出夹爪坐标不是机器人 TCP。将来接实物仍需要夹爪/TCP 变换及相机/机器人标定。

## 自定义 RGB-D 输入

将 `rgb.png`、`depth.npy` 和 `scene.json` 放在同一目录。RGB 必须是彩色图，深度为 H×W 的光轴深度，且已经与 RGB 对齐。当前接口接受去畸变后的针孔图像；不负责相机驱动、对齐或去畸变。

下面的内参及目标框只是格式示例，需要替换为实际设备数据。单位矩阵代表把相机坐标作为本场景世界坐标。

```json
{
  "schema_version": 1,
  "scene_id": "desk_001",
  "rgb": "rgb.png",
  "geometry": {
    "kind": "depth",
    "path": "depth.npy",
    "unit": "mm",
    "aligned_to_rgb": true,
    "depth_convention": "z",
    "pinhole_rectified": true,
    "intrinsics": [[600, 0, 640], [0, 600, 360], [0, 0, 1]]
  },
  "frames": {
    "world": "first_camera",
    "camera_convention": "opencv",
    "T_world_camera": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  },
  "target": {
    "id": "bowl",
    "box_xyxy": [642, 373, 764, 477]
  }
}
```

深度可以是 `.npy` 或保留深度数值的 PNG，不能使用渲染成彩色的深度图。深度 0、负值和非有限值作为无效像素。内参必须对应当前 RGB/对齐深度分辨率。

如果已有逐像素点云，将 `geometry` 改为以下内容：

```json
{
  "kind": "xyz",
  "path": "xyz.npy",
  "unit": "m",
  "frame": "world",
  "aligned_to_rgb": true,
  "valid_mask": "valid.npy"
}
```

XYZ 数组必须是 H×W×3。`frame=world` 已在世界坐标，可省略相机内参与变换，世界 z 可以为负；`frame=camera` 仍需 `camera_convention=opencv` 和 `T_world_camera`。可选 `valid_mask` 为 H×W 二值数组，用于排除 SDK 输出的无效 XYZ。

`target` 也可以写为 `{"id":"bowl","mask":"mask.png"}`，使用显式的 H×W 二值掩码（0/1 或 0/255），此时后端记录为 `provided_mask`，不会加载 SAM2。如果同时给框和掩码，掩码优先。框坐标使用 `[x0,y0,x1,y1]`，必须落在图像内。

官方样例输入转换保存在 `data/offline_official/scene.json`。它使用官方已有的世界 XYZ；目标框通过官方目标点与场景像素的精确对应获得，模拟用户选择目标。推理的目标点云来自 SAM2 掩码，不使用官方目标点云代替分割结果。该样例的目标框来源不能作为独立的检测评测结果。

## 默认算法参数

- SAM2 Hiera Large，BF16，关闭可选 CUDA 后处理；从多个掩码中选择模型预测质量最高的一个。
- 目标点上限 8192，至少 100 点。过滤使用官方代码：K=20，平均邻居 **L1 距离**严格小于 0.014 米；抓取模块不再重复过滤。
- Robotiq-2F-140 官方 PointNet++ 预训练权重，FP32，10 步扩散；一次生成指定数量，保留全部候选，不自动重试或截断为最佳若干个。
- 碰撞背景最多抽样 8192 点，夹爪表面抽样 2000 点，阈值 0.02 米，批量 4。完整背景同时保存以供审查。
- 点云抽样用独立随机数生成器 `seed`；夹爪表面用 `seed+1`；抓取生成前重设 `seed`。本机同参数两次运行的复现检查见 `STAGE1_RESULTS.md`。
- 连续接近默认开启：官方局部+Z方向，预抓取后退100mm，凸部件面偏移余量2mm。使用完整目标和背景观测点，不模拟闭指；未知空间由独立观测模块处理。配置组为 `approach`；`--no-approach` 复现历史范围。旧配置缺少该组时默认关闭，保持兼容。
- 观测门控默认开启：8mm扫掠体素外包络，2mm深度容差；整个包络必须获得有效空闲证据。配置组为 `observation`；`--no-observation` 关闭该门控，旧配置缺少该组默认关闭。`--no-approach` 在未另指定观测选项时同时关闭观测门控。

每个候选的排除原因支持 `low_score`、`invalid_pose`、`invalid_score`、`scene_collision`、`no_scene_geometry`。同一个候选可以有多个原因。通过评分及姿态检查后才做碰撞检查；缺少环境点云时标为未知并排除。保留候选按评分降序排名，同分时按候选 ID 排序。

开启接近检查后新增 `final_background_collision`、`final_target_collision`、`approach_background_collision`、`approach_target_collision`、`approach_missing_geometry`。JSON候选中的 `approach` 保存详细状态及预抓取变换；CSV新增接近状态、首次相交进度及来源；NPZ新增 `approach_state` 与 `pregrasp_poses_world`，未检查的预抓取矩阵用NaN占位。计算耗时包含新增的 `approach_check` 阶段。

## 输出文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 场景/目标、数量、原因统计、排名、参数、限制和性能汇总 |
| `candidates.json` / `candidates.csv` | 全部候选 ID、4×4 姿态、评分、排名、保留状态及排除原因 |
| `scene_arrays.npz` | RGB、掩码、有效像素、目标/背景点云、姿态、评分、判定、夹爪表面抽样 |
| `performance.json` | 各阶段耗时、CUDA 已分配/保留显存峰值、NVML 整卡采样峰值 |
| `segmentation_overlay.png` | 目标框与分割叠加图 |
| `input_manifest.json` | 原始输入声明、素材绝对路径、SHA256、输入/输出单位 |
| `resolved_config.json` | 本次实际使用的配置 |
| `provenance.json` | 权重 SHA256、源码版本/修改摘要、核心源码 SHA256、软件和 GPU 信息 |

NPZ 使用普通数组，可通过 `np.load(path, allow_pickle=False)` 读取。原始非有限数据会在 NPZ 中保留；JSON 对无效数值用 `null` 表示，并以原因字段解释，CSV 留空。

性能统计中的 `compute_stage_seconds` 包括分割、点云、抓取生成、评分过滤、终点碰撞及已启用的接近/观测检查，排除模型加载、素材读取/转换、来源校验及文件导出。壁钟时间从入口参数解析完成后计时，不含 Python 启动及模块导入。显存统计包含常驻模型；NVML 包含桌面和其他进程，不等于本进程显存，且 20ms 采样可能漏掉短峰值。无法读取 NVML 时记为 `null`。

## 验证与环境

从本项目目录运行：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_offline_contract.py -v
.venv\Scripts\python.exe -m tests.verify_collision_geometry
.venv\Scripts\python.exe -m tests.verify_saved_runs runs\offline\stage1_validation runs\offline\stage1_repeat --report runs\offline\stage1_verification.json
```

第一组检查单位、相机/世界坐标、缺少标定、无效掩码、空背景及候选记录；第二项使用实际 GPU 碰撞函数验证已知旋转/平移和障碍距离；第三项核对两次完整运行的 JSON/CSV/NPZ 及同种子复现结果。

当前本机环境已可运行：Windows、Python 3.12.6、PyTorch 2.11.0+cu128、RTX 4070 Laptop 8GB，官方 PointNet++ CUDA 扩展已编译。初始环境准备脚本为 `download_assets.py`、`prepare_windows.py`、`build_pointnet.cmd` 和 `requirements-local.txt`，硬件可行性历史结果见 `RESULTS.md`。当前虚拟环境复用了本机系统依赖；复制代码到新电脑不能直接获得同样的运行环境。

SAM2 默认复用旧项目的代码及权重；GraspGen 的下载代码/权重位于本项目 `third_party/`、`checkpoints/`。这些路径可在配置中更改。课程材料、环境、数据、权重和 `runs/` 均按根目录 `.gitignore` 排除，源代码、默认配置、说明和检查工具可以进入 GitHub。

主抓取入口支持单目标单视角、终点/连续接近筛选及标定深度观测门控。另有多视角融合入口 `fuse_scene.py` 和固定候选复查入口 `recheck_fused_scene.py`；拒绝融合不能进入复查。观测实验9项符合预期，26组融合实验中原视角组合的4组修正因几何退化未成功，其余22组符合预期；45项单元测试通过。融合复查两条链路已跑通，当前仍无最终放行候选。真实噪声、闭指、机械臂可达性和实物稳定性未验证。覆盖率、评分和保留数量不能解释为真实抓取成功率。

```powershell
# 从同一静态场景通过检查的融合结果，复查先前候选；目录和复跑说明见实验报告
.venv\Scripts\python.exe recheck_fused_scene.py --run runs/offline/observation_depth_validation_20261005 --fusion runs/fusion/known_v2_cli_20261005
```
