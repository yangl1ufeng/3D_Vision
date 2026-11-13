# 本机 GraspGen 恢复与虚拟相机推理

2026-11-12：已在本机 RTX 4060 Laptop GPU 上完成官方样例与两个虚拟相机场景的真实 GraspGen 推理。此阶段接通“深度图 → 目标点云 → 抓取候选”，尚未把候选交给仿真机械臂执行。

## 使用

双击 `start_graspgen_check.cmd`，默认处理已保存的第一个虚拟相机场景。结果写入新的 `runs/graspgen_restore/<时间>/`，完成后窗口保留输出。也可将新的 `scene.json` 拖到这个启动文件上。

```powershell
Set-Location 'D:\新建文件夹\3D_Vision-main\projects\graspgen_feasibility'
# 官方样例，使用样例内自带的目标点云
.\.venv-vision\Scripts\python.exe smoke_graspgen.py
# 虚拟 RGB-D，使用场景中显式提供的目标掩码
.\.venv-vision\Scripts\python.exe smoke_graspgen.py --scene runs\virtual_camera\20261111_183317_409604\scene.json
```

每次保存 `result.json`（来源、权重哈希、姿态有效性、评分、耗时）及 `candidates.npz`（姿态、评分、模型输入点云）。点云和位置单位为米；姿态将 GraspGen 夹爪局部列向量变换到输入点云坐标系。虚拟相机输入经过标定转换，因此输出在 `simulation_world`。官方样例则保留官方点云坐标系，不能直接当成仿真世界坐标。

`poses_world_m` 是 NPZ 的通用字段名，具体坐标系以 `result.json` 和输入来源为准。`input_provenance.grasp_generator_used=false` 描述采集输入时尚未生成抓取；顶层 `grasp_generator_used=true` 描述本次推理。

## 实测证据

所有测试固定种子 1000，生成 16 个候选，检查齐次变换合法、旋转合法、评分有限且在 [0,1] 内。未进行抓取成功率测量。

| 输入 | 目标点数 | 有效姿态 | 分数 ≥ 0.8 | 模型推理时间 | 记录目录 |
|---|---:|---:|---:|---:|---|
| 官方真实场景附带的目标点云 | 4881 | 16/16 | 2 | 约 1.15 秒 | `runs/graspgen_restore/20261112_144920_492418` |
| 虚拟方块约 (0.5, 0) | 765 | 16/16 | 13 | 约 0.62 秒 | `runs/graspgen_restore/20261112_145033_785036` |
| 虚拟方块约 (0.6, -0.1) | 911 | 16/16 | 13 | 约 0.62 秒 | `runs/graspgen_restore/20261112_145154_170925` |
| 一键入口复跑第一个虚拟场景 | 765 | 16/16 | 13 | 约 0.66 秒 | `runs/graspgen_restore/20261112_145501_904446` |

最终一键入口实际执行返回 0，复跑评分与首轮一致；最终记录还包含 RGB、深度及掩码文件哈希。`prepare_windows.py` 连续执行两次，编译源文件内容哈希均未变化，确认本机兼容修改已完整保存且可重复应用。

耗时不包含进程启动、导入和模型加载；前两条旧版本计时还包含很短的结果保存时间。模型分数仅用于候选排名，不能解释为物理抓取成功率。运行期间可见未安装 pyrender 的提示，本阶段输入为现成点云，未使用其网格渲染功能。

两份官方权重共 510,864,342 字节，每次执行均对照 `data/model_files.json` 检查 SHA-256：

- 生成器：`fe8497108e39d8fc50be06cd7df22a2f680e0495e713d24c3104616f291e00dd`
- 判别器：`52f68d3cd3decb71499e084fdb88808d174419a72cd9ef38f821083a4409895d`

官方独立物体样例文件尚未下载成功；本次官方测试实际使用已下载真实场景 JSON 内的 `object_info.pc`，没有依赖缺失样例。

## 环境与构建记录

- `.venv-vision`：Python 3.10.11，通过 `--system-site-packages` 复用本机已有 PyTorch 2.12.1+cu126 / torchvision 0.27.1+cu126，其余主要包安装在 D 盘虚拟环境。当前依赖见 `requirements-vision-current.txt`；旧 `requirements-local.txt` 保留为历史环境。
- CUDA 编译工具：仓库根 `envs/graspgen-cuda/Library`，nvcc 12.6.85；缓存及构建临时文件在仓库根 `.downloads`。
- Visual Studio 2022 Professional，MSVC 14.44.35207；编译 GPU 架构 8.9，对应本机 RTX 4060。
- 官方源码固定为 `2dd8852e1be60f5f9d277fafcc621835cdf59110`，原始目录 `third_party/GraspGen-<提交号>` 保留；兼容修改仅应用到工作副本 `third_party/GraspGen`。
- `prepare_windows.py` 保存惰性 PTV3 导入、现代 CUDA 头文件/流接口、Windows 编译选项、编译器输出解码和禁用 Ninja 的修改。不改变所选 PointNet++ 模型数学。
- PointNet2 扩展已编译安装为 3.0.0。中文路径触发编译失败，因此创建了指向本仓库的目录联接 `D:\3D_Vision_build`；它不是另一份源码或安装位置。

本机重新构建：先在项目目录运行 `.\.venv-vision\Scripts\python.exe prepare_windows.py`，然后从英文联接入口执行：

```powershell
& D:\3D_Vision_build\projects\graspgen_feasibility\build_pointnet_current.cmd
```

构建脚本针对本机的 VS 安装路径与 GPU 架构。换机器需要重新核对这两项。成功构建日志为仓库根 `.downloads/pointnet-build-ascii.log`。

环境并非完全隔离：`pip check` 仍报告从系统继承的 SAHI 0.12.1 要求 OpenCV ≥4.12，而当前 GraspGen 环境使用 OpenCV 4.10.0.84 / NumPy 1.26.4。本次流程不导入 SAHI，三次推理实际通过；不能把这个环境称为全部依赖无冲突，也不应直接用它开展 SAHI 工作。未修改系统 Torch 或系统 SAHI。

官方依赖 Meshcat 在导入时创建本地 ZMQ 上下文，受限运行权限曾触发 `Permission denied`。给予该测试所需本机通信权限后通过；本次未启动浏览器可视化服务，未控制机器人。

## 下一接口与边界

目前相机目标掩码仍来自仿真物体 ID，不是 SAM2 自动分割；单视角点云不是完整三维模型。推理有效性也不代表桌面碰撞、整臂可达性和抓取稳定性合格。

下一步验证 GraspGen 夹爪坐标与仿真 2F-140 基座/TCP 的旋转和位移，并核对几何差异（官方沿 x 闭合，仿真基座内沿 y 闭合；仿真加指垫后净开口约 128.5 mm）。通过后再筛选候选、规划接近/搬运轨迹并执行物理抓取。当前 `smoke_graspgen.py` 明确只生成候选，报告 `robot_execution=false`。
