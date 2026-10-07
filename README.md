# 3D Vision：多视角桌面抓取规划与避碰

本文更新于 **2026-10-07**。当前项目面向杂乱桌面分拣：用户选择物体，系统从RGB-D观测中提取目标、生成三维抓取姿态、检查夹爪接近路径，并在遮挡导致观测不足时提示补拍。后续接入机械臂，完成抓取、搬运和放置。

**现阶段已完成离线核心与设备到手前六项准备实验，尚未完成真实相机验收或机械臂执行。** 夹爪固定按 **Robotiq-2F-140** 开发，机械臂型号尚未确定。现有抓取评分、几何检查通过数量均不能当作真实抓取成功率。

## 文档导航

| 想了解什么 | 阅读位置 |
|---|---|
| 完整架构、数据流和模块接口 | [项目README：项目架构](projects/graspgen_feasibility/README.md#architecture) |
| 源码、配置、实验、测试、数据分别放在哪里 | [项目README：文件架构](projects/graspgen_feasibility/README.md#files) |
| 运行入口、输入格式、参数及输出 | [项目README：运行方法](projects/graspgen_feasibility/README.md#run) |
| 拿到相机、机械臂、夹爪后的开发顺序和验收 | [项目README：实机开发方案](projects/graspgen_feasibility/README.md#hardware-plan) |
| 实验结论及尚未解决的问题 | [项目README：当前证据与限制](projects/graspgen_feasibility/README.md#evidence) |
| 下载代码后还缺哪些依赖和资源 | [归档与复现说明](projects/graspgen_feasibility/ARCHIVE.md) |

## 项目架构概览

```mermaid
flowchart LR
    A[已保存的RGB与深度或XYZ] --> B[目标分割]
    B --> C[目标点云与背景几何]
    C --> D[GraspGen生成候选]
    D --> E[评分与姿态检查]
    E --> F[终点碰撞和连续接近检查]
    F --> G[深度观测检查]
    G --> H[保存全部候选及排除原因]
    I[多个标定深度视角] --> J[融合与质量检查]
    J --> K[通过检查的融合结果]
    K --> L[复查已有候选]
    H --> L
```

当前主入口分析单目标、单视角；多视角融合和已有候选复查使用独立入口。融合目标重新生成抓取、真实相机采集、机械臂可达性、整臂轨迹规划和实际执行仍需开发。主算法内部使用米和明确的坐标系，设备接口必须处理单位、工具坐标和时间同步。

## 仓库文件架构

```text
3D_Vision/
├── README.md                         # 仓库入口与项目概览
├── .gitignore                        # 源码、归档产物与本地资源的边界
├── .gitattributes                    # 文本换行；实验归档按原始字节保存
└── projects/
    └── graspgen_feasibility/          # 当前项目
        ├── README.md                 # 详细架构、运行方法与实机开发方案
        ├── offline_core/             # 输入、分割、点云、抓取、避碰、观测、融合
        ├── experiments/              # 合成场景、独立几何参考、输入退化工具
        ├── tests/                    # 单元测试与已保存实验的核验工具
        ├── configs/                  # 默认模型路径和算法参数
        ├── analyze_scene.py          # 单场景分析入口
        ├── fuse_scene.py             # 多视角融合入口
        ├── recheck_fused_scene.py     # 用通过检查的融合结果复查固定候选
        ├── *_experiments.py          # 准备实验入口
        ├── plot_*.py                 # 从已保存结果生成图表
        ├── data/                     # 规范化样例、下载索引；部分原始数据仅本地
        ├── runs/                     # 实验输入、候选、性能、失败样例、图表
        ├── *_RESULTS.md / RESULTS.md  # 分阶段结果报告
        ├── DEVELOPMENT_PLAN.md       # 原始阶段计划与历史进展
        └── ARCHIVE.md                # 上传范围、资源来源及迁移说明
```

本地另有 `.venv/`、`third_party/`、`checkpoints/` 等运行资源，Git不包含这些目录。完整脚本职责和运行产物结构见[详细文件架构](projects/graspgen_feasibility/README.md#files)。

## 当前成果与限制

- 已完成连续接近与碰撞可靠性、遮挡与未观测空间、多视角融合、输入退化鲁棒性、点云规模压力等六项准备实验。
- 后两项共 **207组**：159组鲁棒性、48组压力测试，最大原始点数 **2,517,731**；保留51次预处理中止及6次细障碍采样丢失。
- 最近一次代码验证为 **50项单元测试通过**，结果归档核验通过。实验完成不代表全部实验设置成功。
- 当前仍存在目标选错但候选高分、粗体素与过滤参数不兼容、全量接近检查耗时长等问题；部分合成链路最终候选全部被拒绝，需要真实数据进一步分析。

结果依据：[连续接近与碰撞](projects/graspgen_feasibility/COLLISION_EXPERIMENTS.md)、[遮挡与融合](projects/graspgen_feasibility/VISIBILITY_FUSION_RESULTS.md)、[鲁棒性与规模压力](projects/graspgen_feasibility/ROBUSTNESS_STRESS_RESULTS.md)。

## 设备到手后的开发路线

第一轮实机目标：**固定相机、无遮挡桌面、一个不透明物体、固定放置区、低速执行，完成从拍照到抓放及结果确认的完整流程。**

| 阶段 | 交付内容 |
|---|---|
| 1. 设备接入 | 机械臂状态/运动/停止接口，2F-140开合与反馈，真实几何和工具坐标 |
| 2. 相机与标定 | 可回放RGB-D数据、尺度和对齐检查、相机到机器人坐标标定及独立验证 |
| 3. 指定位置抓放 | 使用人工指定位置验证接近、闭爪、抬起、放下及失败处理 |
| 4. 单视角视觉抓取 | 复用离线核心，新增可达性、整臂规划和执行流程 |
| 5. 障碍与执行复核 | 检查整臂及被抓物体的运动，处理过期观测、目标错误和故障 |
| 6. 多视角补拍 | 获取真实相机位姿、确认同一目标、融合检查及重新生成/复查候选 |
| 7. 实验与展示 | 固定场景对照、真实抓放成功率、失败分类、耗时和可回放界面 |

每阶段的输入、开发工作、验收证据及新增模块见[实机开发方案](projects/graspgen_feasibility/README.md#hardware-plan)。如果只有机械臂先到，先完成阶段1和指定位置抓放；相机先到则先做采集、尺度和离线视觉验证。

## 运行与归档范围

在已恢复模型与依赖的环境中，从项目目录运行归档官方样例：

```powershell
cd projects\graspgen_feasibility
.venv\Scripts\python.exe analyze_scene.py --input data\offline_official\scene.json --no-observation
```

此样例仅含世界XYZ，缺少深度观测所需的相机标定；上述命令用于几何分析，不驱动机械臂。真实标定深度输入默认启用观测检查。环境恢复及其他入口见[项目README](projects/graspgen_feasibility/README.md#run)。

Git保存自编源码、配置、测试、报告，以及当前项目 `runs/` 中除日志外的实验记录和 `data/offline_official/` 的规范化输入。课程资料、下载环境、第三方代码、原始下载数据集、权重和论文不上传。第三方资源遵循各自许可；实验原始字节及历史路径按[归档说明](projects/graspgen_feasibility/ARCHIVE.md)保留。
