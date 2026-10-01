# 3D Vision — GOLDILOCS 独立复现

3D 视觉课程项目：根据 GOLDILOCS 论文，使用 MASt3R 重建几何、跨视角渲染、反向深度过滤和 SAM2 掩码跟踪，识别物体变化。

**当前是独立实现，原论文主实验尚未完整复现，性能优势尚未验证。** 小规模诊断结果不能视为完整基准成绩；实际进度与结果以项目报告为准。

## 目录

主要项目位于 [`projects/goldilocs_reproduction`](projects/goldilocs_reproduction)：

- `goldilocs_local/`：几何、分割、跟踪、变化分类及评测实现。
- `configs/`：明确记录参数与假设的实验配置。
- `scripts/`：数据准备、实验控制、评测及诊断脚本。
- `tests/`：几何、分类、缓存和跟踪语义测试。
- Markdown 报告：实现说明、结果、失败分析与修复记录。

入口文档：[项目说明](projects/goldilocs_reproduction/README.md)、[实现说明](projects/goldilocs_reproduction/IMPLEMENTATION.md)、[第二轮修复](projects/goldilocs_reproduction/FIX_ROUND2_RC3D.md)。

## 仓库范围

本仓库保留自己编写的源码、配置、测试和文本报告。课程文件夹 `理论课/`、`实验课/` 及 `Shares/` 副本，以及 Python/Conda 环境、第三方下载、数据集、权重、论文文件、完整推理输出和审计缓存均通过 `.gitignore` 排除；这些本地文件不会被删除。

报告中的 `data/`、`runs/`、`audit/` 等路径指向本地实验材料，不包含在仓库中。克隆后需要单独准备依赖、模型及数据；修订推理还依赖先前生成的几何缓存。

安装和运行说明见 [SETUP.md](SETUP.md)。

## 来源

- [GOLDILOCS 论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html)
- [GOLDILOCS 作者仓库](https://github.com/almogfri/GOLDILOCS)
- [MASt3R](https://github.com/naver/mast3r)
- [SAM2](https://github.com/facebookresearch/sam2)
- [RC-3D / CYWS-3D](https://github.com/ragavsachdeva/CYWS-3D)

第三方代码、权重和数据的许可按原项目要求处理；此仓库未重新分发它们。
