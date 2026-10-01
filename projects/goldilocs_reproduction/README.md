# GOLDILOCS 主实验复现工作区

更新日期：2026-10-01。**状态：RC-3D 原始分辨率的 100 对独立实现推理和修正评测已完成，诊断 AP50=0.039102；原论文完整主实验复现尚未完成。**

`audit/paper_reported_metrics.json` 中的数值是原文报告值；`runs/` 中是本机独立实现的实际运行结果。当前实现没有复现出 GOLDILOCS 的性能优势。全量结果和评测限制见 `RESULTS_RC3D.md`，最终运行状态见 `runs/rc3d_native/progress.json`。

2026-10-01 已完成失败原因审计，详见 [FAILURE_ANALYSIS_RC3D.md](FAILURE_ANALYSIS_RC3D.md)。当前已修正状态混淆、同视角完全错位检查，补上 RC-3D 原图与渲染图双向传播、双视角独立分类和正确标注入口，29 项测试通过。修订记录和固定 8 对样本的 GPU 对照见 [REVISION_RC3D.md](REVISION_RC3D.md)。修正后尚未重新运行全量；这些事后诊断仍不能证明论文优势，缓存排除对照的 AP50=0.12596 也仅作错误归因。

## 已完成

- 下载并阅读 ICLR 2026 正式版论文，包括实验设置与附录。
- 核查作者 GitHub 仓库的递归目录：只有 README、docs 和示意图，没有算法、评测代码或权重。快照保存在 `audit/author_repo_tree.json` 和 `audit/author_README.md`。
- 建立 `.venv`，继承本机 Python 3.12 环境；PyTorch 2.11.0+cu128 在 RTX 4070 Laptop 8GB 上通过 CUDA 检查，并已完成 MASt3R/SAM2 实际推理。
- 拉取官方 MASt3R、SAM2、RC-3D 作者代码，下载 MASt3R/SAM2 权重并完成实际 GPU 推理。
- 下载 RC-3D 官方数据并核对 100 对、200 张输入图像，保存文件哈希。
- 完成 RGB → 三维重建 → 反向深度清理 → SAM2 分割/跟踪 → 变化分类的独立实现；通过 29 项几何、评测、跟踪状态和修订回归检查。
- 记录论文设置，核算主表与逐场景表的一致性。

## 当前阻碍

1. **原作者完整实现尚未公开。** 可以依据论文独立实现，但关键设置存在缺失或歧义，需要把假设与原论文设置分开记录。
2. **尚未获得所有基准。** RC-3D 已就绪；3DGS-CD 官方 Google Drive 和尝试的 VL-CMU-CD 镜像曾连接超时，ChangeSim 尚未下载。
3. **原实现配置仍有未决项。** 当前 Windows 环境已运行成功，首个样本峰值显存约 3.07 GiB；但 SAM2 checkpoint、跟踪判据、渲染参数和评测约定尚需与作者对齐，见 `IMPLEMENTATION.md`。
4. OpenReview 投稿接口本次返回 HTTP 403，网页触发访问验证，因此不能断言投稿附件中一定没有代码。

## 原论文复现边界

原始输入是 **RGB 图像**，由 MASt3R 估计几何和相机，再通过几何过滤、SAM2 分割与跟踪、SSIM、跨视图投票检测变化。老师提供的 RGB-D 相机可以用于后续应用扩展；把真实深度直接注入算法会改变原实验条件。

| 数据集 | 论文设置与主要指标 | 复现注意事项 |
|---|---|---|
| ChangeSim | 8,212 对，640×480；二类和五类 IoU/mIoU，主表 3 | ZSSCD 要用配对图像设置，不能混用其视频结果 |
| VL-CMU-CD | 429 对，512×512；前景 F1 | 只评价 Removed 类；过滤参数 α 未给出 |
| RC-3D | 100 对，1920×1440；移除对象框 mAP | 需核实原评测器的 IoU 阈值、框合并规则 |
| 3DGS-CD | Mustard、Desk、Swap、Bench、Sill；P/R/F1/IoU，主表 4 | 附录称 Sill 有 30 张测试图，表 4/7 列每场景 4 张，需获得实际清单 |
| NeRFCD | Cats、Block、Go Pieces、Desk、Potting；每场景 10 张测试图，主表 4 | Gaussian Difference 汇总只有 4 场景，排除了 Block，不能当成完全相同样本的比较 |

3RScan 为定性展示。10% ChangeSim 消融未公开子集清单或抽样种子，不能声称精确复现该子集。

## 需要澄清的实现细节

- **α 的含义不一致：** 正文 §4.3 按缺失区域覆盖比例超过 α 丢弃掩码；附录 A.6 按可见比例至少 α 保留掩码。换算后分别相当于可见比例至少 1−α 和至少 α。应分别实验，不能悄悄选一个当作原实现。
- SAM2 具体版本、checkpoint、掩码跟踪成功/失败判定未充分指定。
- MASt3R checkpoint、图像预处理、点云栅格化和深度采样/容差等实现细节需要固定。
- SSIM 核与边界处理、跨视图投票平票规则、ChangeSim 的 replaced 重叠阈值需要固定。
- 各基准须明确 mask 有效区域、忽略标签、空前景处理，以及按像素、图像还是场景聚合。

`configs/paper_settings.json` 是公开参数摘录，含未决字段，**不是已验证的可执行原作者配置**。

## 数值核对：不等于推理实验

运行：

```powershell
& .\.venv\Scripts\python.exe .\scripts\audit_tables.py
```

此脚本只读取人工摘录的原论文数值，输出 `audit/table_consistency.json`。

- 表 4 的 3DGS-CD F1：GOLDILOCS 97.72，基线 97.51，表面提升 0.21 个百分点。
- 表 7 的五场景 F1 等权平均：GOLDILOCS **97.318**，基线 **97.512**，差值 **−0.194** 个百分点。GOLDILOCS 与主表相差 0.402 个百分点；单纯两位小数舍入无法解释。**可能涉及汇总口径或表格错误，未获得逐图预测前不能定论。**
- 表 6 的 GOLDILOCS NeRFCD 场景 IoU 等权平均为 **87.224**，表 4 为 **87.62**，同样待澄清。
- ChangeSim 表 3 报告的二类 mIoU：64.9 对比 C3PO 59.6；五类 mIoU：33.6 对比 27.8。Moved 类 IoU 7.7，低于 CSCDNet 的 17.5。因此论文自身也不支持“所有类别全面领先”。

以上表格存在不同来源和聚合可能，不能从算术核对推断实际模型表现或论文不可信。

## 恢复实验的具体顺序

1. 获得可访问的原始基准数据并记录文件哈希、完整测试清单和标签映射；优先跑 3DGS-CD 小规模场景作流程验证，再用完整 ChangeSim 检验主要性能主张。
2. 若作者释放源码，固定 commit、依赖、权重哈希和评测器；否则按论文做独立实现，对上列未决项逐一记录选择。
3. 首先完成单对图像推理与掩码保存，检查相机坐标、重投影、遮挡和标签，再扩展到完整基准。少数图像成功只能算流程验证。
4. 在相同测试清单和评测代码下比较适用的基线；保存每张预测、逐图/逐场景指标、峰值显存和耗时。论文数字只作为参考列，不能标成自行复现的基线。
5. 对 3D 清理和跨视图投票做消融；分别评估两种 α 解释。参数选择使用验证数据，不能以测试集反复调参。
6. 最终报告实际差值、失败案例及配对置信区间；若优势不成立如实报告。不同 GPU 的耗时不能直接验证论文的速度倍数。

当前已进入独立实现的批量验证阶段；原论文主实验尚未完整复现。

第二轮修复与四组固定候选对照已启动，见 [FIX_ROUND2_RC3D.md](FIX_ROUND2_RC3D.md)。实时进度保存在 `audit/fix_round2/grid_progress.json`，完成结果汇总到 `audit/fix_round2/CONTROL_RESULTS.md`。

## 一手来源

- [ICLR 2026 论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html)
- [作者代码仓库](https://github.com/almogfri/GOLDILOCS)
- [OpenReview 投稿页](https://openreview.net/forum?id=qbCceo3FBE)
- [MASt3R](https://github.com/naver/mast3r)
- [3DGS-CD 数据及基线](https://github.com/520xyxyzq/3DGS-CD)
- [ChangeSim](https://github.com/SAMMiCA/ChangeSim)
- [ZSSCD 基线](https://github.com/kyusik-cho/ZSSCD)
