# 独立实现说明

本目录 `goldilocs_local/` 为根据论文重建的实现，作者未公开实现的细节按下表处理。结果只能先称为独立实现的流程验证，不能宣称精确复现原论文。

| 项目 | 当前处理 |
|---|---|
| MASt3R | 官方 512 metric checkpoint；完整图像配对，500 次 coarse 优化；coarse 按上游 demo 将第二阶段次数设为 0 |
| desc_conf | 论文写 `3d`，当前上游输出键为 `conf` 与 `desc_conf`；暂映射为三维置信度 `conf` |
| 重建输入 | 官方 resize/center crop；square_ok=True；保存精确缩放和裁剪记录 |
| 检测分辨率 | 保留 512×384 原型；新增 `rc3d_native.json` 在 1920×1440 原始 RGB 上检测，按缩放/裁剪记录换算渲染内参 |
| 几何过滤 | 投影到另一相机，按最近像素取深度；严格 z < depth，容差 0；视野外或被遮挡的点保留 |
| 渲染 | CPU z-buffer，正深度，方形 splat；原型半径 1，原始分辨率半径 4 像素；有效区域独立保存，不把黑色物体视作空洞 |
| SAM2 | 官方原版 Hiera Large（非 SAM2.1），BF16；自动分割和视频预测器共用一份模型，降低显存占用 |
| 后处理 | 未编译 SAM2 CUDA 扩展；明确关闭 builder 的额外后处理，包括填孔 |
| 跟踪 | 将源与目标写为两帧 JPEG，quality=100、subsampling=0，通过官方视频接口传播掩码 |
| 跟踪判定（2026-10-01 第二轮） | 新配置将源可见性与 Track 集合成员分开；工程对照加入往返身份验证及同相机 IoU 检查，单独记录拒绝状态与模型存在性 logit；不把身份拒绝伪装成空预测 |
| α | .25；正文和附录分别要求可见比例≥.75 / ≥.25，四组固定候选对照已启动 |
| RC-3D SAM 设置 | 论文没有 RC-3D 参数行，暂用与其室内多视图实验相同的 .9/.9 阈值、64 网格、1 层裁剪；非原作者确认值 |
| 掩码优先级 | Warped > Moved > Removed > Added |
| 非刚体变化 | skimage SSIM，win_size=7，RGB 通道平均；对静态掩码的有效区计算不相似度，超过均值+1标准差 |
| RC-3D 评测 | 两视角分别和合并 AP50；主指标保留最终优先级后的框与 SAM2 predicted IoU，完整对象框和 logit 评分单列补充指标；官方评测器未对齐 |
| RC-3D 双向传播 | I0 与 R0,1 双向传播；第二轮取消额外可见性预过滤和框 NMS，仅合并完全相同掩码 |

尚未实现多视图数据集的跨视图投票和 ChangeSim 的 replaced 类适配。没有使用测试标注搜索阈值，但这些修订是在检查 RC-3D 测试错误后设计的，因此修订样本不能当作独立验证集。错误分析见 `FAILURE_ANALYSIS_RC3D.md`，修订和对照结果见 `REVISION_RC3D.md`。已完成的 `rc3d_native` 使用修正前代码，原源码已按批次哈希冻结保存。修订运行的源码副本保存在 `audit/revision/frozen_source/`。

## 已就绪的组件检查

- 几何、AP、跟踪状态、双视角语义、候选缓存与身份验证共 39 项测试通过。
- SAM2 官方 truck 示例：13 个自动掩码，同图跟踪 IoU=0.9950，全黑目标帧跟踪失败。配置为低密度网格，仅作组件检查。
- RC-3D 官方归档完整下载，按作者 `rc3d.py` 的 0/2 图像配对方式获得 100 对图像，校验 200 张 RGB 输入。真实深度没有用于推理。
- 第二轮 7 个独立 GPU 跟踪控制通过；具体修复、深度诊断与运行进度见 `FIX_ROUND2_RC3D.md`。

## 运行

```powershell
cd D:\3D_Vision\projects\goldilocs_reproduction
# 检查单对端到端流程，结果保存在 runs/rc3d_prototype/part1_banana
& .\.venv\Scripts\python.exe .\run_rc3d.py --limit 1
# 明确请求完整 100 对的独立实现推理，可从已完成的样本继续
& .\.venv\Scripts\python.exe .\run_rc3d.py --limit 100
# 完整性检查默认拒绝缺失样本；仅调试时加 --allow-partial
& .\.venv\Scripts\python.exe .\scripts\evaluate_rc3d_author_order.py --runs .\runs\rc3d_prototype --output .\runs\rc3d_prototype\diagnostic_metrics.json
```

每个样本保存 `run.json`、几何点图/相机、源图重投影、干净场景渲染、有效区、逐掩码跟踪记录、标签图及诊断拼图。程序失败时不会写入完成标记。

`diagnostic_metrics.json` 即使覆盖 100 对，也不能直接与论文表 9 的 0.53 比较；分辨率、掩码到框的转换、目标视图和汇总约定必须先对齐。
