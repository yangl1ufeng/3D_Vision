# 当前任务归档说明

归档日期：2026-10-06。此目录保存当前三维桌面抓取任务的实现、准备实验1–6及前期可行性测试。

## 上传范围

- 自编源码、配置、Windows兼容补丁、依赖清单、测试、实验和绘图脚本。
- 阶段报告、开发计划，以及 `runs/` 下的实验协议、来源记录、候选、统计表、性能记录、图表、输入和失败样例。开发过程中的旧实验也保留；正式结论以各报告明确引用的运行目录为准。
- `data/offline_official/`：由官方样例转换得到的RGB、逐像素世界XYZ、有效像素和场景描述；供检查现有实验输入使用。
- `data/model_files.json` 和 `data/download_manifest.json`：官方资源下载索引、大小及SHA256。

不包含虚拟环境、缓存、模型权重、下载的第三方仓库、重复的官方原始JSON数据、安装或运行日志、课程资料以及已停用的GOLDILOCS项目。SAM2仍是当前项目所需依赖，现有默认配置复用本机旧目录中的SAM2安装；这不表示旧项目已重新上传。

实验结果按原始字节保存，避免Git换行转换破坏融合产物的SHA256校验。结果中记录的绝对路径、机器信息、时间和源码哈希保留原貌。它们是历史来源信息，不能保证换机器后直接执行所有历史场景文件。

## 阅读顺序

1. [项目入口与运行方法](README.md)
2. [连续接近与碰撞实验](COLLISION_EXPERIMENTS.md)
3. [遮挡处理与多视角融合](VISIBILITY_FUSION_RESULTS.md)
4. [输入鲁棒性与点云规模压力](ROBUSTNESS_STRESS_RESULTS.md)
5. [后续开发计划](DEVELOPMENT_PLAN.md)

最后两项共207组实验；51次预处理中止均保留。`runs/preparation_56_verification.json` 是完整产物核验结果。核验通过不表示每组实验成功，也不等于真实机器人验收。

## 在另一台机器复现

先按项目README与 `requirements-local.txt` 准备Python、CUDA/PyTorch及依赖，并恢复第三方算法和模型：

- GraspGen上游为 <https://github.com/NVlabs/GraspGen>；本次使用的提交记录在报告和实验来源文件中，Windows修改见 `windows_compat.patch`。
- GraspGen官方权重及原始样例来自 <https://huggingface.co/adithyamurali/GraspGenModels>，可用 `download_assets.py` 根据归档下载索引恢复。
- SAM2上游为 <https://github.com/facebookresearch/sam2>；需要报告中使用的Hiera Large权重。配置 `configs/offline_default.json` 中的SAM2仓库和模型路径，使其指向新机器的实际安装目录。
- `resolved_config.json` 和模型来源记录可用于比对环境与参数。第三方资源继续遵循各自许可。

在项目目录运行单元测试：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

重跑实验时使用各报告给出的命令，并选用新的输出目录。原始运行中的绝对路径可能需要在复跑输入副本中更新；不要修改已封存的融合产物来绕过来源校验。归档包含结果检查所需数据，但不打包本机已安装的运行环境。
