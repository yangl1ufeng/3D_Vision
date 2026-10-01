# 安装与本地文件恢复

本文使用 Windows PowerShell。进入 `projects/goldilocs_reproduction` 后执行示例命令。现有机器的依赖版本及第三方 commit 记录在根目录 `DEPENDENCIES.json`；没有把虚拟环境打包上传。

## 1. 环境与依赖

本地使用 Python 3.12、CUDA 版 PyTorch 2.11.0、torchvision 0.26.0。GPU 推理需要兼容的 NVIDIA 驱动。以下 CUDA 12.8 安装示例对应原机器；其他机器应按实际平台选择 PyTorch 包。

```powershell
cd projects/goldilocs_reproduction
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force third_party,checkpoints,data | Out-Null
```

下载第三方项目并固定版本；这些目录已被忽略：

```powershell
git clone --recursive https://github.com/naver/mast3r.git third_party/mast3r
git -C third_party/mast3r checkout f5209afc300cec36239a7ac992263f36847bbba0
git -C third_party/mast3r submodule update --init --recursive
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
& .\.venv\Scripts\python.exe -m pip install -r third_party/mast3r/requirements.txt -r third_party/mast3r/dust3r/requirements.txt
$env:SAM2_BUILD_CUDA = '0'
& .\.venv\Scripts\python.exe -m pip install --no-build-isolation -e third_party/sam2
```

SAM2 的可选 CUDA 填孔扩展在原机器未编译，当前实现明确关闭了额外后处理。安装依赖时未在新机器重新验证整个组合；以记录版本与上游依赖要求为依据处理平台差异。

## 2. 模型与图像

按原项目说明取得权重，保存为：

- `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`：见 [MASt3R](https://github.com/naver/mast3r)。
- `checkpoints/sam2_hiera_large.pt`：见 [SAM2](https://github.com/facebookresearch/sam2)，使用原 SAM2 Hiera Large，非 SAM2.1。

自行准备两张同一场景、不同时刻或视角的 RGB 图像。单对推理示例：

```powershell
& .\.venv\Scripts\python.exe run_pair.py --before data/before.jpg --after data/after.jpg --out runs/example --config configs/rc3d_native.json
```

这是流程入口示例，不等于论文主实验或最新 RC-3D 双视角修订。

## 3. RC-3D 与修订实验

数据来源：[CYWS-3D](https://github.com/ragavsachdeva/CYWS-3D)。原始归档地址在 `scripts/prepare_rc3d.py` 中。保留 `data/rc3d.tar`，解压为 `data/rc3d/part1` 至 `part4` 后生成作者顺序的清单：

```powershell
& .\.venv\Scripts\python.exe scripts/prepare_rc3d.py
```

修订脚本 `run_rc3d_revision.py` 会复用 `runs/rc3d_native/<pair>/geometry` 及对应运行记录，**这些缓存不在 GitHub 仓库中**。从零运行时需先生成匹配的原始推理缓存；仅下载源码不能直接恢复历史实验。旧 `run_rc3d.py` 还需要历史 `data/rc3d_manifest.json`，不要把缺少数据或缓存视为模型算法错误。

固定 8 对实验的候选与逐掩码追踪、完整数据清单、权重和诊断图片均留在原机器。报告中的 `audit/`、`runs/` 路径也是本地证据路径。

## 4. 测试

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

原机器已有 39 项单元测试通过。它们验证实现规则，不证明与作者未公开代码完全一致。
