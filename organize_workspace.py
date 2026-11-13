"""One-time documented relocation; runtime assets retain their original paths."""
import json, os, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
P=ROOT/'projects/graspgen_feasibility'
mapping={}
current={'TWO_VIEW_DESK','NEW_DESK_20261113','CLICK_TARGET','SCENE_SELECTION'}
camera={'GEMINI2_CAMERA','MULTIVIEW_CUP_BOX','REAL_SCENE_SIMULATION','VIRTUAL_CAMERA'}
simulation={'SIMULATION','UR5_SIMULATION','RECONSTRUCTED_UR5','MOTION_PLANNING','GRASPGEN_MEASURED_CUP'}
setup={'ARCHIVE','GRASPGEN_RESTORED'}
for src in P.glob('*.md'):
    if src.name=='README.md': continue
    group='current' if src.stem in current else 'camera' if src.stem in camera else 'simulation' if src.stem in simulation else 'setup' if src.stem in setup else 'history'
    mapping[src]=P/'docs'/group/src.name
for src in P.glob('*.cmd'):
    group='launchers' if src.name.startswith('start_') else 'setup'
    mapping[src]=P/group/src.name
mapping[ROOT/'README.md']=P/'docs/history/WORKSPACE_README_legacy.md'
mapping[P/'README.md']=P/'docs/history/PROJECT_README_legacy.md'
for src,dst in mapping.items():
    if dst.exists(): raise RuntimeError(f'Destination exists: {dst}')
    if not dst.resolve().is_relative_to(ROOT): raise RuntimeError('Outside workspace')
records=[]
for src,dst in mapping.items():
    text=src.read_text(encoding='utf-8-sig')
    if src.suffix=='.cmd': text=text.replace('%~dp0','%~dp0..\\')
    else:
        def link(m):
            target=m.group(2)
            if target.startswith(('http:','https:','mailto:','#')): return m.group(0)
            raw,sep,anchor=target.partition('#')
            old=(src.parent/raw).resolve()
            new=mapping.get(old,old)
            relative=os.path.relpath(new,dst.parent).replace('\\','/')
            return m.group(1)+relative+(sep+anchor if sep else '')+')'
        text=re.sub(r'(\]\()([^\n)]+)\)',link,text)
    dst.parent.mkdir(parents=True,exist_ok=True)
    src.rename(dst)
    dst.write_text(text,encoding='utf-8')
    records.append({'from':str(src.relative_to(ROOT)),'to':str(dst.relative_to(ROOT))})
(ROOT/'organization_manifest.json').write_text(json.dumps({'moves':records,'kept_in_place':['source scripts and modules','runs','data','checkpoints','third_party','Python environments','CUDA environment']},ensure_ascii=False,indent=2),encoding='utf-8')
ROOT.joinpath('README.md').write_text('''# 3D 视觉桌面抓取仿真

当前流程：Gemini 2 静态采集 → 物体分割 → 两视角点云融合 → GraspGen 候选 → UR5 仿真抓放。

## 常用入口

- [两视角场景点选](projects/graspgen_feasibility/launchers/start_two_view_selection.cmd)
- [相机实时预览](projects/graspgen_feasibility/launchers/start_gemini2_viewer.cmd)
- [固定机位采集20帧](projects/graspgen_feasibility/launchers/start_gemini2_static.cmd)
- [项目导航](projects/graspgen_feasibility/README.md)
- [最新结果与限制](projects/graspgen_feasibility/docs/current/TWO_VIEW_DESK.md)

## 目录

| 目录 | 用途 |
|---|---|
| projects/graspgen_feasibility | 源码、配置、文档、实验数据与启动入口 |
| envs | CUDA 等编译依赖，运行所需 |
| .downloads | 下载素材、构建记录；可再生成的 pip 缓存已清理 |

原始采集、权重和历史实验保留原路径，避免破坏场景来源校验。文档和启动脚本已经分类搬迁；旧文档在 docs/history，搬迁记录在 organization_manifest.json。
''',encoding='utf-8')
P.joinpath('README.md').write_text('''# 桌面抓取仿真项目导航

最新两视角场景的马克杯已完成抓起、搬运和放置；其他物体的当前状态见 [两视角记录](docs/current/TWO_VIEW_DESK.md)。

## 按用途查找

| 目录 | 内容 |
|---|---|
| launchers/ | 日常启动脚本：相机、点选、仿真 |
| setup/ | PointNet 编译脚本；旧 build_pointnet.cmd 含旧机器路径，仅供历史参考 |
| docs/current/ | 当前场景、点选操作和最新实验结果 |
| docs/camera/ | 相机采集、点云与多视角说明 |
| docs/simulation/ | 机械臂、路径规划与仿真说明 |
| docs/setup/ | 环境安装与归档复现说明 |
| docs/history/ | 历史阶段计划、报告和旧版 README |
| configs/ | 场景与算法配置 |
| offline_core/ | 视觉分析算法模块 |
| simulation/ | 机械臂、碰撞检查、规划与执行模块 |
| tests/ | 验证用例 |
| experiments/ | 实验配置与素材 |
| runs/ | 原始采集及成功、失败运行记录，保持原路径 |
| data/ | 样例和输入数据 |
| checkpoints/ | 模型权重及来源记录 |
| third_party/ | 第三方源码与机械臂资源 |
| .venv-vision / .venv-sim | 视觉与仿真运行环境，禁止直接搬动 |

## 启动

- [两视角照片点选](launchers/start_two_view_selection.cmd)
- [单视角照片点选](launchers/start_target_selection.cmd)
- [相机预览](launchers/start_gemini2_viewer.cmd)
- [20帧静态采集](launchers/start_gemini2_static.cmd)

项目根目录的 Python 脚本是可运行入口及相互导入的模块，保留路径以兼容既有脚本。采集类以 capture/probe/view 开头；重建类以 fit/register/fuse/build 开头；抓取入口为 run_measured_grasp.py；图表类以 plot 开头。完整历史架构见 [归档项目说明](docs/history/PROJECT_README_legacy.md)。

命令行运行仍以本目录为工作目录。移动后的启动脚本会自动切回本目录。
''',encoding='utf-8')
print(f'Moved {len(records)} files; runtime assets retained.')
