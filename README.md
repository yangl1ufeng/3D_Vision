# 3D 视觉桌面抓取仿真

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
