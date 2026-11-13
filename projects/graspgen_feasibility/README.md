# 桌面抓取仿真项目导航

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
