# 在照片中点选抓取目标

当前 [点选窗口](../../launchers/start_target_selection.cmd) 默认打开 2026-11-13 新拍摄的三物体场景。下面的水杯、纸盒示例是此前场景的验证记录；如需打开它，可运行 `select_grasp_target.py --scene configs/measured_table.json`。

双击 [start_target_selection.cmd](../../launchers/start_target_selection.cmd) 打开已保存的 Gemini 2 桌面照片。左键点想抓的物体，右键点需要排除的背景；点击“生成区域”，绿色区域会显示分割范围。若只覆盖物体的一部分，可继续在未选中的表面加点并重新生成。确认区域合适后点击“运行抓取”，程序使用选区内的真实深度点生成候选并运行仿真，完成后自动打开动画。

旧示例场景可选水杯和纸盒；新场景可选马克杯、耳机盒和模型玩偶。点选区域会与保存的 RGB、对齐深度和相机配准一起转为桌面坐标点云；程序核对目标几何、点数、覆盖率和周围物体后才送入 GraspGen。点击选出的点成为**模型实际输入**，原有融合点云仍用于场景显示和碰撞体定位。新物体若尚未完成建模，不能只点击照片就声称得到可靠的碰撞模型。

窗口使用**保存的画面**，不占用相机。要显示第二个已验收视角，可以在项目目录运行：

```powershell
.\.venv-vision\Scripts\python.exe select_grasp_target.py --scene configs/measured_table.json --view-index 1
```

运行记录位于 `runs/click_selection/<时间>/`：`mask.png`、`selection_preview.png`、`target_points_table.npy`、`selection.json` 和绑定该选区的 `scene.json`。点击“运行抓取”后，会在相同目录新建 `grasp_<时间>/` 保存模型候选、规划判定、每次仿真尝试和动画。图像、深度、分割模型、选区、几何和抓取结果均有可追溯记录。

## 本次验证

SAM 2.1 Tiny 的[官方代码](https://github.com/facebookresearch/sam2)与[官方权重](https://huggingface.co/facebook/sam2.1-hiera-tiny)已保存到项目 D 盘目录，权重 SHA-256 与发布记录一致。代码固定在提交 `2b90b9f5ceec907a1c18123530e92e794ad901a4`。权重和新增依赖都在 `.venv-vision` 与项目目录内。

- 在第一张验收照片点击水杯 `(610, 510)`，得到杯身区域；经过深度与几何核对后保留 4,939 个实测点。[查看区域预览](../../runs/click_selection/validation_cup/selection_preview.png)。这份点云实际进入 GraspGen，生成 64 个候选；4 个通过接近规划，第二次尝试完成抓放。[查看任务结果](../../runs/measured_grasp/clicked_cup_validation/execution/task.json)与[动画](../../runs/measured_grasp/clicked_cup_validation/execution/candidate_33/pick_place.gif)。
- 在纸盒标签上单点 `(340, 590)`，只覆盖约 38.5% 的目标可见深度点，现会提示补点；再点纸盒边缘 `(460, 610)` 后，生成覆盖整个纸盒的区域，保留 10,600 个实测点。[查看纸盒预览](../../runs/click_selection/validation_box_refined/selection_preview.png)。
- 相关 17 项测试通过，包括局部区域拒绝、补点后完整区域、错误目标场景拒绝，以及原有坐标和运动规划检查。

当前分割使用保存的单视角照片，后续仍需在新的桌面布局上验证。分割质量分数仅表示模型对区域的估计，不能直接当作抓取成功率。
