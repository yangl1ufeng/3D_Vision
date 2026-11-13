# 纸盒与水杯的实测多视角点云

2026-11-12：用户停止继续拍摄后，使用已保存的前两组 Gemini 2 视角完成融合。每个视角分别固定机位拍摄 20 帧，对有效深度逐像素取中值；未使用用户提供的纸盒尺寸，也未用生成模型补物体背面。

第一组：`runs/gemini_static/20261112_162758_563266`；第二组：`runs/gemini_static/20261112_163515_065591`。桌面平面以及纸盒、水杯的图像轮廓用于估计两次相机的刚体变换。配准记录为 `runs/gemini_multiview/20261112_170625_237128/registration.json`；对齐后纸盒、水杯的相邻实测表面距离中位数分别约 2.5 mm、2.0 mm，P90 约 12.2 mm、14.6 mm。这只是当前重叠表面的检查，不证明隐藏面正确。

彩色 PLY 在同目录 `desk_box_cup_measured.ply`，最终保留 123,935 个点。处理时只裁剪桌面附近，按 2 mm 网格从每个占用格保留一个**原始实测点**，没有用尺寸先验创造新点。`fusion.json` 保存输入配准和输出 PLY 的 SHA-256 与采样参数。可看 `box_cup_3d_preview.png`（盒子和杯子的三维近景）或 `multi_view_pointcloud_preview.png`（桌面整体和俯视图）。PNG 是可视化抽样，完整结果以 PLY 为准。

## 点云到 PyBullet 静态场景

`fit_multiview_proxies.py` 仅用前两视角的实测深度，在手工纸盒/水杯轮廓内截取物体点；由纸盒上表面拟合有朝向的盒体，由杯口附近点估计圆形外径，并从桌面估计物体高度。没有使用之前用户提供的纸盒尺量数据。`geometry.json` 记录估计的纸盒 **19.7 × 12.7 × 7.2 cm**、水杯外径约 **6.5 cm**、高度约 **19.1 cm**，以及相对桌面的中心和角度。纸盒估计宽度几乎等于当前仿真指垫净开口约 12.85 cm，直接从窄边闭合没有可靠余量。

`replay_multiview_scene.py` 将桌面、静态盒体与圆柱碰撞代理放进 PyBullet，并从两个原相机位姿回渲染到 `scene_replay/view_1_render.png`、`view_2_render.png`。两视角与手工实拍轮廓的纸盒 IoU 分别约 **0.887、0.884**，水杯 IoU 约 **0.786、0.775**；相交有效像素上的实测/渲染深度中位绝对差，纸盒约 **1.6、3.0 mm**，水杯约 **4.4、3.3 mm**。纸盒与水杯碰撞形状间隔约 **5.6 cm**，无相互穿透。以上是单场景、手工轮廓上的回放检查，并不能推断普遍重建精度。

水杯采用**实心圆柱碰撞代理**，没有表达中空杯口、杯壁厚度或可能的把手；背面也未被直接测得。桌面坐标是以第一视角拟合出的局部坐标，尚未测得相机/桌面与 UR5 底座之间的变换。因此这份场景可用于静态碰撞和建模展示，不能直接称为机器人已经能按实景坐标抓取。

复跑建模与回渲染：

```powershell
.\.venv-vision\Scripts\python.exe fit_multiview_proxies.py runs\gemini_multiview\20261112_170625_237128\registration.json configs\gemini_cup_box_20261112.json
.\.venv-sim\Scripts\python.exe replay_multiview_scene.py runs\gemini_multiview\20261112_170625_237128\geometry.json
```

第三组 `runs/gemini_static/20261112_164314_709525` 保留为原始记录，但没有并入最终点云。该角度杯身的深度多数落在背后表面，且与前两组的物体表面对齐误差过大；强行合并会产生假杯壁。对应失败记录在 `runs/gemini_multiview/20261112_165904_571755/registration.json`。当前点云因此仍有纸盒背面、杯壁背侧及遮挡处的空洞，不能称为完整闭合网格或抓取碰撞模型。

复跑前两视角：

```powershell
Set-Location 'D:\新建文件夹\3D_Vision-main\projects\graspgen_feasibility'
.\.venv-vision\Scripts\python.exe register_gemini_landmarks.py configs\gemini_cup_box_20261112.json --limit 2
.\.venv-vision\Scripts\python.exe fuse_gemini_multiview.py runs\gemini_multiview\20261112_170625_237128\registration.json --stride 1 --voxel-mm 2
.\.venv-vision\Scripts\python.exe plot_fused_gemini.py runs\gemini_multiview\20261112_170625_237128\desk_box_cup_measured.ply
```

这些实拍图、深度和点云在 D 盘本地，已被 `.gitignore` 排除，避免误加入项目源码。
