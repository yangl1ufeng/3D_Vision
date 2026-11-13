# 两视角桌面模型

最新复测已完成马克杯抓放：排除137个落在杯身内部、归属不明确的玩偶点，保存到 `runs/new_desk/20261113_two_views_resolved/figurine_ambiguous_overlap_points.npy`，保留审查与恢复依据。其余11,077个点用于当前模型，物体位置未改变。该排除依赖杯身圆柱近似，属于归属冲突处理，尚未用第三视角独立验证。

当前[场景](../../runs/new_desk/20261113_two_views_resolved/scene.json)生成64个候选，7个通过接近筛选；第一次抓起后搬运失败，第二次完成抓起、搬运、放置。[成功动画](../../runs/new_desk/20261113_two_views_resolved/grasp_mug/execution/candidate_33/pick_place.gif)与[完整记录](../../runs/new_desk/20261113_two_views_resolved/grasp_mug/execution/task.json)。[两视角点选入口](../../launchers/start_two_view_selection.cmd)已更新。下文保留修正前的失败记录。

第一视角为 `20261113_144008_107123`，第二视角为 `20261113_150944_761219`，各采集20帧。此前玩偶被截断的第二视角作为备份保存，没有用于最终融合。

共有284个带深度的图像匹配，165个通过25毫米内点判定；内点残差中位数1.9毫米，P95为15.5毫米。第二视角桌面映射至第一视角桌面后，高度绝对残差P95为5.0毫米，法向差0.35度。

三件物体点云经2毫米体素去重、删除37个孤立点后共11,214点：马克杯5,767、耳机盒1,437、玩偶4,010。单视角原有7,902点。点数增加不等于完整度或精度提高相同比例。

[点云对比图](../../runs/new_desk/20261113_two_views_final/two_view_pointcloud_preview.png) · [最终场景](../../runs/new_desk/20261113_two_views_compound/scene.json) · [融合记录](../../runs/new_desk/20261113_two_views_compound/fusion_report.json) · [打开两视角场景点选](../../launchers/start_two_view_selection.cmd)

外观使用两次实际深度对应的彩色点。马克杯为杯身圆柱碰撞近似，耳机盒为盒体；玩偶由有测量点支持的3厘米网格分块，各块用24毫米小盒近似，避免一个大盒填满细枝之间的空隙。此分块模型仍会有遗漏和膨胀，不能视为精确网格。

本次融合场景的马克杯复测生成64个候选、5个通过接近筛选，但5次执行均未完成抓放，主要在抬升时与玩偶碰撞近似冲突，另有接近姿态不可达或目标碰撞。[完整失败记录](../../runs/new_desk/20261113_two_views_compound/grasp_mug/execution/task.json)已保留。此前单视角成功动画仍属于单视角场景，不能当作本次两视角的成功结果。相关15项回归检查通过。

下一步需要检查杯子与玩偶之间的碰撞近似和抬升路径，或在明确记录改动的独立虚拟布局中增加物体间距，再验证抓放。
