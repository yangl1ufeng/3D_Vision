"""Read cached results only; posthoc GT attribution, no inference or tuning."""
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.geometry import visibility_accept
from goldilocs_local.metrics import box_iou, detection_ap
from goldilocs_local.segmentation import summarize_track
from scripts.compare_rc3d_revision import predictions


def best(box, truth):
    return float(box_iou(box, truth).max()) if box else 0.


def main():
    run = ROOT/'runs/rc3d_revision_repaired_pilot'
    plan = json.loads((run/'plan.json').read_text())
    pairs = {p['id']:p for p in json.loads((ROOT/'data/rc3d_author_order_manifest.json').read_text())['pairs']}
    counts = Counter()
    changed_by_guard, all_good, nms_losses = [], [], []
    ordered_gt, ordered_pred = {}, []
    for pid in plan['pair_ids']:
        for view, name, key in [(0,'source','before_boxes_xyxy'),(1,'target','after_boxes_xyxy')]:
            folder = run/pid/f'view_{view}'
            truth = pairs[pid][key]
            image_id = pid+'/'+name
            ordered_gt[image_id] = truth
            ordered_pred.extend(dict(p,image_id=image_id) for p in predictions(folder))
            support = np.array(Image.open(folder/'valid.png'))>0
            clean_support = np.array(Image.open(folder/'clean_valid.png'))>0
            settings = json.loads((folder/'prediction.json').read_text())['settings']
            for direction in ('reference_to_target','target_to_reference'):
                trace = json.loads((folder/f'{direction}_tracking.json').read_text())
                with np.load(folder/f'{direction}_tracking.npz') as masks:
                    for t in trace:
                        source = masks[f"source_{t['id']:04d}"]
                        clean_track = masks[f"clean_track_{t['id']:04d}"]
                        raw = summarize_track(clean_track,support,settings)
                        _, clean_fraction = visibility_accept(source,clean_support,settings['alpha'],settings['alpha_rule'])
                        _, cross_fraction = visibility_accept(source,support,settings['alpha'],settings['alpha_rule'])
                        iou = best(t['bbox_xyxy'],truth) if direction=='reference_to_target' else None
                        row = {'pair':pid,'view':name,'direction':direction,'id':t['id'],
                               'bbox_xyxy':t['bbox_xyxy'],'gt_box_iou':iou,
                               'raw_status':raw['status'],'final_status':t['clean_track_status'],
                               'identity_reason':t['clean_track_identity_reason'],
                               'cross_source_coverage':cross_fraction,'clean_source_coverage':clean_fraction,
                               'clean_track_pixels':t['clean_track_pixels'],
                               'aligned_overlap_pixels':t['clean_track_aligned_overlap_pixels']}
                        counts['tracked_proposals'] += 1
                        if raw['status'] != t['clean_track_status']:
                            changed_by_guard.append(row)
                            counts[f"guard_{raw['status']}_to_{t['clean_track_status']}"] += 1
                        if iou is not None and iou>=.5:
                            all_good.append(row)
                            counts['gt_matching_reference_proposals'] += 1
                            if raw['status']=='absent' and t['clean_track_status']=='unobserved':
                                counts['gt_matching_absent_blocked_by_clean_support'] += 1
            if view==1:
                old_folder = ROOT/'runs/rc3d_native'/pid/'view_1'
                if not np.array_equal(np.array(Image.open(old_folder/'cross_render.png')),
                                      np.array(Image.open(folder/'cross_render.png'))):
                    raise ValueError('Cannot audit NMS across different renderings')
                old = json.loads((old_folder/'reference_to_target_tracking.json').read_text())
                selected = json.loads((folder/'reference_to_target_tracking.json').read_text())
                selected_boxes = {tuple(r['bbox_xyxy']) for r in selected}
                for proposal in old:
                    iou = best(proposal['bbox_xyxy'],truth)
                    if iou<.5 or tuple(proposal['bbox_xyxy']) in selected_boxes:
                        continue
                    suppressors = [r for r in selected if best(proposal['bbox_xyxy'],[r['bbox_xyxy']])>.7
                                   and r['predicted_iou']>=proposal['predicted_iou']]
                    if suppressors:
                        nms_losses.append({'pair':pid,'discarded_box':proposal['bbox_xyxy'],
                            'discarded_gt_iou':iou,'discarded_score':proposal['predicted_iou'],
                            'survivors':[{'bbox':r['bbox_xyxy'],'score':r['predicted_iou'],
                                          'gt_iou':best(r['bbox_xyxy'],truth)} for r in suppressors]})
    actual = detection_ap(ordered_gt,ordered_pred)
    # GT is only used to show localization/ranking bounds, never to select scores.
    ideal = [dict(p,score=best(p['bbox'],ordered_gt[p['image_id']])) for p in ordered_pred]
    oracle = detection_ap(ordered_gt,ideal)
    qualities = np.array([p['score'] for p in ordered_pred])
    hits = np.array([best(p['bbox'],ordered_gt[p['image_id']])>=.5 for p in ordered_pred])
    geometry = []
    for pid in plan['pair_ids']:
        with np.load(ROOT/'runs/rc3d_native'/pid/'geometry/scene.npz') as scene:
            for view in (0,1):
                points = scene[f'points_{view}'].reshape(-1,3)
                pose = scene[f'pose_{view}']
                inv_pose = np.linalg.inv(pose)
                camera_z = (points@inv_pose[:3,:3].T + inv_pose[:3,3])[:,2]
                depth = scene[f'depth_{view}'].reshape(-1)
                finite = np.isfinite(camera_z)&np.isfinite(depth)&(depth>0)
                geometry.append({'pair':pid,'view':view,
                    'max_relative_saved_depth_vs_camera_z':float(np.max(np.abs(camera_z[finite]-depth[finite])/depth[finite]))})
    compact = lambda m:{k:v for k,v in m.items() if k not in ('precision','recall')}
    report = {'scope':'read-only cached posthoc attribution; no model inference or parameter selection',
              'audited_run':str(run.relative_to(ROOT)), 'counts':dict(counts),
              'guard_changes':changed_by_guard,'gt_matching_proposals':all_good,
              'fusion_discards_of_good_cached_cross_proposals':nms_losses,
              'actual_metrics':compact(actual),'oracle_ranking_not_a_method':compact(oracle),
              'score_diagnostic':{'unique_scores':len(np.unique(qualities)),
                                  'localized_prediction_mean_score':float(qualities[hits].mean()),
                                  'mislocalized_prediction_mean_score':float(qualities[~hits].mean())},
              'saved_depth_coordinate_consistency':geometry}
    out = ROOT/'audit/algorithm_recheck/cached_evidence.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('counts','fusion_discards_of_good_cached_cross_proposals',
                     'actual_metrics','oracle_ranking_not_a_method','score_diagnostic')},indent=2))
    print('max depth self consistency error:',max(r['max_relative_saved_depth_vs_camera_z'] for r in geometry))


if __name__=='__main__':
    main()
