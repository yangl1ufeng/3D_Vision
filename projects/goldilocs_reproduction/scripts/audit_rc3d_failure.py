"""Post-hoc diagnostics only; uses test GT for attribution, never model selection.

Replays saved masks without modifying the completed inference run. Oracle scores
and counterfactuals are diagnostic bounds, not benchmark or tuned model results.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.metrics import box_iou, detection_ap
from goldilocs_local.pipeline import mask_box
from scripts.evaluate_rc3d_author_order import load_author_order_gt


def compact(gt, preds):
    return {k: v for k, v in detection_ap(gt, preds).items() if k not in ('precision', 'recall')}


def main():
    out = ROOT/'audit/failure_analysis'
    out.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((ROOT/'data/rc3d_manifest.json').read_text())['pairs']
    truth = load_author_order_gt()
    gt = {p['id']: truth[p['after']] for p in pairs}
    variants = {k: [] for k in ('original', 'raw_removed', 'nonempty_clean_is_success',
                                'nms_05', 'area_score_diagnostic', 'all_cross_proposals')}
    rows, counts = [], Counter()
    for num, pair in enumerate(pairs):
        pid = pair['id']
        folder = ROOT/'runs/rc3d_native'/pid/'view_1'
        result = json.loads((folder/'prediction.json').read_text())
        trace = json.loads((folder/'reference_to_target_tracking.json').read_text())
        box = gt[pid][0]
        proposal_iou = [float(box_iou(t['bbox_xyxy'], [box])[0]) for t in trace]
        candidate = [t for t, iou in zip(trace, proposal_iou) if iou >= .5]
        failed = [t for t in trace if not t['clean_track_success']]
        visible_reject = [t for t in failed if t['clean_track_pixels'] > 0]
        counts['cross_proposals'] += len(trace)
        counts['clean_track_failed'] += len(failed)
        counts['nonempty_but_visibility_rejected'] += len(visible_reject)
        counts['clean_track_empty'] += sum(t['clean_track_pixels'] == 0 for t in trace)
        for t in trace:
            variants['all_cross_proposals'].append({'image_id': pid, 'bbox': t['bbox_xyxy'], 'score': t['predicted_iou']})
        label = np.array(Image.open(folder/'labels.png'))
        pred = []
        raw = [o for o in result['objects'] if o['label'] == 'removed']
        invalid_boxes = {tuple(t['bbox_xyxy']) for t in visible_reject}
        with np.load(folder/'object_masks.npz') as masks:
            for obj in raw:
                raw_pred = {'image_id': pid, 'bbox': obj['bbox_xyxy'], 'score': obj['score']}
                variants['raw_removed'].append(raw_pred)
                arr = masks[f"mask_{obj['id']:04d}"]
                final_box = mask_box(arr & (label == 2))
                if final_box is None:
                    continue
                p = {'image_id': pid, 'bbox': final_box, 'score': obj['score'],
                     'pixels': obj['pixels'], 'raw_box': obj['bbox_xyxy']}
                pred.append(p)
                if tuple(obj['bbox_xyxy']) not in invalid_boxes:
                    variants['nonempty_clean_is_success'].append(p)
                variants['area_score_diagnostic'].append(dict(p, score=obj['pixels']))
        variants['original'].extend(pred)
        left = sorted(pred, key=lambda p: -p['score'])
        while left:
            p = left.pop(0)
            variants['nms_05'].append(p)
            left = [q for q in left if box_iou(p['bbox'], [q['bbox']])[0] < .5]
        raw_best = max([float(box_iou(o['bbox_xyxy'], [box])[0]) for o in raw], default=0)
        final_best = max([float(box_iou(p['bbox'], [box])[0]) for p in pred], default=0)
        failed_best = max([float(box_iou(t['bbox_xyxy'], [box])[0]) for t in failed], default=0)
        if max(proposal_iou, default=0) < .5:
            reason = 'no_matching_cross_proposal'
        elif failed_best < .5:
            reason = 'matching_proposal_tracks_to_clean'
        elif raw_best < .5:
            reason = 'matching_changed_proposal_classified_moved'
        elif final_best < .5:
            reason = 'priority_clipping_loses_match'
        else:
            reason = 'final_match_available'
        counts[reason] += 1
        rows.append({'pair': pid, 'reason': reason, 'gt': box,
                     'best_proposal_iou': max(proposal_iou, default=0),
                     'best_changed_iou': failed_best, 'best_raw_removed_iou': raw_best,
                     'best_final_iou': final_best, 'final_predictions': len(pred),
                     'good_proposals': candidate, 'valid_fraction': result['projection_valid_fraction'],
                     'kept_point_fractions': result['kept_point_fractions']})
        if (num+1) % 20 == 0:
            print(f'Analyzed {num+1}/100', flush=True)
    summary = {'purpose': 'posthoc failure attribution; NOT independent validation or tuned benchmark',
               'counts': dict(counts), 'variants': {k: compact(gt, v) for k, v in variants.items()}}
    scores = np.array([p['score'] for p in variants['original']])
    hits = np.array([box_iou(p['bbox'], gt[p['image_id']])[0] >= .5 for p in variants['original']])
    summary['score_diagnostic'] = {'unique_scores': len(np.unique(scores)),
        'localization_hit_mean_score': float(scores[hits].mean()),
        'localization_miss_mean_score': float(scores[~hits].mean()),
        'predicted_pixels_quantiles': np.quantile([p['pixels'] for p in variants['original']], [0,.25,.5,.75,1]).tolist()}
    # Oracle only: perfect ranking shows the bound imposed by missed objects.
    oracle = [dict(p, score=float(box_iou(p['bbox'], gt[p['image_id']])[0])) for p in variants['original']]
    summary['oracle_iou_ranking_not_a_method'] = compact(gt, oracle)
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out/'per_pair.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(variants, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
