"""Attribute saved diagnostic misses; GT is used only after inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.metrics import box_iou
from scripts.compare_rc3d_revision import predictions


def best(boxes, truth):
    return max((float(box_iou(box, truth).max()) for box in boxes if box), default=0.)


def audit(run):
    plan = json.loads((run/'plan.json').read_text())
    if json.loads((run/'progress.json').read_text())['status'] != 'complete':
        raise ValueError('Audit requires a completed run')
    pairs = {p['id']: p for p in json.loads((ROOT/'data/rc3d_author_order_manifest.json').read_text())['pairs']}
    counts = {'source': Counter(), 'target': Counter()}
    rows = []
    for pid in plan['pair_ids']:
        for view, name, key in [(0, 'source', 'before_boxes_xyxy'), (1, 'target', 'after_boxes_xyxy')]:
            folder = run/pid/f'view_{view}'
            truth = pairs[pid][key]
            clean = json.loads((folder/'reference_to_target_tracking.json').read_text())
            other = json.loads((folder/'reference_to_target_other_tracking.json').read_text())
            changed = [t for t in clean if t.get('clean_changed',t['clean_track_status']=='absent')]
            if len(changed) != len(other):
                raise ValueError(f'Inconsistent track count: {pid}/{name}')
            if any(a['bbox_xyxy'] != b['source_bbox_xyxy'] for a, b in zip(changed, other)):
                raise ValueError(f'Inconsistent track order: {pid}/{name}')
            other_by_id = {a['id']: b for a, b in zip(changed, other)}
            final = predictions(folder)
            raw = [o['bbox_xyxy'] for o in json.loads((folder/'prediction.json').read_text())['objects']
                   if o['label'] == 'removed']
            scores = {
                'proposal': best([t['bbox_xyxy'] for t in clean], truth),
                'changed': best([t['bbox_xyxy'] for t in changed], truth),
                'raw_removed': best(raw, truth),
                'final': best([p['bbox'] for p in final], truth),
            }
            if scores['final'] >= .5:
                reason = 'final_match_available'
            elif scores['proposal'] < .5:
                reason = 'no_matching_proposal'
            elif scores['changed'] < .5:
                reason = 'clean_tracking_blocks_candidate'
            elif scores['raw_removed'] < .5:
                reason = 'other_tracking_blocks_candidate'
            else:
                reason = 'priority_clipping_loses_match'
            good = []
            for t in clean:
                iou = best([t['bbox_xyxy']], truth)
                if iou >= .5:
                    good.append(dict(t, box_iou=iou, other_track=other_by_id.get(t['id'])))
            rows.append({'pair': pid, 'view': name, 'reason': reason,
                         'best_iou_by_stage': scores, 'good_candidates': good})
            counts[name][reason] += 1
    return {'scope': 'posthoc failure attribution; never input to inference or threshold selection',
            'run': str(run), 'pairs': len(plan['pair_ids']),
            'counts': {k: dict(v) for k, v in counts.items()}, 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['counts'], indent=2))
