"""Supplemental fixed logit adapter; never rewrites primary run metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.metrics import detection_ap
from goldilocs_local.pipeline import mask_box


def mask_key(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def negative_presence_weight(logit):
    # Numeric stability only; no fitted temperature or data-selected threshold.
    return float(1/(1+np.exp(np.clip(float(logit), -80, 80))))


def evaluate(run):
    plan = json.loads((run/'plan.json').read_text())
    if json.loads((run/'progress.json').read_text())['status'] != 'complete':
        raise ValueError('Supplemental scoring requires a complete run')
    pairs = {p['id']:p for p in json.loads((ROOT/'data/rc3d_author_order_manifest.json').read_text())['pairs']}
    gts, original, adapted, score_rows = {}, [], [], []
    for pid in plan['pair_ids']:
        for view, name, truth_key in ((0, 'source', 'before_boxes_xyxy'), (1, 'target', 'after_boxes_xyxy')):
            image_id = pid+'/'+name
            gts[image_id] = pairs[pid][truth_key]
            folder = run/pid/f'view_{view}'
            clean = json.loads((folder/'reference_to_target_tracking.json').read_text())
            changed = [t for t in clean if t['clean_changed']]
            other = json.loads((folder/'reference_to_target_other_tracking.json').read_text())
            if len(changed) != len(other):
                raise ValueError('Changed/other trace cardinality differs')
            evidence = {}
            with np.load(folder/'reference_to_target_tracking.npz') as masks:
                for a,b in zip(changed, other):
                    if a['bbox_xyxy'] != b['source_bbox_xyxy']:
                        raise ValueError('Changed/other trace order differs')
                    key = mask_key(masks[f"source_{a['id']:04d}"])
                    if key in evidence:
                        raise ValueError('Ambiguous duplicate source mask')
                    evidence[key] = (a, b)
            report = json.loads((folder/'prediction.json').read_text())
            labels = np.array(Image.open(folder/'labels.png'))
            with np.load(folder/'object_masks.npz') as masks:
                for obj in report['objects']:
                    if obj['label'] != 'removed':
                        continue
                    mask = masks[f"mask_{obj['id']:04d}"]
                    box = mask_box(mask & (labels == 2))
                    if box is None:
                        continue
                    a,b = evidence[mask_key(mask)]
                    clean_logit = a['clean_track_object_presence_logit']
                    other_logit = b['object_presence_logit']
                    if clean_logit is None or other_logit is None:
                        raise ValueError('Absence evidence unavailable; no silent score fallback')
                    score = obj['score'] * negative_presence_weight(clean_logit) * negative_presence_weight(other_logit)
                    original.append({'image_id':image_id, 'bbox':box, 'score':obj['score']})
                    adapted.append({'image_id':image_id, 'bbox':box, 'score':score})
                    score_rows.append({'image_id':image_id, 'object_id':obj['id'],
                        'clean_presence_logit':clean_logit, 'other_presence_logit':other_logit,
                        'proposal_iou_score':obj['score'], 'absence_evidence_score':score})
    def metrics(predictions):
        result = {}
        for view in ('source', 'target', 'pooled'):
            gt = {k:v for k,v in gts.items() if view == 'pooled' or k.endswith('/'+view)}
            pred = [p for p in predictions if p['image_id'] in gt]
            result[view] = {k:v for k,v in detection_ap(gt,pred).items() if k not in ('precision','recall')}
        return result
    primary = json.loads((run/'metrics.json').read_text())['metrics']
    original_metrics = metrics(original)
    if original_metrics != primary:
        raise ValueError('Supplemental adapter differs from primary boxes/prediction order')
    return {'scope':'same posthoc diagnostic pairs; independent uncalibrated engineering adapter, not author scoring',
        'formula':'proposal_predicted_iou * sigmoid(-clean_presence_logit) * sigmoid(-other_presence_logit)',
        'formula_parameters_fit_to_ground_truth':False,
        'primary_predictions_or_metrics_modified':False,
        'primary_metrics':original_metrics, 'supplemental_metrics':metrics(adapted), 'rows':score_rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['supplemental_metrics'], indent=2))
