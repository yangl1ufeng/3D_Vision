"""Explicit source/target/pooled AP50; no claim of unpublished official protocol."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.metrics import detection_ap
from goldilocs_local.pipeline import mask_box


def evaluate(runs, expected_ids):
    manifest = json.loads((ROOT/'data/rc3d_author_order_manifest.json').read_text())
    pairs = {p['id']: p for p in manifest['pairs']}
    if len(set(expected_ids)) != len(expected_ids) or not expected_ids:
        raise ValueError('Expected unique nonempty evaluation IDs')
    gts, predictions, object_predictions = {}, [], []
    for pid in expected_ids:
        pair = pairs[pid]
        folder = Path(runs)/pid
        if not (folder/'run.json').exists():
            raise ValueError(f'Missing completed sample: {pid}')
        gts[pid+'/source'] = pair['before_boxes_xyxy']
        gts[pid+'/target'] = pair['after_boxes_xyxy']
        for view,name in [(0,'source'),(1,'target')]:
            view_dir = folder/f'view_{view}'
            result = json.loads((view_dir/'prediction.json').read_text())
            if result['coordinate_system'] != 'original image pixels':
                raise ValueError('Revision evaluator requires native coordinates')
            labels = np.array(Image.open(view_dir/'labels.png'))
            with np.load(view_dir/'object_masks.npz') as masks:
                for obj in result['objects']:
                    if obj['label'] != 'removed':
                        continue
                    raw_bbox = mask_box(masks[f"mask_{obj['id']:04d}"])
                    if raw_bbox is not None:
                        object_predictions.append({'image_id':pid+'/'+name,'bbox':raw_bbox,'score':obj['score']})
                    bbox = mask_box(masks[f"mask_{obj['id']:04d}"] & (labels == 2))
                    if bbox is not None:
                        predictions.append({'image_id':pid+'/'+name,'bbox':bbox,'score':obj['score']})
    metrics, object_metrics = {}, {}
    for view in ('source','target','pooled'):
        gt = {k:v for k,v in gts.items() if view == 'pooled' or k.endswith('/'+view)}
        pred = [p for p in predictions if p['image_id'] in gt]
        metrics[view] = {k:v for k,v in detection_ap(gt,pred).items() if k not in ('precision','recall')}
        obj_pred = [p for p in object_predictions if p['image_id'] in gt]
        object_metrics[view] = {k:v for k,v in detection_ap(gt,obj_pred).items() if k not in ('precision','recall')}
    return {'kind': 'independent_revision_diagnostic', 'completed_pairs': len(expected_ids),
        'full_dataset': len(expected_ids)==100 and set(expected_ids)==set(pairs),
        'pair_ids': expected_ids, 'metrics': metrics,
        'object_box_metrics':object_metrics,
        'object_box_convention':'removed instance boxes before pixel-priority clipping; supplemental adapter diagnostic, not an official score',
        'convention': 'native xyxy; removed after per-view priority; source and target independently classified with T0->T1 fixed; stable all-points AP50; SAM2 proposal IoU score',
        'paper_superiority_comparison_allowed': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads((args.runs/'plan.json').read_text())
    report = evaluate(args.runs, plan['pair_ids'])
    (args.runs/'metrics.json').write_text(json.dumps(report, indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
