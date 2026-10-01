"""Compare identical fixed diagnostic IDs, never subset versus full benchmark."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from goldilocs_local.metrics import box_iou,detection_ap
from goldilocs_local.pipeline import mask_box


def predictions(folder):
    report = json.loads((folder/'prediction.json').read_text())
    labels = np.array(Image.open(folder/'labels.png'))
    result = []
    with np.load(folder/'object_masks.npz') as masks:
        for obj in report['objects']:
            if obj['label']!='removed':
                continue
            box = mask_box(masks[f"mask_{obj['id']:04d}"]&(labels==2))
            if box:
                result.append({'bbox':box,'score':obj['score']})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=ROOT/'runs/rc3d_revision_pilot_v2')
    parser.add_argument('--output',type=Path,default=ROOT/'audit/revision/comparison.json')
    args=parser.parse_args()
    run=args.runs
    status = json.loads((run/'progress.json').read_text())
    if status['status']!='complete':
        raise RuntimeError('Refusing final comparison before fixed pilot completes')
    ids = json.loads((run/'plan.json').read_text())['pair_ids']
    pairs = {p['id']:p for p in json.loads((ROOT/'data/rc3d_author_order_manifest.json').read_text())['pairs']}
    rows,old_predictions,new_predictions,gt = [],[],[],{}
    for pid in ids:
        pair = pairs[pid]
        gt[pid] = pair['after_boxes_xyxy']
        old = predictions(ROOT/'runs/rc3d_native'/pid/'view_1')
        new = predictions(run/pid/'view_1')
        source = predictions(run/pid/'view_0')
        old_predictions.extend([dict(p,image_id=pid) for p in old])
        new_predictions.extend([dict(p,image_id=pid) for p in new])
        best = lambda predictions,boxes: max((float(box_iou(p['bbox'],boxes).max()) for p in predictions),default=0.)
        rows.append({'pair':pid,'old_target_predictions':len(old),'new_target_predictions':len(new),
            'old_target_best_iou':best(old,gt[pid]),'new_target_best_iou':best(new,gt[pid]),
            'new_source_predictions':len(source),'new_source_best_iou':best(source,pair['before_boxes_xyxy'])})
    compact = lambda r:{k:v for k,v in r.items() if k not in ('precision','recall')}
    report = {'scope':'8 fixed posthoc diagnostic examples, not independent validation or full benchmark',
        'old_target_same_ids':compact(detection_ap(gt,old_predictions)),
        'new_target_same_ids':compact(detection_ap(gt,new_predictions)),
        'new_dual_view':json.loads((run/'metrics.json').read_text())['metrics'],
        'per_pair':rows}
    out = args.output
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
