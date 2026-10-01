"""Post-hoc component probes on four deterministic failure examples.

No parameters are tuned. Labels only score already-generated predictions.
This small, failure-selected sample is NOT a validation/benchmark result.
"""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.segmentation import SegmentTracker
from goldilocs_local.pipeline import mask_box
from goldilocs_local.metrics import box_iou


def main():
    import torch
    torch.manual_seed(0)
    out = ROOT/'audit/failure_analysis/probes'
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((out.parent/'per_pair.json').read_text())
    chosen = [r for r in rows if r['reason'] == 'no_matching_cross_proposal'][:3]
    chosen += [r for r in rows if r['reason'] == 'matching_proposal_tracks_to_clean'][:1]
    pairs = {p['id']: p for p in json.loads((ROOT/'data/rc3d_manifest.json').read_text())['pairs']}
    gt0 = {}
    for part in ('part1','part2','part3','part4'):
        data = json.loads((ROOT/'data/rc3d'/part/'coco_annotations.json').read_text())
        for im, ann in zip(data['images'][0::3], data['annotations'][0::3]):
            x,y,w,h = ann['bbox']
            gt0[f"data/rc3d/{part}/{im['file_name']}"] = [x,y,x+w,y+h]
    settings = json.loads((ROOT/'configs/rc3d_native.json').read_text())
    tracker = SegmentTracker(ROOT/'checkpoints/sam2_hiera_large.pt', settings['sam_config'], settings)
    reports = []
    try:
        for row in chosen:
            pid = row['pair']
            destination = out/pid
            destination.mkdir(exist_ok=True)
            if (destination/'probe.json').exists():
                reports.append(json.loads((destination/'probe.json').read_text()))
                continue
            print(f'PROBE {pid}', flush=True)
            pair = pairs[pid]
            folder = ROOT/'runs/rc3d_native'/pid/'view_1'
            source = np.array(Image.open(ROOT/pair['before']).convert('RGB'))
            cross = np.array(Image.open(folder/'cross_render.png'))
            valid = np.array(Image.open(folder/'valid.png')) > 0
            records = tracker.masks(source, np.ones(source.shape[:2], dtype=bool))
            tracks = tracker.track(source, cross, [r['segmentation'] for r in records], valid)
            before_box = gt0[pair['before']]
            proposed = [mask_box(r['segmentation']) for r in records]
            transferred = [mask_box(t['mask']) for t in tracks]
            ious0 = [float(box_iou(b, [before_box])[0]) for b in proposed]
            ious1 = [float(box_iou(b, [row['gt']])[0]) if b else 0. for b in transferred]
            report = {'pair': pid, 'selection': row['reason'], 'original_cross_best_iou': row['best_proposal_iou'],
                'native_source_best_iou': max(ious0, default=0),
                'native_to_cross_best_iou_any_track': max(ious1, default=0),
                'native_to_cross_best_iou_visible_track': max([v for v,t in zip(ious1,tracks) if t['success']], default=0),
                'proposals': [{'source_box': b0, 'cross_box': b1, 'iou_source': i0, 'iou_cross': i1,
                    'visible_fraction': t['visible_fraction'], 'old_track_success': t['success']}
                    for b0,b1,i0,i1,t in zip(proposed,transferred,ious0,ious1,tracks)]}
            best = int(np.argmax(ious0)) if ious0 else None
            report['source_gt_best_track_iou'] = ious1[best] if best is not None else 0
            (destination/'probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            panels = []
            for title, arr, gt, pred in [
                ('Original T0 / green GT, red proposal', source, before_box, proposed[best] if best is not None else None),
                ('Reprojection / transferred proposal', cross, row['gt'], transferred[best] if best is not None else None),
                ('Clean render / green GT location', np.array(Image.open(folder/'clean_render.png')), row['gt'], None),
                ('Target T1 / green GT location', np.array(Image.open(folder/'target.png')), row['gt'], None)]:
                im = Image.fromarray(arr)
                draw = ImageDraw.Draw(im)
                draw.rectangle(gt, outline='#00ff55', width=8)
                if pred:
                    draw.rectangle(pred, outline='#ff4444', width=6)
                im.thumbnail((640,480))
                panel = Image.new('RGB', (640,510), '#202020')
                panel.paste(im, (0,30))
                ImageDraw.Draw(panel).text((10,8), title, fill='white')
                panels.append(panel)
            canvas = Image.new('RGB', (1280,1020))
            for n, panel in enumerate(panels):
                canvas.paste(panel, ((n%2)*640,(n//2)*510))
            canvas.save(destination/'evidence.jpg', quality=93)
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k != 'proposals'}), flush=True)
    finally:
        tracker.close()
    (out/'summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k != 'proposals'} for r in reports], indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
