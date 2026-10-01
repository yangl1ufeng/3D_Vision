"""Inspect saved successful clean tracks for missed annotated removals."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.pipeline import mask_box

out = ROOT/'audit/failure_analysis'
rows = json.loads((out/'per_pair.json').read_text())
results = []
for row in rows:
    if row['reason'] != 'matching_proposal_tracks_to_clean':
        continue
    folder = ROOT/'runs/rc3d_native'/row['pair']/'view_1'
    data = np.load(folder/'reference_to_target_tracking.npz')
    records = []
    for candidate in row['good_proposals']:
        i = candidate['id']
        source, track = data[f'source_{i:04d}'], data[f'clean_track_{i:04d}']
        intersect = int((source & track).sum())
        records.append({'id': i, 'source_box': mask_box(source), 'clean_track_box': mask_box(track),
            'source_area': int(source.sum()), 'track_area': int(track.sum()),
            'track_to_source_area_ratio': float(track.sum()/source.sum()),
            'mask_iou_source_clean': float(intersect/(source | track).sum()),
            'fraction_source_covered_by_track': float(intersect/source.sum())})
    data.close()
    results.append({'pair': row['pair'], 'tracks': records})
(out/'clean_track_analysis.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
ratios = [t['track_to_source_area_ratio'] for r in results for t in r['tracks']]
ious = [t['mask_iou_source_clean'] for r in results for t in r['tracks']]
print(json.dumps({'pairs': len(results), 'good_proposals': len(ratios),
    'track_area_ratio_quantiles': np.quantile(ratios, [0,.25,.5,.75,1]).tolist(),
    'mask_iou_quantiles': np.quantile(ious, [0,.25,.5,.75,1]).tolist(),
    'pairs_all_good_tracks_iou_below_01': [r['pair'] for r in results if all(t['mask_iou_source_clean'] < .1 for t in r['tracks'])]}, indent=2))
