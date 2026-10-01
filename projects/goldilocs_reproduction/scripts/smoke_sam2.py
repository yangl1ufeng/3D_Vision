"""Check installed SAM2 automatic segmentation and identity/blank tracking."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from goldilocs_local.segmentation import SegmentTracker

settings = json.loads((ROOT/'configs/pairwise_prototype.json').read_text())
settings.update(points_per_side=8, crop_n_layers=0, pred_iou_thresh=.8, stability_score_thresh=.8)
image = Image.open(ROOT/'third_party/sam2/notebooks/images/truck.jpg').convert('RGB')
image.thumbnail((512, 512))
rgb = np.array(image)
valid = np.ones(rgb.shape[:2], dtype=bool)
started = time.perf_counter()
tracker = SegmentTracker(ROOT/'checkpoints/sam2_hiera_large.pt', settings['sam_config'], settings)
records = tracker.masks(rgb, valid)
if not records:
    raise RuntimeError('No SAM2 masks returned on official example')
record = sorted(records, key=lambda r: r['predicted_iou'], reverse=True)[0]
identity = tracker.track(rgb, rgb, [record['segmentation']], valid)[0]
blank = tracker.track(rgb, rgb*0, [record['segmentation']], valid)[0]
union = (identity['mask'] | record['segmentation']).sum()
report = {'kind': 'component_smoke_test_not_benchmark', 'automatic_masks': len(records),
          'identity_track_success': identity['success'],
          'identity_mask_iou': float((identity['mask'] & record['segmentation']).sum()/union),
          'blank_track_success': blank['success'], 'seconds': time.perf_counter()-started}
tracker.close()
(ROOT/'audit/sam2_smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
