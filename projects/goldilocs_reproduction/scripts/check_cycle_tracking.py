"""Identity controls from SAM2 demo images, independent of RC-3D labels."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from goldilocs_local.segmentation import SegmentTracker,track_with_cycle

settings=json.loads((ROOT/'configs/rc3d_revision.json').read_text())
settings.update(points_per_side=8,crop_n_layers=0,pred_iou_thresh=.8,stability_score_thresh=.8)
image=Image.open(ROOT/'third_party/sam2/notebooks/images/truck.jpg').convert('RGB')
image.thumbnail((512,512))
rgb=np.array(image)
support=np.ones(rgb.shape[:2],dtype=bool)
tracker=SegmentTracker(ROOT/'checkpoints/sam2_hiera_large.pt',settings['sam_config'],settings)
try:
    records=tracker.masks(rgb,support)
    mask=max(records,key=lambda r:r['predicted_iou'])['segmentation']
    shifted=np.zeros_like(rgb)
    shifted[:,40:]=rgb[:,:-40]
    unrelated=np.array(Image.open(ROOT/'third_party/sam2/notebooks/images/groceries.jpg').convert('RGB').resize(image.size))
    cars=np.array(Image.open(ROOT/'third_party/sam2/notebooks/images/cars.jpg').convert('RGB').resize(image.size))
    rows=[]
    for name,target,present in [('identity',rgb,True),('brightness',np.uint8(rgb*.65),True),
                                ('translation',shifted,True),('blank',rgb*0,False),('unrelated_groceries',unrelated,False),('different_vehicle_instances',cars,False)]:
        result=track_with_cycle(tracker,rgb,target,[mask],support,min_iou=.5)[0]
        rows.append({'case':name,'expected_presence':present,'status':result['status'],
                     'cycle_iou':result.get('cycle_iou'),'reason':result.get('identity_reason'),'passed':(result['status']=='present')==present})
    report={'threshold_fixed_before_test':.5,'rc3d_ground_truth_used':False,'controls':rows,'all_passed':all(r['passed'] for r in rows)}
    (ROOT/'audit/revision/cycle_controls.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
finally:
    tracker.close()
