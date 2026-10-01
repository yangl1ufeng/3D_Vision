"""Fixed identity controls without reading RC-3D images or annotations."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from goldilocs_local.segmentation import SegmentTracker
from goldilocs_local.identity import verify_tracks


def main():
    settings=json.loads((ROOT/'configs/rc3d_verified_main.json').read_text())
    settings.update(points_per_side=8,crop_n_layers=0,pred_iou_thresh=.8,stability_score_thresh=.8)
    image=Image.open(ROOT/'third_party/sam2/notebooks/images/truck.jpg').convert('RGB')
    image.thumbnail((512,512));rgb=np.array(image)
    support=np.ones(rgb.shape[:2],dtype=bool)
    tracker=SegmentTracker(ROOT/'checkpoints/sam2_hiera_large.pt',settings['sam_config'],settings)
    try:
        records=tracker.masks(rgb,support)
        mask=max(records,key=lambda r:r['predicted_iou'])['segmentation']
        shifted=np.zeros_like(rgb);shifted[:,40:]=rgb[:,:-40]
        unrelated=np.array(Image.open(ROOT/'third_party/sam2/notebooks/images/groceries.jpg').convert('RGB').resize(image.size))
        cars=np.array(Image.open(ROOT/'third_party/sam2/notebooks/images/cars.jpg').convert('RGB').resize(image.size))
        ys,xs=np.nonzero(mask)
        residue=np.zeros_like(rgb)
        cy,cx=int(np.median(ys)),int(np.median(xs))
        residue[max(0,cy-4):cy+5,max(0,cx-4):cx+5]=rgb[max(0,cy-4):cy+5,max(0,cx-4):cx+5]
        rows=[]
        cases=[('identity',rgb,True,True),('brightness',np.uint8(rgb*.65),True,True),
               ('translation',shifted,True,False),('blank',rgb*0,False,True),
               ('unrelated_groceries',unrelated,False,True),('different_vehicle_instances',cars,False,False),
               ('tiny_residue',residue,False,True)]
        for name,target,present,aligned in cases:
            result=verify_tracks(tracker,rgb,target,[mask],support,settings,aligned=aligned)[0]
            rows.append({'case':name,'expected_verified_presence':present,'status':result['status'],
                         'identity_status':result['identity_status'],'cycle_iou':result.get('cycle_iou'),
                         'aligned_iou':result.get('aligned_iou'),'object_presence_logit':result.get('object_presence_logit'),
                         'passed':result['success']==present})
        report={'rc3d_data_or_labels_used':False,'fixed_thresholds':{'cycle_iou':.5,'aligned_iou':.5},
                'controls':rows,'all_passed':all(r['passed'] for r in rows)}
        out=ROOT/'audit/fix_round2';out.mkdir(parents=True,exist_ok=True)
        (out/'identity_controls.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
        if not report['all_passed']:raise RuntimeError('Identity controls failed; keep threshold fixed and diagnose')
    finally:
        tracker.close()


if __name__=='__main__':
    main()
