"""Cached geometric transfer versus SAM transfer; GT is diagnostic only."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from goldilocs_local.geometry import rasterize, original_resolution_intrinsic
from goldilocs_local.metrics import box_iou
from goldilocs_local.pipeline import mask_box


def main():
    pairs={p['id']:p for p in json.loads((ROOT/'data/rc3d_author_order_manifest.json').read_text())['pairs']}
    probes=json.loads((ROOT/'audit/failure_analysis/probes/summary.json').read_text())
    rows=[]
    for probe in probes:
        pid=probe['pair'];pair=pairs[pid]
        folder=ROOT/'runs/rc3d_revision_repaired_pilot'/pid/'view_0'
        trace=json.loads((folder/'reference_to_target_tracking.json').read_text())
        best=max(trace,key=lambda t:box_iou(t['bbox_xyxy'],pair['before_boxes_xyxy'])[0])
        with np.load(folder/'reference_to_target_tracking.npz') as masks:
            native=masks[f"source_{best['id']:04d}"]
        geo=ROOT/'runs/rc3d_native'/pid/'geometry'
        meta=json.loads((geo/'geometry.json').read_text())
        transform=meta['image_transforms'][0]
        resized=Image.fromarray(native).resize(tuple(transform['resized_wh']),Image.Resampling.NEAREST)
        left,top=transform['crop_left_top'];w,h=transform['model_wh']
        grid=np.array(resized.crop((left,top,left+w,top+h)))
        with np.load(geo/'scene.npz') as scene:
            colors=np.repeat((grid.ravel()*255).astype('uint8')[:,None],3,axis=1)
            intrinsic=original_resolution_intrinsic(scene['K_1'],meta['image_transforms'][1])
            ow,oh=meta['image_transforms'][1]['original_wh']
            rgb,_,_=rasterize(scene['points_0'],colors,intrinsic,scene['pose_1'],(oh,ow),radius=4)
            # Counterfactual attribution only: preserve estimated cameras and
            # other scene points, replace object depths with aligned sensor data.
            image=ROOT/pair['before']
            sensor=Image.open(image.with_name('depth_'+image.stem+'.png'))
            sensor=sensor.resize(tuple(transform['resized_wh']),Image.Resampling.NEAREST)
            observed=np.array(sensor.crop((left,top,left+w,top+h)),dtype=float)
            predicted=scene['depth_0'].astype(float)
            valid=np.isfinite(predicted)&(predicted>0)&(observed>0)
            sensor_to_model_scale=float(np.median(predicted[valid]/observed[valid]))
            yy,xx=np.indices((h,w))
            rays=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(scene['K_0']).T
            camera_points=rays*(observed*sensor_to_model_scale)[...,None]
            sensor_points=camera_points@scene['pose_0'][:3,:3].T+scene['pose_0'][:3,3]
            replace=grid.astype(bool)&valid
            points=scene['points_0'].copy()
            points[replace]=sensor_points[replace]
            sensor_rgb,_,_=rasterize(points,colors,intrinsic,scene['pose_1'],(oh,ow),radius=4)
        transferred=rgb[:,:,0]>0;box=mask_box(transferred)
        sensor_box=mask_box(sensor_rgb[:,:,0]>0)
        rows.append({'pair':pid,'source_box_iou':float(box_iou(best['bbox_xyxy'],pair['before_boxes_xyxy'])[0]),
                     'geometric_transfer_box':box,'geometric_transfer_iou':float(box_iou(box,pair['after_boxes_xyxy'])[0]) if box else 0.,
                     'sam_transfer_best_iou':probe['native_to_cross_best_iou_any_track'],
                     'sensor_object_depth_counterfactual_box':sensor_box,
                     'sensor_object_depth_counterfactual_iou':float(box_iou(sensor_box,pair['after_boxes_xyxy'])[0]) if sensor_box else 0.,
                     'sensor_to_model_scale_diagnostic_only':sensor_to_model_scale,
                     'object_sensor_coverage':float(replace.sum()/max(1,grid.sum()))})
    out=ROOT/'audit/fix_round2';out.mkdir(parents=True,exist_ok=True)
    report={'purpose':'posthoc transfer attribution only; not a new inference candidate or parameter tuning',
            'estimated_camera_poses_preserved':True,
            'sensor_depth_used_in_counterfactual_only':True,
            'sensor_depth_used_in_inference':False,
            'sensor_alignment_assumption':'Nearest resize, following dataset loader; sensor registration and calibration are not independently verified.',
            'rows':rows}
    (out/'geometric_transfer_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
