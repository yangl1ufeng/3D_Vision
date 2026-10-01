"""Posthoc depth-shape diagnostic only. Sensor depth never enters RGB inference."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
pairs={p['id']:p for p in json.loads((ROOT/'data/rc3d_author_order_manifest.json').read_text())['pairs']}
ids=json.loads((ROOT/'audit/revision/pilot_selection.json').read_text())['ids']
rows=[]
for pid in ids:
    pair=pairs[pid]
    folder=ROOT/'runs/rc3d_native'/pid/'geometry'
    metadata=json.loads((folder/'geometry.json').read_text())
    with np.load(folder/'scene.npz') as scene:
        for view,key,boxkey in [(0,'before','before_boxes_xyxy'),(1,'after','after_boxes_xyxy')]:
            path=ROOT/pair[key]
            sensor=Image.open(path.with_name('depth_'+path.stem+'.png'))
            predicted=scene[f'depth_{view}'].astype(float)
            transform=metadata['image_transforms'][view]
            if transform['crop_left_top']!=[0,0]:
                raise ValueError('Depth diagnostic requires explicit crop mapping for this image')
            h,w=predicted.shape
            observed=np.array(sensor.resize((w,h),Image.Resampling.NEAREST),dtype=float)
            valid=np.isfinite(predicted)&(predicted>0)&(observed>0)
            scale=float(np.median(observed[valid]/predicted[valid]))
            relative=np.abs(scale*predicted-observed)/np.maximum(observed,1.)
            ow,oh=transform['original_wh']
            x0,y0,x1,y1=pair[boxkey][0]
            roi=np.zeros((h,w),dtype=bool)
            roi[max(0,int(y0*h/oh)):min(h,int(np.ceil(y1*h/oh))),max(0,int(x0*w/ow)):min(w,int(np.ceil(x1*w/ow)))]=True
            selected=roi&valid
            rows.append({'pair':pid,'view':view,'scale_fit_to_sensor_for_diagnostic_only':scale,
                'median_absolute_relative_error':float(np.median(relative[valid])),
                'mean_absolute_relative_error':float(relative[valid].mean()),
                'gt_box_roi_median_relative_error':float(np.median(relative[selected])),
                'gt_box_roi_mean_relative_error':float(relative[selected].mean())})
report={'kind':'posthoc depth comparison, not inference or benchmark',
    'assumption':'Depth map is RGB-aligned under nearest resize, following dataset author loader; sensor has finite accuracy.',
    'sensor_depth_used_in_inference':False,'per_view_scale_fit':True,
    'aggregate_median_of_view_median_errors':float(np.median([r['median_absolute_relative_error'] for r in rows])),
    'rows':rows}
(ROOT/'audit/revision/sensor_depth_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
