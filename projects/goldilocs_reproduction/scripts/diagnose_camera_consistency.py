"""CPU-only cached correspondence consistency; no camera ground truth claimed."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from goldilocs_local.geometry import project


def sampson_pixels(f,a,b):
    ax=np.c_[a,np.ones(len(a))];bx=np.c_[b,np.ones(len(b))]
    fa=ax@f.T;fb=bx@f
    numerator=np.sum(bx*fa,axis=1)**2
    denominator=np.sum(fa[:,:2]**2+fb[:,:2]**2,axis=1)
    return np.sqrt(numerator/np.maximum(denominator,1e-20))


def describe(values):
    return {'median_pixels':float(np.median(values)),
            'p90_pixels':float(np.quantile(values,.9))}


def main():
    ids=json.loads((ROOT/'audit/revision/pilot_selection.json').read_text())['ids']
    rows=[]
    for pid in ids:
        geo=ROOT/'runs/rc3d_native'/pid/'geometry'
        meta=json.loads((geo/'geometry.json').read_text())
        hashes=[hashlib.md5(p.encode()).hexdigest() for p in meta['paths']]
        path=geo/'mast3r_cache'/'corres_conf=conf_subsample=8'/f'{hashes[0]}-{hashes[1]}.pth'
        _,(a,b,confidence)=torch.load(path,map_location='cpu',weights_only=False)
        a,b=a.numpy().astype(float),b.numpy().astype(float)
        with np.load(geo/'scene.npz') as scene:
            relative=np.linalg.inv(scene['pose_1'])@scene['pose_0']
            r,t=relative[:3,:3],relative[:3,3]
            cross_t=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
            f=np.linalg.inv(scene['K_1']).T@cross_t@r@np.linalg.inv(scene['K_0'])
            f_fit,inliers=cv2.findFundamentalMat(a,b,cv2.USAC_MAGSAC,1.,.999,10000)
            if f_fit is None or inliers is None:
                raise ValueError('Correspondence fit failed: '+pid)
            inliers=inliers.ravel().astype(bool)
            uv,z,good=project(scene['points_0'][a[:,1].astype(int),a[:,0].astype(int)],scene['K_1'],scene['pose_1'])
            reprojection=np.linalg.norm(uv-b,axis=1)
            usable=inliers&good&np.isfinite(reprojection)
        rows.append({'pair':pid,'cached_matches':len(a),'ransac_inliers':int(inliers.sum()),
            'estimated_camera_sampson_all':describe(sampson_pixels(f,a,b)),
            'estimated_camera_sampson_on_inliers':describe(sampson_pixels(f,a[inliers],b[inliers])),
            'fitted_fundamental_sampson_on_inliers':describe(sampson_pixels(f_fit,a[inliers],b[inliers])),
            'pointmap_reprojection_on_inliers':describe(reprojection[usable])})
    report={'kind':'cached RGB correspondence consistency only; no algorithm changes or new GPU inference',
        'coordinates':'512x384 MASt3R model grid',
        'robust_fit_settings':{'method':'USAC_MAGSAC','threshold_pixels':1.,'confidence':.999,'max_iterations':10000},
        'interpretation':'Both tests reuse MASt3R feature matches. Robust-fit inliers are not ground-truth matches; results cannot establish camera accuracy independently.',
        'rows':rows}
    (ROOT/'audit/fix_round2/camera_consistency.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
