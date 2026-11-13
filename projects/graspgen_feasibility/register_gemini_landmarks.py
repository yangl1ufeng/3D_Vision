"""Register fixed RGB-D views from a table plane and two stationary objects."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from fit_static_scene import fit_plane, points_in_roi, points_in_polygon

ROOT = Path(__file__).resolve().parent


def examine(spec):
    folder = ROOT/spec['capture']
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    depth = np.load(folder/'depth_m.npy', allow_pickle=False)
    K = report['aligned_depth_intrinsic']
    table = np.concatenate([points_in_roi(depth,K,roi)[0] for roi in spec['table_rois']])
    n, offset, inliers, err = fit_plane(table)
    x = np.array([1.,0,0]); x -= n*(x@n); x /= np.linalg.norm(x)
    y = np.cross(n,x)
    T_camera_table = np.eye(4)
    T_camera_table[:3,:3] = np.column_stack((x,y,n))
    T_camera_table[:3,3] = -offset*n
    landmarks = {}
    object_clouds = {}
    for name, min_height in [('box',.035),('cup',.08)]:
        points,coverage = points_in_polygon(depth,K,spec[f'{name}_polygon'])
        local = (points-T_camera_table[:3,3]) @ T_camera_table[:3,:3]
        above = local[local[:,2]>min_height]
        if len(above) < 100:
            raise ValueError(f'Insufficient {name} points for {folder.name}')
        top = np.percentile(above[:,2],95)
        surface = above[above[:,2] >= top-.018]
        if len(surface) < 50:
            raise ValueError(f'Insufficient {name} upper-surface points')
        center,extent,angle = cv2.minAreaRect(surface[:,:2].astype(np.float32))
        landmarks[name] = np.asarray(center,float)
        object_clouds[name] = local
        print(folder.name,name,'depth coverage',round(coverage,3),'top',round(top,3),
              'center',np.round(center,3),'extent',np.round(extent,3),flush=True)
    return {'folder':str(folder.resolve()),'depth':depth,'K':K,'T_camera_table':T_camera_table,
            'landmarks':landmarks,'objects':object_clouds,
            'table_p95_error_m':err,'table_inlier_fraction':float(inliers.mean())}


def align_table(reference,moving):
    a = reference['landmarks']; b = moving['landmarks']
    va = a['cup']-a['box']; vb = b['cup']-b['box']
    if min(np.linalg.norm(va),np.linalg.norm(vb))<.05:
        raise ValueError('Objects too close for reliable yaw registration')
    yaw = np.arctan2(va[1],va[0])-np.arctan2(vb[1],vb[0])
    c,s = np.cos(yaw),np.sin(yaw)
    R = np.array([[c,-s],[s,c]])
    shift = (a['box']+a['cup'])/2 - R @ ((b['box']+b['cup'])/2)
    T = np.eye(4)
    T[:2,:2] = R
    T[:2,3] = shift
    landmark_residual = [float(np.linalg.norm(R@b[name]+shift-a[name])) for name in ('box','cup')]
    # Check overlapping measured surfaces, not merely the two fitted centres.
    distances={}
    for name in ('box','cup'):
        a_obj=reference['objects'][name]
        b_obj=moving['objects'][name].copy()
        b_obj[:,:2] = b_obj[:,:2]@R.T+shift
        a_obj=a_obj[::max(1,len(a_obj)//8000)]
        b_obj=b_obj[::max(1,len(b_obj)//8000)]
        nearest=cKDTree(a_obj).query(b_obj,k=1)[0]
        distances[name]={'p50_m':float(np.percentile(nearest,50)),
                         'p90_m':float(np.percentile(nearest,90)),
                         'fraction_within_3cm':float((nearest<.03).mean())}
    return T,landmark_residual,distances


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--limit',type=int,help='Use only the first N configured views')
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.limit is not None:
        if args.limit < 2:
            parser.error('--limit must be at least 2')
        config['views']=config['views'][:args.limit]
    if len(config['views'])<2:
        raise ValueError('Need at least two views')
    views=[examine(spec) for spec in config['views']]
    transforms=[np.eye(4)]
    stats=[]
    for i in range(1,len(views)):
        T,landmark_error,distances=align_table(views[0],views[i])
        transforms.append(T)
        stats.append({'view':i,'landmark_error_m':landmark_error,
                      'surface_nearest_distance':distances,
                      'passed':max(landmark_error)<.04 and
                               all(v['p50_m']<.03 for v in distances.values())})
    out=ROOT/'runs/gemini_multiview'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    result={'views':[v['folder'] for v in views],
            'T_reference_table_view_table':[T.tolist() for T in transforms],
            'T_view_camera_table':[v['T_camera_table'].tolist() for v in views],
            'table_fit_p95_error_m':[v['table_p95_error_m'] for v in views],
            'landmarks_table_xy_m':[{k:xy.tolist() for k,xy in v['landmarks'].items()} for v in views],
            'registration_quality':stats,
            'passed':all(s['passed'] for s in stats),
            'scope':'Manual object polygons; measured depth surfaces; no size priors or inferred back surfaces.'}
    (out/'registration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print(out)
    if not result['passed']:
        raise SystemExit('Registration quality gate failed; views were not merged')


if __name__=='__main__':
    main()
