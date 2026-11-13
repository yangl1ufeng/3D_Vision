"""Register frozen Gemini 2 RGB-D views using matched static 3D features."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def load_view(folder):
    folder = folder.resolve()
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    rgb = np.asarray(Image.open(folder/'rgb.png').convert('RGB'))
    depth = np.load(folder/'depth_m.npy', allow_pickle=False)
    if rgb.shape[:2] != depth.shape or list(depth.shape) != report['depth_shape']:
        raise ValueError(f'Image/depth size mismatch: {folder}')
    return {'folder': folder, 'rgb': rgb, 'depth': depth,
            'K': report['aligned_depth_intrinsic']}


def backproject(view, xy):
    u = np.rint(xy[:, 0]).astype(int)
    v = np.rint(xy[:, 1]).astype(int)
    depth = view['depth']
    in_bounds = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    z = np.zeros(len(u), dtype=np.float64)
    z[in_bounds] = depth[v[in_bounds], u[in_bounds]]
    good = in_bounds & np.isfinite(z) & (z > .2) & (z < 2.)
    K = view['K']
    xyz = np.column_stack(((u-K['cx'])*z/K['fx'],
                           (v-K['cy'])*z/K['fy'], z))
    return xyz, good


def fit_rigid(source, target):
    """Fit SE(3) mapping source camera points into target camera points."""
    a, b = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source-a).T @ (target-b))
    correction = np.diag([1, 1, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ correction @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = b-R@a
    return T


def match_pair(reference, moving, seed=1000):
    sift = cv2.SIFT_create(nfeatures=6000)
    kp_a, des_a = sift.detectAndCompute(cv2.cvtColor(reference['rgb'],cv2.COLOR_RGB2GRAY),None)
    kp_b, des_b = sift.detectAndCompute(cv2.cvtColor(moving['rgb'],cv2.COLOR_RGB2GRAY),None)
    if des_a is None or des_b is None:
        raise ValueError('No SIFT features in one camera view')
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a,des_b,k=2)
    matches = [m for m,n in pairs if m.distance < .8*n.distance]
    if len(matches) < 30:
        raise ValueError(f'Only {len(matches)} RGB matches; need more common scene')
    a_xy = np.asarray([kp_a[m.queryIdx].pt for m in matches])
    b_xy = np.asarray([kp_b[m.trainIdx].pt for m in matches])
    a_xyz, a_ok = backproject(reference,a_xy)
    b_xyz, b_ok = backproject(moving,b_xy)
    both = a_ok & b_ok
    a_xyz,b_xyz = a_xyz[both],b_xyz[both]
    print(f'{reference["folder"].name} <- {moving["folder"].name}: '
          f'{len(matches)} RGB matches, {len(a_xyz)} with measured depth', flush=True)
    if len(a_xyz) < 8:
        raise ValueError(f'Only {len(a_xyz)} RGB-D matches with depth')
    rng = np.random.default_rng(seed)
    best = np.zeros(len(a_xyz), bool)
    for _ in range(2500):
        ids = rng.choice(len(a_xyz),3,replace=False)
        if np.linalg.norm(np.cross(b_xyz[ids[1]]-b_xyz[ids[0]],
                                   b_xyz[ids[2]]-b_xyz[ids[0]])) < .0001:
            continue
        T = fit_rigid(b_xyz[ids],a_xyz[ids])
        residual = np.linalg.norm(b_xyz@T[:3,:3].T+T[:3,3]-a_xyz,axis=1)
        inliers = residual < .025
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 7:
        raise ValueError(f'Only {best.sum()} RANSAC inliers; cannot trust alignment')
    T = fit_rigid(b_xyz[best],a_xyz[best])
    residual = np.linalg.norm(b_xyz@T[:3,:3].T+T[:3,3]-a_xyz,axis=1)
    inliers = residual < .025
    T = fit_rigid(b_xyz[inliers],a_xyz[inliers])
    residual = np.linalg.norm(b_xyz@T[:3,:3].T+T[:3,3]-a_xyz,axis=1)
    p95 = float(np.percentile(residual[inliers],95))
    if inliers.sum() < 7 or p95 > .025:
        raise ValueError('Alignment residual exceeds 25 mm gate')
    return T, {'rgb_matches':len(matches),'rgbd_matches':len(a_xyz),
               'inliers_25mm':int(inliers.sum()),
               'inlier_fraction':float(inliers.mean()),
               'inlier_p50_m':float(np.median(residual[inliers])),
               'inlier_p95_m':p95,
               'note':'RGB-D feature fit only; does not measure all-surface alignment.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('views',nargs='+',type=Path)
    args=parser.parse_args()
    if len(args.views)<2:
        parser.error('Need at least two camera views')
    views=[load_view(p) for p in args.views]
    transforms=[np.eye(4)]
    quality=[]
    for i in range(1,len(views)):
        # Adjacent views usually have more overlap than first-to-last.
        pair,stats=match_pair(views[i-1],views[i],seed=1000+i)
        transforms.append(transforms[-1]@pair)
        quality.append({'from_view':i,'to_view':i-1,**stats})
    out=ROOT/'runs/gemini_multiview'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    result={'views':[str(v['folder']) for v in views],
            'T_reference_camera':[T.tolist() for T in transforms],
            'pair_quality':quality,
            'frame':'first camera optical frame, metres',
            'scope':'Feature-based rigid alignment of static measured surfaces; no completion.'}
    (out/'registration.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print(out)


if __name__=='__main__':
    main()
