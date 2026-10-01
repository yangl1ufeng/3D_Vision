"""Validate internal camera/pointmap consistency, without using test labels."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.geometry import project

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('scene', type=Path)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
scene = np.load(args.scene, allow_pickle=False)
out = {'kind': 'internal_consistency_not_reconstruction_accuracy', 'views': {}}
for i in (0, 1):
    h, w = scene[f'rgb_{i}'].shape[:2]
    uv, z, valid = project(scene[f'points_{i}'], scene[f'K_{i}'], scene[f'pose_{i}'])
    yy, xx = np.mgrid[:h, :w]
    error = np.linalg.norm(uv-np.stack([xx, yy], axis=-1).reshape(-1, 2), axis=1)
    out['views'][str(i)] = {
        'median_reprojection_px': float(np.median(error[valid])),
        'max_reprojection_px': float(np.max(error[valid])),
        'max_depth_difference': float(np.max(np.abs(z[valid]-scene[f'depth_{i}'].ravel()[valid])))}
args.out.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
