"""Seeded, separate sensor/mask perturbations; no changes to reference geometry."""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation


def degrade(depth, mask, kind, level, seed):
    depth, mask = np.array(depth, copy=True), np.asarray(mask, bool).copy()
    rng = np.random.default_rng(seed)
    valid = depth > 0
    if kind == 'noise':
        depth[valid] += rng.normal(0, level, valid.sum())
        depth[depth <= 0] = 0
    elif kind == 'dropout':
        depth[valid & (rng.random(depth.shape) < level)] = 0
    elif kind == 'hole':
        y, x = np.nonzero(mask)
        cy, cx = (y.min()+y.max())/2, (x.min()+x.max())/2
        hy, hx = np.sqrt(level)*(y.max()-y.min()+1)/2, np.sqrt(level)*(x.max()-x.min()+1)/2
        yy, xx = np.indices(depth.shape)
        depth[(abs(yy-cy) <= hy) & (abs(xx-cx) <= hx)] = 0
    elif kind in ['erode', 'dilate']:
        operation = binary_erosion if kind == 'erode' else binary_dilation
        mask = operation(mask, structure=np.ones((3, 3), bool), iterations=int(level))
    elif kind != 'clean':
        raise ValueError(f'unknown degradation {kind}')
    return depth, mask


def perturb_box(box, shape, kind, level):
    box = np.asarray(box, float)
    center, extent = (box[:2]+box[2:])/2, box[2:]-box[:2]
    if kind == 'box_shift':
        center += [extent[0]*level, 0]
    elif kind == 'box_scale':
        extent *= level
    elif kind != 'box_clean':
        raise ValueError(kind)
    lo, hi = center-extent/2, center+extent/2
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [shape[1]-1, shape[0]-1])
    if np.any(hi <= lo):
        raise ValueError('box outside image')
    return np.r_[lo, hi].tolist()


def mask_metrics(mask, reference):
    mask, reference = np.asarray(mask, bool), np.asarray(reference, bool)
    intersection = np.sum(mask & reference)
    union = np.sum(mask | reference)
    return {'mask_iou': float(intersection/union) if union else 1.,
            'mask_precision': float(intersection/mask.sum()) if mask.any() else 0.,
            'mask_recall': float(intersection/reference.sum()) if reference.any() else None}


def voxel_centroids(points, voxel_m):
    points = np.asarray(points)
    if len(points) == 0:
        return points.copy(), np.empty(0, dtype=int)
    _, inverse, counts = np.unique(np.floor(points/voxel_m).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    reduced = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(reduced, inverse, points)
    return (reduced/counts[:, None]).astype(np.float32), inverse


def bounded_points(points, limit, seed, voxel_m=None):
    from offline_core.pointcloud import select_points
    prepared = voxel_centroids(points, voxel_m)[0] if voxel_m else points
    return select_points(prepared, limit, np.random.default_rng(seed))
