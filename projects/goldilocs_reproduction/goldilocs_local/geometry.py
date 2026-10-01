"""World-coordinate pointmaps, pinhole projection and reverse-depth filtering.

Rasterization choices are local implementation assumptions, not author code.
"""
import numpy as np


def project(points, intrinsic, camera_to_world):
    p = np.asarray(points).reshape(-1, 3)
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = p @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    pixels = camera @ intrinsic.T
    uv = np.full((len(p), 2), np.nan)
    good = np.isfinite(camera).all(axis=1) & (z > 1e-6)
    uv[good] = pixels[good, :2] / pixels[good, 2:3]
    return uv, z, good


def rasterize(points, colors, intrinsic, camera_to_world, shape, radius=1):
    """Nearest-depth z-buffer with square pixel splats; zero RGB means a hole only
    when the separately returned valid map is false. Black objects stay valid.
    """
    if radius < 0:
        raise ValueError('radius must be nonnegative')
    h, w = shape
    colors = np.asarray(colors).reshape(-1, 3)
    uv, z, good = project(points, intrinsic, camera_to_world)
    good &= (uv[:, 0] >= -radius-.5) & (uv[:, 0] < w+radius-.5)
    good &= (uv[:, 1] >= -radius-.5) & (uv[:, 1] < h+radius-.5)
    ids = np.flatnonzero(good)
    # Sort by depth once, rather than constructing radius^2 copies of all points.
    ids = ids[np.lexsort((ids, z[ids]))]
    xy = np.floor(uv[ids] + .5).astype(np.int64)
    rgb = np.zeros((h*w, 3), dtype=np.uint8)
    depth = np.full(h*w, np.inf, dtype=np.float64)
    owner = np.full(h*w, np.iinfo(np.int64).max, dtype=np.int64)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            xx, yy = xy[:, 0]+dx, xy[:, 1]+dy
            inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            pixels, first = np.unique(yy[inside]*w + xx[inside], return_index=True)
            sources = ids[inside][first]
            take = (z[sources] < depth[pixels]) | ((z[sources] == depth[pixels]) & (sources < owner[pixels]))
            pixels, sources = pixels[take], sources[take]
            rgb[pixels] = colors[sources]
            depth[pixels] = z[sources]
            owner[pixels] = sources
    return rgb.reshape(h, w, 3), depth.reshape(h, w).astype(np.float32), np.isfinite(depth).reshape(h, w)


def reverse_depth_keep(points, other_depth, other_intrinsic, other_pose, tolerance=0.0):
    """Remove a source point if it lies in observed free space in the other view.

    Keep points outside the other frustum or behind its foreground. Their status
    is unobserved/occluded, not geometric evidence of change.
    """
    uv, z, good = project(points, other_intrinsic, other_pose)
    h, w = other_depth.shape
    good &= (uv[:, 0] >= -.5) & (uv[:, 0] < w-.5)
    good &= (uv[:, 1] >= -.5) & (uv[:, 1] < h-.5)
    ids = np.flatnonzero(good)
    xy = np.floor(uv[ids] + .5).astype(np.int64)
    observed = other_depth[xy[:, 1], xy[:, 0]]
    remove = np.isfinite(observed) & (observed > 0) & (z[ids] < observed-tolerance)
    keep = np.isfinite(np.asarray(points).reshape(-1, 3)).all(axis=1)
    keep[ids[remove]] = False
    return keep


def visibility_accept(mask, valid, alpha, rule):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha outside [0, 1]')
    if rule not in ('main_text', 'appendix'):
        raise ValueError('alpha rule must be explicit')
    n = np.count_nonzero(mask)
    fraction = np.count_nonzero(mask & valid)/n if n else 0.0
    threshold = 1-alpha if rule == 'main_text' else alpha
    return bool(n and fraction >= threshold), float(fraction)


def original_resolution_intrinsic(intrinsic, transform):
    """Undo resize/crop in the camera matrix, including pixel-center offsets."""
    ow, oh = transform['original_wh']
    rw, rh = transform['resized_wh']
    left, top = transform['crop_left_top']
    sx, sy = ow/rw, oh/rh
    inverse_preprocessing = np.array([[sx, 0, sx*(left+.5)-.5],
                                      [0, sy, sy*(top+.5)-.5], [0, 0, 1]])
    return inverse_preprocessing @ intrinsic


def render_pair_geometry(scene, target, radius=1, tolerance=0.0, disable_cleaning=False,
                         render_intrinsic=None, render_shape=None):
    other = 1-target
    pts = [scene[f'points_{i}'].reshape(-1, 3) for i in (0, 1)]
    rgb = [scene[f'rgb_{i}'].reshape(-1, 3) for i in (0, 1)]
    ks, poses = [scene[f'K_{i}'] for i in (0, 1)], [scene[f'pose_{i}'] for i in (0, 1)]
    depth = [scene[f'depth_{i}'] for i in (0, 1)]
    keep = [reverse_depth_keep(pts[i], depth[1-i], ks[1-i], poses[1-i], tolerance) for i in (0, 1)]
    if disable_cleaning:
        keep = [np.isfinite(p).all(axis=1) for p in pts]
    shape = render_shape if render_shape is not None else scene[f'rgb_{target}'].shape[:2]
    render_k = render_intrinsic if render_intrinsic is not None else ks[target]
    cross, _, valid = rasterize(pts[other], rgb[other], render_k, poses[target], shape, radius)
    clean, _, clean_valid = rasterize(np.concatenate([p[k] for p, k in zip(pts, keep)]),
                                     np.concatenate([c[k] for c, k in zip(rgb, keep)]),
                                     render_k, poses[target], shape, radius)
    return cross, clean, valid, clean_valid, [float(k.mean()) for k in keep]
