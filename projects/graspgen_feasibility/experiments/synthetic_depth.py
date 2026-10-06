"""Deterministic pinhole z-depth from analytic ray/solid-box intersections."""
import numpy as np
from offline_core.observation import DepthView


def look_at(eye, at, up=(0, 0, 1)):
    eye, at, up = map(lambda v: np.asarray(v, dtype=float), (eye, at, up))
    z = at-eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        raise ValueError('camera up and view direction must not be parallel')
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, z])
    pose[:3, 3] = eye
    return pose


def render_boxes(boxes, pose, intrinsics, shape=(120, 160), view_id='view'):
    height, width = shape
    v, u = np.indices(shape)
    rays_camera = np.stack([u, v, np.ones(shape)], -1) @ np.linalg.inv(intrinsics).T
    directions = rays_camera @ pose[:3, :3].T
    origin = pose[:3, 3]
    depth = np.full(shape, np.inf)
    labels = np.full(shape, -1, dtype=np.int32)
    for index, box in enumerate(boxes):
        lower, upper = np.asarray(box['lower']), np.asarray(box['upper'])
        parallel = np.abs(directions) < 1e-12
        safe = np.where(parallel, 1, directions)
        a, b = (lower-origin)/safe, (upper-origin)/safe
        near = np.max(np.where(parallel, -np.inf, np.minimum(a, b)), axis=-1)
        far = np.min(np.where(parallel, np.inf, np.maximum(a, b)), axis=-1)
        possible = np.all(~parallel | ((origin >= lower) & (origin <= upper)), axis=-1)
        hit = possible & (far >= np.maximum(near, 1e-6))
        distance = np.where(near > 1e-6, near, far)
        replace = hit & (distance < depth)
        depth[replace] = distance[replace]
        labels[replace] = index
    depth[~np.isfinite(depth)] = 0
    return DepthView(view_id, depth, intrinsics, pose), labels


def surface_grid(box, spacing=.004):
    lower, upper = np.asarray(box['lower']), np.asarray(box['upper'])
    points = []
    for axis in range(3):
        others = [j for j in range(3) if j != axis]
        coordinates = [np.linspace(lower[k], upper[k], max(2, int(np.ceil((upper[k]-lower[k])/spacing))+1)) for k in others]
        grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), -1).reshape(-1, 2)
        for edge in [lower[axis], upper[axis]]:
            face = np.zeros((len(grid), 3))
            face[:, axis] = edge
            face[:, others] = grid
            points.append(face)
    return np.unique(np.concatenate(points), axis=0)
