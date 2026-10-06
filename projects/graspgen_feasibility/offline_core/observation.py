"""Calibrated depth evidence and conservative swept-volume visibility gating.

Depth is optical-axis z in meters. A return only certifies space IN FRONT of it;
behind a return, outside the image, or at invalid pixels remains unknown.
Voxel envelopes cover the entire straight swept gripper volume. A voxel is
certified free only if its whole projected footprint is before valid returns.
This guarantee is relative to the discrete pinhole depth model, not real sensors.
"""
from dataclasses import dataclass, replace
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from .geometry import depth_to_xyz, transform_points, validate_transform, valid_pose_mask

UNKNOWN, FREE, OCCUPIED, CONFLICT = 0, 1, 2, 3
STATE_NAMES = ['unknown', 'free', 'occupied', 'conflict']


@dataclass
class DepthView:
    view_id: str
    depth_m: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray

    def __post_init__(self):
        self.depth_m = np.asarray(self.depth_m, dtype=np.float64).copy()
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64).copy()
        self.T_world_camera = validate_transform(self.T_world_camera).copy()
        depth_to_xyz(self.depth_m, self.intrinsics)  # Validate shape and pinhole K.
        self.depth_m[~np.isfinite(self.depth_m) | (self.depth_m <= 0)] = 0
        if not isinstance(self.view_id, str) or not self.view_id:
            raise ValueError('view_id must be a nonempty string')

    def with_pose(self, pose):
        return replace(self, T_world_camera=pose)

    def world_image(self):
        return transform_points(depth_to_xyz(self.depth_m, self.intrinsics), self.T_world_camera)

    def points(self, mask=None):
        valid = self.depth_m > 0
        if mask is not None:
            if np.asarray(mask).shape != valid.shape:
                raise ValueError('target mask must match depth image')
            valid &= np.asarray(mask, dtype=bool)
        return self.world_image()[valid]


def classify_in_view(points_world, view, radius_m=0., tolerance_m=.002):
    """Classify a point or enclosing sphere using every pixel of its footprint.

For nonzero radii, use an enclosing camera-axis cube and its projected rectangle.
Occupied means a measured surface intersects this conservative depth envelope.
"""
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('observation query must be finite Nx3 world points')
    if not np.isfinite(radius_m) or radius_m < 0 or not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('radius/tolerance must be finite and nonnegative')
    states = np.full(len(points), UNKNOWN, dtype=np.uint8)
    local = (points - view.T_world_camera[:3, 3]) @ view.T_world_camera[:3, :3]
    valid = local[:, 2] - radius_m > 1e-8
    indices = np.flatnonzero(valid)
    if not len(indices):
        return states
    offsets = np.array(list(product([-radius_m, radius_m], repeat=3)))
    corners = local[indices, None, :] + offsets
    pixel_h = corners @ view.intrinsics.T
    uv = pixel_h[..., :2] / pixel_h[..., 2:]
    lower = np.floor(uv.min(1) + .5).astype(np.int64)
    upper = np.floor(uv.max(1) + .5).astype(np.int64)
    height, width = view.depth_m.shape
    inside = (lower[:, 0] >= 0) & (lower[:, 1] >= 0) & (upper[:, 0] < width) & (upper[:, 1] < height)
    indices, lower, upper = indices[inside], lower[inside], upper[inside]
    sizes = upper - lower + 1
    for size in np.unique(sizes, axis=0):
        selected = np.flatnonzero(np.all(sizes == size, axis=1))
        du, dv = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        for start in range(0, len(selected), 1024):
            chosen = selected[start:start+1024]
            ids = indices[chosen]
            depth = view.depth_m[lower[chosen, 1, None] + dv.ravel(), lower[chosen, 0, None] + du.ravel()]
            near, far = local[ids, 2] - radius_m - tolerance_m, local[ids, 2] + radius_m + tolerance_m
            free = np.all(depth > far[:, None], axis=1)
            occupied = np.any((depth > 0) & (depth >= near[:, None]) & (depth <= far[:, None]), axis=1)
            states[ids[free]] = FREE
            states[ids[occupied]] = OCCUPIED
    return states


def classify_space(points_world, views, radius_m=0., tolerance_m=.002):
    points = np.asarray(points_world)
    per_view = np.array([classify_in_view(points, v, radius_m, tolerance_m) for v in views], dtype=np.uint8)
    if not views:
        return np.zeros(len(points), dtype=np.uint8), np.empty((0, len(points)), dtype=np.uint8)
    free, occupied = np.any(per_view == FREE, axis=0), np.any(per_view == OCCUPIED, axis=0)
    combined = np.zeros(len(points), dtype=np.uint8)
    combined[free] = FREE
    combined[occupied] = OCCUPIED
    combined[free & occupied] = CONFLICT
    return combined, per_view


def swept_volume_cells(approach, voxel_m=.008, max_cells=500000):
    """Outer voxel cover of convex component sweeps, including voxel interiors."""
    if not np.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('voxel_m must be positive and finite')
    cells = []
    for normals, offsets, lower, upper in approach.parts:
        planes = np.column_stack([normals, offsets - approach.clearance])
        # The midpoint of an envelope AABB need not be inside an arbitrary hull.
        from scipy.optimize import linprog
        center = linprog([0, 0, 0, -1], A_ub=np.column_stack([normals, np.ones(len(normals))]),
                         b_ub=-planes[:, 3], bounds=[(None, None)]*3+[(0, None)], method='highs')
        if not center.success or center.x[3] <= 0:
            raise ValueError('could not find an interior point in gripper component')
        vertices = HalfspaceIntersection(planes, center.x[:3]).intersections
        vertices = np.vstack([vertices, vertices - approach.direction * approach.distance])
        sweep = ConvexHull(vertices).equations
        lo = np.floor(vertices.min(0) / voxel_m).astype(int)
        hi = np.floor(vertices.max(0) / voxel_m).astype(int)
        counts = hi - lo + 1
        if np.prod(counts) > max_cells * 8:
            raise ValueError('sweep grid too large; use a larger observation voxel size')
        grid = np.stack(np.meshgrid(*(np.arange(lo[k], hi[k]+1) for k in range(3)), indexing='ij'), axis=-1).reshape(-1, 3)
        keep = []
        support = .5 * voxel_m * np.abs(sweep[:, :3]).sum(1)
        for start in range(0, len(grid), 4096):
            part = grid[start:start+4096]
            centers = (part + .5) * voxel_m
            possible = np.all(np.einsum('ij,kj->ik', centers, sweep[:, :3]) + sweep[:, 3] <= support + 1e-10, axis=1)
            keep.append(part[possible])
        cells.append(np.concatenate(keep))
    grid = np.unique(np.concatenate(cells), axis=0)
    if len(grid) > max_cells:
        raise ValueError('sweep grid exceeds observation max_cells')
    return (grid + .5) * voxel_m


class ObservationChecker:
    def __init__(self, approach, views, voxel_m=.008, tolerance_m=.002, max_cells=500000):
        self.views = list(views)
        if len({v.view_id for v in self.views}) != len(self.views):
            raise ValueError('observation view IDs must be unique')
        self.voxel = voxel_m
        self.tolerance = tolerance_m
        self.local_cells = swept_volume_cells(approach, voxel_m, max_cells) if self.views else np.empty((0, 3))
        self.radius = np.sqrt(3) * voxel_m / 2

    def check(self, poses):
        poses = np.asarray(poses, dtype=np.float64)
        if not valid_pose_mask(poses).all():
            raise ValueError('observation poses must be rigid transforms')
        results = []
        for pose in poses:
            if not self.views:
                results.append({'state': 'insufficient_observation', 'reasons': ['observation_missing_calibration'],
                                'needs_more_views': True, 'message': '缺少带内外参的深度观测，需要采集或补充标定信息。',
                                'cells': 0, 'free_cells': 0, 'unknown_cells': 0, 'occupied_cells': 0, 'conflict_cells': 0})
                continue
            world = transform_points(self.local_cells, pose)
            states, _ = classify_space(world, self.views, self.radius, self.tolerance)
            counts = np.bincount(states, minlength=4)
            reasons = []
            if counts[CONFLICT]: reasons.append('observation_conflict')
            if counts[OCCUPIED]: reasons.append('observed_surface_in_sweep_envelope')
            if counts[UNKNOWN]: reasons.append('insufficient_observation')
            state = 'conflict' if counts[CONFLICT] else 'occupied' if counts[OCCUPIED] else 'insufficient_observation' if counts[UNKNOWN] else 'observed_free'
            results.append({'state': state, 'reasons': reasons, 'needs_more_views': bool(counts[UNKNOWN]),
                            'message': '观测冲突，需要重新配准或采集。' if counts[CONFLICT] else '扫掠包络附近存在观测表面。' if counts[OCCUPIED] else '观测不足，需要补拍。' if counts[UNKNOWN] else '整个接近扫掠包络已有空闲观测。',
                            'cells': len(states), 'free_cells': int(counts[FREE]), 'unknown_cells': int(counts[UNKNOWN]),
                            'occupied_cells': int(counts[OCCUPIED]), 'conflict_cells': int(counts[CONFLICT]),
                            'unknown_fraction': float(counts[UNKNOWN]/len(states)), 'voxel_m': self.voxel,
                            'view_ids': [v.view_id for v in self.views],
                            'unknown_examples_world_m': world[states == UNKNOWN][:12].tolist()})
        return results


def apply_observation_decisions(decisions, indices, results):
    if len(indices) != len(results):
        raise ValueError('observation count does not match candidates')
    for index, result in zip(indices, results):
        record = decisions[int(index)]
        record['observation'] = result
        record['reasons'].extend(r for r in result['reasons'] if r not in record['reasons'])
        record['accepted'] = record['accepted'] and result['state'] == 'observed_free'
        if not record['accepted']:
            record['rank'] = None
