"""Static-scene multi-view registration, quality gates and provenance-preserving fusion.

The first camera anchors the world frame. Gates use ONLY depth observations,
never the experiment's true geometry/poses. Rejected attempts export diagnostics
but cannot provide a usable fused cloud or visibility views to planning.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import minimum_filter, maximum_filter
from .geometry import transform_points

DEFAULT_FUSION = {
    'voxel_m': .004, 'registration_voxel_m': .006, 'max_iterations': 35,
    'max_correspondence_m': .045, 'min_correspondences': 100,
    'min_constraint_ratio': .001, 'max_correction_translation_m': .05,
    'max_correction_rotation_deg': 8., 'overlap_distance_m': .012,
    'min_overlap_fraction': .20, 'max_reprojection_p90_m': .008,
    'max_free_conflict_fraction': .05, 'min_target_evidence': 30,
    'min_scene_evidence': 100, 'depth_edge_m': .015,
    'min_normal_agreement': .95, 'correspondence_neighbors': 12,
    'normal_patch_agreement': .995,
}


def first_voxel_indices(points, voxel):
    if len(points) == 0:
        return np.empty(0, dtype=int)
    return np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)[1]


def points_normals(view, voxel, patch_agreement=.995):
    xyz = view.world_image()
    dx = xyz[1:-1, 2:] - xyz[1:-1, :-2]
    dy = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1)
    d = view.depth_m
    valid = (minimum_filter(d, size=3, mode='constant') > 0)[1:-1, 1:-1]
    valid &= ((maximum_filter(d, size=3) - minimum_filter(d, size=3)) < .025)[1:-1, 1:-1]
    valid &= norm > 1e-10
    # A finite-difference normal spanning two box faces is not a measured plane.
    # Such blended edge normals can falsely supply missing registration DOFs.
    unit = normals / np.maximum(norm[..., None], 1e-20)
    padded = np.pad(unit, ((1, 1), (1, 1), (0, 0)))
    for y in range(3):
        for x in range(3):
            neighbor = padded[y:y+unit.shape[0], x:x+unit.shape[1]]
            valid &= np.einsum('ijk,ijk->ij', unit, neighbor) >= patch_agreement
    points = xyz[1:-1, 1:-1][valid]
    normals = normals[valid] / norm[valid, None]
    indices = first_voxel_indices(points, voxel)
    return points[indices], normals[indices]


def register_view(moving, references, cfg):
    source, source_normals = points_normals(moving, cfg['registration_voxel_m'], cfg['normal_patch_agreement'])
    prepared = [points_normals(v, cfg['registration_voxel_m'], cfg['normal_patch_agreement']) for v in references]
    target = np.concatenate([p for p, _ in prepared])
    normals = np.concatenate([n for _, n in prepared])
    tree = cKDTree(target)
    correction = np.eye(4)
    history, reasons = [], []
    if len(source) < cfg['min_correspondences'] or len(target) < cfg['min_correspondences']:
        return moving, {'accepted': False, 'reasons': ['insufficient_registration_points'], 'iterations': []}
    constraint = 0.
    converged = False
    for iteration in range(cfg['max_iterations']):
        current = transform_points(source, correction)
        distances_all, matches = tree.query(current, k=min(cfg['correspondence_neighbors'], len(target)))
        if distances_all.ndim == 1:
            distances_all, matches = distances_all[:, None], matches[:, None]
        moved_normals = source_normals @ correction[:3, :3].T
        agreement = np.einsum('ij,ikj->ik', moved_normals, normals[matches])
        distances_all[agreement < cfg['min_normal_agreement']] = np.inf
        best = distances_all.argmin(axis=1)
        distances, match = distances_all[np.arange(len(current)), best], matches[np.arange(len(current)), best]
        good = distances < cfg['max_correspondence_m']
        if good.sum() < cfg['min_correspondences']:
            reasons.append('insufficient_overlap_correspondences')
            break
        # Drop the most distant 10%, preserving the same rule for every run.
        good &= distances <= np.quantile(distances[good], .9)
        p, q, n = current[good], target[match[good]], normals[match[good]]
        residual = np.einsum('ij,ij->i', q-p, n)
        # Scale angular variables by 0.2m for an interpretable rank test.
        A = np.column_stack([np.cross(p, n) / .2, n])
        singular = np.linalg.svd(A, compute_uv=False)
        constraint = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.
        history.append({'iteration': iteration, 'correspondences': len(p),
                        'rmse_point_plane_m': float(np.sqrt(np.mean(residual**2))),
                        'constraint_ratio': constraint})
        if constraint < cfg['min_constraint_ratio']:
            reasons.append('degenerate_registration_geometry')
            break
        solution = np.linalg.lstsq(A, residual, rcond=None)[0]
        angular, translation = solution[:3] / .2, solution[3:]
        step = np.eye(4)
        step[:3, :3] = Rotation.from_rotvec(angular).as_matrix()
        step[:3, 3] = translation
        correction = step @ correction
        if np.linalg.norm(translation) < 2e-5 and np.linalg.norm(angular) < 2e-5:
            converged = True
            break
    translation_m = float(np.linalg.norm(correction[:3, 3]))
    rotation_deg = float(np.rad2deg(Rotation.from_matrix(correction[:3, :3]).magnitude()))
    if translation_m > cfg['max_correction_translation_m'] or rotation_deg > cfg['max_correction_rotation_deg']:
        reasons.append('correction_exceeds_trusted_initial_pose')
    if not converged and not reasons:
        reasons.append('registration_not_converged')
    corrected = moving.with_pose(correction @ moving.T_world_camera)
    return corrected, {'accepted': not reasons, 'reasons': reasons, 'converged': converged,
                       'correction_world': correction.tolist(), 'translation_m': translation_m,
                       'rotation_deg': rotation_deg, 'constraint_ratio': constraint, 'iterations': history}


def reproject_quality(source, source_mask, reference, cfg):
    points = source.points(source_mask)
    camera = (points - reference.T_world_camera[:3, 3]) @ reference.T_world_camera[:3, :3]
    valid = camera[:, 2] > 0
    pixels = camera @ reference.intrinsics.T
    uv = np.floor(pixels[:, :2] / np.where(pixels[:, 2:] > 0, pixels[:, 2:], 1) + .5).astype(int)
    h, w = reference.depth_m.shape
    valid &= (uv[:, 0] >= 1) & (uv[:, 0] < w-1) & (uv[:, 1] >= 1) & (uv[:, 1] < h-1)
    ids = np.flatnonzero(valid)
    d = reference.depth_m
    dmin, dmax = minimum_filter(d, size=3), maximum_filter(d, size=3)
    x, y = uv[ids, 0], uv[ids, 1]
    stable = (dmin[y, x] > 0) & (dmax[y, x]-dmin[y, x] < cfg['depth_edge_m'])
    ids, x, y = ids[stable], x[stable], y[stable]
    delta = d[y, x] - camera[ids, 2]
    # Points behind reference returns are occluded, not registration errors.
    visible = delta >= -.004
    residual = delta[visible]
    return {'evidence_points': len(residual),
            'free_conflicts': int(np.sum(residual > cfg['max_reprojection_p90_m'])),
            'free_conflict_fraction': float(np.mean(residual > cfg['max_reprojection_p90_m'])) if len(residual) else None,
            'reprojection_p90_m': float(np.quantile(np.abs(residual), .9)) if len(residual) else None}


def quality_gate(views, masks, cfg):
    pairs, reasons = [], []
    evidence_for_view = [False] * len(views)
    if len(views) == 1:
        return {'accepted': True, 'reasons': [], 'pairs': [], 'note': 'single-view reference; no registration claim'}
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            a, b = views[i].points(), views[j].points()
            da = cKDTree(b).query(a)[0]
            db = cKDTree(a).query(b)[0]
            overlap = float(min(np.mean(da < cfg['overlap_distance_m']), np.mean(db < cfg['overlap_distance_m'])))
            record = {'views': [views[i].view_id, views[j].view_id], 'overlap_fraction': overlap, 'directions': []}
            scene_evidence = False
            for source, target in [(i, j), (j, i)]:
                for scope, mask, minimum in [('scene', None, cfg['min_scene_evidence']), ('target', masks[source], cfg['min_target_evidence'])]:
                    quality = reproject_quality(views[source], mask, views[target], cfg)
                    record['directions'].append({'source': views[source].view_id, 'scope': scope, **quality})
                    if quality['evidence_points'] >= minimum:
                        if scope == 'scene': scene_evidence = True
                        if quality['free_conflict_fraction'] > cfg['max_free_conflict_fraction'] or quality['reprojection_p90_m'] > cfg['max_reprojection_p90_m']:
                            reasons.append(f'double_contour_or_depth_conflict:{views[source].view_id}:{views[target].view_id}:{scope}')
            if overlap >= cfg['min_overlap_fraction'] and scene_evidence:
                evidence_for_view[i] = evidence_for_view[j] = True
            pairs.append(record)
    if not all(evidence_for_view):
        reasons.append('insufficient_multiview_overlap')
    return {'accepted': not reasons, 'reasons': sorted(set(reasons)), 'pairs': pairs}


def fuse_arrays(views, masks, voxel_m):
    if len(views) > 63:
        raise ValueError('source bitmask supports at most 63 views')
    raw, targets, sources, pixels = [], [], [], []
    for i, (view, mask) in enumerate(zip(views, masks)):
        valid = view.depth_m > 0
        raw.append(view.world_image()[valid])
        targets.append(np.asarray(mask, dtype=bool)[valid])
        sources.append(np.full(valid.sum(), i, dtype=np.int16))
        pixels.append(np.argwhere(valid))
    points, target, source, pixel = map(np.concatenate, [raw, targets, sources, pixels])
    # Never average target and background into the same fused voxel.
    keys = np.column_stack([np.floor(points/voxel_m).astype(np.int64), target])
    unique, inverse, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    fused = np.zeros((len(unique), 3))
    np.add.at(fused, inverse, points)
    fused /= count[:, None]
    provenance = np.zeros(len(unique), dtype=np.uint64)
    np.bitwise_or.at(provenance, inverse, np.left_shift(np.uint64(1), source.astype(np.uint64)))
    return {'observed_points_world_m': points, 'observed_target_mask': target, 'source_view_index': source,
            'source_pixel_yx': pixel, 'fused_points_world_m': fused, 'fused_target_mask': unique[:, 3].astype(bool),
            'fused_source_view_bits': provenance, 'fused_observation_count': count}


def fuse_views(views, masks, refine=False, config=None):
    cfg = {**DEFAULT_FUSION, **(config or {})}
    for key, value in cfg.items():
        if key not in DEFAULT_FUSION or not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'invalid fusion setting: {key}')
        if isinstance(DEFAULT_FUSION[key], int) and type(value) is not int:
            raise ValueError(f'fusion setting {key} must be an integer')
    for key in ['min_overlap_fraction', 'min_normal_agreement', 'normal_patch_agreement', 'max_free_conflict_fraction', 'min_constraint_ratio']:
        if cfg[key] > 1:
            raise ValueError(f'fusion setting {key} must not exceed 1')
    views, masks = list(views), list(masks)
    if not views or len(views) != len(masks) or len({v.view_id for v in views}) != len(views):
        raise ValueError('provide nonempty uniquely named views and one target mask per view')
    for view, mask in zip(views, masks):
        if np.asarray(mask).shape != view.depth_m.shape or not np.isin(mask, [0, 1]).all() or not np.any((view.depth_m > 0) & mask):
            raise ValueError('each view requires a valid same-target mask with observed points')
    corrected, registration, failures = [views[0]], [], []
    for view in views[1:]:
        if refine:
            candidate, record = register_view(view, corrected, cfg)
            registration.append({'view_id': view.view_id, **record})
            if not record['accepted']:
                failures.extend(f'{view.view_id}:{r}' for r in record['reasons'])
            corrected.append(candidate)
        else:
            corrected.append(view)
    gate = quality_gate(corrected, masks, cfg)
    failures.extend(gate['reasons'])
    # Keep rejected clouds ONLY as diagnostics. No usable output can be consumed
    # by downstream observation/planning unless every gate passes.
    diagnostic = fuse_arrays(corrected, masks, cfg['voxel_m'])
    return {'accepted': not failures, 'reasons': failures, 'registration': registration,
            'quality': gate, 'config': cfg, 'corrected_views': corrected,
            'usable_views': corrected if not failures else [],
            'usable_cloud': diagnostic if not failures else None, 'diagnostic_cloud': diagnostic}
