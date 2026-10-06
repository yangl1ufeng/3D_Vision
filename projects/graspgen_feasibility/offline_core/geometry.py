"""Explicit units and column-vector SE(3) conventions."""
import numpy as np

UNIT_SCALES = {'m': 1.0, 'mm': .001}


def meters_per_unit(unit, scale=None):
    if unit == 'raw':
        if scale is None:
            raise ValueError('raw depth needs an explicit meters_per_unit')
        factor = float(scale)
    elif unit in UNIT_SCALES:
        factor = UNIT_SCALES[unit]
        if scale is not None and not np.isclose(float(scale), factor, rtol=1e-9, atol=0):
            raise ValueError('unit and meters_per_unit disagree')
    else:
        raise ValueError('unit must explicitly be m, mm or raw; units are never guessed')
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('meters_per_unit must be finite and positive')
    return factor


def validate_transform(value, name='T_world_camera'):
    T = np.asarray(value, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f'{name} must be a finite 4x4 transform')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f'{name} must have bottom row [0,0,0,1]')
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError(f'{name} must contain a proper rotation, not scale/reflection')
    return T


def transform_points(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def depth_to_xyz(depth_m, intrinsics):
    """Pinhole, rectified optical-axis depth; camera x right, y down, z forward."""
    depth = np.asarray(depth_m, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError('depth must be HxW')
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('intrinsics must be a finite 3x3 matrix with positive focal lengths')
    if not np.allclose(K[2], [0, 0, 1]) or not np.isclose(K[1, 0], 0):
        raise ValueError('intrinsics must use the pinhole camera convention')
    v, u = np.indices(depth.shape)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    return (pixels @ np.linalg.inv(K).T * depth[..., None]).astype(np.float32)


def valid_pose_mask(poses, tolerance=1e-3):
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError('grasp poses must have shape Nx4x4')
    finite = np.isfinite(poses).all(axis=(1, 2))
    result = np.zeros(len(poses), dtype=bool)
    R = poses[finite, :3, :3]
    result[finite] = (
        (np.max(np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)), axis=(1, 2)) <= tolerance)
        & (np.abs(np.linalg.det(R) - 1) <= tolerance)
        & (np.max(np.abs(poses[finite, 3] - [0, 0, 0, 1]), axis=1) <= tolerance)
    ) if len(R) else np.zeros(0, dtype=bool)
    return result
