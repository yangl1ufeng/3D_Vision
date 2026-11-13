"""Fuse repeated depth frames from a fixed camera and unchanged scene."""
import warnings

import numpy as np


def fuse_static_depth(frames, near=.2, far=2., min_observations=5):
    """Return median depth, valid-count, and a scene-motion diagnostic.

    Every frame must share a camera pose, resolution and intrinsics. Pixels with
    too few valid measurements remain unknown, never inferred as free space.
    """
    stack = np.asarray(frames, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 3:
        raise ValueError('Need at least three HxW depth frames')
    if not 0 < near < far or not 1 <= min_observations <= len(stack):
        raise ValueError('Invalid depth range or minimum observation count')
    valid = np.isfinite(stack) & (stack >= near) & (stack <= far)
    count = valid.sum(axis=0)
    clean = np.where(valid, stack, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        median = np.nanmedian(clean, axis=0)
    fused_valid = count >= min_observations
    fused = np.where(fused_valid, median, 0).astype(np.float32)
    # Evaluate on pixels seen in most frames, avoiding edges and holes.
    stable = count >= max(min_observations, int(np.ceil(.8 * len(stack))))
    drift = []
    if stable.any():
        for frame, frame_valid in zip(stack, valid):
            overlap = stable & frame_valid
            drift.append(float(np.mean(np.abs(frame[overlap] - fused[overlap]) > .05))
                         if overlap.any() else 1.0)
    return fused, count.astype(np.uint16), {
        'frames_used': len(stack), 'valid_fraction': float(fused_valid.mean()),
        'stable_pixel_fraction': float(stable.mean()),
        'per_frame_fraction_over_5cm': drift,
        'stationarity_warning': bool(drift and max(drift) > .1),
        'note': 'Depth-change metric is a warning, not a camera-pose measurement.'}
