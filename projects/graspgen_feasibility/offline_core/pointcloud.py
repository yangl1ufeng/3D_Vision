"""Extract world-frame point clouds and bound expensive preprocessing."""
import numpy as np
from .types import Clouds


def select_points(points, limit, rng):
    if len(points) > limit:
        # Sort selected indices to preserve the source pixel order.
        indices = np.sort(rng.choice(len(points), limit, replace=False))
        return points[indices]
    return points.copy()


def prepare_clouds(scene, segmentation, cfg, seed):
    import torch
    from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
    target = scene.xyz_world_m[segmentation.mask & scene.valid]
    background = scene.xyz_world_m[(~segmentation.mask) & scene.valid]
    if len(target) < cfg['min_target_points']:
        raise ValueError(f'target has {len(target)} valid points; requires {cfg["min_target_points"]}')
    rng = np.random.default_rng(seed)
    bounded = select_points(target, cfg['max_target_points'], rng)
    gpu = torch.as_tensor(bounded, dtype=torch.float32, device='cuda')
    filtered, outliers = point_cloud_outlier_removal(gpu, threshold=cfg['outlier_threshold_m'], K=cfg['outlier_neighbors'])
    filtered = filtered.cpu().numpy()
    outliers = outliers.cpu().numpy()
    if len(filtered) < cfg['min_target_points']:
        raise ValueError(f'only {len(filtered)} target points remain after outlier removal')
    collision = select_points(background, cfg['max_scene_points'], rng)
    return Clouds(target, filtered, background, collision, outliers,
                  {'target_raw_points': len(target), 'target_bounded_points': len(bounded),
                   'target_model_points': len(filtered), 'outliers': len(outliers),
                   'target_sampling_dropped': len(target) - len(bounded),
                   'background_full_points': len(background), 'background_collision_points': len(collision),
                   'target_bounds_m': [target.min(0).tolist(), target.max(0).tolist()],
                   'outlier_method': 'official mean distance to K neighbors using L1 norm, strict < threshold',
                   'collision_sampling': 'seeded random subset; full background retained in export',
                   'sampling_seed': seed})
