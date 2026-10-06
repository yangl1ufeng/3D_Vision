import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def load_config(path=None, candidates=None, seed=None):
    path = Path(path) if path else ROOT / 'configs/offline_default.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg.get('schema_version') != 1:
        raise ValueError('unsupported config schema_version')
    cfg.setdefault('approach', {'enabled': False, 'distance_m': .10, 'clearance_m': .002, 'point_chunk': 8192})
    cfg.setdefault('observation', {'enabled': False, 'voxel_m': .008, 'tolerance_m': .002, 'max_cells': 500000})
    observation = cfg['observation']
    if type(observation.get('enabled')) is not bool:
        raise ValueError('observation.enabled must be boolean')
    for key in ['voxel_m', 'tolerance_m']:
        value = observation.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'observation.{key} must be positive and finite')
    if type(observation.get('max_cells')) is not int or observation['max_cells'] < 1:
        raise ValueError('observation.max_cells must be a positive integer')
    approach = cfg['approach']
    if type(approach.get('enabled')) is not bool:
        raise ValueError('approach.enabled must be boolean')
    for key, zero_allowed in [('distance_m', False), ('clearance_m', True)]:
        value = approach.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or (value < 0 if zero_allowed else value <= 0):
            raise ValueError(f'approach.{key} is invalid')
    if type(approach.get('point_chunk')) is not int or approach['point_chunk'] < 1:
        raise ValueError('approach.point_chunk must be a positive integer')
    if candidates is not None:
        cfg['grasp']['candidates'] = candidates
    if seed is not None:
        cfg['seed'] = seed
    if type(cfg['seed']) is not int or cfg['seed'] < 0 or cfg['seed'] >= 2**32:
        raise ValueError('seed must be an integer in [0,2**32)')
    for group, names in [('grasp', ['candidates']), ('pointcloud', ['max_target_points', 'max_scene_points', 'min_target_points', 'outlier_neighbors']), ('collision', ['surface_samples', 'batch_size'])]:
        for key in names:
            if type(cfg[group][key]) is not int or cfg[group][key] < 1:
                raise ValueError(f'{group}.{key} must be a positive integer')
    if cfg['pointcloud']['max_target_points'] < cfg['pointcloud']['min_target_points']:
        raise ValueError('max_target_points is below min_target_points')
    if cfg['pointcloud']['min_target_points'] <= cfg['pointcloud']['outlier_neighbors']:
        raise ValueError('min_target_points must exceed outlier_neighbors')
    threshold = cfg['grasp']['score_threshold']
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('score_threshold must be in [0,1]')
    for group, key in [('pointcloud', 'outlier_threshold_m'), ('collision', 'threshold_m')]:
        if not np.isfinite(cfg[group][key]) or cfg[group][key] <= 0:
            raise ValueError(f'{group}.{key} must be positive and finite')
    if cfg['device'] != 'cuda:0':
        raise ValueError('this official GraspGen adapter currently requires device cuda:0')
    for group, names in [('segmentation', ['sam2_repo', 'checkpoint']), ('grasp', ['repo', 'gripper_config'])]:
        for name in names:
            cfg[group][name] = str(project_path(cfg[group][name]))
    return cfg
