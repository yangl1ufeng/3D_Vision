"""Recheck fixed saved grasp candidates using accepted static-scene depth fusion."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from offline_core.config import ROOT, load_config
from offline_core.scene_io import file_hash
from offline_core.types import Grasps
from offline_core.geometry import transform_points, valid_pose_mask
from offline_core.collision import initial_decisions, collision_eligible_indices, apply_collision_decisions
from offline_core.approach import ApproachChecker, apply_approach_decisions
from offline_core.observation import DepthView, ObservationChecker, apply_observation_decisions
from offline_core.export import export_candidates, rank_candidates, write_json


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_accepted_fusion(folder, original):
    quality = read_json(folder/'fusion_quality.json')
    if quality.get('accepted') is not True or quality.get('planning_allowed') is not True or quality.get('reasons'):
        raise ValueError('fusion was rejected; diagnostic clouds cannot be used for candidate rechecking')
    hashes = read_json(folder/'artifact_sha256.json')
    for name in ['fusion_quality.json', 'fusion_usable.npz', 'accepted_depth_views.npz', 'input_provenance.json']:
        if name not in hashes or file_hash(folder/name) != hashes[name]:
            raise ValueError(f'fusion artifact missing or changed: {name}')
    sources = read_json(folder/'input_provenance.json')
    manifest = original['input_manifest']
    for source in sources:
        if source['world_frame'] != original['world_frame'] or source['input_manifest']['target']['id'] != manifest['target']['id']:
            raise ValueError('fusion target or world frame does not match saved candidates')
    matches = [s for s in sources if s['input_manifest']['scene_id'] == manifest['scene_id']]
    if len(matches) != 1 or matches[0]['asset_sha256'] != original['asset_sha256']:
        raise ValueError('candidate source must exactly match one original fusion view, including calibration and mask')
    views = []
    with np.load(folder/'accepted_depth_views.npz', allow_pickle=False) as data:
        for i, source in enumerate(sources):
            view = DepthView(str(data[f'view_id_{i}']), data[f'depth_{i}_m'], data[f'intrinsics_{i}'], data[f'T_world_camera_{i}'])
            if view.view_id != source['input_manifest']['scene_id'] or not np.allclose(view.T_world_camera, quality['corrected_poses'][view.view_id], atol=1e-12, rtol=0):
                raise ValueError('fusion depth evidence does not match the accepted camera poses')
            views.append(view)
    with np.load(folder/'fusion_usable.npz', allow_pickle=False) as data:
        cloud = {key: data[key] for key in data.files}
    points = cloud['observed_points_world_m']
    target = cloud['observed_target_mask']
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or target.shape != (len(points),) or target.dtype != bool:
        raise ValueError('invalid fused raw geometry or target mask')
    if not target.any() or target.all():
        raise ValueError('fusion requires both target and background observations')
    for i, view in enumerate(views):
        chosen = cloud['source_view_index'] == i
        pixel = cloud['source_pixel_yx'][chosen]
        expected_pixels = np.argwhere(view.depth_m > 0)
        if not np.array_equal(pixel, expected_pixels) or not np.allclose(points[chosen], view.world_image()[view.depth_m > 0], atol=1e-12, rtol=0):
            raise ValueError('fused point provenance does not match accepted depth evidence')
    if not np.isin(cloud['source_view_index'], np.arange(len(views))).all():
        raise ValueError('unknown fused point source')
    source_view = next(v for v in views if v.view_id == manifest['scene_id'])
    original_pose = np.asarray(manifest['frames']['T_world_camera'])
    correction = source_view.T_world_camera @ np.linalg.inv(original_pose)
    if not valid_pose_mask(correction[None]).all():
        raise ValueError('invalid source-camera correction')
    return views, points, target, correction, hashes


def recheck(run, fusion, output):
    original = read_json(run/'input_manifest.json')
    views, points, target, correction, fusion_hashes = load_accepted_fusion(fusion, original)
    cfg = load_config(run/'resolved_config.json')
    if not cfg['approach']['enabled'] or not cfg['observation']['enabled']:
        raise ValueError('source run must enable approach and observation checks for a comparable recheck')
    previous = read_json(run/'candidates.json')['candidates']
    summary = read_json(run/'summary.json')
    with np.load(run/'scene_arrays.npz', allow_pickle=False) as data:
        original_poses, scores = data['poses_world_graspgen_gripper'], data['scores']
        samples = data['gripper_surface_local_m']
    if len(previous) != len(scores):
        raise ValueError('saved candidate count mismatch')
    for i, record in enumerate(previous):
        if record['candidate_id'] != i or record['score'] != float(scores[i]) or not np.allclose(record['T_world_graspgen_gripper'], original_poses[i], atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError('saved candidate JSON and arrays disagree')
    grasps = Grasps(correction @ original_poses, scores.copy(), {'source_run': str(run), 'regenerated': False})
    decisions = initial_decisions(grasps, cfg['grasp']['score_threshold'])
    eligible = collision_eligible_indices(decisions)
    sys.path.insert(0, cfg['grasp']['repo'])
    from grasp_gen.robot import get_gripper_info
    gripper = summary['grasp_generation']['gripper']
    if gripper != cfg['grasp']['expected_gripper']:
        raise ValueError('saved gripper does not match configuration')
    approach = ApproachChecker.from_official(get_gripper_info(gripper), cfg['approach'])
    # Same saved gripper samples and distance rule, but all observed background
    # points and a CPU nearest-neighbor query. No model loading or resampling.
    background = points[~target]
    tree = cKDTree(background)
    free = [bool(np.all(tree.query(transform_points(samples, pose))[0] >= cfg['collision']['threshold_m']))
            for pose in grasps.poses_world_m[eligible]]
    apply_collision_decisions(decisions, eligible, free, len(background) > 0)
    apply_approach_decisions(decisions, eligible, approach.check(background, points[target], grasps.poses_world_m[eligible]))
    settings = cfg['observation']
    observer = ObservationChecker(approach, views, settings['voxel_m'], settings['tolerance_m'], settings['max_cells'])
    apply_observation_decisions(decisions, eligible, observer.check(grasps.poses_world_m[eligible]))
    accepted = rank_candidates(decisions, scores)
    def counts(records):
        observed = [d['observation'] for d in records if 'observation' in d]
        return {'accepted': sum(d['accepted'] for d in records), 'observation_checked': len(observed),
                'unknown_cells': sum(d['unknown_cells'] for d in observed),
                'candidates_with_unknown': sum(d['unknown_cells'] > 0 for d in observed),
                'observation_states': dict(Counter(d['state'] for d in observed))}
    report = {'status': 'complete', 'world_frame': original['world_frame'], 'target_id': manifest_target(original),
              'candidate_count': len(scores), 'source_run': str(run), 'fusion_run': str(fusion),
              'accepted_candidate_ids_by_rank': accepted, 'source_camera_correction_world': correction,
              'before': counts(previous), 'after': counts(decisions),
              'collision_scope': 'saved gripper surface samples vs FULL observed fused background, CPU Euclidean nearest-neighbor',
              'notes': ['Candidate scores and IDs preserved; poses follow the corrected source camera.',
                        'All rejection reasons recomputed using accepted depth views and full observed geometry.',
                        'Static same-target scene required; this is not robot execution or regenerated grasp inference.']}
    export_candidates(output, grasps, decisions, original['world_frame'])
    np.savez_compressed(output/'rechecked_arrays.npz', original_poses_world_m=original_poses,
                        poses_world_m=grasps.poses_world_m, scores=scores,
                        accepted=np.asarray([d['accepted'] for d in decisions]),
                        source_camera_correction_world=correction, gripper_surface_local_m=samples)
    write_json(output/'recheck_summary.json', report)
    write_json(output/'resolved_config.json', cfg)
    write_json(output/'provenance.json', {'fusion_artifacts_sha256': fusion_hashes,
               'source_run_sha256': {name: file_hash(run/name) for name in ['input_manifest.json', 'resolved_config.json', 'summary.json', 'candidates.json', 'scene_arrays.npz']},
               'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__), ROOT/'offline_core/observation.py', ROOT/'offline_core/approach.py']}})
    return report


def manifest_target(metadata):
    return metadata['input_manifest']['target']['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing analyze_scene output directory')
    parser.add_argument('--fusion', type=Path, required=True, help='Accepted fuse_scene output directory')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT/'runs/recheck'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = recheck(args.run.resolve(), args.fusion.resolve(), output)
        print(json.dumps({'before': result['before'], 'after': result['after'], 'output': str(output)}, ensure_ascii=False))
    except Exception as exc:
        write_json(output/'failure.json', {'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
