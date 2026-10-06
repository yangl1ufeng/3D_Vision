"""Orchestrate the four independent offline stages."""
from collections import Counter
import time
from .collision import CollisionChecker, apply_collision_decisions, collision_eligible_indices, initial_decisions
from .approach import ApproachChecker, apply_approach_decisions
from .observation import ObservationChecker, apply_observation_decisions
from .export import export_artifacts, rank_candidates, write_json
from .grasp import GraspGenerator
from .pointcloud import prepare_clouds
from .runtime import initialize, provenance, set_seed
from .scene_io import file_hash, load_scene
from .segmentation import SAM2Segmenter, provided_segmentation


def analyze(manifest, output, cfg, profiler):
    if cfg['observation']['enabled'] and not cfg['approach']['enabled']:
        raise ValueError('observation checking requires approach geometry')
    started = time.perf_counter()
    with profiler.stage('input_loading'):
        scene = load_scene(manifest)
    with profiler.stage('runtime_initialization'):
        initialize(cfg)
        set_seed(cfg['seed'])
    segmenter = None
    if not scene.target.get('mask_path'):
        with profiler.stage('segmentation_model_loading', gpu=True):
            segmenter = SAM2Segmenter(cfg['segmentation'])
    with profiler.stage('grasp_model_loading', gpu=True):
        generator = GraspGenerator(cfg['grasp'])
    with profiler.stage('collision_geometry_loading'):
        checker = CollisionChecker(generator.gripper_name, cfg['collision'], cfg['seed'] + 1)
        approach = ApproachChecker.from_official(checker.gripper_info, cfg['approach']) if cfg['approach']['enabled'] else None
    with profiler.stage('segmentation', gpu=segmenter is not None):
        segmentation = segmenter.segment(scene) if segmenter else provided_segmentation(scene)
    with profiler.stage('pointcloud', gpu=True):
        clouds = prepare_clouds(scene, segmentation, cfg['pointcloud'], cfg['seed'])
    with profiler.stage('grasp_generation', gpu=True):
        grasps = generator.generate(clouds.target_model_m, cfg['grasp']['candidates'], cfg['seed'])
    with profiler.stage('score_filter'):
        decisions = initial_decisions(grasps, cfg['grasp']['score_threshold'])
        eligible = collision_eligible_indices(decisions)
    with profiler.stage('collision_check', gpu=True):
        free = checker.check(clouds.background_collision_m, grasps.poses_world_m[eligible])
        apply_collision_decisions(decisions, eligible, free, len(clouds.background_collision_m) > 0)
    if approach is not None:
        with profiler.stage('approach_check'):
            # Check all score/pose-eligible candidates, retaining overlapping reasons.
            # Use full observed geometry, not background/model inference subsamples.
            results = approach.check(clouds.background_full_m, clouds.target_raw_m, grasps.poses_world_m[eligible])
            apply_approach_decisions(decisions, eligible, results)
    if cfg['observation']['enabled']:
        with profiler.stage('observation_check'):
            observer = ObservationChecker(approach, [scene.depth_view] if scene.depth_view else [],
                                          cfg['observation']['voxel_m'], cfg['observation']['tolerance_m'],
                                          cfg['observation']['max_cells'])
            observed = observer.check(grasps.poses_world_m[eligible])
            apply_observation_decisions(decisions, eligible, observed)
    accepted = rank_candidates(decisions, grasps.scores)
    inference_wall = time.perf_counter() - started
    with profiler.stage('provenance_hashing'):
        source = provenance(cfg, generator, segmenter)
        source['gripper_config_sha256'] = file_hash(generator.model_config)
    with profiler.stage('artifact_export'):
        export_artifacts(output, scene, segmentation, clouds, grasps, decisions, checker, cfg, source)
    counts = Counter(reason for d in decisions for reason in d['reasons'])
    summary = {
        'schema_version': 1, 'status': 'ok', 'scene_id': scene.scene_id, 'target_id': scene.target_id,
        'world_frame': scene.metadata['world_frame'], 'translation_unit': 'm',
        'pose_convention': scene.metadata['pose_convention'],
        'segmentation': {'backend': segmentation.backend, 'mask_pixels': int(segmentation.mask.sum()),
                         'predicted_quality': segmentation.predicted_quality, **segmentation.metadata},
        'pointcloud': clouds.stats, 'grasp_generation': grasps.metadata,
        'collision': {**cfg['collision'], 'surface_seed': checker.seed,
                      'scene_points': len(clouds.background_collision_m), 'scope': 'observed geometry, final gripper pose'},
        'approach': approach.metadata if approach else {'enabled': False},
        'observation': {**cfg['observation'], 'calibrated_depth_available': scene.depth_view is not None,
                        'scope': 'conservative voxel cover of straight gripper sweep; pinhole depth evidence'},
        'counts': {'raw': len(decisions), 'eligible_for_collision': len(eligible), 'accepted': len(accepted),
                   'rejected': len(decisions) - len(accepted), 'exclusion_reasons': dict(counts)},
        'accepted_candidate_ids_by_rank': accepted,
        'inference_wall_with_loading_seconds': inference_wall,
        'limitations': ['This entry analyzes one view; multi-view registration/fusion is available through fuse_scene.py.',
                        'No arm reachability, gripper closing simulation, or physical execution.',
                        'Approach and observation checks cover only a fixed-orientation straight translation when enabled.',
                        'Observation uses a conservative voxel envelope and a discrete calibrated depth model; sensor uncertainties are not calibrated.',
                        'Approach uses full observed points and convex component envelopes, including a conservative convexified base.',
                        'Collision uses sampled observed background and gripper surface; unobserved geometry can be missed.',
                        'T_world_graspgen_gripper is not a calibrated robot-base TCP command.',
                        'Scores and accepted counts do not measure physical grasp success.'],
        'output_directory': str(output)}
    performance = profiler.summary()
    performance['analysis_wall_seconds'] = time.perf_counter() - profiler.started
    performance['wall_note'] = 'From CLI argument parsing completion through export; Python startup/imports, final summary/performance JSON writes and console output excluded.'
    write_json(output / 'performance.json', performance)
    summary['performance'] = performance
    write_json(output / 'summary.json', summary)
    return summary
