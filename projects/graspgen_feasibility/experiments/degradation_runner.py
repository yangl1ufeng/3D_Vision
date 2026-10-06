"""Shared measured inference and independent evaluation for preparation 5/6."""
import csv
import time
import numpy as np
from scipy.spatial import cKDTree
from offline_core.config import load_config
from offline_core.runtime import initialize, provenance
from offline_core.grasp import GraspGenerator
from offline_core.collision import CollisionChecker, initial_decisions, collision_eligible_indices, apply_collision_decisions
from offline_core.approach import ApproachChecker, apply_approach_decisions
from offline_core.observation import ObservationChecker, apply_observation_decisions
from offline_core.export import write_json, export_candidates, rank_candidates
from offline_core.profiling import Profiler
from offline_core.pointcloud import prepare_clouds
from offline_core.types import Segmentation
from experiments.reference_geometry import mesh_boxes_collision


def save_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class Runner:
    def __init__(self):
        self.cfg = load_config()
        self.torch = initialize(self.cfg)
        self.generator = GraspGenerator(self.cfg['grasp'])
        self.checker = CollisionChecker(self.generator.gripper_name, self.cfg['collision'], 1001)
        self.approach = ApproachChecker.from_official(self.checker.gripper_info, self.cfg['approach'])
        self.parts = self.checker.gripper_info.collision_mesh.split(only_watertight=False)

    def save_provenance(self, output):
        write_json(output/'model_provenance.json', provenance(self.cfg, self.generator, None))
        write_json(output/'resolved_config.json', self.cfg)

    def screen(self, grasps, clouds, views, profiler):
        decisions = initial_decisions(grasps, self.cfg['grasp']['score_threshold'])
        eligible = collision_eligible_indices(decisions)
        with profiler.stage('collision_check', gpu=True):
            clear = self.checker.check(clouds.background_collision_m, grasps.poses_world_m[eligible])
            apply_collision_decisions(decisions, eligible, clear, len(clouds.background_collision_m) > 0)
        with profiler.stage('approach_check'):
            apply_approach_decisions(decisions, eligible, self.approach.check(clouds.background_full_m, clouds.target_raw_m, grasps.poses_world_m[eligible]))
        geometric = np.asarray([d['accepted'] for d in decisions])
        if views is not None:
            with profiler.stage('observation_check'):
                observer = ObservationChecker(self.approach, views)
                apply_observation_decisions(decisions, eligible, observer.check(grasps.poses_world_m[eligible]))
        rank_candidates(decisions, grasps.scores)
        return decisions, eligible, geometric

    def run_scene(self, scene, mask, seed, output, truth_boxes=None, fixed=None, segmentation_seconds=None):
        output.mkdir(parents=True, exist_ok=True)
        profiler = Profiler()
        arrays = {'segmentation_mask': mask, 'valid_mask': scene.valid}
        result = {'status': 'ok', 'seed': seed, 'segmentation_seconds': segmentation_seconds,
                  'target_raw_points': int((mask & scene.valid).sum()), 'mask_pixels': int(mask.sum())}
        grasps = None
        try:
            with profiler.stage('pointcloud', gpu=True):
                if not mask.any(): raise ValueError('segmentation produced an empty target mask')
                clouds = prepare_clouds(scene, Segmentation(mask, 'experiment_mask', None, {}), self.cfg['pointcloud'], seed)
            arrays.update(target_model_world_m=clouds.target_model_m, target_raw_world_m=clouds.target_raw_m,
                          background_collision_world_m=clouds.background_collision_m)
            result['target_model_points'] = len(clouds.target_model_m)
            with profiler.stage('grasp_generation', gpu=True):
                grasps = self.generator.generate(clouds.target_model_m, 100, seed)
            views = [scene.depth_view] if scene.depth_view else None
            decisions, eligible, geometric = self.screen(grasps, clouds, views, profiler)
            arrays.update(poses_world_m=grasps.poses_world_m, scores=grasps.scores, geometric_accepted=geometric,
                          accepted=np.asarray([d['accepted'] for d in decisions]))
            result.update(candidate_count=len(decisions), eligible=len(eligible), geometric_accepted=int(geometric.sum()),
                          accepted=sum(d['accepted'] for d in decisions), mean_score=float(grasps.scores.mean()),
                          unknown_candidate_fraction=float(np.mean([d['observation']['unknown_cells'] > 0 for d in decisions if 'observation' in d])) if len(eligible) and views else None)
            export_candidates(output, grasps, decisions, scene.metadata['world_frame'])
            if truth_boxes is not None:
                # Original mesh + solid boxes: no observed/noisy cloud enters reference.
                reference = np.full(len(decisions), -1, dtype=np.int8)
                started = time.perf_counter()
                for i in eligible:
                    reference[i] = mesh_boxes_collision(self.parts, grasps.poses_world_m[i], truth_boxes, .1)
                result['reference_seconds'] = time.perf_counter()-started
                arrays['reference_collision'] = reference
                result.update(reference_clear_eligible=int(np.sum(reference == 0)),
                              false_accept_geometry=int(np.sum(geometric & (reference == 1))),
                              false_accept_final=int(np.sum(arrays['accepted'] & (reference == 1))),
                              retained_reference_clear_final=int(np.sum(arrays['accepted'] & (reference == 0))))
            if fixed is not None:
                # Fixed clean candidates isolate filtering changes from generation changes.
                with profiler.stage('fixed_probe_check'):
                    fixed_results = self.approach.check(clouds.background_full_m, clouds.target_raw_m, fixed['poses'])
                fixed_clear = np.asarray([r['state']=='clear' for r in fixed_results])
                arrays.update(fixed_poses_world_m=fixed['poses'], fixed_geometric_clear=fixed_clear,
                              fixed_reference_collision=fixed['reference'], fixed_clean_clear=fixed['clear'])
                result.update(fixed_decision_flip_fraction=float(np.mean(fixed_clear != fixed['clear'])) if len(fixed_clear) else None,
                              fixed_missed_reference_collisions=int(np.sum(fixed_clear & fixed['reference'])))
            return result, arrays, grasps
        except (ValueError, RuntimeError) as exc:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            if isinstance(exc, self.torch.cuda.OutOfMemoryError): self.torch.cuda.empty_cache()
            return result, arrays, grasps
        finally:
            result['performance'] = profiler.summary()
            write_json(output/'result.json', result)
            np.savez_compressed(output/'arrays.npz', **arrays)


def target_comparison(model_points, reference_points):
    if len(model_points) == 0:
        return {'target_surface_p95_m': None, 'target_coverage_6mm': 0., 'target_centroid_shift_m': None}
    return {'target_surface_p95_m': float(np.quantile(cKDTree(reference_points).query(model_points)[0], .95)),
            'target_coverage_6mm': float(np.mean(cKDTree(model_points).query(reference_points)[0] <= .006)),
            'target_centroid_shift_m': float(np.linalg.norm(model_points.mean(0)-reference_points.mean(0)))}
