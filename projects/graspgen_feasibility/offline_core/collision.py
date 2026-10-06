"""Final-pose collision only; explicit reasons for every candidate."""
import numpy as np
from .geometry import valid_pose_mask


def initial_decisions(grasps, threshold):
    valid_pose = valid_pose_mask(grasps.poses_world_m)
    valid_score = np.isfinite(grasps.scores) & (grasps.scores >= 0) & (grasps.scores <= 1)
    decisions = []
    for i, score in enumerate(grasps.scores):
        reasons = []
        if not valid_pose[i]: reasons.append('invalid_pose')
        if not valid_score[i]: reasons.append('invalid_score')
        elif score < threshold: reasons.append('low_score')
        decisions.append({'candidate_id': i, 'accepted': False, 'rank': None,
                          'collision_checked': False, 'collision_state': 'not_checked',
                          'reasons': reasons})
    return decisions


def collision_eligible_indices(decisions):
    return np.array([d['candidate_id'] for d in decisions if not d['reasons']], dtype=np.int64)


def apply_collision_decisions(decisions, indices, free_mask, scene_available=True):
    if len(indices) != len(free_mask):
        raise ValueError('collision results do not match checked candidates')
    for i, free in zip(indices, free_mask):
        d = decisions[int(i)]
        if not scene_available:
            d['collision_state'] = 'unknown_no_scene_geometry'
            d['reasons'].append('no_scene_geometry')
        else:
            d['collision_checked'] = True
            d['collision_state'] = 'clear' if free else 'collision'
            if not free: d['reasons'].append('scene_collision')
            d['accepted'] = bool(free)
    return decisions


class CollisionChecker:
    def __init__(self, gripper_name, cfg, seed):
        import trimesh
        from grasp_gen.robot import get_gripper_info
        info = get_gripper_info(gripper_name)
        self.gripper_info = info
        samples, _ = trimesh.sample.sample_surface(info.collision_mesh, cfg['surface_samples'], seed=seed)
        self.samples_local_m = np.asarray(samples, dtype=np.float32)
        self.cfg = cfg
        self.seed = seed

    def check(self, background_world_m, poses_world_m):
        if len(background_world_m) == 0:
            # Upstream treats empty geometry as free; here it is explicitly unknown.
            return np.zeros(len(poses_world_m), dtype=bool)
        from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps_fast
        return filter_colliding_grasps_fast(scene_pc=background_world_m, grasp_poses=poses_world_m,
                                          gripper_surface_points=self.samples_local_m,
                                          collision_threshold=self.cfg['threshold_m'],
                                          batch_size=self.cfg['batch_size'], device='cuda:0')
