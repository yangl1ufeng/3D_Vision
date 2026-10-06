"""Continuous, fixed-orientation translation of convex gripper components.

Scene observations remain points: this does not infer free/unknown space. Each
connected mesh component is convexified separately (never the whole gripper),
preserving the opening. Plane offsets are a conservative clearance envelope,
not an exact Euclidean distance. There is no temporal or surface sampling.
"""
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from .geometry import valid_pose_mask


def finite_points(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f'{name} must be finite Nx3 points')
    return points


class ApproachChecker:
    def __init__(self, component_vertices, direction_local, distance_m=.10,
                 clearance_m=.002, point_chunk=8192):
        direction = np.asarray(direction_local, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1):
            raise ValueError('approach direction must be a finite unit vector')
        if not np.isfinite(distance_m) or distance_m <= 0:
            raise ValueError('approach distance must be positive and finite')
        if not np.isfinite(clearance_m) or clearance_m < 0:
            raise ValueError('approach clearance must be nonnegative and finite')
        if type(point_chunk) is not int or point_chunk < 1:
            raise ValueError('point_chunk must be a positive integer')
        self.direction = direction
        self.distance = float(distance_m)
        self.clearance = float(clearance_m)
        self.point_chunk = point_chunk
        self.parts = []
        for vertices in component_vertices:
            vertices = finite_points(vertices, 'component vertices')
            hull = ConvexHull(vertices)
            # Qhull repeats planes for coplanar triangles. Deduplicate for speed.
            planes = np.unique(np.round(hull.equations, 12), axis=0)
            expanded = planes.copy()
            expanded[:, 3] -= self.clearance
            envelope = HalfspaceIntersection(expanded, vertices.mean(0)).intersections
            self.parts.append((planes[:, :3], planes[:, 3], envelope.min(0), envelope.max(0)))
        if not self.parts:
            raise ValueError('at least one solid gripper component is required')
        self.metadata = {'method': 'continuous_point_vs_convex_component_translation',
                         'component_count': len(self.parts), 'direction_local': direction.tolist(),
                         'distance_m': self.distance, 'clearance_m': self.clearance,
                         'target_contact_policy': 'fixed_open_gripper_strict_material_collision',
                         'clearance_note': 'Offset convex face planes; conservative, not Euclidean mesh distance.'}

    @classmethod
    def from_official(cls, info, cfg):
        from grasp_gen.robot import load_default_gripper_config
        config = load_default_gripper_config(info.gripper_name)
        # The supported model's YAML and canonical convention explicitly agree.
        # Other grippers need their own convention/contact review before enabling.
        if info.gripper_name != 'robotiq_2f_140' or not np.allclose(info.offset_transform, np.eye(4)):
            raise ValueError('approach convention has only been verified for Robotiq-2F-140 with identity asset offset')
        direction = config['approach_direction']
        parts = info.collision_mesh.split(only_watertight=False)
        if any(not p.is_watertight for p in parts):
            raise ValueError('approach requires closed collision mesh components')
        checker = cls([p.vertices for p in parts], direction, cfg['distance_m'], cfg['clearance_m'], cfg['point_chunk'])
        checker.metadata.update({'gripper': info.gripper_name,
                                 'convexified_nonconvex_components': sum(not p.is_convex for p in parts),
                                 'source': 'config/grippers/robotiq_2f_140.yaml: approach_direction; grasp_gen/robot.py: canonical +Z',
                                 'gripper_model_origin': 'GraspGen asset base frame, not robot TCP',
                                 'nominal_tip_depth_m': config['depth']})
        return checker

    def pregrasp_poses(self, poses):
        poses = np.asarray(poses, dtype=np.float64)
        if not valid_pose_mask(poses).all():
            raise ValueError('approach poses must be valid rigid transforms')
        result = poses.copy()
        result[:, :3, 3] -= np.einsum('nij,j->ni', poses[:, :3, :3], self.direction * self.distance)
        return result

    def _first_hit(self, points_local, distance):
        """Slab intersection of point trajectories with convex half-spaces."""
        best = np.inf
        start_shift = self.direction * distance
        for normals, offsets, lower, upper in self.parts:
            # Bounds come from the offset polyhedron, including acute corners.
            sweep_lower = lower + np.minimum(-start_shift, 0)
            sweep_upper = upper + np.maximum(-start_shift, 0)
            points = points_local[np.all((points_local >= sweep_lower - 1e-10) & (points_local <= sweep_upper + 1e-10), axis=1)]
            speed = normals @ (-start_shift)
            stationary = np.abs(speed) < 1e-12
            for start in range(0, len(points), self.point_chunk):
                q = points[start:start + self.point_chunk] + start_shift
                bound = self.clearance - offsets - np.einsum('ij,kj->ik', q, normals)
                feasible = np.all(bound[:, stationary] >= -1e-10, axis=1)
                enter = np.zeros(len(q))
                leave = np.ones(len(q))
                neg, pos = speed < -1e-12, speed > 1e-12
                if neg.any():
                    enter = np.maximum(enter, np.max(bound[:, neg] / speed[neg], axis=1))
                if pos.any():
                    leave = np.minimum(leave, np.min(bound[:, pos] / speed[pos], axis=1))
                feasible &= enter <= leave + 1e-10
                if feasible.any():
                    best = min(best, float(enter[feasible].min()))
                    if best <= 0:
                        return 0.0
        return None if not np.isfinite(best) else best

    def check(self, background_world_m, target_world_m, poses_world_m):
        background = finite_points(background_world_m, 'background')
        target = finite_points(target_world_m, 'target')
        poses = np.asarray(poses_world_m, dtype=np.float64)
        pregrasp = self.pregrasp_poses(poses)
        results = []
        for pose, pre in zip(poses, pregrasp):
            record = {'state': 'clear', 'reasons': [], 'pregrasp_pose_world': pre.tolist(),
                      'first_hit_fraction': None, 'first_hit_source': None,
                      'final_background_collision': False, 'final_target_collision': False}
            # Missing either input cannot establish that the whole segment is clear.
            if not len(background) or not len(target):
                record.update(state='unknown', reasons=['approach_missing_geometry'])
                results.append(record)
                continue
            first = []
            for label, points in [('background', background), ('target', target)]:
                local = (points - pose[:3, 3]) @ pose[:3, :3]
                hit = self._first_hit(local, self.distance)
                final_hit = self._first_hit(local, 0)
                record[f'final_{label}_collision'] = final_hit is not None
                if final_hit is not None:
                    record['reasons'].append(f'final_{label}_collision')
                if hit is not None:
                    first.append((hit, label))
                    if hit < 1 - 1e-8:
                        record['reasons'].append(f'approach_{label}_collision')
            if first:
                fraction, source = min(first)
                record.update(state='collision', first_hit_fraction=fraction, first_hit_source=source)
            results.append(record)
        return results


def apply_approach_decisions(decisions, indices, results):
    if len(indices) != len(results):
        raise ValueError('approach result count does not match candidates')
    for index, result in zip(indices, results):
        decision = decisions[int(index)]
        decision['approach'] = result
        decision['reasons'].extend(r for r in result['reasons'] if r not in decision['reasons'])
        decision['accepted'] = decision['accepted'] and result['state'] == 'clear'
        if not decision['accepted']:
            decision['rank'] = None
