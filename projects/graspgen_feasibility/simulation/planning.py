"""Isolated collision scene and bounded bidirectional RRT-Connect.

Only the private DIRECT client is reset for geometric queries. Execution bodies
are never teleported. Collision checking is discretized, not continuous CCD.
"""
import math
import os
from pathlib import Path
import time

import numpy as np
import pybullet as p
from pybullet_utils.bullet_client import BulletClient

from .ur5_robotiq import SOURCE


class PlanningError(RuntimeError):
    pass


class CollisionScene:
    def __init__(self, demo, obstacles):
        self.demo = demo
        self.b = BulletClient(connection_mode=p.DIRECT)
        previous = Path.cwd()
        try:
            os.chdir(SOURCE)
            self.robot = self.b.loadURDF('urdf/ur5_robotiq_140_project.urdf', useFixedBase=True)
        finally:
            os.chdir(previous)
        self.arm, self.tcp = demo.arm, demo.tcp
        self.limits = demo.adapter.limits
        self.infos = {i: self.b.getJointInfo(self.robot, i) for i in range(self.b.getNumJoints(self.robot))}
        self.names = {-1: 'world', **{i: v[12].decode() for i, v in self.infos.items()}}
        self.groups = {i: self.group(i) for i in self.names}
        self.adjacent = {tuple(sorted((self.group(i), self.group(v[16])))) for i, v in self.infos.items()}
        # The tree URDF omits the mechanical four-bar closure connections.
        # Exclude only these same-side touching linkage pairs (including fixed pads).
        self.linkage = set()
        links = demo.adapter.links
        for side in ['left', 'right']:
            for other in ['outer_knuckle', 'inner_finger']:
                self.linkage.add(tuple(sorted((self.group(links[f'{side}_inner_knuckle']),
                                              self.group(links[f'{side}_{other}'])))))
        self.environment = []
        for name, position, half in [
            ('table', [.55, 0, -.04], [.35, .45, .04]),
            ('pedestal', [0, 0, -.04], [.14, .16, .04]), *obstacles]:
            self.environment.append((name, self.box(position, half)))
        self.target = self.box([.5, 0, demo.cube_size/2], [demo.cube_size/2]*3)
        self.attached = False
        self.allow_pads = False
        self.allow_support = False
        self.attachment = None
        self.calls = 0
        self.last_reason = None
        self.sync()

    def group(self, link):
        while link >= 0 and self.infos[link][2] == p.JOINT_FIXED:
            link = self.infos[link][16]
        return link

    def box(self, position, half, orientation=(0, 0, 0, 1)):
        shape = self.b.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        return self.b.createMultiBody(0, shape, basePosition=position,
                                      baseOrientation=orientation)

    def reconstructed_object(self, item, offset):
        center = (np.asarray(item['center_table_m']) + offset).tolist()
        if item['shape'] == 'voxel_compound_proxy':
            shape = self.b.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*len(item['parts']),
                halfExtents=[part['half_extents_m'] for part in item['parts']],
                collisionFramePositions=[part['center_local_m'] for part in item['parts']])
            return self.b.createMultiBody(0, shape, basePosition=center)
        if item['shape'] == 'box':
            return self.box(center, (np.asarray(item['dimensions_m'])/2).tolist(),
                            p.getQuaternionFromEuler([0, 0, item['yaw_table_rad']]))
        if item['shape'] == 'solid_cylinder_proxy':
            shape = self.b.createCollisionShape(p.GEOM_CYLINDER,
                                                 radius=item['radius_m'], height=item['height_m'])
            return self.b.createMultiBody(0, shape, basePosition=center)
        raise ValueError(f'Unsupported reconstructed shape: {item["shape"]}')

    def use_reconstructed_scene(self, geometry, target_name, offset, scene_spec=None):
        """Replace synthetic query bodies with measured table-relative proxies."""
        if target_name not in geometry['objects']:
            raise ValueError(f'Unknown target {target_name}')
        for _, body in self.environment:
            self.b.removeBody(body)
        self.b.removeBody(self.target)
        self.environment = [
            ('table', self.box([.55, 0, -.04], [.45, .45, .04])),
            ('pedestal', self.box([0, 0, -.04], [.14, .16, .04]))]
        if scene_spec is not None:
            for _, body in self.environment:
                self.b.removeBody(body)
            self.environment = [(name, self.box(scene_spec.data[name]['center_robot_m'],
                                                scene_spec.data[name]['half_extents_m']))
                                for name in ('table', 'pedestal')]
        for name, item in geometry['objects'].items():
            body = self.reconstructed_object(item, offset)
            if name == target_name:
                self.target = body
            else:
                self.environment.append((name, body))
        self.sync()

    def sync(self):
        for i in self.infos:
            state = p.getJointState(self.demo.robot, i, physicsClientId=self.demo.client)
            self.b.resetJointState(self.robot, i, state[0])
        pose = p.getBasePositionAndOrientation(self.demo.cube, physicsClientId=self.demo.client)
        self.b.resetBasePositionAndOrientation(self.target, *pose)

    def configure(self, stage):
        self.sync()
        self.attached = stage in {'lift', 'hold', 'transfer', 'lower'}
        self.allow_pads = stage in {'close', 'lift', 'hold', 'transfer', 'lower', 'release', 'retreat', 'settle'}
        self.allow_support = stage in {'lift', 'lower'}
        if self.attached:
            state = self.b.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)
            self.attachment = self.b.multiplyTransforms(*self.b.invertTransform(state[4], state[5]),
                                                       *self.b.getBasePositionAndOrientation(self.target))

    def set_q(self, q):
        for joint, angle in zip(self.arm, q):
            self.b.resetJointState(self.robot, joint, float(angle))
        if self.attached:
            state = self.b.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)
            pose = self.b.multiplyTransforms(state[4], state[5], *self.attachment)
            self.b.resetBasePositionAndOrientation(self.target, *pose)

    def collision(self, q, margin=.004):
        self.calls += 1
        self.last_reason = None
        q = np.asarray(q)
        if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            self.last_reason = {'kind': 'joint_limits'}
            return self.last_reason
        self.set_q(q)
        for c in self.b.getClosestPoints(self.robot, self.robot, max(0, margin)):
            i, j = c[3:5]
            groups = tuple(sorted((self.groups[i], self.groups[j])))
            if i >= j or self.groups[i] == self.groups[j] or groups in self.adjacent or groups in self.linkage:
                continue
            if c[8] < margin:
                self.last_reason = {'kind': 'self_collision', 'links': [self.names[i], self.names[j]], 'distance_m': c[8]}
                return self.last_reason
        for name, body in self.environment:
            for c in self.b.getClosestPoints(self.robot, body, max(0, margin)):
                if name == 'pedestal' and self.groups[c[3]] == -1:
                    continue  # Fixed mounting contact only.
                if c[8] < margin:
                    self.last_reason = {'kind': 'environment_collision', 'object': name,
                                        'link': self.names[c[3]], 'distance_m': c[8]}
                    return self.last_reason
        for c in self.b.getClosestPoints(self.robot, self.target, max(0, margin)):
            if self.allow_pads and c[3] in self.demo.fingers:
                continue
            if c[8] < margin:
                self.last_reason = {'kind': 'target_collision', 'link': self.names[c[3]], 'distance_m': c[8]}
                return self.last_reason
        if self.attached:
            for name, body in self.environment:
                threshold = -.001 if name == 'table' and self.allow_support else margin
                for c in self.b.getClosestPoints(self.target, body, max(0, threshold)):
                    if c[8] < threshold:
                        self.last_reason = {'kind': 'payload_collision', 'object': name, 'distance_m': c[8]}
                        return self.last_reason
        return None

    def edge(self, start, goal, resolution=.04):
        count = max(1, math.ceil(float(np.max(np.abs(np.array(goal)-start))) / resolution))
        for t in np.linspace(0, 1, count+1):
            if self.collision(np.array(start)*(1-t)+np.array(goal)*t):
                return False
        return True

    def ik(self, position, orientation, seed):
        self.set_q(seed)
        solved = self.b.calculateInverseKinematics(self.robot, self.tcp, position, orientation,
                                                  maxNumIterations=200, residualThreshold=1e-6)
        q = np.asarray(solved[:6])
        q = (q + np.pi) % (2*np.pi) - np.pi
        self.set_q(q)
        state = self.b.getLinkState(self.robot, self.tcp, computeForwardKinematics=True)
        error = float(np.linalg.norm(np.asarray(position)-state[4]))
        quat = np.asarray(state[5]); quat /= np.linalg.norm(quat)
        angular = 2*math.acos(float(np.clip(abs(np.dot(quat, orientation)), 0, 1)))
        if error > .002 or angular > math.radians(2):
            raise PlanningError(f'ik_unreachable: position_error={error:.4f}, angle_error={angular:.4f}')
        if np.max(np.abs(q-seed)) > np.pi:
            raise PlanningError('ik_branch_discontinuity')
        return q

    def close(self):
        self.b.disconnect()


def plan_rrt(scene, start, goal, seed=7, max_iterations=2000, timeout=15):
    """Bidirectional RRT-Connect; a budget failure is not proof of no path."""
    start, goal = np.asarray(start), np.asarray(goal)
    if scene.collision(start):
        raise PlanningError(f'invalid_start: {scene.last_reason}')
    if scene.collision(goal):
        raise PlanningError(f'invalid_goal: {scene.last_reason}')
    if scene.edge(start, goal):
        return [start, goal], {'method': 'direct', 'iterations': 0, 'direct_blocked': False}
    direct_reason = scene.last_reason
    rng = np.random.default_rng(seed)
    trees = [([start], [-1]), ([goal], [-1])]
    began = time.perf_counter()

    def extend(tree, target):
        nodes, parents = tree
        index = int(np.argmin([np.linalg.norm(n-target) for n in nodes]))
        delta = target-nodes[index]
        distance = np.linalg.norm(delta)
        if distance < 1e-9:
            return index, True
        new = nodes[index] + delta * min(1, .25/distance)
        if not scene.edge(nodes[index], new):
            return None, False
        nodes.append(new); parents.append(index)
        return len(nodes)-1, distance <= .25

    def ancestry(tree, index):
        result = []
        while index >= 0:
            result.append(tree[0][index]); index = tree[1][index]
        return result[::-1]

    for iteration in range(max_iterations):
        if time.perf_counter()-began > timeout:
            break
        a = iteration % 2; b = 1-a
        target = trees[b][0][0] if rng.random() < .15 else rng.uniform(scene.limits[:, 0], scene.limits[:, 1])
        ia, _ = extend(trees[a], target)
        if ia is None:
            continue
        for _ in range(100):
            ib, reached = extend(trees[b], trees[a][0][ia])
            if ib is None:
                break
            if reached:
                first, second = ancestry(trees[a], ia), ancestry(trees[b], ib)
                path = first + second[-2::-1]
                if a == 1:
                    path.reverse()
                # Greedy shortcuts retain checked edges.
                shortened = [path[0]]; index = 0
                while index < len(path)-1:
                    end = len(path)-1
                    while end > index+1 and not scene.edge(path[index], path[end]):
                        end -= 1
                    shortened.append(path[end]); index = end
                return shortened, {'method': 'rrt_connect', 'iterations': iteration+1,
                                   'direct_blocked': True, 'direct_rejection': direct_reason}
    raise PlanningError('search_budget_exhausted: no path found within this budget')


def validate_path(scene, path, resolution=.005):
    for a, b in zip(path, path[1:]):
        if not scene.edge(a, b, resolution):
            raise PlanningError(f'fine_validation_rejected: {scene.last_reason}')
