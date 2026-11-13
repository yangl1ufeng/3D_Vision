"""Collision regression tests on the real downloaded model, not mock geometry."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pybullet as p

from simulation.planned_demo import PlannedDemo
from simulation.planning import PlanningError, plan_rrt


class MotionPlanningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.demo = PlannedDemo(Path(self.tmp.name), scenario='free')
        self.scene = self.demo.scene
        self.scene.configure('pregrasp')
        self.q = np.array([p.getJointState(self.demo.robot, j)[0] for j in self.demo.arm])

    def tearDown(self):
        self.scene.close()
        p.disconnect(self.demo.client)
        self.tmp.cleanup()

    def test_queries_do_not_move_execution_world(self):
        before = p.getLinkState(self.demo.robot, self.demo.tcp, computeForwardKinematics=True)[4:6]
        object_before = p.getBasePositionAndOrientation(self.demo.cube)
        self.scene.collision(np.zeros(6))
        self.scene.ik([.5, 0, .22], self.demo.orientation, self.q)
        after = p.getLinkState(self.demo.robot, self.demo.tcp, computeForwardKinematics=True)[4:6]
        self.assertEqual(before, after)
        self.assertEqual(object_before, p.getBasePositionAndOrientation(self.demo.cube))

    def test_folded_arm_self_collision_rejected(self):
        folded = [1.916389889, 1.934849046, .09629334, -1.345849621, -2.802736057, -.732814935]
        reason = self.scene.collision(folded)
        self.assertEqual(reason['kind'], 'self_collision')
        self.assertEqual(set(reason['links']), {'base_link', 'upper_arm_link'})

    def test_forearm_collision_not_just_tcp(self):
        link = self.demo.adapter.links['forearm_link']
        position = p.getLinkState(self.demo.robot, link, computeForwardKinematics=True)[0]
        tcp = p.getLinkState(self.demo.robot, self.demo.tcp, computeForwardKinematics=True)[4]
        self.assertGreater(np.linalg.norm(np.array(position)-tcp), .15)
        obstacle = self.scene.box(position, [.02]*3)
        self.scene.environment.append(('forearm_probe', obstacle))
        self.assertEqual(self.scene.collision(self.q)['object'], 'forearm_probe')

    def test_payload_collision_detected_when_arm_is_clear(self):
        self.scene.set_q(self.q)
        tcp = self.scene.b.getLinkState(self.scene.robot, self.demo.tcp, computeForwardKinematics=True)
        pose = ([1.2, 0, .7], [0, 0, 0, 1])
        self.scene.attachment = p.multiplyTransforms(*p.invertTransform(tcp[4], tcp[5]), *pose)
        self.scene.attached = True
        self.scene.allow_pads = True
        self.assertIsNone(self.scene.collision(self.q))
        obstacle = self.scene.box(pose[0], [.01]*3)
        self.scene.environment.append(('payload_probe', obstacle))
        self.assertEqual(self.scene.collision(self.q)['kind'], 'payload_collision')

    def test_pad_permission_does_not_remove_whole_target(self):
        base = self.scene.b.getLinkState(self.scene.robot, self.demo.adapter.links['robotiq_arg2f_base_link'], computeForwardKinematics=True)
        position = p.multiplyTransforms(base[4], base[5], [0, 0, .04], [0, 0, 0, 1])[0]
        self.scene.b.resetBasePositionAndOrientation(self.scene.target, position, [0, 0, 0, 1])
        self.scene.allow_pads = True
        self.assertEqual(self.scene.collision(self.q)['kind'], 'target_collision')

    def test_unreachable_goal_and_limits(self):
        with self.assertRaisesRegex(PlanningError, 'ik_unreachable'):
            self.scene.ik([2, 0, .22], self.demo.orientation, self.q)
        self.assertEqual(self.scene.collision([10]*6)['kind'], 'joint_limits')

    def test_direct_edge_blocked_and_zero_budget_rejected(self):
        obstacle = self.scene.box([.5, .11, .12], [.1, .008, .12])
        self.scene.environment.append(('barrier', obstacle))
        goal = self.scene.ik([.5, 0, .22], self.demo.orientation, self.q)
        self.assertIsNone(self.scene.collision(self.q))
        self.assertIsNone(self.scene.collision(goal))
        self.assertFalse(self.scene.edge(self.q, goal))
        with self.assertRaisesRegex(PlanningError, 'search_budget_exhausted'):
            plan_rrt(self.scene, self.q, goal, max_iterations=0)


if __name__ == '__main__':
    unittest.main()
