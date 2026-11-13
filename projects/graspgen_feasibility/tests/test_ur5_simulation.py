"""Physical adapter checks; run with the isolated .venv-sim interpreter."""
import math
import unittest

import numpy as np
import pybullet as p

from simulation.ur5_robotiq import UR5Robotiq140


class UR5ModelTests(unittest.TestCase):
    def setUp(self):
        p.connect(p.DIRECT)
        self.robot = UR5Robotiq140()
        p.setTimeStep(1/240)
        p.setPhysicsEngineParameter(numSolverIterations=150)

    def tearDown(self):
        p.disconnect()

    def test_real_model_opening_and_invalid_requests(self):
        self.assertGreater(self.robot.max_width, 0.12)
        self.assertLess(self.robot.max_width, 0.14)
        for bad in [-0.01, 0.14, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.robot.angle_for_width(bad)

    def test_motor_and_linkage_reach_requested_gaps(self):
        for width in [0.1, 0.05, 0.02, 0.12]:
            self.robot.command(self.robot.home, width)
            for _ in range(480):
                p.stepSimulation()
            self.assertAlmostEqual(self.robot.pad_gap(), width, delta=0.002)
            self.assertLess(self.robot.mimic_error(), 0.01)

    def test_tcp_is_fixed_to_gripper_base(self):
        for angle in [-0.5, 0, 0.5]:
            p.resetJointState(self.robot.body, self.robot.arm[0], angle)
            base = p.getLinkState(self.robot.body, self.robot.links['robotiq_arg2f_base_link'], computeForwardKinematics=True)
            tcp = p.getLinkState(self.robot.body, self.robot.tcp, computeForwardKinematics=True)
            inv = p.invertTransform(base[4], base[5])
            pos, quat = p.multiplyTransforms(*inv, tcp[4], tcp[5])
            np.testing.assert_allclose(pos, [0, 0, .2], atol=1e-6)
            self.assertAlmostEqual(abs(quat[3]), 1, places=6)

    def test_added_pad_inertias_are_positive(self):
        for pad in self.robot.pads:
            mass, _, inertia, *_ = p.getDynamicsInfo(self.robot.body, pad)
            self.assertAlmostEqual(mass, .01)
            self.assertTrue(all(x > 0 for x in inertia))

    def test_wrapping_cannot_command_full_wrist_turn(self):
        p.resetJointState(self.robot.body, self.robot.arm[-1], math.pi - .01)
        q = list(self.robot.home)
        q[-1] = -math.pi + .01
        with self.assertRaisesRegex(ValueError, 'Discontinuous IK'):
            self.robot.command(q, .05)

    def test_equivalent_legal_angle_does_not_mutate_input(self):
        q = np.array(self.robot.home)
        q[0] += 2 * math.pi
        original = q.copy()
        self.robot.command(q, .05)
        np.testing.assert_array_equal(q, original)


if __name__ == '__main__':
    unittest.main()
