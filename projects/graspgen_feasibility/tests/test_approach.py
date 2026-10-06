import unittest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from offline_core.approach import ApproachChecker, apply_approach_decisions
from experiments.reference_geometry import mesh_boxes_collision, moving_triangles_box


def box(lower, upper):
    return {'lower': list(lower), 'upper': list(upper)}


class ContinuousApproachTests(unittest.TestCase):
    def setUp(self):
        self.mesh = trimesh.creation.box(extents=[.02, .02, .02])
        self.checker = ApproachChecker([self.mesh.vertices], [0, 0, 1], .1, 0)
        self.pose = np.eye(4)[None]
        self.far = np.array([[1., 1., 1.]])

    def test_thin_obstacle_between_endpoints(self):
        obstacle = np.array([[0., 0., -.05]])
        result = self.checker.check(obstacle, self.far, self.pose)[0]
        self.assertEqual(result['state'], 'collision')
        self.assertFalse(result['final_background_collision'])
        self.assertAlmostEqual(result['first_hit_fraction'], .4)
        self.assertEqual(result['reasons'], ['approach_background_collision'])
        slab = box([-.03, -.03, -.0501], [.03, .03, -.0499])
        self.assertFalse(mesh_boxes_collision([self.mesh], self.pose[0], [slab]))
        self.assertTrue(mesh_boxes_collision([self.mesh], self.pose[0], [slab], .1))

    def test_target_is_checked_during_approach(self):
        result = self.checker.check(self.far, np.array([[0., 0., -.05]]), self.pose)[0]
        self.assertIn('approach_target_collision', result['reasons'])
        self.assertFalse(result['final_target_collision'])

    def test_target_final_solid_and_open_gap(self):
        result = self.checker.check(self.far, np.zeros((1, 3)), self.pose)[0]
        self.assertTrue(result['final_target_collision'])
        left, right = self.mesh.copy(), self.mesh.copy()
        left.apply_translation([-.04, 0, 0])
        right.apply_translation([.04, 0, 0])
        checker = ApproachChecker([left.vertices, right.vertices], [0, 0, 1], .1, 0)
        self.assertEqual(checker.check(self.far, np.zeros((1, 3)), self.pose)[0]['state'], 'clear')

    def test_world_rotation_and_pregrasp_direction(self):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', [35, 67, -12], degrees=True).as_matrix()
        T[:3, 3] = [1.2, -.7, .5]
        obstacle = np.array([[0., 0., -.05]]) @ T[:3, :3].T + T[:3, 3]
        result = self.checker.check(obstacle, self.far, T[None])[0]
        self.assertEqual(result['state'], 'collision')
        np.testing.assert_allclose(np.asarray(result['pregrasp_pose_world'])[:3, 3], T[:3, 3] - .1 * T[:3, 2])

    def test_clear_empty_invalid_and_empty_batch(self):
        self.assertEqual(self.checker.check(self.far, self.far, self.pose)[0]['state'], 'clear')
        self.assertEqual(self.checker.check(np.empty((0, 3)), self.far, self.pose)[0]['state'], 'unknown')
        self.assertEqual(self.checker.check(self.far, np.empty((0, 3)), self.pose)[0]['state'], 'unknown')
        self.assertEqual(self.checker.check(self.far, self.far, np.empty((0, 4, 4))), [])
        with self.assertRaises(ValueError):
            self.checker.check(np.array([[np.nan, 0, 0]]), self.far, self.pose)
        with self.assertRaises(ValueError):
            self.checker.check(self.far, self.far, np.zeros((1, 4, 4)))

    def test_endpoint_contact_and_clearance(self):
        result = self.checker.check(np.array([[0, 0, .01]]), self.far, self.pose)[0]
        self.assertTrue(result['final_background_collision'])
        self.assertAlmostEqual(result['first_hit_fraction'], 1)
        self.assertNotIn('approach_background_collision', result['reasons'])
        margin = ApproachChecker([self.mesh.vertices], [0, 0, 1], .1, .002)
        self.assertEqual(margin.check(np.array([[.011, 0, 0]]), self.far, self.pose)[0]['state'], 'collision')

    def test_reference_containment_both_directions(self):
        small = box([-.001]*3, [.001]*3)
        big = box([-.1]*3, [.1]*3)
        away = box([.1]*3, [.2]*3)
        self.assertTrue(mesh_boxes_collision([self.mesh], self.pose[0], [small]))
        self.assertTrue(mesh_boxes_collision([self.mesh], self.pose[0], [big]))
        self.assertFalse(mesh_boxes_collision([self.mesh], self.pose[0], [away]))

    def test_reference_against_analytic_moving_boxes(self):
        # Independent analytic swept AABB oracle for axis-aligned translations.
        rng = np.random.default_rng(14)
        for _ in range(100):
            center = rng.uniform([-.025, -.025, -.13], [.025, .025, .03])
            half = rng.uniform(.0001, .008, 3)
            obstacle = box(center - half, center + half)
            analytic = bool(np.all(center + half >= [-.01, -.01, -.11]) and np.all(center - half <= [.01, .01, .01]))
            self.assertEqual(mesh_boxes_collision([self.mesh], self.pose[0], [obstacle], .1), analytic)

    def test_decisions_preserve_baseline_rejection(self):
        decisions = [{'accepted': False, 'rank': None, 'reasons': ['scene_collision']},
                     {'accepted': True, 'rank': 1, 'reasons': []}]
        apply_approach_decisions(decisions, [0, 1], [{'state': 'clear', 'reasons': []},
                                                   {'state': 'collision', 'reasons': ['approach_target_collision']}])
        self.assertFalse(any(d['accepted'] for d in decisions))
        self.assertIsNone(decisions[1]['rank'])


if __name__ == '__main__':
    unittest.main()
