import unittest
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from offline_core.approach import ApproachChecker
from offline_core.observation import (DepthView, classify_space, classify_in_view, ObservationChecker,
                                      swept_volume_cells, apply_observation_decisions, FREE, UNKNOWN, OCCUPIED, CONFLICT)
from offline_core.fusion import fuse_views, fuse_arrays
from fuse_scene import save_result


class DepthEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[60., 0, 31.5], [0, 60., 31.5], [0, 0, 1]])
        self.view = DepthView('front', np.ones((64, 64)), self.K, np.eye(4))
        mesh = trimesh.creation.box(extents=[.02]*3)
        self.approach = ApproachChecker([mesh.vertices], [0, 0, 1], .1, 0)

    def test_free_surface_behind_outside_and_behind_camera(self):
        points = [[0, 0, .5], [0, 0, 1], [0, 0, 1.5], [4, 0, .5], [0, 0, -.5]]
        np.testing.assert_array_equal(classify_in_view(points, self.view), [FREE, OCCUPIED, UNKNOWN, UNKNOWN, UNKNOWN])

    def test_invalid_depth_and_secondary_view(self):
        depth = np.ones((64, 64))
        depth[28:36, 28:36] = np.nan
        holes = DepthView('holes', depth, self.K, np.eye(4))
        points = np.array([[0, 0, .5]])
        self.assertEqual(classify_space(points, [holes])[0][0], UNKNOWN)
        self.assertEqual(classify_space(points, [holes, self.view])[0][0], FREE)

    def test_conflict_is_not_overwritten_by_free(self):
        front = DepthView('near', np.full((64, 64), .5), self.K, np.eye(4))
        self.assertEqual(classify_space(np.array([[0, 0, .5]]), [front, self.view])[0][0], CONFLICT)

    def test_entire_footprint_required_not_only_center(self):
        depth = self.view.depth_m.copy()
        depth[29, 29] = 0
        view = DepthView('partial', depth, self.K, np.eye(4))
        point = np.array([[0, 0, .5]])
        self.assertEqual(classify_in_view(point, view)[0], FREE)
        self.assertEqual(classify_in_view(point, view, radius_m=.04)[0], UNKNOWN)

    def test_rotated_camera_preserves_classification(self):
        T = np.array([[0, 0, 1, 2], [1, 0, 0, -1], [0, 1, 0, .3], [0, 0, 0, 1]], dtype=float)
        view = self.view.with_pose(T)
        points = np.array([[0, 0, .5], [0, 0, 1.2]]) @ T[:3, :3].T + T[:3, 3]
        np.testing.assert_array_equal(classify_in_view(points, view), [FREE, UNKNOWN])

    def test_sweep_outer_cover_contains_random_interior(self):
        cells = swept_volume_cells(self.approach, .008)
        indices = {tuple(p) for p in np.floor(cells/.008).astype(int)}
        random = np.random.default_rng(12).uniform([-.01, -.01, -.11], [.01, .01, .01], (1000, 3))
        self.assertTrue(all(tuple(p) in indices for p in np.floor(random/.008).astype(int)))

    def test_complete_sweep_free_then_occluded(self):
        pose = np.eye(4)
        pose[2, 3] = .7
        visible = ObservationChecker(self.approach, [self.view]).check(pose[None])[0]
        self.assertEqual(visible['state'], 'observed_free')
        depth = np.ones((64, 64))
        depth[20:44, 20:44] = .55
        occluded = ObservationChecker(self.approach, [DepthView('occluded', depth, self.K, np.eye(4))]).check(pose[None])[0]
        self.assertEqual(occluded['state'], 'insufficient_observation')
        self.assertTrue(occluded['needs_more_views'])
        self.assertEqual(occluded['unknown_cells'], occluded['cells'])

    def test_missing_camera_rejects_candidate(self):
        record = ObservationChecker(self.approach, []).check(np.eye(4)[None])[0]
        self.assertIn('observation_missing_calibration', record['reasons'])
        decisions = [{'accepted': True, 'rank': 1, 'reasons': []}]
        apply_observation_decisions(decisions, [0], [record])
        self.assertFalse(decisions[0]['accepted'])
        self.assertIsNone(decisions[0]['rank'])

    def test_view_validation(self):
        with self.assertRaises(ValueError):
            DepthView('bad', np.ones((3, 3)), self.K, np.diag([2, 1, 1, 1]))
        with self.assertRaises(ValueError):
            classify_in_view([[np.nan, 0, 1]], self.view)


class FusionGateTests(unittest.TestCase):
    def setUp(self):
        K = np.array([[80., 0, 31.5], [0, 80., 31.5], [0, 0, 1]])
        self.a = DepthView('a', np.ones((64, 64)), K, np.eye(4))
        self.b = DepthView('b', np.ones((64, 64)), K, np.eye(4))
        self.mask = np.ones((64, 64), dtype=bool)

    def test_known_identical_views_preserve_provenance(self):
        result = fuse_views([self.a, self.b], [self.mask]*2)
        self.assertTrue(result['accepted'])
        cloud = result['usable_cloud']
        self.assertEqual(int(cloud['fused_observation_count'].sum()), 8192)
        self.assertTrue(np.all(cloud['fused_source_view_bits'] == 3))
        self.assertEqual(set(cloud['source_view_index']), {0, 1})

    def test_planar_icp_is_rejected_even_if_residual_zero(self):
        result = fuse_views([self.a, self.b], [self.mask]*2, refine=True)
        self.assertFalse(result['accepted'])
        self.assertTrue(any('degenerate' in r for r in result['reasons']))
        self.assertIsNone(result['usable_cloud'])
        self.assertEqual(result['usable_views'], [])

    def test_displaced_surface_rejected(self):
        T = np.eye(4)
        T[2, 3] = .03
        result = fuse_views([self.a, self.b.with_pose(T)], [self.mask]*2)
        self.assertFalse(result['accepted'])
        self.assertTrue(any('depth_conflict' in r for r in result['reasons']))
        self.assertIsNone(result['usable_cloud'])

    def test_target_background_not_averaged_together(self):
        opposite = ~self.mask
        cloud = fuse_arrays([self.a, self.b], [self.mask, opposite], .03)
        self.assertTrue(np.all(np.isin(cloud['fused_source_view_bits'], [1, 2])))

    def test_rejected_output_has_no_usable_artifact_and_cannot_overwrite(self):
        rejected = fuse_views([self.a, self.b], [self.mask]*2, refine=True)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'result'
            save_result(output, rejected)
            self.assertFalse((output/'fusion_usable.npz').exists())
            self.assertFalse((output/'accepted_depth_views.npz').exists())
            with self.assertRaises(ValueError):
                save_result(output, rejected)


if __name__ == '__main__':
    unittest.main()
