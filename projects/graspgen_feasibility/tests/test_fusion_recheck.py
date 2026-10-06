import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from offline_core.observation import DepthView
from offline_core.fusion import fuse_views, points_normals
from offline_core.export import write_json
from fuse_scene import save_result, seal_artifacts
from recheck_fused_scene import load_accepted_fusion
from experiments.synthetic_depth import render_boxes, look_at
from experiments.collision_suite import make_box
from visibility_fusion_experiments import perturb


class RegistrationObservabilityTests(unittest.TestCase):
    def test_box_edge_normals_cannot_supply_unobserved_plane(self):
        K = np.array([[170., 0, 79.5], [0, 170., 59.5], [0, 0, 1]])
        boxes = [make_box('table', [0, 0, -.01], [1, .8, .02]),
                 make_box('target', [0, 0, .09], [.12, .1, .18])]
        a, labels = render_boxes(boxes, look_at([0, -.65, .48], [0, 0, .08]), K, view_id='front')
        b, other = render_boxes(boxes, look_at([.48, -.34, .42], [0, 0, .08]), K, view_id='side')
        _, normals = points_normals(a, .006)
        self.assertLess(np.max(np.abs(normals[:, 0])), 1e-8)
        result = fuse_views([a, perturb(b, [.01, 0, 0], [0, 0, 1])], [labels == 1, other == 1], refine=True)
        self.assertFalse(result['accepted'])
        self.assertTrue(any('degenerate_registration_geometry' in r for r in result['reasons']))


class AcceptedFusionLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)/'fusion'
        K = np.array([[80., 0, 31.5], [0, 80., 31.5], [0, 0, 1]])
        self.a = DepthView('a', np.ones((64, 64)), K, np.eye(4))
        self.b = DepthView('b', np.ones((64, 64)), K, np.eye(4))
        mask = np.zeros((64, 64), bool)
        mask[20:44, 20:44] = True
        shift = np.eye(4)
        shift[0, 3] = .002
        result = fuse_views([self.a.with_pose(shift), self.b.with_pose(shift)], [mask, mask])
        save_result(self.folder, result)
        self.original = {'world_frame': 'table', 'asset_sha256': {'manifest': 'fixture', 'geometry': 'depth', 'target_mask': 'mask'},
                         'input_manifest': {'scene_id': 'a', 'target': {'id': 'target'}, 'frames': {'T_world_camera': np.eye(4).tolist()}}}
        second = copy.deepcopy(self.original)
        second['input_manifest']['scene_id'] = 'b'
        write_json(self.folder/'input_provenance.json', [self.original, second])
        seal_artifacts(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_camera_correction_is_carried_to_saved_poses(self):
        views, points, target, correction, _ = load_accepted_fusion(self.folder, self.original)
        self.assertEqual(len(views), 2)
        self.assertEqual(len(points), 8192)
        self.assertTrue(target.any() and (~target).any())
        np.testing.assert_allclose(correction[:3, 3], [.002, 0, 0], atol=1e-12)

    def test_changed_calibration_or_mask_source_is_rejected(self):
        altered = copy.deepcopy(self.original)
        altered['asset_sha256']['manifest'] = 'different'
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            load_accepted_fusion(self.folder, altered)

    def test_different_target_or_world_is_rejected(self):
        for key in ['target', 'world']:
            altered = copy.deepcopy(self.original)
            if key == 'target': altered['input_manifest']['target']['id'] = 'other'
            else: altered['world_frame'] = 'robot'
            with self.assertRaisesRegex(ValueError, 'target or world'):
                load_accepted_fusion(self.folder, altered)

    def test_changed_depth_file_is_rejected(self):
        with (self.folder/'accepted_depth_views.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            load_accepted_fusion(self.folder, self.original)

    def test_rejected_fusion_cannot_use_stale_usable_files(self):
        write_json(self.folder/'fusion_quality.json', {'accepted': False, 'planning_allowed': False, 'reasons': ['test rejection']})
        seal_artifacts(self.folder)
        with self.assertRaisesRegex(ValueError, 'fusion was rejected'):
            load_accepted_fusion(self.folder, self.original)


if __name__ == '__main__':
    unittest.main()
