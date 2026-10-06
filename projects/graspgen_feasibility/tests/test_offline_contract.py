"""Exercise geometry, input validation, and audit accounting without model loading."""
import csv
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import numpy as np
from PIL import Image
from offline_core.collision import CollisionChecker, apply_collision_decisions, collision_eligible_indices, initial_decisions
from offline_core.export import export_candidates, rank_candidates
from offline_core.geometry import meters_per_unit, validate_transform, valid_pose_mask
from offline_core.scene_io import load_scene
from offline_core.segmentation import provided_segmentation
from offline_core.types import Grasps
from offline_core.config import ROOT


class SceneContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(self.root / 'rgb.png')
        np.save(self.root / 'depth.npy', np.array([[1000, 2000], [0, np.nan]], dtype=np.float32))
        np.save(self.root / 'mask.npy', np.array([[255, 0], [0, 0]], dtype=np.uint8))
        self.T = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]], dtype=float)
        self.manifest = {
            'schema_version': 1, 'scene_id': 'fixture', 'rgb': 'rgb.png',
            'geometry': {'kind': 'depth', 'path': 'depth.npy', 'unit': 'mm', 'aligned_to_rgb': True,
                         'depth_convention': 'z', 'pinhole_rectified': True,
                         'intrinsics': [[2, 0, 0], [0, 2, 0], [0, 0, 1]]},
            'frames': {'world': 'test_world', 'camera_convention': 'opencv', 'T_world_camera': self.T.tolist()},
            'target': {'id': 'test', 'mask': 'mask.npy'}}

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        path = self.root / 'scene.json'
        path.write_text(json.dumps(self.manifest), encoding='utf-8')
        return load_scene(path)

    def test_mm_depth_camera_to_world_and_mask(self):
        scene = self.load()
        np.testing.assert_allclose(scene.xyz_world_m[0, 0], [10, 20, 31])
        np.testing.assert_allclose(scene.xyz_world_m[0, 1], [10, 21, 32])
        np.testing.assert_array_equal(scene.valid, [[True, True], [False, False]])
        mask = provided_segmentation(scene)
        self.assertEqual(mask.backend, 'provided_mask')
        self.assertEqual(int(mask.mask.sum()), 1)

    def test_camera_xyz_and_depth_agree(self):
        depth_scene = self.load()
        np.save(self.root / 'xyz.npy', np.array([[[0, 0, 1000], [1000, 0, 2000]],
                                              [[0, 0, 0], [0, 0, np.nan]]], dtype=np.float32))
        self.manifest['geometry'] = {'kind': 'xyz', 'path': 'xyz.npy', 'unit': 'mm',
                                     'frame': 'camera', 'aligned_to_rgb': True}
        xyz_scene = self.load()
        np.testing.assert_allclose(xyz_scene.xyz_world_m[depth_scene.valid], depth_scene.xyz_world_m[depth_scene.valid])
        np.testing.assert_array_equal(xyz_scene.valid, depth_scene.valid)

    def test_world_xyz_negative_z_allowed(self):
        np.save(self.root / 'xyz.npy', np.full((2, 2, 3), -1, dtype=np.float32))
        self.manifest['geometry'] = {'kind': 'xyz', 'path': 'xyz.npy', 'unit': 'm',
                                     'frame': 'world', 'aligned_to_rgb': True}
        self.manifest['frames'] = {'world': 'world'}
        scene = self.load()
        self.assertTrue(scene.valid.all())
        np.testing.assert_array_equal(scene.xyz_world_m, -np.ones((2, 2, 3)))

    def test_unaligned_geometry_rejected(self):
        self.manifest['geometry']['aligned_to_rgb'] = False
        with self.assertRaisesRegex(ValueError, 'aligned'):
            self.load()

    def test_unknown_calibration_rejected(self):
        del self.manifest['frames']['T_world_camera']
        with self.assertRaisesRegex(ValueError, 'T_world_camera'):
            self.load()

    def test_depth_resolution_mismatch_rejected(self):
        np.save(self.root / 'depth.npy', np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, 'resolution'):
            self.load()

    def test_nonbinary_valid_mask_rejected(self):
        np.save(self.root / 'valid.npy', [[1, np.nan], [0, 1]])
        self.manifest['geometry']['valid_mask'] = 'valid.npy'
        with self.assertRaisesRegex(ValueError, 'valid_mask'):
            self.load()

    def test_target_mask_resolution_rejected(self):
        np.save(self.root / 'mask.npy', np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, 'mask must match'):
            provided_segmentation(self.load())

    def test_empty_target_mask_rejected(self):
        np.save(self.root / 'mask.npy', np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, 'empty'):
            provided_segmentation(self.load())

    def test_cli_saves_diagnostic_for_bad_input(self):
        self.manifest['geometry']['aligned_to_rgb'] = False
        path = self.root / 'scene.json'
        path.write_text(json.dumps(self.manifest), encoding='utf-8')
        output = self.root / 'failed_run'
        run = subprocess.run([sys.executable, str(ROOT / 'analyze_scene.py'), '--input', str(path),
                              '--output', str(output)], capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 2)
        failure = json.loads((output / 'failure.json').read_text(encoding='utf-8'))
        self.assertEqual(failure['status'], 'failed')
        self.assertIn('aligned', failure['error'])
        self.assertEqual(failure['performance']['stages']['input_loading']['status'], 'failed')

    def test_cli_preserves_nonempty_output(self):
        output = self.root / 'existing'
        output.mkdir()
        sentinel = output / 'summary.json'
        sentinel.write_text('previous result', encoding='utf-8')
        run = subprocess.run([sys.executable, str(ROOT / 'analyze_scene.py'), '--official-sample',
                              '--output', str(output)], capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'previous result')


class GeometryValidation(unittest.TestCase):
    def test_units_explicit_and_consistent(self):
        self.assertEqual(meters_per_unit('mm'), .001)
        self.assertEqual(meters_per_unit('raw', .0001), .0001)
        for unit, scale in [('mm', 1), ('raw', None), ('unknown', None), ('raw', -1), ('raw', np.nan)]:
            with self.subTest(unit=unit, scale=scale), self.assertRaises(ValueError):
                meters_per_unit(unit, scale)

    def test_scale_reflection_and_bad_row_rejected(self):
        for T in [np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]), np.diag([1, 1, 1, 0])]:
            with self.assertRaises(ValueError):
                validate_transform(T)


class CandidateAudit(unittest.TestCase):
    def setUp(self):
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 6, axis=0)
        poses[1, 0, 0] = -1  # Reflection.
        poses[2, 0, 3] = np.nan
        self.grasps = Grasps(poses, np.array([.2, .9, np.nan, .85, .95, .9]), {})

    def test_reasons_ranks_and_export_preserve_every_candidate(self):
        decisions = initial_decisions(self.grasps, .8)
        indices = collision_eligible_indices(decisions)
        np.testing.assert_array_equal(indices, [3, 4, 5])
        apply_collision_decisions(decisions, indices, [False, True, True])
        self.assertEqual(rank_candidates(decisions, self.grasps.scores), [4, 5])
        self.assertEqual(decisions[0]['reasons'], ['low_score'])
        self.assertEqual(decisions[1]['reasons'], ['invalid_pose'])
        self.assertEqual(decisions[2]['reasons'], ['invalid_pose', 'invalid_score'])
        self.assertEqual(decisions[3]['reasons'], ['scene_collision'])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            export_candidates(output, self.grasps, decisions, 'world')
            text = (output / 'candidates.json').read_text(encoding='utf-8')
            self.assertNotIn('NaN', text)
            records = json.loads(text)['candidates']
            self.assertEqual(len(records), 6)
            self.assertIsNone(records[2]['score'])
            self.assertIsNone(records[2]['T_world_graspgen_gripper'][0][3])
            with (output / 'candidates.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[3]['reasons'], 'scene_collision')
            self.assertEqual(rows[4]['rank'], '1')

    def test_empty_background_is_unknown_and_rejected(self):
        decisions = initial_decisions(self.grasps, .8)
        indices = collision_eligible_indices(decisions)
        checker = CollisionChecker.__new__(CollisionChecker)
        free = checker.check(np.empty((0, 3)), self.grasps.poses_world_m[indices])
        apply_collision_decisions(decisions, indices, free, scene_available=False)
        self.assertFalse(any(d['accepted'] for d in decisions))
        for i in indices:
            self.assertEqual(decisions[i]['reasons'], ['no_scene_geometry'])
            self.assertFalse(decisions[i]['collision_checked'])

    def test_empty_pose_batch_is_valid(self):
        self.assertEqual(len(valid_pose_mask(np.empty((0, 4, 4)))), 0)


if __name__ == '__main__':
    unittest.main()
