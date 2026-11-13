"""Scene/target isolation, frame conversion and actual Bullet body checks."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pybullet as p

from simulation.scene_spec import SceneSpec, load_scene, object_rotation
from simulation.measured_visuals import MeasuredVisuals
from simulation.graspgen_bridge import validate_candidate_source
from simulation.reconstructed_animation import RecordedDemo, add_measured_bodies
from simulation.graspgen_task import execute


class SceneSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.spec = load_scene()

    def tearDown(self):
        self.tmp.cleanup()

    def test_arbitrary_names_and_rotated_target_points(self):
        data = copy.deepcopy(self.spec.data)
        data['geometry']['objects'] = {'parcel': data['geometry']['objects']['box'],
                                       'bottle': data['geometry']['objects']['cup']}
        data['target'] = 'parcel'
        data['table_to_robot_translation_m'] = [.62, -.37, 0]
        spec = SceneSpec(data)
        view = MeasuredVisuals(spec.data['pointcloud_path'], spec.geometry, spec.offset, spec.target)
        item = spec.geometry['objects'][spec.target]
        recovered = view.target_local_xyz @ object_rotation(item).T + item['center_table_m'] + spec.offset
        np.testing.assert_allclose(recovered, view.target_world_xyz, atol=1e-12)
        self.assertGreater(view.target_points, 10000)
        self.assertEqual(set(view.object_point_counts), {'parcel', 'bottle'})

    def test_box_becomes_dynamic_target_and_cup_is_obstacle(self):
        spec = SceneSpec(self.spec.data, target='box')
        demo = RecordedDemo(self.path)
        try:
            center, item = add_measured_bodies(demo, spec.geometry, 'proxy', spec)
            self.assertEqual(item['shape'], 'box')
            self.assertEqual(demo.cube, demo.object_bodies['box'])
            self.assertGreater(p.getDynamicsInfo(demo.cube, -1)[0], 0)
            self.assertEqual(p.getDynamicsInfo(demo.object_bodies['cup'], -1)[0], 0)
            self.assertIn('cup', dict(demo.scene.environment))
            np.testing.assert_allclose(demo.scene.b.getBasePositionAndOrientation(demo.scene.target)[0], center)
            np.testing.assert_allclose(p.getAABB(demo.table), demo.scene.b.getAABB(dict(demo.scene.environment)['table']))
        finally:
            demo.scene.close()
            p.disconnect(demo.client)

    def test_snapshot_roundtrip_and_wrong_target_candidate_rejected(self):
        snapshot = self.path/'scene.json'
        self.spec.save(snapshot)
        self.assertEqual(load_scene(snapshot).fingerprint, self.spec.fingerprint)
        report = {'scene_fingerprint': self.spec.fingerprint, 'target': 'cup'}
        (self.path/'result.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'scene/target mismatch'):
            validate_candidate_source(self.path/'candidates.npz', load_scene(snapshot, 'box'))

    def test_changed_scene_rejected_before_execution(self):
        snapshot = self.path/'scene.json'
        altered = copy.deepcopy(self.spec.data)
        altered['drop_xy_robot_m'] = [.65, -.2]
        SceneSpec(altered).save(snapshot)
        (self.path/'screening.json').write_text(json.dumps({'scene_snapshot': str(snapshot),
                                      'scene_fingerprint': self.spec.fingerprint}))
        with self.assertRaisesRegex(ValueError, 'differs from screened'):
            execute(self.path/'screening.json', self.path/'execution')
        self.assertFalse((self.path/'execution').exists())

    def test_bad_geometry_and_missing_target_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown target'):
            SceneSpec(self.spec.data, target='missing')
        data = copy.deepcopy(self.spec.data)
        data['geometry']['objects']['cup']['radius_m'] = -1
        with self.assertRaisesRegex(ValueError, 'Invalid cup'):
            SceneSpec(data)

    def test_no_candidate_still_records_selected_target(self):
        spec = SceneSpec(self.spec.data, target='box')
        snapshot = self.path/'scene.json'
        spec.save(snapshot)
        screening = {'scene_snapshot': str(snapshot), 'scene_fingerprint': spec.fingerprint,
                     'mapping': {'passed': True}, 'candidate_count': 1,
                     'validated_count': 0, 'records': [{'candidate_id': 0, 'score': .5, 'status': 'rejected'}]}
        path = self.path/'screening.json'
        path.write_text(json.dumps(screening))
        task = execute(path, self.path/'execution')
        self.assertEqual(task['target'], 'box')
        self.assertEqual(task['status'], 'no_valid_candidate')
        self.assertFalse(task['task_success'])
        self.assertTrue((self.path/'execution/task.json').is_file())


if __name__ == '__main__':
    unittest.main()
