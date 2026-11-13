"""Saved RGB-D click masks actually govern exported model input."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from click_selection import observation, export_selection
from simulation.scene_spec import load_scene

PROJECT = Path(__file__).resolve().parents[1]


class ClickSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_scene()
        cls.obs = observation(cls.spec)

    def mask(self, name):
        return np.asarray(Image.open(PROJECT/f'runs/click_selection/{name}/mask.png')) > 0

    def test_cup_mask_exports_distinct_measured_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'selection'
            report = export_selection(self.spec, self.obs, self.mask('validation_cup'),
                                      {'sam2_quality': .9}, path, 'saved_mask_regression')
            self.assertEqual(report['target'], 'cup')
            self.assertGreater(report['visible_measured_target_coverage'], .9)
            selected = np.load(path/'target_points_table.npy')
            self.assertEqual(len(selected), report['exported_points'])
            self.assertGreater(len(selected), 4000)
            scene = load_scene(path/'scene.json')
            self.assertEqual(scene.target, 'cup')
            self.assertEqual(scene.data['target_observation']['target'], 'cup')
            with self.assertRaisesRegex(ValueError, 'another target'):
                load_scene(path/'scene.json', 'box')

    def test_partial_box_selection_rejected_but_refined_selection_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'partial'
            with self.assertRaisesRegex(ValueError, 'only part'):
                export_selection(self.spec, self.obs, self.mask('validation_box'),
                                 {'sam2_quality': .8}, path, 'saved_mask_regression')
            self.assertFalse(path.exists())
            path = Path(folder)/'refined'
            report = export_selection(self.spec, self.obs, self.mask('validation_box_refined'),
                                      {'sam2_quality': .95}, path, 'saved_mask_regression')
            self.assertEqual(report['target'], 'box')
            self.assertGreater(report['visible_measured_target_coverage'], .9)
            self.assertGreater(report['exported_points'], 10000)


if __name__ == '__main__':
    unittest.main()
