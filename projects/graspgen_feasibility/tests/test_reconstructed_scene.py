"""End-to-end geometry-to-UR5 planning check on the saved Gemini capture."""
import tempfile
from pathlib import Path
import unittest

from simulation.reconstructed_scene import DEFAULT_GEOMETRY, run


class ReconstructedSceneTests(unittest.TestCase):
    def test_saved_capture_reaches_cup_with_box_present(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(DEFAULT_GEOMETRY, Path(directory))
            self.assertTrue(result['cup_opening_fit'])
            self.assertFalse(result['box_robust_pinch_fit'])
            self.assertEqual(result['cup_plan']['status'], 'pregrasp_path_validated')
            self.assertEqual(result['cup_plan']['approach_status'], 'sampled_path_clear')
            self.assertGreater(len(result['cup_plan']['safe_approach_path_rad']), 10)
            self.assertTrue((Path(directory) / 'scene_approach.png').is_file())


if __name__ == '__main__':
    unittest.main()
