import unittest

import numpy as np

from offline_core.static_depth import fuse_static_depth


class StaticDepthTests(unittest.TestCase):
    def test_median_rejects_one_bad_reading_and_keeps_unseen_unknown(self):
        frames = np.array([
            [[1.00, 0], [1.5, 0]],
            [[1.01, 0], [1.5, 0]],
            [[.99, 1.2], [1.5, 0]],
            [[1.00, 0], [1.5, 0]],
            [[1.8, 0], [1.5, 0]],
        ], dtype=np.float32)
        fused, count, quality = fuse_static_depth(frames, min_observations=3)
        self.assertAlmostEqual(float(fused[0, 0]), 1., places=5)
        self.assertEqual(int(count[0, 0]), 5)
        self.assertEqual(float(fused[0, 1]), 0.)
        self.assertEqual(float(fused[1, 1]), 0.)
        self.assertTrue(quality['stationarity_warning'])

    def test_stationary_noise_does_not_trigger_motion_warning(self):
        frames = np.stack([np.full((6, 8), 1 + x, np.float32)
                           for x in (-.003, .002, 0, -.001, .004)])
        fused, _, quality = fuse_static_depth(frames, min_observations=3)
        self.assertTrue(np.allclose(fused, 1., atol=.001))
        self.assertFalse(quality['stationarity_warning'])


if __name__ == '__main__':
    unittest.main()
