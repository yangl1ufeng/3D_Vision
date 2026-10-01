import unittest
import numpy as np
from goldilocs_local.geometry import project, rasterize, reverse_depth_keep, visibility_accept, original_resolution_intrinsic


class GeometryTests(unittest.TestCase):
    def test_native_projection_undoes_resize_and_crop(self):
        t = {'original_wh': [1200, 800], 'resized_wh': [600, 400], 'crop_left_top': [20, 10]}
        k = np.array([[400., 0, 280], [0, 400., 190], [0, 0, 1.]])
        p = np.array([[.1, .2, 2.]])
        small, _, _ = project(p, k, np.eye(4))
        native, _, _ = project(p, original_resolution_intrinsic(k, t), np.eye(4))
        np.testing.assert_allclose(native, (small+[20, 10]+.5)*2-.5)

    def test_splat_overlaps_choose_nearest_point(self):
        rgb, _, _ = rasterize([[0, 0, 2], [1, 0, 1]], [[255, 0, 0], [0, 255, 0]],
                              np.eye(3), np.eye(4), (3, 3), radius=1)
        np.testing.assert_array_equal(rgb[0, 0], [0, 255, 0])

    def test_occluded_and_outside_are_not_free_space(self):
        k, pose = np.eye(3), np.eye(4)
        pts = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3], [20, 0, 1]])
        keep = reverse_depth_keep(pts, np.full((3, 3), 2.), k, pose)
        np.testing.assert_array_equal(keep, [False, True, True, True])

    def test_black_surface_is_valid_and_nearer_wins(self):
        rgb, depth, valid = rasterize([[0, 0, 3], [0, 0, 2], [0, 0, -1]],
            [[255, 0, 0], [0, 0, 0], [0, 255, 0]], np.eye(3), np.eye(4), (2, 2), 0)
        self.assertTrue(valid[0, 0])
        self.assertEqual(depth[0, 0], 2)
        self.assertEqual(rgb[0, 0].sum(), 0)
        self.assertFalse(valid[1, 1])

    def test_camera_convention(self):
        pose = np.eye(4)
        pose[:3, 3] = [1, 2, 3]
        uv, z, valid = project([[3, 6, 5]], np.eye(3), pose)
        np.testing.assert_allclose(uv, [[1, 2]])
        self.assertEqual(z[0], 2)
        self.assertTrue(valid[0])

    def test_alpha_interpretations_and_empty_masks(self):
        mask = np.ones((2, 2), dtype=bool)
        valid = np.array([[1, 1], [0, 0]], dtype=bool)
        self.assertFalse(visibility_accept(mask, valid, .25, 'main_text')[0])
        self.assertTrue(visibility_accept(mask, valid, .25, 'appendix')[0])
        self.assertFalse(visibility_accept(mask & False, valid, 0, 'appendix')[0])


if __name__ == '__main__':
    unittest.main()
