import unittest
import numpy as np
from goldilocs_local.surface import rasterize_surface


class SurfaceRendererTests(unittest.TestCase):
    def scene(self, depth):
        y,x = np.mgrid[:2,:2]
        points = np.stack([x*depth,y*depth,depth],axis=-1).astype(float)
        return points,np.zeros((2,2,3),dtype=np.uint8)

    def test_black_plane_is_valid_surface(self):
        depth = np.ones((2,2))
        points,colors = self.scene(depth)
        k = np.diag([4.,4.,1.])
        rgb,z,valid = rasterize_surface(points,colors,depth,k,np.eye(4),(5,5))
        self.assertTrue(valid.all())
        self.assertEqual(rgb.sum(),0)
        np.testing.assert_allclose(z,1.)

    def test_depth_discontinuity_is_not_bridged(self):
        depth = np.array([[1.,3.],[1.,3.]])
        points,colors = self.scene(depth)
        _,_,valid = rasterize_surface(points,colors,depth,np.diag([4.,4.,1.]),np.eye(4),(5,5))
        self.assertFalse(valid.any())

    def test_removed_vertices_are_not_filled_back_in(self):
        depth = np.ones((2,2))
        points,colors = self.scene(depth)
        keep = np.array([[False,True],[True,True]])
        _,_,valid = rasterize_surface(points,colors,depth,np.diag([4.,4.,1.]),np.eye(4),(5,5),keep=keep)
        self.assertFalse(valid.any())


if __name__=='__main__':
    unittest.main()
