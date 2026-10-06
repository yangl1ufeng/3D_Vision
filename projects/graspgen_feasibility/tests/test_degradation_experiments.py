import unittest
import numpy as np
from experiments.input_degradation import degrade, perturb_box, mask_metrics, voxel_centroids, bounded_points


class InputDegradationTests(unittest.TestCase):
    def setUp(self):
        self.depth=np.ones((64,64))
        self.depth[:4]=0
        self.mask=np.zeros((64,64),bool)
        self.mask[20:44,20:44]=True

    def test_noise_reproducible_and_never_fills_unobserved_depth(self):
        a,m=degrade(self.depth,self.mask,'noise',.003,123)
        b,_=degrade(self.depth,self.mask,'noise',.003,123)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(a[:4]==0))
        np.testing.assert_array_equal(m,self.mask)
        self.assertTrue(np.all(self.depth[4:]==1))

    def test_dropout_and_holes_remove_without_inventing_or_moving_points(self):
        for family,level in [('dropout',.5),('hole',.5)]:
            d,m=degrade(self.depth,self.mask,family,level,12)
            self.assertTrue(np.all((d==0)|(d==self.depth)))
            self.assertLess(np.sum(d>0),np.sum(self.depth>0))
            np.testing.assert_array_equal(m,self.mask)

    def test_morphology_changes_labels_only_and_can_fail_empty(self):
        d,eroded=degrade(self.depth,self.mask,'erode',12,1)
        _,dilated=degrade(self.depth,self.mask,'dilate',3,1)
        np.testing.assert_array_equal(d,self.depth)
        self.assertFalse(eroded.any())
        self.assertTrue(np.all(dilated[self.mask]))
        self.assertGreater(dilated.sum(),self.mask.sum())
        self.assertEqual(mask_metrics(eroded,self.mask)['mask_iou'],0)

    def test_box_shift_is_relative_to_object_width_and_clipped(self):
        self.assertEqual(perturb_box([20,20,40,40],(64,64),'box_shift',.5),[30,20,50,40])
        self.assertEqual(perturb_box([20,20,40,40],(64,64),'box_scale',.5),[25,25,35,35])

    def test_voxel_centroids_keep_counts_and_bounded_points_obey_budget(self):
        p=np.array([[.001,0,0],[.003,0,0],[.009,0,0]],dtype=float)
        reduced,inverse=voxel_centroids(p,.004)
        self.assertEqual(len(reduced),2)
        np.testing.assert_allclose(reduced[:,0],[.002,.009])
        self.assertEqual(len(inverse),len(p))
        a=bounded_points(p,2,123)
        np.testing.assert_array_equal(a,bounded_points(p,2,123))
        self.assertEqual(len(a),2)


if __name__=='__main__':
    unittest.main()
