"""Regression: rejected evidence must not be treated as observed disappearance."""
import unittest
import tempfile
from unittest.mock import patch
import numpy as np
from goldilocs_local.segmentation import summarize_track, track_with_cycle
from goldilocs_local.pipeline import predict_view


class TrackingSemanticsTests(unittest.TestCase):
    settings = {'alpha': .25, 'alpha_rule': 'main_text', 'min_track_pixels': 1}

    def test_nonempty_out_of_view_track_is_unobserved(self):
        mask = np.array([[1,1],[0,0]], dtype=bool)
        valid = ~mask
        result = summarize_track(mask, valid, self.settings)
        self.assertEqual(result['status'], 'unobserved')
        self.assertNotEqual(result['status'], 'absent')

    def test_empty_track_is_absent(self):
        result = summarize_track(np.zeros((2,2), dtype=bool), np.ones((2,2), dtype=bool), self.settings)
        self.assertEqual(result['status'], 'absent')

    def test_visible_nonempty_track_is_present(self):
        mask = np.ones((2,2), dtype=bool)
        result = summarize_track(mask, mask, self.settings)
        self.assertEqual(result['status'], 'present')
        self.assertTrue(result['success'])

    def check_pipeline(self, clean_status, other_status, expected):
        mask = np.ones((4,4), dtype=bool)
        rgb = np.full((4,4,3), 100, dtype=np.uint8)
        class Tracker:
            calls = 0
            tracks = 0
            def masks(self, source, valid):
                self.calls += 1
                return [{'segmentation': mask, 'predicted_iou': .99}] if self.calls == 1 else []
            def track(self, source, target, masks, valid):
                if not masks:
                    return []
                self.tracks += 1
                status = clean_status if self.tracks == 1 else other_status
                return [{'mask': mask if status != 'absent' else mask & False,
                         'status': status, 'success': status == 'present',
                         'visible_fraction': 0. if status != 'present' else 1.}]
        settings = {'splat_radius': 1, 'depth_tolerance': 0., 'detect_warped': False}
        with tempfile.TemporaryDirectory() as folder, patch(
            'goldilocs_local.pipeline.render_pair_geometry', return_value=(rgb,rgb,mask,mask,[1.,1.])):
            result = predict_view({'rgb_0': rgb, 'rgb_1': rgb}, 1, Tracker(), settings, folder)
        self.assertEqual([o['label'] for o in result['objects']], expected)

    def test_unobserved_clean_track_never_becomes_removed(self):
        self.check_pipeline('unobserved', 'absent', [])

    def test_unobserved_second_track_never_becomes_removed(self):
        self.check_pipeline('absent', 'unobserved', [])

    def test_absent_in_both_frames_still_becomes_removed(self):
        self.check_pipeline('absent', 'absent', ['removed'])

    def test_cycle_rejects_return_to_a_different_object(self):
        first = np.zeros((8,8),dtype=bool)
        first[1:3,1:3] = True
        other = np.zeros_like(first)
        other[5:7,5:7] = True
        class Tracker:
            def track(self,source,target,masks,valid):
                return [{'mask':other,'status':'present','success':True} for m in masks]
        rgb = np.zeros((8,8,3),dtype=np.uint8)
        result = track_with_cycle(Tracker(),rgb,rgb,[first],np.ones_like(first))[0]
        self.assertEqual(result['status'],'absent')
        self.assertEqual(result['cycle_iou'],0.)

    def test_cycle_allows_a_moved_object_that_returns_to_its_source(self):
        first = np.zeros((8,8),dtype=bool)
        first[1:3,1:3] = True
        moved = np.zeros_like(first)
        moved[5:7,5:7] = True
        class Tracker:
            calls=0
            def track(self,source,target,masks,valid):
                self.calls+=1
                return [{'mask':moved if self.calls==1 else first,'status':'present','success':True} for m in masks]
        rgb = np.zeros((8,8,3),dtype=np.uint8)
        result = track_with_cycle(Tracker(),rgb,rgb,[first],np.ones_like(first))[0]
        self.assertEqual(result['status'],'present')
        self.assertEqual(result['cycle_iou'],1.)


if __name__ == '__main__':
    unittest.main()
