import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from goldilocs_local.pipeline import predict_view
from goldilocs_local.rc3d import fuse_proposals, cached_proposals
from goldilocs_local.identity import verify_tracks
from PIL import Image


class PaperFlowTests(unittest.TestCase):
    def run_case(self,source,valid,clean_status,other_status,clean_valid=None):
        rgb = np.full((*source.shape,3),100,dtype=np.uint8)
        class Tracker:
            calls=0
            def track(self,a,b,masks,support):
                if not masks:return []
                self.calls+=1
                state=clean_status if self.calls==1 else other_status
                return [{'mask':source.copy() if state!='absent' else np.zeros_like(source),
                         'status':state,'success':state=='present','visible_fraction':1. if state=='present' else 0.} for m in masks]
        settings={'classification_policy':'paper_set_difference','alpha':.25,'alpha_rule':'main_text',
                  'splat_radius':1,'depth_tolerance':0.,'detect_warped':False}
        with tempfile.TemporaryDirectory() as temp:
            result=predict_view({'rgb_0':rgb,'rgb_1':rgb},1,Tracker(),settings,temp,
                reference_records=[{'segmentation':source,'predicted_iou':.9}],current_records=[],
                rendered=(rgb,rgb,valid,valid if clean_valid is None else clean_valid,[1.,1.]))
            trace=json.loads((Path(temp)/'reference_to_target_tracking.json').read_text())
        return result,trace

    def test_bad_track_location_does_not_discard_observed_source(self):
        source=np.ones((4,4),dtype=bool)
        result,_=self.run_case(source,source,'absent','unobserved')
        self.assertEqual([o['label'] for o in result['objects']],['removed'])

    def test_filtered_clean_track_participates_in_set_difference(self):
        source=np.ones((4,4),dtype=bool)
        result,_=self.run_case(source,source,'unobserved','absent')
        self.assertEqual([o['label'] for o in result['objects']],['removed'])

    def test_actual_source_outside_shared_support_stays_unclassified(self):
        source=np.ones((4,4),dtype=bool)
        result,trace=self.run_case(source,~source,'absent','absent')
        self.assertEqual(result['objects'],[])
        self.assertFalse(trace[0]['source_visibility_accepted'])

    def test_clean_render_hole_does_not_veto_shared_source(self):
        source=np.ones((4,4),dtype=bool)
        result,_=self.run_case(source,source,'absent','absent',clean_valid=~source)
        self.assertEqual([o['label'] for o in result['objects']],['removed'])

    def test_same_boxes_with_different_masks_are_preserved(self):
        first=np.eye(8,dtype=bool)
        second=np.fliplr(first)
        rows=[{'segmentation':first,'predicted_iou':.9},
              {'segmentation':second,'predicted_iou':.95},
              {'segmentation':first.copy(),'predicted_iou':.8}]
        kept=fuse_proposals(rows,None,mode='exact_masks')
        self.assertEqual(len(kept),2)
        self.assertTrue(any(np.array_equal(r['segmentation'],first) for r in kept))

    def test_small_residue_does_not_confirm_aligned_identity(self):
        source=np.ones((8,8),dtype=bool)
        small=np.zeros_like(source);small[:1,:1]=True
        class Tracker:
            def track(self,a,b,masks,support):
                return [{'mask':source,'status':'present','success':True,'visible_fraction':1.} for m in masks]
        forward=[{'mask':small,'status':'present','success':True,'visible_fraction':1.}]
        rgb=np.zeros((8,8,3),dtype=np.uint8)
        result=verify_tracks(Tracker(),rgb,rgb,[source],source,
            {'cycle_min_iou':.5,'clean_aligned_min_iou':.5},aligned=True,tracks=forward)[0]
        self.assertEqual(result['cycle_iou'],1.)
        self.assertEqual(result['identity_status'],'rejected')
        self.assertEqual(result['status'],'unobserved')
        self.assertTrue(result['mask'].any())  # Rejection is not an empty prediction.

    def test_true_move_does_not_require_same_location_overlap(self):
        source=np.zeros((8,8),dtype=bool);source[:2,:2]=True
        moved=np.zeros_like(source);moved[-2:,-2:]=True
        class Tracker:
            def track(self,a,b,masks,support):
                return [{'mask':source,'status':'present','success':True,'visible_fraction':1.} for m in masks]
        rgb=np.zeros((8,8,3),dtype=np.uint8)
        result=verify_tracks(Tracker(),rgb,rgb,[source],np.ones_like(source),
            {'cycle_min_iou':.5,'clean_aligned_min_iou':.5},aligned=False,
            tracks=[{'mask':moved,'status':'present','success':True,'visible_fraction':1.}])[0]
        self.assertEqual(result['identity_status'],'confirmed')
        self.assertEqual(result['status'],'present')

    def test_missing_return_is_unverified_not_observed_absence(self):
        source=np.ones((8,8),dtype=bool)
        class Tracker:
            def track(self,a,b,masks,support):
                return [{'mask':~source,'status':'absent','success':False,'visible_fraction':0.} for m in masks]
        rgb=np.zeros((8,8,3),dtype=np.uint8)
        result=verify_tracks(Tracker(),rgb,rgb,[source],source,
            {'cycle_min_iou':.5,'clean_aligned_min_iou':.5},
            tracks=[{'mask':source,'status':'present','success':True,'visible_fraction':1.}])[0]
        self.assertEqual(result['status'],'unobserved')
        self.assertEqual(result['identity_status'],'rejected')

    def cache_fixture(self,folder,settings):
        mask=np.ones((4,4),dtype=bool)
        rgb=np.full((4,4,3),100,dtype=np.uint8)
        Image.fromarray(rgb).save(folder/'cross_render.png')
        Image.fromarray(mask.astype('uint8')*255).save(folder/'valid.png')
        (folder/'prediction.json').write_text(json.dumps({'settings':settings}))
        (folder/'reference_to_target_tracking.json').write_text(json.dumps([{'id':0,'predicted_iou':.9}]))
        np.savez(folder/'reference_to_target_tracking.npz',source_0000=mask)
        return rgb,mask

    def test_filtered_cache_cannot_supply_unfiltered_proposals(self):
        old={'alpha':.25,'alpha_rule':'main_text'}
        new=dict(old,generation_visibility_filter=False,propagation_visibility_filter=False)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);rgb,mask=self.cache_fixture(folder,old)
            self.assertIsNone(cached_proposals(folder,'reference_to_target',rgb,mask,new))

    def test_unfiltered_cache_can_reuse_identical_candidates_across_alpha_rules(self):
        old={'alpha':.25,'alpha_rule':'main_text','generation_visibility_filter':False,'propagation_visibility_filter':False}
        new=dict(old,alpha_rule='appendix')
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);rgb,mask=self.cache_fixture(folder,old)
            records=cached_proposals(folder,'reference_to_target',rgb,mask,new)
            self.assertEqual(len(records),1)
            np.testing.assert_array_equal(records[0]['segmentation'],mask)


if __name__=='__main__':
    unittest.main()
