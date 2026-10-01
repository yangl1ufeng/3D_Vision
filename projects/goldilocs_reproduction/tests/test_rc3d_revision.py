import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from goldilocs_local.segmentation import summarize_track, validate_aligned_track
from goldilocs_local.rc3d import fuse_proposals, predict_rc3d_pair
from scripts.evaluate_rc3d_revision import evaluate


class RevisionTests(unittest.TestCase):
    settings = {'alpha': .25, 'alpha_rule': 'main_text', 'min_track_pixels': 1}

    def test_same_camera_jump_is_not_static_presence(self):
        source = np.zeros((8,8),dtype=bool)
        source[1:3,1:3] = True
        jumped = np.zeros_like(source)
        jumped[5:7,5:7] = True
        visible = np.ones_like(source)
        track = summarize_track(jumped,visible,self.settings)
        self.assertEqual(track['status'],'present')  # Allowed for a moved object.
        checked = validate_aligned_track(source,track,visible,self.settings)
        self.assertEqual(checked['status'],'absent')  # Impossible static match.
        self.assertEqual(track['status'],'present')  # Guard never mutates free tracking.

    def test_missing_clean_geometry_cannot_prove_removal(self):
        source = np.ones((8,8),dtype=bool)
        empty = np.zeros_like(source)
        track = summarize_track(empty,source,self.settings)
        checked = validate_aligned_track(source,track,empty,self.settings)
        self.assertEqual(checked['status'],'unobserved')

    def test_disjoint_ghost_outside_shared_fov_does_not_hide_observed_absence(self):
        source = np.zeros((8,8),dtype=bool)
        source[1:3,1:3] = True
        ghost = np.zeros_like(source)
        ghost[5:7,5:7] = True
        shared = source.copy()
        track = summarize_track(ghost,shared,self.settings)
        self.assertEqual(track['status'],'unobserved')
        clean_valid = np.ones_like(source)
        checked = validate_aligned_track(source,track,clean_valid,self.settings)
        self.assertEqual(checked['status'],'absent')
        checked_without_geometry = validate_aligned_track(source,track,~source,self.settings)
        self.assertEqual(checked_without_geometry['status'],'unobserved')

    def test_fusion_keeps_source_association_and_removes_duplicate(self):
        a = np.zeros((8,8),dtype=bool)
        a[1:4,1:4] = True
        b = np.zeros_like(a)
        b[5:7,5:7] = True
        rows = [{'segmentation':a,'source_mask':b,'predicted_iou':.95},
                {'segmentation':a.copy(),'source_mask':a,'predicted_iou':.90},
                {'segmentation':b,'source_mask':a,'predicted_iou':.8}]
        merged = fuse_proposals(rows,.7)
        self.assertEqual(len(merged),2)
        np.testing.assert_array_equal(merged[0]['source_mask'],b)

    def test_source_classification_preserves_old_to_new_time_direction(self):
        old = np.full((4,4,3),10,dtype=np.uint8)
        new = np.full_like(old,20)
        old_at_new = np.full_like(old,11)
        new_at_old = np.full_like(old,22)
        valid = np.ones((4,4),dtype=bool)
        clean = np.full_like(old,30)
        class Tracker:
            def masks(self,image,support):
                return [{'segmentation':valid,'predicted_iou':.95}]
            def track(self,source,target,masks,support):
                return [{'mask':m,'status':'present','success':True} for m in masks]
        metadata = {'image_transforms':[{'original_wh':[4,4],'resized_wh':[4,4],'crop_left_top':[0,0]}]*2}
        settings = dict(self.settings,splat_radius=1,depth_tolerance=0.,proposal_box_nms=.7)
        renders = [(old_at_new,clean,valid,valid,[1.,1.]),(new_at_old,clean,valid,valid,[1.,1.])]
        with tempfile.TemporaryDirectory() as folder, patch('goldilocs_local.rc3d.render_pair_geometry',side_effect=renders), patch(
            'goldilocs_local.rc3d.predict_view',side_effect=[{'objects':[]},{'objects':[]}]) as prediction:
            predict_rc3d_pair({'K_0':np.eye(3),'K_1':np.eye(3)},metadata,[old,new],Tracker(),settings,folder)
        target_call,source_call = prediction.call_args_list
        self.assertEqual(target_call.args[1],1)
        self.assertEqual(source_call.args[1],0)
        np.testing.assert_array_equal(target_call.kwargs['rendered'][0],old_at_new)
        np.testing.assert_array_equal(source_call.kwargs['rendered'][0],old)
        np.testing.assert_array_equal(source_call.kwargs['current_image'],new_at_old)

    def test_paired_evaluation_does_not_reverse_time_or_drop_missing_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'data').mkdir()
            pair = {'id':'sample','before_boxes_xyxy':[[1,1,3,3]],'after_boxes_xyxy':[[4,4,6,6]]}
            (root/'data/rc3d_author_order_manifest.json').write_text(json.dumps({'pairs':[pair]}))
            folder = root/'runs/sample/view_1'
            folder.mkdir(parents=True)
            (folder.parent/'run.json').write_text('{}')
            (folder/'prediction.json').write_text(json.dumps({'coordinate_system':'original image pixels',
                'objects':[{'id':0,'label':'removed','score':.9}]}))
            target = np.zeros((8,8),dtype=bool)
            target[4:6,4:6] = True
            source = np.zeros_like(target)
            source[1:3,1:3] = True
            Image.fromarray(target.astype('uint8')*2).save(folder/'labels.png')
            np.savez(folder/'object_masks.npz',mask_0000=target,source_mask_0000=source)
            source_folder = folder.parent/'view_0'
            source_folder.mkdir()
            (source_folder/'prediction.json').write_text((folder/'prediction.json').read_text())
            Image.fromarray(source.astype('uint8')*2).save(source_folder/'labels.png')
            np.savez(source_folder/'object_masks.npz',mask_0000=source)
            with patch('scripts.evaluate_rc3d_revision.ROOT',root):
                good = evaluate(root/'runs',['sample'])
                self.assertEqual(good['metrics']['pooled']['ap'],1.)
                (source_folder/'prediction.json').write_text(json.dumps({'coordinate_system':'original image pixels','objects':[]}))
                missing = evaluate(root/'runs',['sample'])
                self.assertEqual(missing['metrics']['source']['ap'],0.)
                self.assertEqual(missing['metrics']['pooled']['gt_objects'],2)
                self.assertEqual(missing['metrics']['pooled']['ap'],.5)


if __name__ == '__main__':
    unittest.main()
