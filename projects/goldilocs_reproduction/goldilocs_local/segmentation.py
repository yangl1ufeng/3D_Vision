"""SAM2 automatic masks and two-frame mask propagation.

Uses the public video API. Presence and visibility are separate decisions:
an out-of-view track is unobserved, not evidence that an object disappeared.
Identity validation of nonempty tracks is still an unresolved limitation.
"""
import gc
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import visibility_accept


def summarize_track(mask, valid, settings):
    """Keep absence, observed presence, and unreliable evidence distinct."""
    pixels = int(np.count_nonzero(mask))
    accepted, fraction = visibility_accept(mask, valid, settings['alpha'], settings['alpha_rule'])
    if pixels == 0:
        status = 'absent'
    elif accepted and pixels >= settings['min_track_pixels']:
        status = 'present'
    else:
        status = 'unobserved'
    return {'mask': mask, 'visible_fraction': fraction, 'status': status,
            'success': status == 'present'}


def validate_aligned_track(source_mask, track, target_valid, settings):
    """Same-camera clean scene: an unrelated disjoint track cannot prove presence.

    This conservative identity guard is an independent implementation choice,
    not an author-confirmed tracking rule. No threshold is fitted to RC-3D GT.
    Missing target support stays unobserved. Moved tracking must not use it.
    """
    result = dict(track)
    supported, fraction = visibility_accept(source_mask, target_valid,
        settings['alpha'], settings['alpha_rule'])
    overlap = int(np.count_nonzero(source_mask & track['mask']))
    result.update(aligned_overlap_pixels=overlap, source_target_visible_fraction=fraction)
    if not supported:
        result.update(status='unobserved', success=False, identity_reason='source_location_unobserved')
    elif np.any(track['mask']) and overlap == 0:
        result.update(status='absent', success=False, identity_reason='disjoint_in_same_camera')
    else:
        result.setdefault('identity_reason', 'unchanged')
    return result


def track_with_cycle(tracker, source, target, masks, target_valid, min_iou=.5):
    """Optional identity verification; forward match must return to its source.

    The .5 threshold is an explicit engineering choice, not a paper parameter.
    Rejected or unobserved forward matches remain distinguishable in the trace.
    """
    tracks = tracker.track(source,target,masks,target_valid)
    indices = [i for i,t in enumerate(tracks) if t['status']=='present']
    if not indices:
        return tracks
    backwards = tracker.track(target,source,[tracks[i]['mask'] for i in indices],
        np.ones(source.shape[:2],dtype=bool))
    for i,back in zip(indices,backwards):
        union = np.count_nonzero(masks[i] | back['mask'])
        iou = float(np.count_nonzero(masks[i] & back['mask'])/union) if union else 0.
        tracks[i] = dict(tracks[i],cycle_iou=iou,cycle_status=back['status'])
        if back['status']!='present' or iou<min_iou:
            tracks[i].update(status='absent',success=False,identity_reason='failed_cycle_consistency')
    return tracks


class SegmentTracker:
    def __init__(self, checkpoint, sam_config, settings):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self.torch = torch
        self.settings = settings
        # Custom CUDA hole-filling extension is optional and unavailable locally.
        # Disable the builder's optional postprocessing overrides explicitly.
        self.model = build_sam2_video_predictor(sam_config, str(checkpoint), device='cuda',
            hydra_overrides_extra=['++model.fill_hole_area=0'], apply_postprocessing=False)
        self.generator = SAM2AutomaticMaskGenerator(self.model,
            points_per_side=settings['points_per_side'], points_per_batch=settings['points_per_batch'],
            pred_iou_thresh=settings['pred_iou_thresh'], stability_score_thresh=settings['stability_score_thresh'],
            crop_n_layers=settings['crop_n_layers'], crop_n_points_downscale_factor=settings['crop_n_points_downscale_factor'],
            multimask_output=settings['multimask_output'], use_m2m=False, min_mask_region_area=0)

    def masks(self, rgb, valid):
        with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
            records = self.generator.generate(rgb)
        if self.settings.get('generation_visibility_filter',True):
            records = [r for r in records if visibility_accept(r['segmentation'], valid,
                self.settings['alpha'], self.settings['alpha_rule'])[0]]
        return records

    def track(self, source, target, masks, valid):
        if not masks:
            return []
        answers = []
        batch_size = self.settings['track_batch_size']
        with tempfile.TemporaryDirectory(prefix='sam2_pair_') as temp:
            # SAM2's public frame loader takes a numerically named JPEG directory.
            for i, rgb in enumerate((source, target)):
                Image.fromarray(rgb).save(Path(temp)/f'{i:05d}.jpg', quality=100, subsampling=0)
            with self.torch.inference_mode(), self.torch.autocast('cuda', dtype=self.torch.bfloat16):
                for offset in range(0, len(masks), batch_size):
                    state = self.model.init_state(temp, offload_video_to_cpu=True, offload_state_to_cpu=True)
                    batch = masks[offset:offset+batch_size]
                    for i, mask in enumerate(batch):
                        self.model.add_new_mask(state, frame_idx=0, obj_id=i, mask=mask)
                    found = {}
                    for frame, ids, logits in self.model.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=1):
                        if frame == 1:
                            for i, object_id in enumerate(ids):
                                pred = (logits[i, 0] > 0).cpu().numpy()
                                found[object_id] = summarize_track(pred, valid, self.settings)
                                obj_idx = state['obj_id_to_idx'][object_id]
                                saved = state['output_dict_per_obj'][obj_idx]['non_cond_frame_outputs'][1]
                                score = saved.get('object_score_logits')
                                found[object_id]['object_presence_logit'] = float(score.float().item()) if score is not None else None
                    if len(found) != len(batch):
                        raise RuntimeError('SAM2 did not return every object on target frame')
                    answers.extend(found[i] for i in range(len(batch)))
                    del state
        return answers

    def close(self):
        del self.generator, self.model
        gc.collect()
        self.torch.cuda.empty_cache()
