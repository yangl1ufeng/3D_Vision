"""RC-3D-specific original <-> rendered mask propagation (paper appendix A.2).

No ground truth is read by this module. Proposal fusion and tracking rules remain
documented independent choices because the authors' exact implementation is absent.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import render_pair_geometry, original_resolution_intrinsic, visibility_accept
from .pipeline import predict_view, mask_box
from .metrics import box_iou


def cached_proposals(folder, direction, image, valid, settings):
    """Reuse only identical image, support mask, and SAM2 generation settings."""
    if folder is None:
        return None
    folder = Path(folder)
    old = json.loads((folder/'prediction.json').read_text())['settings']
    keys = ('sam_config','points_per_side','points_per_batch','pred_iou_thresh',
            'stability_score_thresh','crop_n_layers','crop_n_points_downscale_factor',
            'multimask_output','sam2_build_postprocessing','precision')
    if any(old.get(k)!=settings.get(k) for k in keys):
        return None
    for key in ('generation_visibility_filter','propagation_visibility_filter'):
        if old.get(key,True)!=settings.get(key,True):
            return None
    if settings.get('generation_visibility_filter',True) or settings.get('propagation_visibility_filter',True):
        if any(old.get(k)!=settings.get(k) for k in ('alpha','alpha_rule')):
            return None
    image_file = 'cross_render.png' if direction=='reference_to_target' else 'target.png'
    if not np.array_equal(np.array(Image.open(folder/image_file)),image):
        return None
    if not np.array_equal(np.array(Image.open(folder/'valid.png'))>0,valid):
        return None
    trace = json.loads((folder/f'{direction}_tracking.json').read_text())
    with np.load(folder/f'{direction}_tracking.npz') as masks:
        return [{'segmentation':masks[f"source_{r['id']:04d}"], 'predicted_iou':r['predicted_iou'],
                 'proposal_origin':r.get('proposal_origin',direction)}
                for r in trace]


def fuse_proposals(records, threshold, mode='box_nms'):
    """Exact deduplication preserves distinct object hypotheses before classification."""
    if mode not in ('box_nms','exact_masks'):
        raise ValueError('Unknown proposal fusion policy')
    ranked = sorted(records, key=lambda r: -float(r['predicted_iou']))
    chosen, boxes, seen = [], [], set()
    for record in ranked:
        box = mask_box(record['segmentation'])
        if box is None:
            continue
        digest = hashlib.sha256(np.packbits(record['segmentation']).tobytes()).digest()
        if mode=='exact_masks' and digest in seen:
            continue
        if mode=='box_nms' and boxes and threshold is not None and np.max(box_iou(box, boxes)) > threshold:
            continue
        chosen.append(record)
        boxes.append(box)
        seen.add(digest)
    return chosen


def predict_rc3d_pair(scene, metadata, native_images, tracker, settings, outdir, proposal_cache=None, paired_cache=None):
    """Keep T0/T1 semantics fixed. Never obtain view 0 by reversing time."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    k = original_resolution_intrinsic(scene['K_1'], metadata['image_transforms'][1])
    rendered = render_pair_geometry(scene, 1, settings['splat_radius'], settings['depth_tolerance'],
        settings.get('disable_cleaning', False), render_intrinsic=k, render_shape=native_images[1].shape[:2])
    cross, clean, valid, clean_valid, kept = rendered
    original = native_images[0]
    if paired_cache is not None:
        paired_cache = Path(paired_cache)
        old_settings = json.loads((paired_cache/'view_1/prediction.json').read_text())['settings']
        if old_settings.get('proposal_box_nms')!=settings.get('proposal_box_nms'):
            raise ValueError('Cannot reuse fused proposals with a different fusion rule')
        if old_settings.get('proposal_fusion','box_nms')!=settings.get('proposal_fusion','box_nms'):
            raise ValueError('Cannot reuse a different proposal fusion policy')
        k0 = original_resolution_intrinsic(scene['K_0'],metadata['image_transforms'][0])
        after0,clean0,valid0,clean_valid0,kept0 = render_pair_geometry(scene,0,settings['splat_radius'],
            settings['depth_tolerance'],settings.get('disable_cleaning',False),render_intrinsic=k0,
            render_shape=original.shape[:2])
        records1 = cached_proposals(paired_cache/'view_1','reference_to_target',cross,valid,settings)
        current1 = cached_proposals(paired_cache/'view_1','target_to_reference',native_images[1],valid,settings)
        records0 = cached_proposals(paired_cache/'view_0','reference_to_target',original,valid0,settings)
        current0 = cached_proposals(paired_cache/'view_0','target_to_reference',after0,valid0,settings)
        if any(r is None for r in (records1,current1,records0,current0)):
            raise ValueError('Paired proposal cache failed image/support/settings verification')
        trace = json.loads((paired_cache/'rc3d_propagation.json').read_text())
        trace['identical_fused_proposals_reused_from'] = str(paired_cache)
        (outdir/'rc3d_propagation.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
        report = predict_view(scene,1,tracker,settings,outdir/'view_1',native_images,k,
            reference_records=records1,rendered=rendered,current_records=current1)
        source_report = predict_view(scene,0,tracker,settings,outdir/'view_0',native_images,k0,
            reference_records=records0,rendered=(original,clean0,valid0,clean_valid0,kept0),
            current_image=after0,reference_display=original,current_records=current0)
        report['source_view_objects'] = len(source_report['objects'])
        return report
    original_valid = np.ones(original.shape[:2], dtype=bool)
    cross_records = cached_proposals(proposal_cache,'reference_to_target',cross,valid,settings)
    current_records = cached_proposals(proposal_cache,'target_to_reference',native_images[1],valid,settings)
    cached_cross = cross_records is not None
    cached_current = current_records is not None
    if cross_records is None:
        cross_records = tracker.masks(cross, valid)
    original_records = tracker.masks(original, original_valid)
    to_original = tracker.track(cross, original, [r['segmentation'] for r in cross_records], original_valid)
    to_cross = tracker.track(original, cross, [r['segmentation'] for r in original_records], valid)
    admitted = lambda t: t['status']=='present' if settings.get('propagation_visibility_filter',True) else bool(np.any(t['mask']))
    records, source_records = [], []
    for record, track in zip(cross_records, to_original):
        records.append(dict(record, source_mask=track['mask'] if track['status'] == 'present' else None,
                            proposal_origin='cross_to_original'))
        if admitted(track):
            source_records.append(dict(record, segmentation=track['mask'], proposal_origin='cross_to_original'))
    for record, track in zip(original_records, to_cross):
        source_records.append(dict(record, proposal_origin='original_to_cross'))
        if admitted(track):
            records.append(dict(record, segmentation=track['mask'], source_mask=record['segmentation'],
                                proposal_origin='original_to_cross'))
    before_fusion = len(records)
    records = fuse_proposals(records, settings.get('proposal_box_nms'),settings.get('proposal_fusion','box_nms'))
    source_records = fuse_proposals(source_records, settings.get('proposal_box_nms'),settings.get('proposal_fusion','box_nms'))
    trace = {'original_proposals': len(original_records), 'cross_proposals': len(cross_records),
             'identical_cross_proposals_reused':cached_cross,'identical_current_proposals_reused':cached_current,
             'original_to_cross_success': sum(t['success'] for t in to_cross),
             'cross_to_original_success': sum(t['success'] for t in to_original),
             'before_fusion': before_fusion, 'after_fusion': len(records),
             'source_after_fusion': len(source_records),
             'proposal_box_nms': settings.get('proposal_box_nms'),
             'proposal_fusion':settings.get('proposal_fusion','box_nms'),
             'time_direction': 'T0 original to T1 viewpoint; unchanged',
             'records': [{'origin': r['proposal_origin'], 'score': float(r['predicted_iou']),
                          'target_box': mask_box(r['segmentation']),
                          'source_box': mask_box(r['source_mask']) if r['source_mask'] is not None else None}
                         for r in records]}
    (outdir/'rc3d_propagation.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    report = predict_view(scene, 1, tracker, settings, outdir/'view_1', native_images, k,
                          reference_records=records, rendered=rendered,current_records=current_records)
    k0 = original_resolution_intrinsic(scene['K_0'], metadata['image_transforms'][0])
    after_at_source,clean0,valid0,clean_valid0,kept0 = render_pair_geometry(scene, 0,
        settings['splat_radius'], settings['depth_tolerance'], settings.get('disable_cleaning',False),
        render_intrinsic=k0,render_shape=original.shape[:2])
    if settings.get('generation_visibility_filter',True):
        source_records = [r for r in source_records if visibility_accept(r['segmentation'],valid0,
            settings['alpha'],settings['alpha_rule'])[0]]
    # Source viewpoint retains OLD=I0, NEW=R1,0. Merely swapping T0/T1 is wrong.
    source_report = predict_view(scene, 0, tracker, settings, outdir/'view_0',native_images,k0,
        reference_records=source_records, rendered=(original,clean0,valid0,clean_valid0,kept0),
        current_image=after_at_source,reference_display=original)
    report['source_view_objects'] = len(source_report['objects'])
    return report
