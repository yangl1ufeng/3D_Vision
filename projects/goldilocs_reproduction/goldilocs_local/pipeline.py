"""Pairwise GOLDILOCS-style inference; no multiview voting or benchmark claims."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from .geometry import render_pair_geometry, visibility_accept
from .segmentation import validate_aligned_track, track_with_cycle
from .identity import verify_tracks

LABELS = {'static': 0, 'added': 1, 'removed': 2, 'moved': 3, 'warped': 4}
COLORS = np.array([[0, 0, 0], [60, 210, 80], [235, 65, 75], [255, 180, 40], [175, 90, 230]], dtype=np.uint8)


def mask_box(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)] if len(x) else None


def predict_view(scene, target, tracker, settings, outdir, native_images=None, render_intrinsic=None,
                 reference_records=None, rendered=None, current_image=None, reference_display=None,
                 current_records=None):
    started = time.perf_counter()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    current = native_images[target] if native_images is not None else scene[f'rgb_{target}']
    if current_image is not None:
        current = current_image
    cross, clean, valid, clean_valid, kept = rendered if rendered is not None else render_pair_geometry(scene, target,
        settings['splat_radius'], settings['depth_tolerance'], settings.get('disable_cleaning', False),
        render_intrinsic=render_intrinsic, render_shape=current.shape[:2])
    geometry_seconds = time.perf_counter()-started
    for name, image in [('cross_render', cross), ('clean_render', clean), ('target', current),
                        ('valid', valid.astype(np.uint8)*255), ('clean_valid', clean_valid.astype(np.uint8)*255)]:
        Image.fromarray(image).save(outdir/f'{name}.png')
    grouped = {name: [] for name in LABELS if name != 'static'}
    static_current = []
    diagnostics = []
    policy = settings.get('classification_policy','legacy_tristate')
    if policy not in ('legacy_tristate','paper_set_difference'):
        raise ValueError('Unknown classification policy')
    for direction, source, other_image, fallback_label in [
        ('reference_to_target', cross, current, 'removed'), ('target_to_reference', current, cross, 'added')]:
        print(f'[{direction}] generating masks', flush=True)
        stage_start = time.perf_counter()
        supplied = reference_records if direction == 'reference_to_target' else current_records
        records = supplied if supplied is not None else tracker.masks(source, valid)
        segmentation_seconds = time.perf_counter()-stage_start
        track_masks = lambda a,b,m,v: track_with_cycle(tracker,a,b,m,v,settings['cycle_min_iou']) if settings.get('cycle_identity_check',False) else tracker.track(a,b,m,v)
        tracked_clean = track_masks(source, clean, [r['segmentation'] for r in records], valid)
        if settings.get('identity_verification',False):
            tracked_clean = verify_tracks(tracker,source,clean,[r['segmentation'] for r in records],
                                         valid,settings,aligned=True,tracks=tracked_clean)
        if settings.get('aligned_clean_identity_guard', False):
            tracked_clean = [validate_aligned_track(r['segmentation'], t, clean_valid, settings)
                             for r, t in zip(records, tracked_clean)]
        trace = []
        traced_masks = {}
        visibility = [visibility_accept(r['segmentation'],valid,settings['alpha'],settings['alpha_rule'])
                      if policy=='paper_set_difference' else (True,1.) for r in records]
        changed_flags = [visible[0] and (not t['success'] if policy=='paper_set_difference' else t['status']=='absent')
                         for visible,t in zip(visibility,tracked_clean)]
        for index, (record, tracked) in enumerate(zip(records, tracked_clean)):
            traced_masks[f'source_{index:04d}'] = record['segmentation']
            traced_masks[f'clean_track_{index:04d}'] = tracked['mask']
            trace.append({'id': index, 'bbox_xyxy': mask_box(record['segmentation']),
                'predicted_iou': float(record['predicted_iou']), 'source_pixels': int(record['segmentation'].sum()),
                'clean_track_pixels': int(tracked['mask'].sum()), 'clean_track_success': tracked['success'],
                'clean_track_status': tracked['status'],
                'clean_changed':bool(changed_flags[index]),
                'source_visibility_accepted':bool(visibility[index][0]),
                'source_visible_fraction':visibility[index][1],
                'proposal_origin':record.get('proposal_origin',direction),
                'clean_track_identity_reason': tracked.get('identity_reason'),
                'clean_track_aligned_overlap_pixels': tracked.get('aligned_overlap_pixels'),
                'clean_track_cycle_iou': tracked.get('cycle_iou'),
                'clean_track_aligned_iou':tracked.get('aligned_iou'),
                'clean_track_identity_status':tracked.get('identity_status'),
                'clean_track_object_presence_logit':tracked.get('object_presence_logit'),
                'clean_track_visible_fraction': tracked['visible_fraction']})
        np.savez_compressed(outdir/f'{direction}_tracking.npz', **traced_masks)
        (outdir/f'{direction}_tracking.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
        # An unreliable nonempty track must not turn a static object into a change.
        changed = [r for r, flag in zip(records, changed_flags) if flag]
        if direction == 'target_to_reference':
            static_current = [r for r, t, v in zip(records,tracked_clean,visibility) if t['success'] and v[0]]
        print(f'[{direction}] {len(records)} visible masks, {len(changed)} failed clean tracking', flush=True)
        tracked_other = track_masks(source, other_image, [r['segmentation'] for r in changed], valid)
        if settings.get('identity_verification',False):
            tracked_other = verify_tracks(tracker,source,other_image,[r['segmentation'] for r in changed],
                                         valid,settings,aligned=False,tracks=tracked_other)
        other_trace = []
        for record, tracked in zip(changed, tracked_other):
            other_trace.append({'source_bbox_xyxy': mask_box(record['segmentation']),
                'target_bbox_xyxy': mask_box(tracked['mask']), 'status': tracked['status'],
                'cycle_iou':tracked.get('cycle_iou'),
                'identity_status':tracked.get('identity_status'),
                'identity_reason':tracked.get('identity_reason'),
                'object_presence_logit':tracked.get('object_presence_logit'),
                'visible_fraction': tracked['visible_fraction']})
            if policy=='legacy_tristate' and tracked['status'] == 'unobserved':
                continue
            label = 'moved' if tracked['status'] == 'present' else fallback_label
            grouped[label].append({'mask': record['segmentation'], 'score': float(record['predicted_iou']),
                                   'origin': direction, 'source_mask': record.get('source_mask'),
                                   'proposal_origin': record.get('proposal_origin', direction)})
            if label == 'moved':
                grouped[label].append({'mask': tracked['mask'], 'score': float(record['predicted_iou']),
                                       'origin': direction+'_tracked'})
        (outdir/f'{direction}_other_tracking.json').write_text(json.dumps(other_trace, indent=2), encoding='utf-8')
        diagnostics.append({'direction': direction, 'visible_masks': len(records), 'changed_masks': len(changed),
                            'segmentation_seconds': segmentation_seconds,
                            'direction_seconds': time.perf_counter()-stage_start,
                            'unobserved_clean_tracks': sum(t['status'] == 'unobserved' for t in tracked_clean),
                            'unobserved_other_tracks': sum(t['status'] == 'unobserved' for t in tracked_other),
                            'tracked_to_other': sum(t['success'] for t in tracked_other)})
    if settings['detect_warped'] and static_current:
        _, ssim = structural_similarity(cross, current, data_range=255, channel_axis=2,
                                       win_size=7, full=True)
        dissimilarity = (1-ssim.mean(axis=2))/2
        scores = [float(dissimilarity[r['segmentation'] & valid].mean()) for r in static_current]
        threshold = float(np.mean(scores)+np.std(scores))
        for record, score in zip(static_current, scores):
            if score > threshold:
                grouped['warped'].append({'mask': record['segmentation'], 'score': score, 'origin': 'ssim'})
    labels = np.zeros(valid.shape, dtype=np.uint8)
    # Low-to-high priority assignment, exactly the class priority in section 4.5.
    objects, arrays = [], {}
    for name in ['added', 'removed', 'moved', 'warped']:
        for record in grouped[name]:
            mask = record['mask']
            labels[mask] = LABELS[name]
            idx = len(objects)
            arrays[f'mask_{idx:04d}'] = mask
            source_mask = record.get('source_mask')
            if source_mask is not None:
                arrays[f'source_mask_{idx:04d}'] = source_mask
            objects.append({'id': idx, 'label': name, 'score': record['score'], 'bbox_xyxy': mask_box(mask),
                            'origin': record['origin'], 'pixels': int(mask.sum()),
                            'source_bbox_xyxy': mask_box(source_mask) if source_mask is not None else None,
                            'proposal_origin': record.get('proposal_origin', record['origin'])})
    np.savez_compressed(outdir/'object_masks.npz', **arrays)
    Image.fromarray(labels).save(outdir/'labels.png')
    overlay = current.copy()
    changed_pixels = labels > 0
    overlay[changed_pixels] = (current[changed_pixels]*.45 + COLORS[labels[changed_pixels]]*.55).astype(np.uint8)
    Image.fromarray(overlay).save(outdir/'overlay.png')
    h, w = valid.shape
    panel = Image.new('RGB', (w*3, h*2+64), '#222222')
    for i, (name, im) in enumerate([('Input reference', reference_display if reference_display is not None else scene[f'rgb_{1-target}']), ('Target', current),
        ('Reference rendered at target', cross), ('Clean canonical render', clean),
        ('Projection validity', np.repeat(valid[:, :, None]*255, 3, axis=2).astype(np.uint8)), ('Prediction (unvalidated)', overlay)]):
        x, y = (i%3)*w, (i//3)*(h+32)
        panel.paste(Image.fromarray(im).resize((w, h)), (x, y+32))
        ImageDraw.Draw(panel).text((x+8, y+8), name, fill='white')
    panel.save(outdir/'diagnostic_panel.jpg', quality=92)
    report = {'implementation': 'independent_prototype', 'benchmark_evaluation': False,
        'target_view': target, 'coordinate_system': 'original image pixels' if native_images is not None else 'MASt3R resized/cropped image grid',
        'output_shape_hw': list(current.shape[:2]), 'geometry_seconds': geometry_seconds,
        'settings': settings, 'kept_point_fractions': kept, 'projection_valid_fraction': float(valid.mean()),
        'clean_valid_fraction': float(clean_valid.mean()), 'tracking': diagnostics,
        'objects': objects, 'label_ids': LABELS, 'seconds': time.perf_counter()-started}
    (outdir/'prediction.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
