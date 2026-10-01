"""Optional identity checks, explicitly separate from paper set-difference rules."""
import numpy as np


def mask_iou(a,b):
    union = np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else 0.


def verify_tracks(tracker,source,target,masks,support,settings,aligned=False,tracks=None):
    """Keep an unverified match distinct from an actually empty model prediction.

    Thresholds are fixed engineering assumptions, not GOLDILOCS parameters.
    Aligned comparison uses shared source support, never clean-render coverage.
    Free-position moved tracking does not require same-location overlap.
    """
    tracks = [dict(t) for t in (tracker.track(source,target,masks,support) if tracks is None else tracks)]
    ids = [i for i,t in enumerate(tracks) if np.any(t['mask'])]
    backs = tracker.track(target,source,[tracks[i]['mask'] for i in ids],
        np.ones(source.shape[:2],dtype=bool)) if ids else []
    for i,t in enumerate(tracks):
        t['identity_status'] = 'no_forward_mask' if not np.any(t['mask']) else 'unchecked'
    for i,back in zip(ids,backs):
        t = tracks[i]
        cycle = mask_iou(masks[i],back['mask'])
        alignment = mask_iou(masks[i]&support,t['mask']&support) if aligned else None
        cycle_ok = bool(np.any(back['mask'])) and cycle>=settings['cycle_min_iou']
        aligned_ok = alignment is None or alignment>=settings['clean_aligned_min_iou']
        t.update(cycle_iou=cycle,aligned_iou=alignment,
                 identity_status='confirmed' if cycle_ok and aligned_ok else 'rejected')
        if not cycle_ok or not aligned_ok:
            t.update(status='unobserved',success=False,
                     identity_reason='failed_cycle_identity' if not cycle_ok else 'failed_aligned_identity')
    return tracks
