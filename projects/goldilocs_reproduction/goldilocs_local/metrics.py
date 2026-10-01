"""Explicit standalone diagnostic metrics; not the unpublished author evaluator."""
import numpy as np


def box_iou(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    box = np.asarray(box, dtype=float)
    overlap = np.maximum(0, np.minimum(box[2:], boxes[:, 2:])-np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    a = np.maximum(0, box[2:]-box[:2]).prod()
    b = np.maximum(0, boxes[:, 2:]-boxes[:, :2]).prod(axis=1)
    return overlap/np.maximum(a+b-overlap, 1e-12)


def detection_ap(ground_truth, predictions, iou_threshold=.5):
    """Single-class all-points interpolated AP. Half-open xyxy boxes.

    ground_truth maps image IDs to boxes; predictions contain image_id, bbox, score.
    Images with zero detections MUST remain in ground_truth. An empty GT set has
    undefined AP and returns None, never a misleading perfect score.
    """
    n_gt = sum(len(b) for b in ground_truth.values())
    matched = {k: np.zeros(len(b), dtype=bool) for k, b in ground_truth.items()}
    predictions = sorted(predictions, key=lambda p: -p['score'])
    tp, fp = [], []
    for p in predictions:
        key = p['image_id']
        if key not in ground_truth:
            raise ValueError(f'Prediction outside evaluation manifest: {key}')
        ious = box_iou(p['bbox'], ground_truth[key])
        best = int(ious.argmax()) if len(ious) else -1
        hit = best >= 0 and ious[best] >= iou_threshold and not matched[key][best]
        if hit:
            matched[key][best] = True
        tp.append(int(hit))
        fp.append(int(not hit))
    cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
    precision = cumulative_tp/np.maximum(cumulative_tp+cumulative_fp, 1)
    recall = cumulative_tp/max(n_gt, 1)
    envelope = np.maximum.accumulate(np.r_[0., precision, 0.][::-1])[::-1]
    r = np.r_[0., recall, 1.]
    ap = float(np.sum(np.diff(r)*envelope[1:])) if n_gt else None
    return {'ap': ap, 'iou_threshold': iou_threshold, 'gt_objects': n_gt,
            'prediction_count': len(predictions), 'true_positives': sum(tp),
            'false_positives': sum(fp), 'precision': precision.tolist(), 'recall': recall.tolist()}
