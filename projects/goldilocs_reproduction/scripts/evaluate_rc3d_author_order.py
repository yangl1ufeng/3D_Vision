"""Diagnostic AP50 with RC-3D's annotation order, matching CYWS-3D/rc3d.py.

This file is intentionally separate from the running inference batch. The
original COCO files repeat image_id=2023 for many images in parts 2-4, so
mapping annotations by image_id yields false ground truth. Pair images and
annotations by their order in triplets, as the dataset author's loader does.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.metrics import detection_ap
from goldilocs_local.pipeline import mask_box
from goldilocs_local.reconstruct import sha256


def load_author_order_gt():
    by_path = {}
    for part in ('part1', 'part2', 'part3', 'part4'):
        source = ROOT/'data/rc3d'/part/'coco_annotations.json'
        data = json.loads(source.read_text(encoding='utf-8'))
        images, ann = data['images'], data['annotations']
        if len(images) != len(ann) or len(images) % 3:
            raise ValueError(f'Unexpected ordered triplets in {source}')
        for image, annotation in zip(images[2::3], ann[2::3]):
            x, y, w, h = annotation['bbox']
            key = f'data/rc3d/{part}/{image["file_name"]}'
            if key in by_path:
                raise ValueError(f'Duplicate target path: {key}')
            by_path[key] = [[x, y, x+w, y+h]]
    if len(by_path) != 100:
        raise ValueError(f'Expected 100 target boxes, got {len(by_path)}')
    return by_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    pairs = json.loads((ROOT/'data/rc3d_manifest.json').read_text(encoding='utf-8'))['pairs']
    source_gt = load_author_order_gt()
    gt, predictions, missing = {}, [], []
    for pair in pairs:
        folder = args.runs/pair['id']
        prediction = folder/'view_1/prediction.json'
        if not prediction.exists() or not (folder/'run.json').exists():
            missing.append(pair['id'])
            continue
        result = json.loads(prediction.read_text(encoding='utf-8'))
        transform = json.loads((folder/'geometry/geometry.json').read_text(encoding='utf-8'))['image_transforms'][1]
        ow, oh = transform['original_wh']
        rw, rh = transform['resized_wh']
        left, top = transform['crop_left_top']
        gt[pair['id']] = source_gt[pair['after']]
        label_map = np.array(Image.open(folder/'view_1/labels.png'))
        with np.load(folder/'view_1/object_masks.npz', allow_pickle=False) as masks:
            for obj in result['objects']:
                if obj['label'] != 'removed' or obj['bbox_xyxy'] is None:
                    continue
                mask = masks[f"mask_{obj['id']:04d}"] & (label_map == result['label_ids']['removed'])
                box = mask_box(mask)
                if box is None:
                    continue
                box = np.array(box, dtype=float)
                if result['coordinate_system'] != 'original image pixels':
                    box = (box + [left, top, left, top])*[ow/rw, oh/rh, ow/rw, oh/rh]
                predictions.append({'image_id': pair['id'], 'bbox': box.tolist(), 'score': obj['score']})
    if missing and not args.allow_partial:
        raise RuntimeError(f'{len(missing)} pairs missing; refusing to report a full result')
    if not gt:
        raise RuntimeError('No completed predictions')
    score = detection_ap(gt, predictions)
    report = {
        'kind': 'independent_prototype_diagnostic_not_paper_reproduction',
        'ground_truth_source': 'Original RC-3D coco_annotations.json annotation[2::3], paired with images[2::3] as in CYWS-3D/rc3d.py',
        'ground_truth_warning': 'RC-3D parts 2-4 reuse image_id=2023, therefore image_id-based COCO grouping is invalid',
        'evaluator_status': 'corrected_author_order',
        'evaluator_sha256': sha256(__file__),
        'input_manifest_sha256': sha256(ROOT/'data/rc3d_manifest.json'),
        'expected_pairs': len(pairs), 'completed_pairs': len(gt), 'missing_pairs': missing,
        'ap50': score, 'paper_superiority_comparison_allowed': False,
        'prediction_convention': 'target view removed masks after final class priority; SAM2 mask IoU as detection confidence; no extra NMS',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'completed_pairs': len(gt), 'missing_pairs': len(missing),
        'gt_objects': score['gt_objects'], 'predictions': score['prediction_count'],
        'true_positives': score['true_positives'], 'false_positives': score['false_positives'],
        'diagnostic_ap50': score['ap'], 'interpretation': report['kind']
    }, indent=2))


if __name__ == '__main__':
    main()
