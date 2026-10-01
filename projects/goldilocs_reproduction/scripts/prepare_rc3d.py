"""Validate official RC-3D archive and reproduce author loader's 0/2 pairing."""
import json
import argparse
import sys
from pathlib import Path

from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from goldilocs_local.reconstruct import sha256

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, default=ROOT/'data/rc3d_author_order_manifest.json')
args = p.parse_args()
records = []
for part in ['part1', 'part2', 'part3', 'part4']:
    folder = ROOT/'data/rc3d'/part
    data = json.loads((folder/'coco_annotations.json').read_text())
    images = data['images']
    annotations = data['annotations']
    if len(images) % 3 or len(images) != len(annotations):
        raise ValueError(f'Unexpected triplet count in {part}')
    def box(annotation):
        x,y,w,h = annotation['bbox']
        return [[x,y,x+w,y+h]]
    for before, after, ann0, ann1 in zip(images[0::3], images[2::3], annotations[0::3], annotations[2::3]):
        paths = [folder/x['file_name'] for x in [before, after]]
        for path, meta in zip(paths, [before, after]):
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                if im.size != (meta['width'], meta['height']):
                    raise ValueError(f'Image size mismatch: {path}')
        records.append({'id': part+'_'+Path(before['file_name']).stem.removesuffix('_0'),
            'before': str(paths[0].relative_to(ROOT)).replace('\\', '/'),
            'after': str(paths[1].relative_to(ROOT)).replace('\\', '/'),
            'before_sha256': sha256(paths[0]), 'after_sha256': sha256(paths[1]),
            'before_size_wh': [before['width'], before['height']],
            'after_size_wh': [after['width'], after['height']],
            'before_boxes_xyxy': box(ann0), 'after_boxes_xyxy': box(ann1)})
if len(records) != 100:
    raise ValueError(f'Expected 100 pairs, got {len(records)}')
manifest = {'source': 'https://thor.robots.ox.ac.uk/cyws-3d/rc3d.tar',
    'archive_sha256': sha256(ROOT/'data/rc3d.tar'),
    'pairing_source': 'third_party/cyws3d/rc3d.py: images[0::3] and images[2::3]',
    'pairs': records, 'count': len(records), 'image_files_validated': len(records)*2,
    'ground_truth_depth_used': False}
manifest['annotation_pairing'] = 'ordered annotations[0::3] and [2::3]; image_id is not unique'
args.output.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k != 'pairs'}, indent=2))
