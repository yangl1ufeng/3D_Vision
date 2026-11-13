"""Segment named objects in a saved static Gemini 2 capture with point prompts."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from click_selection import PointSegmenter, preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='JSON with capture and object point prompts')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    capture = (args.config.parent / config['capture']).resolve()
    rgb = np.array(Image.open(capture/'rgb.png').convert('RGB'), copy=True)
    depth = np.load(capture/'depth_m.npy', allow_pickle=False)
    if rgb.shape[:2] != depth.shape:
        raise ValueError('RGB and depth dimensions differ')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = PointSegmenter()
    result = {'capture': str(capture), 'objects': {}}
    masks = {}
    for name, item in config['objects'].items():
        mask, report = model.segment(rgb, item['positive'], item.get('negative', []))
        masks[name] = mask
        result['objects'][name] = report
    # Later entries are foreground when two objects touch in the image. Only
    # tiny boundary disagreements are resolved automatically; larger ones need
    # better prompts to avoid assigning a real surface to the wrong object.
    names = list(masks)
    for i, earlier in enumerate(names):
        for later in names[i+1:]:
            overlap = masks[earlier] & masks[later]
            count = int(overlap.sum())
            if count > .03*min(masks[earlier].sum(), masks[later].sum()):
                raise ValueError(f'{earlier} and {later} overlap by {count} pixels; refine the prompts')
            masks[earlier] &= ~overlap
            result['objects'][earlier].setdefault('foreground_overlap_removed', {})[later] = count
    for name, mask in masks.items():
        valid = mask & np.isfinite(depth) & (depth > .2) & (depth < 2.)
        if valid.sum() < 100:
            raise ValueError(f'{name} has too few valid depth pixels')
        Image.fromarray(mask.astype(np.uint8)*255).save(out/f'{name}_mask.png')
        preview(rgb, mask).save(out/f'{name}_preview.png')
        result['objects'][name].update(mask_pixels=int(mask.sum()),
                                        valid_depth_pixels=int(valid.sum()),
                                        valid_depth_fraction=float(valid.sum()/mask.sum()))
        print(name, json.dumps(result['objects'][name], ensure_ascii=False), flush=True)
    (out/'segmentation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
