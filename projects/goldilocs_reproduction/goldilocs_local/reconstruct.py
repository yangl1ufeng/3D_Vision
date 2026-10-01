"""Thin adapter around official MASt3R sparse global alignment."""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def reconstruct(paths, checkpoint, outdir, iterations=500, image_size=512):
    import torch
    sys.path.insert(0, str(ROOT/'third_party/mast3r'))
    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [str(Path(p).resolve()) for p in paths]
    signature = {'input_sha256': [sha256(p) for p in paths], 'image_size': image_size,
                 'desc_conf_effective': 'conf',
                 'iterations': iterations, 'checkpoint_sha256': sha256(checkpoint)}
    scene_path, metadata_path = outdir/'scene.npz', outdir/'geometry.json'
    if scene_path.exists() and metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get('signature') == signature:
            return dict(np.load(scene_path, allow_pickle=False)), old
        raise ValueError('Output cache settings differ. Choose a new output directory.')
    started = time.perf_counter()
    model = AsymmetricMASt3R.from_pretrained(str(checkpoint)).to('cuda').eval()
    imgs = load_images(paths, size=image_size, square_ok=True)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    scene = sparse_global_alignment(paths, pairs, str(outdir/'mast3r_cache'), model,
        lr1=.07, niter1=iterations, lr2=.014, niter2=0, device='cuda',
        opt_depth=False, shared_intrinsics=False, matching_conf_thr=0, desc_conf='conf')
    points, depths, confs = scene.get_dense_pts3d(clean_depth=False)
    result = {}
    for i in range(2):
        rgb = (scene.imgs[i]*255).round().clip(0, 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        result.update({f'rgb_{i}': rgb,
            f'points_{i}': points[i].detach().cpu().numpy().reshape(h, w, 3),
            f'depth_{i}': depths[i].detach().cpu().numpy().reshape(h, w),
            f'confidence_{i}': confs[i].detach().cpu().numpy().reshape(h, w),
            f'K_{i}': scene.intrinsics[i].detach().cpu().numpy(),
            f'pose_{i}': scene.cam2w[i].detach().cpu().numpy()})
    torch.cuda.synchronize()
    transforms = []
    for i, path in enumerate(paths):
        with Image.open(path) as original:
            ow, oh = ImageOps.exif_transpose(original).size
        rw, rh = [round(v*image_size/max(ow, oh)) for v in (ow, oh)]
        h, w = result[f'rgb_{i}'].shape[:2]
        left, top = rw//2-w//2, rh//2-h//2
        transforms.append({'original_wh': [ow, oh], 'resized_wh': [rw, rh],
                           'crop_left_top': [left, top], 'model_wh': [w, h]})
    metadata = {'signature': signature, 'paths': paths, 'seconds': time.perf_counter()-started,
                'optim_level': 'coarse', 'niter2_effective': 0, 'square_ok': True,
                'image_transforms': transforms,
                'desc_conf_note': 'Paper says 3d; upstream accepts dictionary keys conf or desc_conf. Using 3D confidence key conf (assumption).',
                'notes': 'coarse mode in official demo sets niter2=0; model-grid output, not original resolution'}
    np.savez_compressed(scene_path, **result)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    del model, scene, points, depths, confs
    gc.collect()
    torch.cuda.empty_cache()
    return result, metadata
