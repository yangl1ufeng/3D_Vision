"""Run a pairwise independent reproduction and save all intermediate images."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

from goldilocs_local.reconstruct import reconstruct, sha256
from goldilocs_local.pipeline import predict_view
from goldilocs_local.segmentation import SegmentTracker
from goldilocs_local.geometry import original_resolution_intrinsic

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/pairwise_prototype.json')
    p.add_argument('--geometry-only', action='store_true')
    p.add_argument('--bidirectional', action='store_true')
    args = p.parse_args()
    settings = json.loads(args.config.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'run.json').exists():
        raise FileExistsError('Completed output already exists; use a new directory.')
    (args.out/'effective_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    # Upstream local caches contain trusted numpy objects. Scope compatibility
    # flag to this process; never use this runner with untrusted checkpoints.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This runner requires a working CUDA device')
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checkpoint = ROOT/'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    sam_checkpoint = ROOT/'checkpoints/sam2_hiera_large.pt'
    scene, metadata = reconstruct([args.before, args.after], checkpoint, args.out/'geometry',
        settings['mast3r_iterations'], settings['mast3r_image_size'])
    reports = []
    native_images = None
    if settings.get('detection_resolution') == 'original':
        native_images = [np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
                         for path in [args.before, args.after]]
    if not args.geometry_only:
        tracker = SegmentTracker(sam_checkpoint, settings['sam_config'], settings)
        try:
            for target in ([1, 0] if args.bidirectional else [1]):
                render_k = original_resolution_intrinsic(scene[f'K_{target}'], metadata['image_transforms'][target]) if native_images is not None else None
                reports.append(predict_view(scene, target, tracker, settings, args.out/f'view_{target}',
                    native_images=native_images, render_intrinsic=render_k))
        finally:
            tracker.close()
    report = {'status': 'geometry_complete' if args.geometry_only else 'pairwise_inference_complete',
        'benchmark_evaluation': False, 'reproduction_claim': 'independent implementation, unvalidated',
        'seconds': time.perf_counter()-started, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'geometry': metadata, 'sam_checkpoint_sha256': sha256(sam_checkpoint) if reports else None,
        'prediction_views': len(reports)}
    (args.out/'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
