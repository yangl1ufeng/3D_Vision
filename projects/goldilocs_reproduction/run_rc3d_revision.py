"""Frozen, resumable RC-3D revision using verified original geometry caches."""
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps
from goldilocs_local.reconstruct import sha256
from goldilocs_local.segmentation import SegmentTracker
from goldilocs_local.rc3d import predict_rc3d_pair
from scripts.evaluate_rc3d_revision import evaluate

ROOT = Path(__file__).resolve().parent


def completed_sample_matches(folder, checkpoint_hash, input_hashes):
    """Never silently reuse mixed-model or incomplete results on resume."""
    marker = folder/'run.json'
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record.get('status') != 'complete':
        raise ValueError(f'Invalid completion marker: {folder}')
    if record.get('sam_checkpoint_sha256') != checkpoint_hash:
        raise ValueError(f'Completed sample checkpoint mismatch: {folder}')
    if record.get('input_sha256') != input_hashes:
        raise ValueError(f'Completed sample input mismatch: {folder}')
    for view in ('view_0', 'view_1'):
        for name in ('prediction.json', 'labels.png', 'object_masks.npz'):
            if not (folder/view/name).is_file():
                raise ValueError(f'Incomplete completed sample: {folder/view/name}')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/rc3d_paper_main.json')
    p.add_argument('--ids', nargs='+')
    p.add_argument('--paired-cache',type=Path,help='Verified fused proposals from a completed two-view run')
    args = p.parse_args()
    settings = json.loads(args.config.read_text())
    pairs = json.loads((ROOT/'data/rc3d_author_order_manifest.json').read_text())['pairs']
    wanted = args.ids or [r['id'] for r in pairs]
    if len(set(wanted)) != len(wanted) or not set(wanted) <= {r['id'] for r in pairs}:
        raise ValueError('Invalid sample IDs')
    pairs = [r for r in pairs if r['id'] in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT/'checkpoints/sam2_hiera_large.pt'
    checkpoint_hash = sha256(checkpoint)
    mast3r_hash = sha256(ROOT/'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
    sources = list((ROOT/'goldilocs_local').glob('*.py')) + [Path(__file__),ROOT/'scripts/evaluate_rc3d_revision.py']
    plan = {'pair_ids': [r['id'] for r in pairs], 'settings': settings,
        'manifest_sha256': sha256(ROOT/'data/rc3d_author_order_manifest.json'),
        'source_sha256': {str(f.relative_to(ROOT)): sha256(f) for f in sources},
        'checkpoint_sha256': {'sam2': checkpoint_hash, 'mast3r': mast3r_hash},
        'geometry_source': 'runs/rc3d_native', 'ground_truth_used_in_inference': False,
        'paired_cache':str(args.paired_cache.resolve()) if args.paired_cache else None}
    plan_path = args.out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen plan changed; use a new run directory')
    plan_path.write_text(json.dumps(plan,indent=2), encoding='utf-8')
    import torch
    torch.manual_seed(0)
    tracker = None
    status = {'status':'running', 'completed':[], 'requested':len(pairs)}
    def save():
        status['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        (args.out/'progress.json').write_text(json.dumps(status,indent=2), encoding='utf-8')
    try:
        for pair in pairs:
            if any(sha256(f) != plan['source_sha256'][str(f.relative_to(ROOT))] for f in sources):
                raise RuntimeError('Source changed during frozen run')
            folder = args.out/pair['id']
            status['current_pair'] = pair['id']
            save()
            before, after = ROOT/pair['before'], ROOT/pair['after']
            hashes = [sha256(before),sha256(after)]
            if hashes != [pair['before_sha256'],pair['after_sha256']]:
                raise ValueError('Input data hash mismatch')
            if not completed_sample_matches(folder, checkpoint_hash, hashes):
                geo = ROOT/'runs/rc3d_native'/pair['id']/'geometry'
                meta = json.loads((geo/'geometry.json').read_text())
                sig = meta['signature']
                if sig['input_sha256'] != hashes or sig['checkpoint_sha256'] != mast3r_hash:
                    raise ValueError('Geometry cache input/checkpoint mismatch')
                if sig['iterations'] != settings['mast3r_iterations'] or sig['image_size'] != settings['mast3r_image_size']:
                    raise ValueError('Geometry cache config mismatch')
                with np.load(geo/'scene.npz') as z:
                    scene = dict(z)
                native = [np.array(ImageOps.exif_transpose(Image.open(f)).convert('RGB')) for f in (before,after)]
                if tracker is None:
                    tracker = SegmentTracker(checkpoint, settings['sam_config'], settings)
                started = time.perf_counter()
                print('START '+pair['id'],flush=True)
                old_run = json.loads((geo.parent/'run.json').read_text())
                proposal_cache = geo.parent/'view_1' if old_run['sam_checkpoint_sha256']==checkpoint_hash else None
                paired_cache = args.paired_cache/pair['id'] if args.paired_cache else None
                if paired_cache:
                    paired_meta = json.loads((paired_cache/'run.json').read_text())
                    if paired_meta['sam_checkpoint_sha256']!=checkpoint_hash or paired_meta['geometry_sha256']!=sha256(geo/'scene.npz'):
                        raise ValueError('Paired cache model/geometry mismatch')
                report = predict_rc3d_pair(scene,meta,native,tracker,settings,folder,
                    proposal_cache=proposal_cache,paired_cache=paired_cache)
                (folder/'run.json').write_text(json.dumps({'status':'complete',
                    'input_sha256':hashes,
                    'geometry_source':str(geo),'geometry_sha256':sha256(geo/'scene.npz'),
                    'sam_checkpoint_sha256':checkpoint_hash,'seconds':time.perf_counter()-started,
                    'objects':len(report['objects'])},indent=2), encoding='utf-8')
                del scene, native
            status['completed'].append(pair['id'])
            save()
            print(f"DONE {len(status['completed'])}/{len(pairs)} {pair['id']}",flush=True)
        result = evaluate(args.out,plan['pair_ids'])
        (args.out/'metrics.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
        status.update(status='complete',current_pair=None,metrics=result['metrics'])
    except Exception as exc:
        status.update(status='failed',error=str(exc))
        raise
    finally:
        save()
        if tracker is not None:
            tracker.close()


if __name__ == '__main__':
    main()
