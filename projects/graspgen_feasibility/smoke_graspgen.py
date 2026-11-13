"""One real GraspGen inference, with verified official weights; no robot execution."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'third_party/GraspGen'))
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, help='Calibrated scene with explicit provided target mask')
    parser.add_argument('--candidates', type=int, default=16)
    parser.add_argument('--points', type=Path, help='Saved measured Nx3 points in virtual robot frame, metres')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--provenance', type=Path, help='Exported target input_provenance.json')
    args = parser.parse_args()
    if not 1 <= args.candidates <= 200:
        parser.error('candidates must be 1..200')
    if args.points and args.scene:
        parser.error('Use only one of --points and --scene')
    out = args.output or ROOT/'runs/graspgen_restore'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    report = {'passed': False, 'robot_execution': False, 'seed': 1000}
    try:
        entries = json.loads((ROOT/'data/model_files.json').read_text())
        hashes = {}
        for e in entries:
            if e['path'].startswith('checkpoints/graspgen_robotiq_2f_140') and e['path'].endswith('.pth'):
                path = ROOT/e['path']
                actual = digest(path)
                if actual != e['lfs']['oid']:
                    raise RuntimeError(f'Weight hash mismatch: {path.name}')
                hashes[path.name] = actual
        report['weight_sha256'] = hashes
        if len(hashes) != 2:
            raise ValueError('Expected both official 2F-140 weight hashes in model_files.json')
        import numpy as np
        import torch
        from PIL import Image
        from offline_core.geometry import valid_pose_mask
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        if args.points:
            points = np.load(args.points, allow_pickle=False)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError('Expected Nx3 points')
            report.update(input=str(args.points.resolve()), input_sha256=digest(args.points),
                          segmentation_source='measured target points cropped from registered RGB-D PLY',
                          output_frame='virtual robot base, metres')
            if args.provenance:
                provenance = json.loads(args.provenance.read_text(encoding='utf-8'))
                if provenance['input_points_sha256'] != digest(args.points):
                    raise ValueError('Input points do not match provenance')
                report.update(scene_fingerprint=provenance['scene_fingerprint'], target=provenance['target'],
                              input_provenance=provenance)
                report['segmentation_source'] = provenance['segmentation']
        elif args.scene:
            from offline_core.scene_io import load_scene
            scene = load_scene(args.scene)
            if not scene.target.get('mask_path'):
                raise ValueError('This smoke test requires an explicit mask; it does not run SAM2')
            mask = np.asarray(Image.open(scene.target['mask_path'])) > 0
            points = scene.xyz_world_m[scene.valid & mask]
            report.update(input=str(args.scene.resolve()), input_sha256=digest(args.scene),
                          input_asset_sha256=scene.metadata['asset_sha256'],
                          input_provenance=scene.metadata['input_manifest'].get('provenance', {}),
                          segmentation_source='provided target mask; not SAM2',
                          output_frame=scene.metadata['input_manifest']['frames']['world'])
        else:
            source = ROOT/'data/sample_data/real_scene_pc/1745619598_467125.json'
            data = json.loads(source.read_text())
            points = np.asarray(data['object_info']['pc'], dtype=np.float32)
            report.update(input=str(source), input_sha256=digest(source),
                          segmentation_source='official supplied object point cloud; not SAM2',
                          output_frame='official input point-cloud frame')
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 100:
            raise ValueError('Insufficient points')
        if len(points) > 8192:
            points = points[np.random.default_rng(1000).choice(len(points), 8192, replace=False)]
        torch.manual_seed(1000); torch.cuda.manual_seed_all(1000); np.random.seed(1000)
        print('Loading verified generator and discriminator', flush=True)
        sampler = GraspGenSampler(load_grasp_cfg(str(ROOT/'checkpoints/graspgen_robotiq_2f_140.yml')))
        pc = torch.tensor(points, dtype=torch.float32, device='cuda')
        torch.cuda.synchronize(); began = time.perf_counter()
        with torch.inference_mode():
            poses, scores = sampler.sample(pc, threshold=0., num_grasps=args.candidates, remove_outliers=False)[:2]
            # Same normalization as the upstream public inference wrapper.
            poses[:, 3, 3] = 1
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-began
        poses, scores = poses.cpu().numpy(), scores.cpu().numpy()
        np.savez_compressed(out/'candidates.npz', poses_world_m=poses, scores=scores, input_points_m=points)
        report['candidate_sha256'] = digest(out/'candidates.npz')
        passed = (len(poses) == args.candidates and bool(valid_pose_mask(poses).all())
                  and bool(np.isfinite(scores).all()) and bool(((scores>=0)&(scores<=1)).all()))
        report.update(passed=passed, point_count=len(points), candidate_count=len(poses),
                      grasp_generator_used=True, output_unit='m',
                      pose_convention='T_inputframe_graspgen_gripper; column vectors',
                      inference_seconds=inference_seconds, torch=torch.__version__,
                      gpu=torch.cuda.get_device_name(0), above_08=int((scores>=.8).sum()),
                      min_score=float(scores.min()), max_score=float(scores.max()),
                      note='Scores are model rankings, not grasp success; TCP mapping unverified.')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        import traceback
        traceback.print_exc()
    (out/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(out)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
