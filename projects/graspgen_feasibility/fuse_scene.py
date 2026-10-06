"""Validate/refine calibrated static-scene views and save gated multi-view fusion."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from offline_core.config import ROOT
from offline_core.scene_io import load_scene, file_hash
from offline_core.segmentation import provided_segmentation
from offline_core.fusion import fuse_views
from offline_core.export import write_json


def seal_artifacts(output):
    names = ['fusion_quality.json', 'fusion_diagnostic.npz', 'fusion_usable.npz',
             'accepted_depth_views.npz', 'input_provenance.json']
    write_json(output/'artifact_sha256.json', {name: file_hash(output/name) for name in names if (output/name).exists()})


def save_result(output, result):
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('fusion output must be empty; a rejected run cannot leave stale usable artifacts')
    output.mkdir(parents=True, exist_ok=True)
    metadata = {k: v for k, v in result.items() if k not in ['corrected_views', 'usable_views', 'usable_cloud', 'diagnostic_cloud']}
    metadata['corrected_poses'] = {v.view_id: v.T_world_camera for v in result['corrected_views']}
    metadata['planning_allowed'] = result['accepted']
    write_json(output/'fusion_quality.json', metadata)
    np.savez_compressed(output/'fusion_diagnostic.npz', **result['diagnostic_cloud'])
    if result['accepted']:
        np.savez_compressed(output/'fusion_usable.npz', **result['usable_cloud'])
        # Preserve the accepted depth evidence, not just a fused point surface.
        evidence = {}
        for i, v in enumerate(result['usable_views']):
            evidence[f'depth_{i}_m'] = v.depth_m
            evidence[f'intrinsics_{i}'] = v.intrinsics
            evidence[f'T_world_camera_{i}'] = v.T_world_camera
            evidence[f'view_id_{i}'] = np.asarray(v.view_id)
        np.savez_compressed(output/'accepted_depth_views.npz', **evidence)
    seal_artifacts(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Bundle JSON with a views list of single-view scene.json paths')
    parser.add_argument('--refine', action='store_true', help='Refine supplied camera poses using constrained point-to-plane ICP')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT/'runs/fusion'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    try:
        bundle = json.loads(args.input.read_text(encoding='utf-8'))
        scenes = [load_scene(args.input.resolve().parent / p) for p in bundle['views']]
        if len({s.target_id for s in scenes}) != 1 or len({s.metadata['world_frame'] for s in scenes}) != 1:
            raise ValueError('all views must identify the same target and world frame')
        if any(s.depth_view is None for s in scenes):
            raise ValueError('fusion requires calibrated depth manifests')
        if any(not s.target.get('mask_path') for s in scenes):
            raise ValueError('fusion CLI requires a confirmed target mask for each view')
        result = fuse_views([s.depth_view for s in scenes], [provided_segmentation(s).mask for s in scenes], args.refine)
        save_result(output, result)
        write_json(output/'input_provenance.json', [s.metadata for s in scenes])
        seal_artifacts(output)
        print(json.dumps({'accepted': result['accepted'], 'reasons': result['reasons'], 'output': str(output)}, ensure_ascii=False))
        return 0 if result['accepted'] else 3
    except Exception as exc:
        import traceback
        write_json(output/'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
