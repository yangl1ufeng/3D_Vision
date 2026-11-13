"""Export selected object points in the configured virtual robot frame."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from simulation.measured_visuals import MeasuredVisuals
from simulation.scene_spec import DEFAULT_SCENE, load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--target')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = load_scene(args.scene, args.target)
    ply = Path(spec.data['pointcloud_path'])
    visual = MeasuredVisuals(ply, spec.geometry, spec.offset, spec.target)
    points = visual.target_grasp_world_xyz
    observation = spec.data.get('target_observation')
    if observation is not None:
        measured = np.load(observation['points_table_path'], allow_pickle=False)
        if measured.ndim != 2 or measured.shape[1] != 3 or not np.isfinite(measured).all():
            raise ValueError('Clicked target points must be finite Nx3 table coordinates')
        points = measured + spec.offset
    if len(points) < 100:
        raise ValueError(f'Target {spec.target} has only {len(points)} measured points')
    path = args.output/'target_robot_points.npy'
    np.save(path, points)
    spec.save(args.output/'scene_snapshot.json')
    report = {'source_ply': str(ply), 'source_ply_sha256': hashlib.sha256(ply.read_bytes()).hexdigest(),
              'geometry_source': spec.data['geometry_path'],
              'scene_fingerprint': spec.fingerprint, 'target': spec.target,
              'scene_snapshot': str((args.output/'scene_snapshot.json').resolve()),
              'input_points_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'points': len(points), 'virtual_table_translation_m': spec.offset.tolist(),
              'frame': 'virtual robot base, metres',
              'segmentation': ('SAM2 point-prompt mask with measured aligned depth and geometry association'
                               if observation else
                               'SAM2 point-prompt masks with measured aligned depth'
                               + ('; model input cropped to collision proxy' if spec.geometry['objects'][spec.target].get('grasp_points_proxy_crop') else '')
                               if spec.geometry.get('segmentation_path') else
                               'geometric crop using selected reconstructed object; not automatic segmentation'),
              'target_observation': observation,
              'unseen_surfaces_completed': False}
    (args.output/'input_provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
