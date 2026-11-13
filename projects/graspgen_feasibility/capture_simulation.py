"""Export calibrated simulated RGB-D; instance masks are explicitly oracle data."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pybullet as p

from simulation.pick_place import Demo, PROJECT
from offline_core.geometry import depth_to_xyz, transform_points


def capture(demo, output):
    width, height = 640, 480
    near, far = .05, 3.0
    view = p.computeViewMatrix([.9, -.65, .8], [.5, .08, .05], [0, 0, 1])
    projection = p.computeProjectionMatrixFOV(55, width/height, near, far)
    _, _, rgba, buffer, labels = p.getCameraImage(width, height, view, projection,
                                                 renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
    buffer = np.asarray(buffer).reshape(height, width)
    labels = np.asarray(labels).reshape(height, width)
    valid = (buffer < 1-1e-6) & (labels >= 0)
    depth = far*near/(far-(far-near)*buffer)
    depth = np.where(valid, depth, 0).astype(np.float32)
    matrix = np.asarray(projection).reshape(4, 4, order='F')
    K = np.array([[matrix[0, 0]*width/2, 0, (width-1)/2],
                  [0, matrix[1, 1]*height/2, (height-1)/2], [0, 0, 1]])
    # OpenGL camera: right/up/back; OpenCV: right/down/forward.
    transform = np.linalg.inv(np.asarray(view).reshape(4, 4, order='F')) @ np.diag([1, -1, -1, 1])
    ids = labels & ((1 << 24)-1)
    target = valid & (ids == demo.cube)
    if target.sum() < 100:
        raise RuntimeError('Target has insufficient rendered visibility')
    Image.fromarray(rgb).save(output/'rgb.png')
    Image.fromarray(target.astype(np.uint8)*255).save(output/'target_mask.png')
    np.save(output/'depth_m.npy', depth)
    np.save(output/'valid_mask.npy', valid.astype(np.uint8))
    manifest = {'schema_version': 1, 'scene_id': output.name, 'rgb': 'rgb.png',
                'geometry': {'kind': 'depth', 'path': 'depth_m.npy', 'unit': 'm',
                             'aligned_to_rgb': True, 'depth_convention': 'z', 'pinhole_rectified': True,
                             'intrinsics': K.tolist(), 'valid_mask': 'valid_mask.npy'},
                'frames': {'world': 'simulation_world', 'camera_convention': 'opencv',
                           'T_world_camera': transform.tolist()},
                'target': {'id': 'cube', 'mask': 'target_mask.png'},
                'provenance': {'sensor': 'PyBullet TinyRenderer', 'mask_source': 'simulator_instance_id_oracle',
                               'segmentation_model_used': False, 'grasp_generator_used': False}}
    (output/'scene.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    xyz = transform_points(depth_to_xyz(depth, K), transform)
    # Independent known-geometry checks: table plane and cube surface, not a
    # mere projection/unprojection round trip.
    table = valid & (ids == demo.table)
    lo, hi = map(np.asarray, p.getAABB(demo.cube))
    points = xyz[target]
    distance = np.min(np.concatenate([abs(points-lo), abs(points-hi)], axis=1), axis=1)
    outside = np.maximum(np.maximum(lo-points, points-hi), 0).max(axis=1)
    metrics = {'target_pixels': int(target.sum()),
               'table_z_p95_error_m': float(np.percentile(abs(xyz[table, 2]), 95)),
               'target_surface_p95_error_m': float(np.percentile(distance, 95)),
               'target_aabb_p95_outside_m': float(np.percentile(outside, 95)),
               'cube_center_ground_truth_m': p.getBasePositionAndOrientation(demo.cube)[0],
               'note': 'Partial visible surfaces only. Oracle mask is not SAM2 or learned segmentation.'}
    metrics['passed'] = all(metrics[k] < .003 for k in ['table_z_p95_error_m', 'target_surface_p95_error_m', 'target_aabb_p95_outside_m'])
    (output/'capture_validation.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    if not metrics['passed']:
        raise RuntimeError('RGB-D calibration validation failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cube-xy', nargs=2, type=float, default=[.5, 0])
    args = parser.parse_args()
    xy = np.asarray(args.cube_xy)
    if not np.isfinite(xy).all() or not (.3 <= xy[0] <= .75 and abs(xy[1]) <= .3):
        parser.error('cube-xy must be within the table: x=.3.. .75, y=-.3.. .3')
    output = PROJECT/'runs/virtual_camera'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    try:
        demo = Demo(output, robot_model='ur5_robotiq140')
        # Scene construction only; no object teleportation during a grasp task.
        p.resetBasePositionAndOrientation(demo.cube, [*xy, .025], [0, 0, 0, 1])
        demo.tick(240)
        capture(demo, output)
    finally:
        if p.isConnected():
            p.disconnect()
    print(output)


if __name__ == '__main__':
    main()
