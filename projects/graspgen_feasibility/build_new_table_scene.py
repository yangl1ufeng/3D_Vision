"""Fit a tabletop and simple collision proxies from named RGB-D object masks.

Measured visible object points remain the visual surface; proxies are explicit
approximations for simulation, not claims about unseen geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fit_static_scene import fit_plane, points_in_roi
from simulation.scene_spec import SceneSpec

PLY_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('view', 'u1')])


def measured_camera_points(depth, intrinsic, mask):
    valid = mask & np.isfinite(depth) & (depth > .2) & (depth < 2.)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    xyz = np.column_stack(((u-intrinsic['cx'])*z/intrinsic['fx'],
                           (v-intrinsic['cy'])*z/intrinsic['fy'], z))
    return xyz, u, v


def fit_object(xyz, shape, label):
    height = xyz[:, 2]
    top = float(np.percentile(height, 98))
    if top < .015:
        raise ValueError(f'Object {label} has too little height above the table')
    if shape == 'solid_cylinder_proxy':
        upper = xyz[height > np.percentile(height, 80)]
        lower_xy, upper_xy = np.percentile(upper[:, :2], [2, 98], axis=0)
        center_xy = (lower_xy + upper_xy)/2
        radius = float(max(upper_xy-lower_xy)/2 + .003)
        if not .015 < radius < .10:
            raise ValueError(f'Object {label} cylinder radius is implausible: {radius:.3f} m')
        return {'shape': shape, 'center_table_m': [*center_xy.tolist(), top/2],
                'radius_m': radius, 'height_m': top, 'visual_label': label,
                'grasp_points_proxy_crop': True,
                'collision_note': 'Solid cylinder body; hollow interior and handle omitted.'}
    if shape != 'box':
        raise ValueError(f'Unsupported proxy shape: {shape}')
    footprint = xyz[height > .005, :2]
    lower_xy, upper_xy = np.percentile(footprint, [1, 99], axis=0)
    footprint = footprint[np.all((footprint >= lower_xy) & (footprint <= upper_xy), axis=1)].astype(np.float32)
    (cx, cy), (a, b), angle = cv2.minAreaRect(footprint)
    if min(a, b) < .01:
        raise ValueError(f'Object {label} footprint is too small')
    if b > a:
        a, b = b, a
        angle += 90
    return {'shape': 'box', 'center_table_m': [float(cx), float(cy), top/2],
            'dimensions_m': [float(a+.006), float(b+.006), top],
            'yaw_table_rad': float(np.deg2rad(angle)), 'visual_label': label,
            'collision_note': 'Oriented bounding box around measured visible surfaces; unseen back approximated.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('segmentation', type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    capture = (config_path.parent/config['capture']).resolve()
    output = args.segmentation.resolve()
    segmentation = json.loads((output/'segmentation.json').read_text(encoding='utf-8'))
    if Path(segmentation['capture']).resolve() != capture:
        raise ValueError('Segmentation belongs to a different capture')
    rgb = np.array(Image.open(capture/'rgb.png').convert('RGB'))
    depth = np.load(capture/'depth_m.npy', allow_pickle=False)
    report = json.loads((capture/'report.json').read_text(encoding='utf-8'))
    intrinsic = report['aligned_depth_intrinsic']
    table_parts = [points_in_roi(depth, intrinsic, roi) for roi in config['table_rois']]
    table_points = np.concatenate([part[0] for part in table_parts])
    normal, plane_offset, inliers, residual = fit_plane(table_points)
    if residual > .01:
        raise ValueError(f'Table fit p95 residual too large: {residual:.4f} m')
    x_axis = np.array([1., 0., 0.])
    x_axis -= normal*np.dot(x_axis, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    rotation = np.column_stack((x_axis, y_axis, normal))
    observations = {}
    for name, item in config['objects'].items():
        mask = np.array(Image.open(output/f'{name}_mask.png').convert('L')) > 0
        if mask.shape != depth.shape:
            raise ValueError(f'{name} mask dimensions do not match depth')
        camera, u, v = measured_camera_points(depth, intrinsic, mask)
        observations[name] = (camera, u, v)
    # Centre the local table frame among the selected objects, making the virtual
    # placement independent of their position in the camera image.
    centers = np.array([np.median(points[0], axis=0) for points in observations.values()])
    origin = centers.mean(axis=0)
    origin -= normal*(np.dot(origin, normal)+plane_offset)
    T_camera_table = np.eye(4)
    T_camera_table[:3, :3] = rotation
    T_camera_table[:3, 3] = origin
    geometry = {'registration': str(output/'registration.json'),
                'segmentation_path': str(output/'segmentation.json'),
                'segmentation_sha256': hashlib.sha256((output/'segmentation.json').read_bytes()).hexdigest(),
                'frame': 'single-view fitted tabletop, metres, z up',
                'table': {'shape': 'plane', 'z_m': 0.0}, 'objects': {},
                'limitations': 'Only visible depth surfaces measured. Collision proxies approximate hidden geometry.'}
    chunks = []
    per_object = {}
    for label, (name, item) in enumerate(config['objects'].items(), 1):
        camera, u, v = observations[name]
        table_xyz = (camera-origin) @ rotation
        above = (table_xyz[:, 2] > .003) & (table_xyz[:, 2] < .5)
        table_xyz, u, v = table_xyz[above], u[above], v[above]
        if len(table_xyz) < 100:
            raise ValueError(f'{name} has too few points above the table')
        # A 2 mm voxel only deduplicates observations; the output remains measured.
        _, unique = np.unique(np.floor(table_xyz/.002).astype(np.int32), axis=0, return_index=True)
        table_xyz, u, v = table_xyz[unique], u[unique], v[unique]
        proxy = fit_object(table_xyz, item['shape'], label)
        proxy['mask_path'] = str(output/f'{name}_mask.png')
        proxy['mask_sha256'] = hashlib.sha256((output/f'{name}_mask.png').read_bytes()).hexdigest()
        proxy['measured_surface_points'] = int(len(table_xyz))
        geometry['objects'][name] = proxy
        cloud = np.empty(len(table_xyz), dtype=PLY_DTYPE)
        for index, axis in enumerate(('x','y','z')):
            cloud[axis] = table_xyz[:, index]
        for index, color in enumerate(('red','green','blue')):
            cloud[color] = rgb[v, u, index]
        cloud['view'] = label
        chunks.append(cloud)
        per_object[name] = {'points': len(table_xyz), 'height_m': float(proxy.get('height_m', proxy.get('dimensions_m', [0,0,0])[2])),
                            'proxy': proxy}
    data = np.concatenate(chunks)
    ply_path = output/'three_objects_measured.ply'
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment Three selected objects; actual visible Gemini 2 samples in table frame\n'
              f'element vertex {len(data)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'property uchar view\nend_header\n')
    with ply_path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(data.tobytes())
    (output/'geometry.json').write_text(json.dumps(geometry, indent=2, ensure_ascii=False), encoding='utf-8')
    registration = {'passed': True, 'views': [str(capture)],
                    'T_view_camera_table': [T_camera_table.tolist()],
                    'T_reference_table_view_table': [np.eye(4).tolist()],
                    'scope': 'single fixed-camera view, fitted tabletop, no robot extrinsic calibration'}
    (output/'registration.json').write_text(json.dumps(registration, indent=2), encoding='utf-8')
    scene = {'version': 1, 'scene_id': 'new_desk_20261113',
             'geometry_path': str(output/'geometry.json'),
             'pointcloud_path': str(ply_path), 'target': 'mug',
             'table_to_robot_translation_m': [.55, 0, 0],
             'drop_xy_robot_m': [.50, .31],
             'table': {'center_robot_m': [.55, 0, -.04], 'half_extents_m': [.45, .45, .04]},
             'pedestal': {'center_robot_m': [0, 0, -.04], 'half_extents_m': [.14, .16, .04]},
             'target_physics': {'mass_kg': .09, 'lateral_friction': 1.2}}
    SceneSpec(scene).save(output/'scene.json')
    result = {'capture': str(capture), 'table_fit_inlier_fraction': float(inliers.mean()),
              'table_fit_p95_error_m': residual,
              'table_roi_valid_fractions': [x[1] for x in table_parts],
              'table_plane_camera': {'normal': normal.tolist(), 'offset_m': float(plane_offset)},
              'T_camera_table': T_camera_table.tolist(), 'objects': per_object,
              'ply_sha256': hashlib.sha256(ply_path.read_bytes()).hexdigest(),
              'scope': 'Visible measured RGB-D points, single-view table fit, simple collision proxies; virtual robot placement only.'}
    (output/'scene_fit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(output/'scene.json')


if __name__ == '__main__':
    main()
