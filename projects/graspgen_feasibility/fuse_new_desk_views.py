"""Fuse the three selected objects from two registered static RGB-D views."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from build_new_table_scene import fit_object, measured_camera_points, PLY_DTYPE
from fit_static_scene import fit_plane, points_in_roi
from simulation.scene_spec import SceneSpec, load_scene


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def load_labeled_ply(path):
    with Path(path).open('rb') as stream:
        lines = []
        while True:
            line = stream.readline().decode('ascii')
            if not line:
                raise ValueError('Incomplete PLY header')
            lines.append(line)
            if line == 'end_header\n':
                break
        header = ''.join(lines)
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError('Expected binary little-endian PLY')
        count = int(next(x.split()[2] for x in lines if x.startswith('element vertex ')))
        data = np.frombuffer(stream.read(), dtype=PLY_DTYPE)
    if len(data) != count:
        raise ValueError('PLY vertex count does not match header')
    xyz = np.column_stack([data[x] for x in ('x', 'y', 'z')]).astype(np.float64)
    rgb = np.column_stack([data[x] for x in ('red', 'green', 'blue')]).astype(np.uint8)
    return xyz, rgb, data['view'].copy()


def write_ply(path, xyz, rgb, labels):
    data = np.empty(len(xyz), dtype=PLY_DTYPE)
    for index, axis in enumerate(('x', 'y', 'z')):
        data[axis] = xyz[:, index]
    for index, color in enumerate(('red', 'green', 'blue')):
        data[color] = rgb[:, index]
    data['view'] = labels
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment Selected real RGB-D points from two views; view field stores object label\n'
              f'element vertex {len(data)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'property uchar view\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(data.tobytes())


def voxel_compound_proxy(xyz, label):
    """Represent a complex object by occupied 3 cm cells, not one filled box."""
    height = float(np.percentile(xyz[:, 2], 98))
    core = xyz[(xyz[:, 2] > .02) & (xyz[:, 2] < .08)]
    center_xy = np.median(core[:, :2], axis=0)
    center = np.array([*center_xy, height/2])
    voxel = np.floor(xyz/.03).astype(np.int32)
    cells, inverse, counts = np.unique(voxel, axis=0, return_inverse=True, return_counts=True)
    parts = []
    for index, count in enumerate(counts):
        if count < 5:
            continue
        observed_center = np.median(xyz[inverse == index], axis=0)
        parts.append({'center_local_m': (observed_center-center).tolist(),
                      'half_extents_m': [.012, .012, .012], 'measured_points': int(count)})
    if not 8 <= len(parts) <= 80:
        raise ValueError(f'Implausible compound collision part count: {len(parts)}')
    low, high = np.percentile(xyz, [1, 99], axis=0)
    return {'shape': 'voxel_compound_proxy', 'center_table_m': center.tolist(),
            'dimensions_m': (high-low+.006).tolist(), 'height_m': height,
            'parts': parts, 'visual_label': label,
            'collision_note': 'Occupied measured cells; each collision box is 24 mm across. Gaps remain empty.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first_scene', type=Path)
    parser.add_argument('second_config', type=Path)
    parser.add_argument('second_segmentation', type=Path)
    parser.add_argument('feature_registration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    first_scene = load_scene(args.first_scene)
    first_registration = json.loads(Path(first_scene.geometry['registration']).read_text(encoding='utf-8'))
    first_capture = Path(first_registration['views'][0]).resolve()
    second_config = json.loads(args.second_config.read_text(encoding='utf-8'))
    second_capture = (args.second_config.resolve().parent / second_config['capture']).resolve()
    second_seg = args.second_segmentation.resolve()
    seg_report = json.loads((second_seg/'segmentation.json').read_text(encoding='utf-8'))
    if Path(seg_report['capture']).resolve() != second_capture:
        raise ValueError('Second-view segmentation belongs to another capture')
    feature_path = args.feature_registration.resolve()
    feature = json.loads(feature_path.read_text(encoding='utf-8'))
    if [Path(p).resolve() for p in feature['views']] != [first_capture, second_capture]:
        raise ValueError('Feature registration belongs to another view pair')
    match = feature['pair_quality'][0]
    if match['inliers_25mm'] < 30 or match['inlier_p95_m'] > .025:
        raise ValueError('Too few reliable RGB-D feature matches')
    T_first_camera_table = np.asarray(first_registration['T_view_camera_table'][0], float)
    T_first_camera_second_camera = np.asarray(feature['T_reference_camera'][1], float)
    T_table_second_camera = np.linalg.inv(T_first_camera_table) @ T_first_camera_second_camera
    first_xyz, first_rgb, first_labels = load_labeled_ply(first_scene.data['pointcloud_path'])
    rgb_image = np.asarray(Image.open(second_capture/'rgb.png').convert('RGB'))
    depth = np.load(second_capture/'depth_m.npy', allow_pickle=False)
    camera_report = json.loads((second_capture/'report.json').read_text(encoding='utf-8'))
    intrinsic = camera_report['aligned_depth_intrinsic']
    table_parts = [points_in_roi(depth, intrinsic, roi) for roi in second_config['table_rois']]
    second_table_camera = np.concatenate([part[0] for part in table_parts])
    normal, plane_offset, inliers, table_fit_error = fit_plane(second_table_camera)
    second_table_reference = transform(second_table_camera[inliers], T_table_second_camera)
    table_z_error = float(np.percentile(np.abs(second_table_reference[:, 2]), 95))
    reference_normal = T_first_camera_table[:3, 2]
    moved_normal = T_first_camera_second_camera[:3, :3] @ normal
    table_normal_angle = float(np.degrees(np.arccos(np.clip(np.dot(reference_normal, moved_normal), -1, 1))))
    if table_fit_error > .01 or table_z_error > .025 or table_normal_angle > 8:
        raise ValueError(f'Table alignment failed: fit={table_fit_error:.3f} m, '
                         f'height={table_z_error:.3f} m, angle={table_normal_angle:.1f} deg')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    all_xyz, all_rgb, all_labels = [], [], []
    geometry = dict(first_scene.geometry)
    geometry['objects'] = {name: dict(item) for name, item in first_scene.geometry['objects'].items()}
    geometry['registration'] = str(output/'registration.json')
    geometry['segmentation_views'] = [
        {'path': first_scene.geometry.get('segmentation_path'),
         'sha256': first_scene.geometry.get('segmentation_sha256')},
        {'path': str(second_seg/'segmentation.json'), 'sha256': digest(second_seg/'segmentation.json')}]
    geometry['frame'] = 'first-view fitted tabletop, two RGB-D views, metres, z up'
    geometry['limitations'] = 'Only visible measured surfaces; collision proxies approximate unseen geometry.'
    stats = {}
    for name, item in geometry['objects'].items():
        label = int(item['visual_label'])
        first_part = first_xyz[first_labels == label]
        first_color = first_rgb[first_labels == label]
        mask_path = second_seg/f'{name}_mask.png'
        mask = np.asarray(Image.open(mask_path).convert('L')) > 0
        camera, u, v = measured_camera_points(depth, intrinsic, mask)
        second_part = transform(camera, T_table_second_camera)
        raised = (second_part[:, 2] > .003) & (second_part[:, 2] < .5)
        second_part, u, v = second_part[raised], u[raised], v[raised]
        if len(second_part) < 100:
            raise ValueError(f'{name} has too few measured points above table in second view')
        # Reject depth-edge spill far from the stable first-view object region.
        first_center = np.median(first_part[:, :2], axis=0)
        spread = np.percentile(np.linalg.norm(first_part[:, :2]-first_center, axis=1), 99)
        near = np.linalg.norm(second_part[:, :2]-first_center, axis=1) < max(.07, spread+.05)
        second_part, u, v = second_part[near], u[near], v[near]
        if len(second_part) < 100:
            raise ValueError(f'{name} second-view points do not align with first view')
        second_color = rgb_image[v, u]
        first_sample = first_part[::max(1, len(first_part)//5000)]
        second_sample = second_part[::max(1, len(second_part)//5000)]
        distances = cKDTree(first_sample).query(second_sample, k=1)[0]
        centroid_shift = float(np.linalg.norm(np.median(second_part[:, :2], axis=0)-first_center))
        within_3cm = float((distances < .03).mean())
        if centroid_shift > .05 or within_3cm < .15:
            raise ValueError(f'{name} object alignment failed: center shift {centroid_shift:.3f} m, '
                             f'overlap {within_3cm:.1%}')
        combined = np.concatenate([first_part, second_part])
        combined_rgb = np.concatenate([first_color, second_color])
        _, unique = np.unique(np.floor(combined/.002).astype(np.int32), axis=0, return_index=True)
        combined, combined_rgb = combined[unique], combined_rgb[unique]
        supported = cKDTree(combined).query_ball_point(combined, .008, return_length=True) >= 3
        isolated_removed = int((~supported).sum())
        combined, combined_rgb = combined[supported], combined_rgb[supported]
        ambiguous_removed = 0
        if second_config['objects'][name].get('collision_proxy') == 'voxel_compound':
            ambiguous = np.zeros(len(combined), dtype=bool)
            for other_name, other in geometry['objects'].items():
                if other_name == name or other['shape'] != 'solid_cylinder_proxy':
                    continue
                delta = combined - np.asarray(other['center_table_m'])
                ambiguous |= ((np.linalg.norm(delta[:, :2], axis=1) < other['radius_m']-.002)
                              & (np.abs(delta[:, 2]) < other['height_m']/2))
            ambiguous_removed = int(ambiguous.sum())
            np.save(output/f'{name}_ambiguous_overlap_points.npy', combined[ambiguous])
            combined, combined_rgb = combined[~ambiguous], combined_rgb[~ambiguous]
        proxy = (voxel_compound_proxy(combined, label)
                 if second_config['objects'][name].get('collision_proxy') == 'voxel_compound'
                 else fit_object(combined, item['shape'], label))
        proxy['measured_surface_points'] = len(combined)
        proxy['mask_paths'] = [item.get('mask_path'), str(mask_path)]
        proxy['mask_sha256s'] = [item.get('mask_sha256'), digest(mask_path)]
        geometry['objects'][name] = proxy
        all_xyz.append(combined)
        all_rgb.append(combined_rgb)
        all_labels.append(np.full(len(combined), label, np.uint8))
        stats[name] = {'first_points': len(first_part), 'second_points': len(second_part),
                       'fused_points': len(combined), 'second_to_first_surface_p50_m': float(np.median(distances)),
                       'isolated_points_removed': isolated_removed,
                       'ambiguous_cross_object_points_removed': ambiguous_removed,
                       'second_to_first_surface_p90_m': float(np.percentile(distances, 90)),
                       'fraction_within_3cm': within_3cm, 'centroid_shift_m': centroid_shift,
                       'proxy': proxy}
    ply_path = output/'three_objects_two_views_measured.ply'
    write_ply(ply_path, np.concatenate(all_xyz), np.concatenate(all_rgb), np.concatenate(all_labels))
    (output/'geometry.json').write_text(json.dumps(geometry, indent=2, ensure_ascii=False), encoding='utf-8')
    second_camera_table = np.linalg.inv(T_table_second_camera)
    registration = {'passed': True, 'views': [str(first_capture), str(second_capture)],
                    'T_view_camera_table': [T_first_camera_table.tolist(), second_camera_table.tolist()],
                    'T_reference_table_view_table': [np.eye(4).tolist(), np.eye(4).tolist()],
                    'feature_registration_path': str(feature_path), 'feature_registration_sha256': digest(feature_path),
                    'table_alignment_p95_m': table_z_error, 'table_normal_angle_deg': table_normal_angle,
                    'object_quality': {k: {f: v[f] for f in ('centroid_shift_m', 'fraction_within_3cm')}
                                       for k, v in stats.items()},
                    'scope': 'First tabletop frame; two RGB-D views aligned by measured static features.'}
    (output/'registration.json').write_text(json.dumps(registration, indent=2), encoding='utf-8')
    scene = dict(first_scene.data)
    scene.pop('geometry', None)
    scene.pop('pointcloud_sha256', None)
    scene.pop('target_observation', None)
    scene.update(scene_id='new_desk_20261113_two_views', geometry_path=str(output/'geometry.json'),
                 pointcloud_path=str(ply_path))
    SceneSpec(scene).save(output/'scene.json')
    result = {'first_capture': str(first_capture), 'second_capture': str(second_capture),
              'feature_registration': str(feature_path), 'feature_quality': match,
              'second_table_fit_p95_m': table_fit_error, 'second_table_inlier_fraction': float(inliers.mean()),
              'second_table_reference_p95_z_m': table_z_error,
              'table_normal_angle_deg': table_normal_angle, 'objects': stats,
              'ply_sha256': digest(ply_path),
              'scope': 'Accepted measured surfaces only; collision bodies are geometric proxies.'}
    (output/'fusion_report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'objects'}, indent=2, ensure_ascii=False))
    for name, item in stats.items():
        print(name, {k: v for k, v in item.items() if k != 'proxy'})
    print(output/'scene.json')


if __name__ == '__main__':
    main()
