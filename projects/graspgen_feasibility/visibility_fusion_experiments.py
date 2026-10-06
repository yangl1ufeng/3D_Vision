"""Reproducible synthetic-depth occlusion and 1/2/3-view fusion experiments."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from offline_core.config import ROOT
from offline_core.export import write_json
from offline_core.scene_io import file_hash, load_scene
from offline_core.approach import ApproachChecker
from offline_core.observation import DepthView, ObservationChecker, classify_space, STATE_NAMES
from offline_core.fusion import fuse_views, quality_gate, DEFAULT_FUSION
from experiments.synthetic_depth import render_boxes, look_at, surface_grid
from experiments.collision_suite import make_box
from experiments.reference_geometry import mesh_boxes_collision
from fuse_scene import save_result


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_view(folder, view, labels, target_label=1):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder/'depth.npy', view.depth_m.astype(np.float32))
    mask = labels == target_label
    np.save(folder/'target_mask.npy', mask)
    palette = np.array([[195, 184, 166], [65, 142, 225], [213, 98, 68], [112, 173, 121]], dtype=np.uint8)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels >= 0
    rgb[valid] = palette[labels[valid] % len(palette)]
    Image.fromarray(rgb).save(folder/'rgb.png')
    manifest = {'schema_version': 1, 'scene_id': view.view_id, 'rgb': 'rgb.png',
                'geometry': {'kind': 'depth', 'path': 'depth.npy', 'unit': 'm', 'aligned_to_rgb': True,
                             'depth_convention': 'z', 'pinhole_rectified': True, 'intrinsics': view.intrinsics},
                'frames': {'world': 'synthetic_table', 'camera_convention': 'opencv', 'T_world_camera': view.T_world_camera},
                'target': {'id': 'target', 'mask': 'target_mask.npy'}}
    write_json(folder/'scene.json', manifest)


def visibility_experiment(output, gripper):
    folder = output/'visibility'
    folder.mkdir()
    K = np.array([[160., 0, 79.5], [0, 160., 59.5], [0, 0, 1]])
    back = make_box('back_wall', [0, 0, 1.21], [2., 2., .02])
    side = make_box('side_wall', [-.71, 0, .75], [.02, 2., 2.])
    blocker = make_box('occluder', [0, 0, .55], [.30, .24, .04])
    hidden_obstacle = make_box('hidden_finger_obstacle', [.068, 0, .88], [.018, .035, .03])
    closed = make_box('enclosing_solid', [0, 0, .8], [.6, .6, .6])
    front_pose = np.eye(4)
    side_pose = look_at([.8, 0, .78], [0, 0, .78])
    clear_boxes = [back, side]
    shadow_boxes = [back, side, blocker]
    collision_boxes = shadow_boxes+[hidden_obstacle]
    front_clear, _ = render_boxes(clear_boxes, front_pose, K, view_id='front_clear')
    front_shadow, shadow_labels = render_boxes(shadow_boxes, front_pose, K, view_id='front_shadow')
    side_shadow, _ = render_boxes(shadow_boxes, side_pose, K, view_id='side_shadow')
    front_hidden, _ = render_boxes(collision_boxes, front_pose, K, view_id='front_hidden')
    side_hidden, _ = render_boxes(collision_boxes, side_pose, K, view_id='side_hidden')
    front_enclosed, _ = render_boxes([back, side, closed], front_pose, K, view_id='front_enclosed')
    hole_depth = front_clear.depth_m.copy()
    hole_depth[45:75, 60:100] = 0
    holes = DepthView('missing_depth', hole_depth, K, front_pose)
    pose = np.eye(4)
    pose[2, 3] = .72
    outside_pose = pose.copy()
    outside_pose[0, 3] = 1.
    approach = ApproachChecker([p.vertices for p in gripper], [0, 0, 1], .10, .002)
    specs = [
        ('clear_single', [front_clear], pose, clear_boxes, 'observed_free'),
        ('occluded_single', [front_shadow], pose, shadow_boxes, 'insufficient_observation'),
        ('occluded_plus_side', [front_shadow, side_shadow], pose, shadow_boxes, 'observed_free'),
        ('hidden_collision_single', [front_hidden], pose, collision_boxes, 'insufficient_observation'),
        ('hidden_collision_plus_side', [front_hidden, side_hidden], pose, collision_boxes, 'occupied'),
        ('missing_depth', [holes], pose, clear_boxes, 'insufficient_observation'),
        ('outside_fov', [front_clear], outside_pose, clear_boxes, 'insufficient_observation'),
        ('enclosed_in_solid', [front_enclosed], pose, [back, side, closed], 'insufficient_observation'),
        ('missing_calibration', [], pose, clear_boxes, 'insufficient_observation'),
    ]
    rows, detailed, evidence = [], [], {}
    for name, views, candidate, boxes, expected in specs:
        started = time.perf_counter()
        checker = ObservationChecker(approach, views, voxel_m=.008, tolerance_m=.002)
        result = checker.check(candidate[None])[0]
        seconds = time.perf_counter()-started
        true_collision = mesh_boxes_collision(gripper, candidate, boxes, .1)
        rows.append({'case': name, 'views': len(views), 'expected_state': expected, 'state': result['state'],
                     'passed': result['state'] == expected, 'reference_collision': true_collision,
                     'planning_allowed_by_observation': result['state']=='observed_free',
                     'cells': result['cells'], 'free_cells': result['free_cells'], 'unknown_cells': result['unknown_cells'],
                     'occupied_cells': result['occupied_cells'], 'conflict_cells': result['conflict_cells'],
                     'needs_more_views': result['needs_more_views'], 'seconds': seconds})
        detailed.append({'case': name, 'pose': candidate, 'result': result, 'boxes_for_evaluation_only': boxes})
        for v in views:
            evidence[v.view_id+'_depth_m'] = v.depth_m
            evidence[v.view_id+'_T_world_camera'] = v.T_world_camera
        print(f'Visibility {name}: {result["state"]}, unknown={result["unknown_cells"]}/{result["cells"]}', flush=True)
    # Dense x/z slice illustrates that absent returns behind the front blocker
    # do NOT mean free. It is classified from the saved depth evidence only.
    x, z = np.meshgrid(np.linspace(-.30, .30, 201), np.linspace(.2, 1.15, 201))
    queries = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
    single, _ = classify_space(queries, [front_shadow])
    multiple, _ = classify_space(queries, [front_shadow, side_shadow])
    np.savez_compressed(folder/'visibility_arrays.npz', **evidence, intrinsics=K,
                        query_x=x, query_z=z, single_states=single.reshape(x.shape),
                        multi_states=multiple.reshape(x.shape), gripper_pose=pose,
                        gripper_vertices=np.concatenate([p.vertices for p in gripper]))
    write_json(folder/'details.json', detailed)
    csv_write(folder/'summary.csv', rows)
    return rows


def perturb(view, translation, rotation_deg):
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler('xyz', rotation_deg, degrees=True).as_matrix()
    delta[:3, 3] = translation
    return view.with_pose(delta @ view.T_world_camera)


def target_metrics(points, box, reference_points):
    lower, upper = np.asarray(box['lower']), np.asarray(box['upper'])
    outside = np.maximum(np.maximum(lower-points, points-upper), 0)
    distance = np.linalg.norm(outside, axis=1)
    interior = np.all((points >= lower) & (points <= upper), axis=1)
    distance[interior] = np.minimum(points[interior]-lower, upper-points[interior]).min(1)
    nearest = cKDTree(points).query(reference_points)[0]
    extent = np.ptp(points, axis=0)
    return {'target_coverage_fraction': float(np.mean(nearest <= .006)),
            'target_surface_error_p95_m': float(np.quantile(distance, .95)),
            'target_extent_x_m': float(extent[0]), 'target_extent_y_m': float(extent[1]), 'target_extent_z_m': float(extent[2]),
            'max_target_extent_error_m': float(np.max(np.abs(extent-(upper-lower))))}


def fusion_experiment(output):
    root = output/'fusion'
    root.mkdir()
    K = np.array([[170., 0, 79.5], [0, 170., 59.5], [0, 0, 1]])
    camera_poses = [look_at([0, -.65, .48], [0, 0, .08]),
                    look_at([.48, -.34, .42], [0, 0, .08]),
                    look_at([-.45, .28, .40], [0, 0, .08])]
    rows = []
    for layout in ['A', 'B']:
        folder = root/layout
        folder.mkdir()
        target = make_box('target', [0, 0, .09] if layout=='A' else [.02, .01, .09], [.12, .10, .18] if layout=='A' else [.10, .08, .18])
        occluder = make_box('occluder', [-.055, -.12, .08] if layout=='A' else [.045, -.10, .10], [.065, .035, .16] if layout=='A' else [.06, .03, .20])
        boxes = [make_box('table', [0, 0, -.01], [1., .8, .02]), target, occluder,
                 make_box('asymmetric_neighbor', [.17, .07, .065], [.07, .09, .13])]
        views, masks, labels_all = [], [], []
        for i, pose in enumerate(camera_poses):
            view, labels = render_boxes(boxes, pose, K, view_id=f'{layout}_view_{i}')
            save_view(folder/f'view_{i}', view, labels)
            loaded = load_scene(folder/f'view_{i}'/'scene.json')
            views.append(loaded.depth_view)
            masks.append(labels==1)
            labels_all.append(labels)
        write_json(folder/'bundle.json', {'views': [f'view_{i}/scene.json' for i in range(3)]})
        truth_by_view = {v.view_id: v.T_world_camera for v in views}
        scene_paths = {v.view_id: f'view_{i}/scene.json' for i, v in enumerate(views)}
        error2 = perturb(views[1], [.016, -.012, .006], [1., -1., 2.])
        error3 = perturb(views[2], [-.012, .014, -.004], [-1., 1., -1.5])
        gross = perturb(views[1], [.12, -.07, .04], [5., 8., 18.])
        distant = perturb(views[1], [.60, 0, 0], [0, 0, 0])
        scenarios = [('known_1', views[:1], masks[:1], False, True),
                     ('known_2', views[:2], masks[:2], False, True),
                     ('known_3', views, masks, False, True),
                     ('perturbed_2_raw', [views[0], error2], masks[:2], False, False),
                     ('perturbed_2_refined', [views[0], error2], masks[:2], True, True),
                     ('perturbed_3_raw', [views[0], error2, error3], masks, False, False),
                     ('perturbed_3_refined', [views[0], error2, error3], masks, True, True),
                     ('gross_error_refined', [views[0], gross], masks[:2], True, False),
                     ('insufficient_overlap', [views[0], distant], masks[:2], False, False)]
        # Additional recovery controls: all three cameras share three actual
        # surface directions. Keep the original difficult cases and expectations.
        overlap_poses = [look_at([.40, -.48, .48], [0, 0, .08]), camera_poses[1],
                         look_at([.35, -.45, .50], [0, 0, .08])]
        overlap_views, overlap_masks, overlap_labels = [], [], []
        for i, pose in enumerate(overlap_poses):
            v, labels = render_boxes(boxes, pose, K, view_id=f'{layout}_overlap_{i}')
            save_view(folder/f'overlap_{i}', v, labels)
            v = load_scene(folder/f'overlap_{i}'/'scene.json').depth_view
            overlap_views.append(v)
            overlap_masks.append(labels == 1)
            overlap_labels.append(labels)
            truth_by_view[v.view_id] = pose
            scene_paths[v.view_id] = f'overlap_{i}/scene.json'
        overlap_errors = [overlap_views[0],
                          perturb(overlap_views[1], [.016, -.012, .006], [1., -1., 2.]),
                          perturb(overlap_views[2], [-.012, .014, -.004], [-1., 1., -1.5])]
        for n in [2, 3]:
            scenarios.extend([(f'overlap_{n}_raw', overlap_errors[:n], overlap_masks[:n], False, False),
                              (f'overlap_{n}_refined', overlap_errors[:n], overlap_masks[:n], True, True)])
        for i, v in enumerate(overlap_errors):
            save_view(folder/f'overlap_error_{i}', v, overlap_labels[i])
        write_json(folder/'overlap_erroneous_bundle.json', {'views': [f'overlap_error_{i}/scene.json' for i in range(3)]})
        write_json(folder/'truth_for_evaluation_only.json', {'boxes': boxes, 'true_poses': camera_poses,
                                                           'true_poses_by_view': truth_by_view})
        write_json(folder/'input_scenes.json', scene_paths)
        truth_samples = surface_grid(target)
        for name, inputs, target_masks, refine, expected in scenarios:
            destination = folder/name
            started = time.perf_counter()
            result = fuse_views(inputs, target_masks, refine)
            seconds = time.perf_counter()-started
            save_result(destination, result)
            input_gate = quality_gate(inputs, target_masks, DEFAULT_FUSION)
            write_json(destination/'initial_pose_quality.json', input_gate)
            write_json(destination/'initial_poses.json', {v.view_id: v.T_world_camera for v in inputs})
            cloud = result['diagnostic_cloud']
            target_points = cloud['fused_points_world_m'][cloud['fused_target_mask']]
            metric = target_metrics(target_points, target, truth_samples)
            translation_errors, rotation_errors = [], []
            for v in result['corrected_views']:
                truth = truth_by_view[v.view_id]
                translation_errors.append(float(np.linalg.norm(v.T_world_camera[:3, 3]-truth[:3, 3])))
                rotation_errors.append(float(np.rad2deg(Rotation.from_matrix(v.T_world_camera[:3, :3] @ truth[:3, :3].T).magnitude())))
            row = {'layout': layout, 'case': name, 'views': len(inputs), 'refine': refine,
                   'expected_accepted': expected, 'accepted': result['accepted'], 'passed': result['accepted']==expected,
                   'reasons': '|'.join(result['reasons']), **metric,
                   'max_camera_translation_error_m': max(translation_errors), 'max_camera_rotation_error_deg': max(rotation_errors),
                   'observed_points': len(cloud['observed_points_world_m']), 'fused_points': len(cloud['fused_points_world_m']),
                   'fused_target_points': len(target_points), 'seconds': seconds}
            rows.append(row)
            write_json(destination/'evaluation.json', row)
            print(f'Fusion {layout}/{name}: accepted={result["accepted"]}, coverage={metric["target_coverage_fraction"]:.3f}, reasons={result["reasons"]}', flush=True)
        # Preserve a reusable erroneous bundle for the real CLI refusal check.
        for i, view in enumerate([views[0], error2]):
            save_view(folder/f'erroneous_view_{i}', view, labels_all[i])
        write_json(folder/'erroneous_bundle.json', {'views': [f'erroneous_view_{i}/scene.json' for i in range(2)]})
    csv_write(root/'summary.csv', rows)
    return rows


def planar_experiment(output):
    K = np.array([[80., 0, 31.5], [0, 80., 31.5], [0, 0, 1]])
    a = DepthView('plane_anchor', np.ones((64, 64)), K, np.eye(4))
    b = perturb(DepthView('plane_moving', np.ones((64, 64)), K, np.eye(4)), [.015, 0, 0], [0, 0, 0])
    result = fuse_views([a, b], [np.ones((64, 64), dtype=bool)]*2, refine=True)
    save_result(output/'planar_degeneracy', result)
    return {'accepted': result['accepted'], 'reasons': result['reasons'],
            'passed': not result['accepted'] and any('degenerate' in r for r in result['reasons'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT/'runs/visibility_fusion'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    mesh_path = ROOT/'third_party/GraspGen/assets/robotiq/robotiq_140_collision.obj'
    mesh = trimesh.load(mesh_path, force='mesh')
    parts = mesh.split(only_watertight=False)
    sources = [Path(__file__), ROOT/'offline_core/observation.py', ROOT/'offline_core/fusion.py', ROOT/'experiments/synthetic_depth.py', ROOT/'fuse_scene.py']
    write_json(output/'protocol.json', {'revision': 2, 'date': '2026-10-05', 'camera_size': [160, 120],
                                      'visibility_voxel_m': .008, 'depth_tolerance_m': .002, 'fusion_config': DEFAULT_FUSION,
                                      'coverage_reference_spacing_m': .004, 'coverage_distance_m': .006,
                                      'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                                      'gripper_mesh_sha256': file_hash(mesh_path),
                                      'notes': ['Exact ray/box z-depth, no added noise.', 'True geometry and poses are evaluation-only.',
                                                'Two deterministic layouts, not a statistical real-camera benchmark.',
                                                'Original 18 expectations retained; 8 added shared-face controls. Edge normal stability checked before registration.',
                                                'Rejected fusion clouds are diagnostic only and never usable for planning.']})
    try:
        observation = visibility_experiment(output, parts)
        fusion = fusion_experiment(output)
        planar = planar_experiment(output)
        # Completion means the protocol ran. A failed recovery remains an
        # explicit experimental finding; never tune acceptance to make it pass.
        summary = {'status': 'complete',
                   'visibility_cases': len(observation), 'visibility_passed': sum(r['passed'] for r in observation),
                   'fusion_cases': len(fusion), 'fusion_passed': sum(r['passed'] for r in fusion),
                   'unexpected_fusion_outcomes': [r for r in fusion if not r['passed']],
                   'planar_degeneracy': planar, 'output': str(output)}
        write_json(output/'summary.json', summary)
        print(summary, flush=True)
    except Exception as exc:
        import traceback
        write_json(output/'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
