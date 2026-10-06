"""Audit saved depth/fusion evidence, source pixels, metrics and refusal artifacts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from offline_core.export import write_json
from offline_core.scene_io import load_scene, file_hash
from offline_core.config import ROOT
from experiments.synthetic_depth import surface_grid


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def audit(root):
    visibility = read_csv(root/'visibility/summary.csv')
    fusion = read_csv(root/'fusion/summary.csv')
    protocol = json.loads((root/'protocol.json').read_text())
    assert all(file_hash(ROOT/name) == expected for name, expected in protocol['source_sha256'].items())
    assert len(visibility) == 9 and len(fusion) == 26
    for row in visibility:
        assert row['state'] == row['expected_state']
        assert int(row['cells']) == sum(int(row[key]) for key in ['free_cells', 'unknown_cells', 'occupied_cells', 'conflict_cells'])
        if row['reference_collision'] == 'True':
            assert row['planning_allowed_by_observation'] == 'False'
        if row['state'] == 'observed_free':
            assert int(row['unknown_cells']) == int(row['occupied_cells']) == int(row['conflict_cells']) == 0
    counts = {'accepted_fusions': 0, 'rejected_fusions': 0}
    for row in fusion:
        folder = root/'fusion'/row['layout']/row['case']
        quality = json.loads((folder/'fusion_quality.json').read_text())
        accepted = row['accepted'] == 'True'
        assert quality['accepted'] == quality['planning_allowed'] == accepted
        assert (folder/'fusion_usable.npz').exists() == accepted
        assert (folder/'accepted_depth_views.npz').exists() == accepted
        if not accepted: assert quality['reasons']
        counts['accepted_fusions' if accepted else 'rejected_fusions'] += 1
        with np.load(folder/'fusion_diagnostic.npz', allow_pickle=False) as data:
            points = data['observed_points_world_m']
            assert len(points) == int(row['observed_points']) == len(data['source_pixel_yx'])
            assert data['fused_observation_count'].sum() == len(points)
            assert np.all(data['fused_source_view_bits'] > 0)
            # Every retained raw point must come from the recorded image pixel
            # transformed by the recorded corrected camera pose.
            for index, (view_id, pose) in enumerate(quality['corrected_poses'].items()):
                sources = json.loads((root/'fusion'/row['layout']/'input_scenes.json').read_text())
                scene = load_scene(root/'fusion'/row['layout']/sources[view_id])
                correct = scene.depth_view.with_pose(pose)
                choose = data['source_view_index'] == index
                pixels = data['source_pixel_yx'][choose]
                assert choose.sum() == (correct.depth_m > 0).sum()
                np.testing.assert_allclose(points[choose], correct.world_image()[pixels[:, 0], pixels[:, 1]], atol=1e-12, rtol=0)
            target = data['fused_points_world_m'][data['fused_target_mask']]
            truth = json.loads((root/'fusion'/row['layout']/'truth_for_evaluation_only.json').read_text())
            box = truth['boxes'][1]
            reference = surface_grid(box)
            coverage = float(np.mean(cKDTree(target).query(reference)[0] <= .006))
            assert abs(coverage-float(row['target_coverage_fraction'])) < 1e-12
            extent = np.ptp(target, axis=0)
            np.testing.assert_allclose(extent, [float(row['target_extent_'+axis+'_m']) for axis in 'xyz'], atol=1e-12)
            expected_error = np.max(np.abs(extent-(np.asarray(box['upper'])-box['lower'])))
            assert abs(expected_error-float(row['max_target_extent_error_m'])) < 1e-12
        if row['case'] in ['perturbed_2_raw', 'perturbed_3_raw', 'gross_error_refined', 'insufficient_overlap', 'overlap_2_raw', 'overlap_3_raw']:
            assert not accepted
        if accepted and row['refine'] == 'True':
            assert float(row['max_camera_translation_error_m']) < .005
            assert float(row['max_camera_rotation_error_deg']) < .1
    for layout in ['A', 'B']:
        known = [next(r for r in fusion if r['layout']==layout and r['case']==f'known_{n}') for n in [1, 2, 3]]
        assert all(r['accepted'] == 'True' for r in known)
        assert float(known[0]['target_coverage_fraction']) < float(known[1]['target_coverage_fraction']) < float(known[2]['target_coverage_fraction'])
        assert float(known[2]['max_target_extent_error_m']) < .001
    planar = json.loads((root/'planar_degeneracy/fusion_quality.json').read_text())
    assert not planar['accepted'] and any('degenerate' in reason for reason in planar['reasons'])
    assert not (root/'planar_degeneracy/fusion_usable.npz').exists()
    unexpected = [r for r in fusion if r['passed']=='False']
    return {'status': 'passed', 'visibility_cases': 9, 'fusion_cases': len(fusion), **counts,
            'unsuccessful_small_error_recoveries': [{'layout': r['layout'], 'case': r['case'], 'reasons': r['reasons']} for r in unexpected],
            'checks': ['source hashes', 'visibility counters and no release of reference collisions',
                       'refusal prevents usable cloud/depth artifacts', 'all raw point pixel/pose provenance',
                       'coverage/extent metrics recomputed', 'accepted registration error bounds', 'planar degeneracy refusal']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    write_json(args.root/'verification.json', result)
    print(json.dumps(result, indent=2))
