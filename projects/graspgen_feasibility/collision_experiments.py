"""Run hardware-free approach and sampled final-collision reliability studies."""
import argparse
import csv
from datetime import datetime
from itertools import product
from pathlib import Path
import platform
import time
import numpy as np
import trimesh
from offline_core.approach import ApproachChecker
from offline_core.collision import CollisionChecker
from offline_core.config import ROOT, load_config
from offline_core.export import write_json
from offline_core.runtime import initialize
from offline_core.scene_io import file_hash
from experiments.collision_suite import scenes, sample_boxes
from experiments.reference_geometry import mesh_boxes_collision


def save_csv(path, records):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def metrics(rows):
    accepted = [r for r in rows if not r['predicted_collision']]
    false_accept = sum(r['reference_collision'] for r in accepted)
    false_reject = sum(r['predicted_collision'] and not r['reference_collision'] for r in rows)
    reference_collisions = sum(r['reference_collision'] for r in rows)
    return {'total': len(rows), 'accepted': len(accepted), 'false_accept': false_accept,
            'false_accept_among_accepted': false_accept / len(accepted) if accepted else None,
            'reference_collisions': reference_collisions,
            'miss_rate_among_collisions': false_accept / reference_collisions if reference_collisions else None,
            'false_reject': false_reject, 'reference_clear': len(rows) - reference_collisions,
            'retained_reference_clear': len(accepted) - false_accept}


def timed_gpu(torch, checker, points, poses):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    free = checker.check(points, poses)
    torch.cuda.synchronize()
    return free, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**20


def run(output):
    cfg = load_config()
    torch = initialize(cfg)
    baseline = CollisionChecker(cfg['grasp']['expected_gripper'], cfg['collision'], 1001)
    info = baseline.gripper_info
    parts = info.collision_mesh.split(only_watertight=False)
    suite = scenes()
    seeds = [1000, 1001, 1002]
    budgets, surfaces, margins = [2048, 8192, 32768], [500, 2000, 4000], [.005, .01, .02]
    protocol = {'schema_version': 1, 'scene_count': len(suite), 'candidates_per_scene': 6,
                'seeds': seeds, 'background_budgets': budgets, 'surface_budgets': surfaces,
                'baseline_margins_m': margins, 'approach_margins_m': [0, .002, .005],
                'approach_distance_m': .1, 'full_observed_background_points': 65536,
                'candidate_origin': 'fixed geometric perturbations, not learned GraspGen outputs',
                'sampling': 'area-weighted whole-box surfaces, nested prefixes per seed; not a depth-camera simulator',
                'reference': 'original mesh triangles vs solid boxes; continuous 13-axis SAT plus component winding containment',
                'reference_clearance_m': 0, 'contact': 'touching counts as solid collision; open gripper only',
                'split': '6 development and 6 held-out scene templates; frozen before running, not real-data generalization',
                'timing': 'one warmed call per scene/config/seed; includes transform, transfer and synchronization, excludes sampling/reference',
                'sources_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in
                    [Path(__file__), ROOT/'experiments/collision_suite.py', ROOT/'experiments/reference_geometry.py', ROOT/'offline_core/approach.py', ROOT/'offline_core/collision.py']},
                'gripper_mesh_sha256': file_hash(Path(cfg['grasp']['repo'])/'assets/robotiq/robotiq_140_collision.obj'),
                'python': platform.python_version(), 'numpy': np.__version__, 'trimesh': trimesh.__version__,
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    write_json(output/'protocol.json', protocol)
    write_json(output/'scenes.json', suite)
    reference = []
    ref = {}
    for scene in suite:
        for candidate, pose in enumerate(scene['poses']):
            final = mesh_boxes_collision(parts, pose, scene['background'])
            final_all = mesh_boxes_collision(parts, pose, scene['background'] + scene['target'])
            path = mesh_boxes_collision(parts, pose, scene['background'] + scene['target'], .1)
            row = {'scene_id': scene['scene_id'], 'split': scene['split'], 'candidate': candidate,
                   'final_background_collision': final, 'final_all_collision': final_all,
                   'full_approach_collision': path}
            reference.append(row)
            ref[(scene['scene_id'], candidate)] = row
    save_csv(output/'reference.csv', reference)
    print(f'Reference ready: {len(reference)} fixed candidates', flush=True)
    # Warm CUDA and the upstream filter once, without counting startup as runtime.
    baseline.check(sample_boxes(suite[0]['background'], 8192, 0), suite[0]['poses'])
    baseline_rows, timing, approach_rows, path_comparison = [], [], [], []
    checker_by_margin = {margin: ApproachChecker.from_official(info, {**cfg['approach'], 'clearance_m': margin}) for margin in [0, .002, .005]}
    write_json(output/'approach_geometry.json', checker_by_margin[.002].metadata)
    for seed in seeds:
        surface = trimesh.sample.sample_surface(info.collision_mesh, max(surfaces), seed=seed + 1)[0].astype(np.float32)
        for scene in suite:
            name, poses = scene['scene_id'], scene['poses']
            cloud = sample_boxes(scene['background'], 65536, seed)
            target = sample_boxes(scene['target'], 4096, seed + 10)
            default_prediction = None
            for budget, samples, margin in product(budgets, surfaces, margins):
                baseline.samples_local_m = surface[:samples]
                baseline.cfg = {**cfg['collision'], 'threshold_m': margin}
                free, seconds, peak = timed_gpu(torch, baseline, cloud[:budget], poses)
                timing.append({'scene_id': name, 'seed': seed, 'background_points': budget,
                               'surface_points': samples, 'margin_m': margin, 'seconds': seconds, 'peak_cuda_mib': peak})
                for index, clear in enumerate(free):
                    baseline_rows.append({'scene_id': name, 'split': scene['split'], 'candidate': index, 'seed': seed,
                                          'background_points': budget, 'surface_points': samples, 'margin_m': margin,
                                          'predicted_collision': not bool(clear),
                                          'reference_collision': ref[(name, index)]['final_background_collision']})
                if (budget, samples, margin) == (8192, 2000, .02):
                    default_prediction = ~free
            for margin, checker in checker_by_margin.items():
                started = time.perf_counter()
                results = checker.check(cloud, target, poses)
                seconds = time.perf_counter() - started
                for index, result in enumerate(results):
                    approach_rows.append({'scene_id': name, 'split': scene['split'], 'candidate': index, 'seed': seed,
                                          'margin_m': margin, 'predicted_collision': result['state'] != 'clear',
                                          'reference_collision': ref[(name, index)]['full_approach_collision'],
                                          'first_hit_fraction': result['first_hit_fraction'],
                                          'reasons': '|'.join(result['reasons']), 'scene_batch_seconds': seconds})
                    if margin == .002:
                        for method, collision in [('final_only_default', default_prediction[index]),
                                                   ('continuous_approach_only', result['state'] != 'clear'),
                                                   ('combined_pipeline', bool(default_prediction[index]) or result['state'] != 'clear')]:
                            path_comparison.append({'scene_id': name, 'split': scene['split'], 'candidate': index, 'seed': seed,
                                                    'method': method, 'predicted_collision': bool(collision),
                                                    'reference_collision': ref[(name, index)]['full_approach_collision']})
            print(f'Seed {seed}: {name} done', flush=True)
        # Checkpoint raw data so a later failure cannot erase finished experiments.
        save_csv(output/'baseline_candidates.csv', baseline_rows)
        save_csv(output/'baseline_timings.csv', timing)
        save_csv(output/'approach_candidates.csv', approach_rows)
        save_csv(output/'path_comparison.csv', path_comparison)
    summary_rows = []
    for split in ['all', 'development', 'held_out']:
        for budget, samples, margin in product(budgets, surfaces, margins):
            rows = [r for r in baseline_rows if (r['background_points'], r['surface_points'], r['margin_m']) == (budget, samples, margin) and (split == 'all' or r['split'] == split)]
            times = [r for r in timing if (r['background_points'], r['surface_points'], r['margin_m']) == (budget, samples, margin) and (split == 'all' or next(s['split'] for s in suite if s['scene_id'] == r['scene_id']) == split)]
            summary_rows.append({'split': split, 'background_points': budget, 'surface_points': samples, 'margin_m': margin,
                                 **metrics(rows), 'mean_scene_seconds': float(np.mean([r['seconds'] for r in times])),
                                 'max_cuda_mib': max(r['peak_cuda_mib'] for r in times)})
    save_csv(output/'baseline_summary.csv', summary_rows)
    approach_summary = [{'split': split, 'margin_m': margin, **metrics([r for r in approach_rows if r['margin_m'] == margin and (split == 'all' or r['split'] == split)])} for split in ['all', 'development', 'held_out'] for margin in [0, .002, .005]]
    path_summary = [{'split': split, 'method': method, **metrics([r for r in path_comparison if r['method'] == method and (split == 'all' or r['split'] == split)])} for split in ['all', 'development', 'held_out'] for method in ['final_only_default', 'continuous_approach_only', 'combined_pipeline']]
    save_csv(output/'approach_summary.csv', approach_summary)
    save_csv(output/'path_summary.csv', path_summary)
    # Controlled time-step ablation with an explicitly observed 0.5mm plate.
    thin = next(s for s in suite if s['scene_id'] == 'thin_between_endpoints')
    observed_plate = sample_boxes([thin['background'][1]], 4000, 42)
    strict = checker_by_margin[0]
    discrete = []
    for step in [.10, .02, .005, .001]:
        fractions = np.linspace(0, 1, int(np.ceil(.1 / step)) + 1)
        collision = any(strict._first_hit(observed_plate + [0, 0, .1 * (1-f)], 0) is not None for f in fractions)
        discrete.append({'requested_step_m': step, 'poses': len(fractions), 'detected': collision})
    continuous = strict.check(observed_plate, sample_boxes(thin['target'], 1000, 3), thin['poses'][:1])[0]
    write_json(output/'thin_obstacle.json', {'plate_thickness_m': .0005, 'reference_continuous_collision': ref[('thin_between_endpoints', 0)]['full_approach_collision'],
                                          'reference_final_collision': ref[('thin_between_endpoints', 0)]['final_background_collision'],
                                          'discrete': discrete, 'continuous': continuous})
    default = next(r for r in summary_rows if (r['split'], r['background_points'], r['surface_points'], r['margin_m']) == ('all', 8192, 2000, .02))
    result = {'status': 'complete', 'output': str(output), 'reference_candidates': len(reference),
              'baseline_candidate_evaluations': len(baseline_rows), 'baseline_configurations': 27,
              'default_final_only': default, 'path_comparison': [r for r in path_summary if r['split'] == 'all'],
              'approach': [r for r in approach_summary if r['split'] == 'all'],
              'thin_discrete': discrete, 'thin_continuous_detected': continuous['state'] == 'collision',
              'limitations': ['Synthetic whole-surface box observations, not sensor data or real grasp success.',
                             'Rates pool 3 sampling repeats of the same 72 candidates; not 216 independent scenes.',
                             'Point observations cannot detect all solid containment; unknown-space processing is not implemented.',
                             'Convexified base and face-plane clearance can conservatively reject mesh-clear candidates.',
                             'No parameter was selected using held-out scenes; current defaults are not hardware-calibrated.']}
    write_json(output/'summary.json', result)
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT/'runs/collision_experiments'/datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('output must be empty; existing experiments are never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    try:
        run(output)
    except Exception as exc:
        import traceback
        write_json(output/'failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
