"""Preparation 5: seeded depth/mask degradation and actual SAM2 box perturbations."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
from offline_core.config import ROOT
from offline_core.scene_io import load_scene, file_hash
from offline_core.segmentation import SAM2Segmenter
from offline_core.export import write_json
from experiments.input_degradation import degrade, perturb_box, mask_metrics
from experiments.degradation_runner import Runner, save_csv, target_comparison

SEEDS = [1000, 1001, 1002]
VARIANTS = [('clean', 0)] + [('noise', x) for x in [.001, .003, .006, .010, .020]] + \
    [('dropout', x) for x in [.25, .50, .75, .90, .97, .995]] + \
    [('hole', x) for x in [.25, .50, .75]] + \
    [(kind, x) for kind in ['erode', 'dilate'] for x in [1, 3, 6, 12]]
BOX_VARIANTS = [('box_clean', 0)] + [('box_shift', x) for x in [.1, .25, .5, 1.]] + [('box_shrink', .5), ('box_expand', 1.5)]


def json_read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def case_manifest(original, folder, depth=None, mask=None, box=None):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(original.metadata['input_manifest'])
    assets = original.metadata['asset_paths']
    manifest['rgb'] = assets['rgb']
    if depth is not None:
        np.save(folder/'depth.npy', depth.astype(np.float32))
        manifest['geometry']['path'] = 'depth.npy'
    else:
        manifest['geometry']['path'] = assets['geometry']
        if 'valid_mask' in manifest['geometry']:
            manifest['geometry']['valid_mask'] = str((original.manifest_path.parent / manifest['geometry']['valid_mask']).resolve())
    if mask is not None:
        np.save(folder/'mask.npy', mask)
        manifest['target'] = {'id': original.target_id, 'mask': 'mask.npy'}
    else:
        manifest['target'] = {'id': original.target_id, 'box_xyxy': box, 'prompt_source': 'seeded preparation experiment'}
    write_json(folder/'scene.json', manifest)
    return load_scene(folder/'scene.json')


def flag_row(row, baseline):
    failures = []
    if row['status'] != 'ok': failures.append('pipeline_failure')
    if row.get('mask_iou', 1) < .70: failures.append('mask_iou_below_0.70')
    if row.get('target_surface_p95_m') is not None and row['target_surface_p95_m'] > .010:
        failures.append('target_surface_error_over_10mm')
    if row.get('target_coverage_6mm', 0) < .5 * baseline.get('target_coverage_6mm', 0):
        failures.append('target_coverage_below_half_clean')
    if baseline.get('eligible', 0) >= 10 and row.get('eligible', 0) < .5*baseline['eligible']:
        failures.append('eligible_candidates_below_half_clean')
    if (row.get('fixed_decision_flip_fraction') or 0) > .20:
        failures.append('fixed_geometry_decisions_changed_over_20pct')
    row['material_degradation'] = bool(failures)
    row['degradation_reasons'] = '|'.join(failures)


def run(output, source_root):
    protocol = {'seeds': SEEDS, 'synthetic_variants': VARIANTS, 'box_variants': BOX_VARIANTS,
                'candidate_budget': 100, 'source_root': str(source_root),
                'planned_synthetic_runs': 2*len(VARIANTS)*len(SEEDS), 'planned_prompt_runs': len(BOX_VARIANTS)*len(SEEDS),
                'failure_criteria': ['pipeline exception', 'mask IoU < 0.70', 'model point nearest-clean-surface P95 > 10mm',
                                    'coverage at 6mm < half clean', 'score-qualified count < half clean when clean >= 10',
                                    'fixed clean-candidate approach decisions change > 20%'],
                'notes': ['One degradation family at a time. Three seeded replicates; not a sensor-specific confidence interval.',
                          'Noise is additive Gaussian z-depth, dropout independent per valid pixel; hole is a central box-area fraction.',
                          'Mask morphology uses a 3x3 square per iteration. Geometry and RGB unchanged for mask experiments.',
                          'Synthetic truth remains evaluation-only. SAM2 prompt experiment uses official real RGB and its existing XYZ.',
                          'Real-image clean SAM2 mask is a stability reference, not a human ground-truth annotation.',
                          'Final all-rejected baseline cannot demonstrate improved grasp success; inspect earlier stages and fixed probes.'],
                'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__), ROOT/'experiments/input_degradation.py', ROOT/'experiments/degradation_runner.py']}}
    write_json(output/'protocol.json', protocol)
    runner = Runner()
    runner.save_provenance(output)
    rows, clean, fixed = [], {}, {}
    for layout in ['A', 'B']:
        original = load_scene(source_root/'fusion'/layout/'view_1/scene.json')
        reference_mask = np.load(original.target['mask_path']) > 0
        reference_points = original.xyz_world_m[reference_mask & original.valid]
        boxes = json_read(source_root/'fusion'/layout/'truth_for_evaluation_only.json')['boxes']
        write_json(output/f'{layout}_source.json', {'input': original.metadata, 'boxes_evaluation_only': boxes})
        for kind, level in VARIANTS:
            for seed in SEEDS:
                folder = output/'synthetic'/layout/f'{kind}_{level:g}'/str(seed)
                depth, mask = degrade(original.depth_view.depth_m, reference_mask, kind, level, seed+50000)
                scene = case_manifest(original, folder/'input', depth=depth, mask=mask)
                record, arrays, grasps = runner.run_scene(scene, mask, seed, folder, boxes, fixed.get((layout, seed)))
                row = {k:v for k,v in record.items() if k != 'performance'}
                row.update(layout=layout, family=kind, level=level, folder=str(folder.relative_to(output)),
                           target_observed_fraction=float(np.sum(reference_mask & scene.valid)/np.sum(reference_mask & original.valid)),
                           **mask_metrics(mask, reference_mask),
                           **target_comparison(arrays.get('target_model_world_m', np.empty((0, 3))), reference_points))
                if kind == 'clean':
                    clean[(layout, seed)] = dict(row)
                    if record['status'] == 'ok':
                        decisions = json_read(folder/'candidates.json')['candidates']
                        ids = [i for i,d in enumerate(decisions) if 'approach' in d]
                        fixed[(layout, seed)] = {'poses': grasps.poses_world_m[ids],
                                                'clear': np.asarray([decisions[i]['approach']['state']=='clear' for i in ids]),
                                                'reference': arrays['reference_collision'][ids].astype(bool)}
                flag_row(row, clean[(layout, seed)])
                write_json(folder/'evaluation.json', row)
                rows.append(row)
                save_csv(output/'summary.csv', rows)
                print(f"{layout} {kind} {level:g} seed={seed}: {row['status']}, eligible={row.get('eligible')}, material={row['degradation_reasons']}", flush=True)
    # Prompt perturbations run actual SAM2 on the existing real RGB scene.
    segmenter = SAM2Segmenter(runner.cfg['segmentation'])
    original = load_scene(ROOT/'data/offline_official/scene.json')
    write_json(output/'prompt_source.json', original.metadata)
    write_json(output/'sam2_provenance.json', {'checkpoint': str(segmenter.checkpoint), 'sha256': file_hash(segmenter.checkpoint), 'precision': segmenter.precision})
    original_box = original.target['box_xyxy']
    reference_mask = None
    for kind, level in BOX_VARIANTS:
        prompt_folder = output/'prompts'/f'{kind}_{level:g}'
        box = perturb_box(original_box, original.rgb.shape[:2], 'box_scale' if kind in ['box_shrink', 'box_expand'] else kind, level)
        scene = case_manifest(original, prompt_folder/'input', box=box)
        from offline_core.profiling import Profiler
        profiler = Profiler()
        with profiler.stage('segmentation', gpu=True):
            segmentation = segmenter.segment(scene)
        np.save(prompt_folder/'sam2_mask.npy', segmentation.mask)
        write_json(prompt_folder/'segmentation.json', {'box_xyxy': box, 'predicted_quality': segmentation.predicted_quality, 'performance': profiler.summary()})
        if reference_mask is None:
            reference_mask = segmentation.mask.copy()
            reference_points = scene.xyz_world_m[reference_mask & scene.valid]
        for seed in SEEDS:
            folder = prompt_folder/str(seed)
            record, arrays, _ = runner.run_scene(scene, segmentation.mask, seed, folder, segmentation_seconds=profiler.stages['segmentation']['seconds'])
            row = {k:v for k,v in record.items() if k != 'performance'}
            row.update(layout='real_prompt', family=kind, level=level, folder=str(folder.relative_to(output)),
                       **mask_metrics(segmentation.mask, reference_mask),
                       **target_comparison(arrays.get('target_model_world_m', np.empty((0, 3))), reference_points))
            if kind == 'box_clean': clean[('real_prompt', seed)] = dict(row)
            flag_row(row, clean[('real_prompt', seed)])
            write_json(folder/'evaluation.json', row)
            rows.append(row)
            save_csv(output/'summary.csv', rows)
            print(f"Real SAM2 {kind} {level:g} seed={seed}: {row['status']}, IoU={row['mask_iou']:.3f}, material={row['degradation_reasons']}", flush=True)
    groups = []
    for layout in ['A', 'B', 'real_prompt']:
        variants = BOX_VARIANTS if layout == 'real_prompt' else VARIANTS
        for kind, level in variants:
            chosen = [r for r in rows if (r['layout'],r['family'],r['level']) == (layout,kind,level)]
            groups.append({'layout':layout, 'family':kind, 'level':level, 'repeats':len(chosen),
                           'pipeline_failures':sum(r['status']!='ok' for r in chosen),
                           'material_degradation_runs':sum(r['material_degradation'] for r in chosen),
                           'median_mask_iou':float(np.median([r['mask_iou'] for r in chosen])),
                           'median_eligible':float(np.median([r.get('eligible',0) for r in chosen])),
                           'failures': [r['folder'] for r in chosen if r['material_degradation']]})
    write_json(output/'group_summary.json', groups)
    write_json(output/'summary.json', {'status':'complete','runs':len(rows), 'pipeline_failures':sum(r['status']!='ok' for r in rows),
                                      'material_degradation_runs':sum(r['material_degradation'] for r in rows), 'failure_examples':[r for r in rows if r['material_degradation']]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/visibility_fusion/prehardware_v2_20261005')
    args = parser.parse_args()
    output = (args.output or ROOT/'runs/robustness'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and any(output.iterdir()): parser.error('output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    try:
        run(output, args.source.resolve())
    except Exception as exc:
        import traceback
        write_json(output/'failure.json', {'error':str(exc), 'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
