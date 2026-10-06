"""Preparation 6: real denser depth rays, bounded preprocessing and obstacle retention."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import time
import numpy as np
from offline_core.config import ROOT
from offline_core.export import write_json, rank_candidates
from offline_core.scene_io import file_hash
from offline_core.profiling import Profiler
from offline_core.collision import initial_decisions, collision_eligible_indices, apply_collision_decisions
from experiments.degradation_runner import Runner, save_csv
from experiments.input_degradation import bounded_points
from experiments.synthetic_depth import render_boxes
from experiments.collision_suite import make_box
from experiments.reference_geometry import mesh_boxes_collision

SCALES = [1, 2, 4, 8]
SEEDS = [1000, 1001, 1002]
METHODS = [('random8192', None, 8192), ('voxel4mm8192', .004, 8192),
           ('voxel12mm8192', .012, 8192), ('voxel4mm32768', .004, 32768)]


def stage_value(profiler, names, field='seconds'):
    values = [profiler.stages[n][field] for n in names if n in profiler.stages and field in profiler.stages[n]]
    return sum(values) if values else None


def run(output, source_root):
    truth = json.loads((source_root/'fusion/A/truth_for_evaluation_only.json').read_text())
    post = make_box('thin_post', [.30, -.05, .27], [.001, .001, .025])
    boxes = truth['boxes']+[post]
    protocol = {'scales': SCALES, 'base_size': [160, 120], 'views': 3, 'seeds': SEEDS, 'methods': METHODS,
                'target_budget':8192, 'candidate_budget':100, 'planned_runs':len(SCALES)*len(SEEDS)*len(METHODS),
                'thin_post':post, 'true_camera_poses':truth['true_poses'],
                'notes':['Input consists of newly ray-rendered pixels at each resolution, not duplicated old points.',
                         'Known calibrated static-scene poses; raw observed points concatenated before reduction. Registration is not timed.',
                         'Target and background reduced separately; all full observations retained for geometric checks.',
                         'Same frozen K=20, L1 outlier threshold 14mm after every target reduction; incompatibility is a measured failure.',
                         'Fixed 100 clean generated poses for collision timing; 6 fixed poses for full-geometry approach timing.',
                         'Final-pose thin-post probe is reference-clear of all other boxes, colliding with the thin post.',
                         'GPU synchronized stage times; models loaded once, warmed once; CUDA peaks include resident model.',
                         'No observation-envelope timing here; full depth observation cost is a separate stage from point-cloud stress.'],
                'source_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__), ROOT/'experiments/input_degradation.py', ROOT/'experiments/degradation_runner.py', ROOT/'experiments/synthetic_depth.py']}}
    write_json(output/'protocol.json', protocol)
    runner = Runner()
    runner.save_provenance(output)
    torch = runner.torch
    from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
    probe = np.eye(4)
    probe[:3, 3] = [.232, -.05, .08]
    reference_hit = mesh_boxes_collision(runner.parts, probe, [post], 0)
    reference_other = mesh_boxes_collision(runner.parts, probe, truth['boxes'], 0)
    if not reference_hit or reference_other: raise AssertionError('thin-obstacle reference probe is not isolated')
    write_json(output/'reference_probe.json', {'pose':probe, 'post_collision':reference_hit, 'other_boxes_collision':reference_other})
    fixed = {}
    rows = []
    K = np.array([[170.,0,79.5],[0,170.,59.5],[0,0,1]])
    lower, upper = np.asarray(post['lower']), np.asarray(post['upper'])
    for scale in SCALES:
        folder = output/f'scale_{scale}'
        folder.mkdir()
        started = time.perf_counter()
        points, labels, evidence = [], [], {}
        # Scale pixel centers consistently with image resizing: u' = s(u+.5)-.5.
        Ks = K.copy()
        Ks[:2, :2] *= scale
        Ks[:2, 2] = (K[:2, 2]+.5)*scale-.5
        for i, pose in enumerate(truth['true_poses']):
            view, lab = render_boxes(boxes, np.asarray(pose), Ks, shape=(120*scale,160*scale), view_id=f'pressure_{scale}_{i}')
            valid = view.depth_m > 0
            points.append(view.world_image()[valid].astype(np.float32))
            labels.append(lab[valid])
            evidence[f'depth_{i}_m'], evidence[f'labels_{i}'] = view.depth_m.astype(np.float32), lab
        xyz, lab = np.concatenate(points), np.concatenate(labels)
        del points, labels
        target, background = xyz[lab==1], xyz[lab!=1]
        observed_post = xyz[lab==4]
        other_local = xyz[(lab!=1)&(lab!=4)].astype(np.float64)-probe[:3,3]
        other_observed_hit = runner.approach._first_hit(other_local,0) is not None
        if other_observed_hit: raise AssertionError('non-post observations mask obstacle-loss probe')
        del other_local
        np.savez_compressed(folder/'input.npz', points_world_m=xyz, labels=lab, intrinsics=Ks, **evidence)
        generation_seconds = time.perf_counter()-started
        del evidence, xyz, lab
        full_local = (background-probe[:3,3]) @ probe[:3,:3]
        full_probe_hit = runner.approach._first_hit(full_local, 0) is not None
        del full_local
        if scale == 1:
            # Warmup outside measurements; freeze one set of 100 poses per seed.
            warm_target = bounded_points(target,8192,0)
            filtered, _ = point_cloud_outlier_removal(torch.as_tensor(warm_target, device='cuda'))
            for seed in SEEDS:
                fixed[seed] = runner.generator.generate(filtered.cpu().numpy(),100,seed).poses_world_m
            runner.checker.check(bounded_points(background,8192,0), fixed[SEEDS[0]])
            np.savez_compressed(output/'fixed_candidates.npz', **{f'poses_{seed}':p for seed,p in fixed.items()})
            del filtered
        for method, voxel, budget in METHODS:
            for seed in SEEDS:
                destination = folder/method/str(seed)
                destination.mkdir(parents=True)
                profiler = Profiler()
                row = {'scale':scale,'width':160*scale,'height':120*scale,'method':method,'seed':seed,
                       'input_points':len(target)+len(background),'raw_target_points':len(target),
                       'raw_background_points':len(background),'raw_post_points':len(observed_post),
                       'input_generation_and_export_seconds':generation_seconds,'status':'ok',
                       'reference_post_collision':reference_hit,'reference_other_collision':reference_other,
                       'full_geometry_probe_detected':full_probe_hit, 'background_budget':budget,
                       'folder':str(destination.relative_to(output))}
                arrays = {}
                # Release free cached blocks between trials. Resident tensors remain.
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    with profiler.stage('pointcloud_reduction', gpu=True):
                        model_input = bounded_points(target,8192,seed,voxel)
                        bg = bounded_points(background,budget,seed+10,voxel)
                    arrays.update(target_bounded_world_m=model_input, background_collision_world_m=bg)
                    row.update(target_bounded_points=len(model_input),background_collision_points=len(bg),
                               sampled_post_points=int(np.sum(np.all((bg>=lower-2e-6)&(bg<=upper+2e-6),axis=1))))
                    with profiler.stage('small_obstacle_check'):
                        local = (bg-probe[:3,3]) @ probe[:3,:3]
                        sampled_probe = runner.approach._first_hit(local,0) is not None
                    row['sampled_geometry_probe_detected'] = sampled_probe
                    row['sampling_lost_observed_obstacle'] = bool(full_probe_hit and not sampled_probe)
                    with profiler.stage('fixed_collision_check',gpu=True):
                        arrays['fixed_final_clear'] = runner.checker.check(bg,fixed[seed])
                    with profiler.stage('full_approach_check'):
                        full_results = runner.approach.check(background,target,fixed[seed][:6])
                    arrays['fixed_full_approach_clear'] = np.asarray([r['state']=='clear' for r in full_results])
                    with profiler.stage('pointcloud_filter',gpu=True):
                        if len(model_input)<100: raise ValueError(f'only {len(model_input)} target points before filtering')
                        gpu = torch.as_tensor(model_input, dtype=torch.float32, device='cuda')
                        filtered, outliers = point_cloud_outlier_removal(gpu,threshold=.014,K=20)
                        model_points = filtered.cpu().numpy()
                        row['target_model_points'] = len(model_points)
                        arrays['target_model_world_m'] = model_points
                        del gpu, filtered, outliers
                        if len(model_points)<100: raise ValueError(f'only {len(model_points)} target points after fixed outlier filter')
                    with profiler.stage('grasp_generation',gpu=True):
                        grasps = runner.generator.generate(model_points,100,seed)
                    decisions = initial_decisions(grasps,.8)
                    eligible = collision_eligible_indices(decisions)
                    with profiler.stage('collision_check',gpu=True):
                        clear = runner.checker.check(bg,grasps.poses_world_m[eligible])
                        apply_collision_decisions(decisions,eligible,clear,len(bg)>0)
                    rank_candidates(decisions,grasps.scores)
                    row.update(candidate_count=len(grasps.scores),eligible=len(eligible),final_sampled_clear=int(clear.sum()))
                    arrays.update(poses_world_m=grasps.poses_world_m,scores=grasps.scores)
                    # These records deliberately stop at sampled final geometry:
                    # no implication of full-path/observation planning acceptance.
                    write_json(destination/'final_only_candidates.json', {'scope':'sampled final pose only; not planning authorization',
                                                                         'decisions':decisions})
                except (RuntimeError, ValueError) as exc:
                    row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                    if isinstance(exc,torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
                performance = profiler.summary()
                row.update(preprocess_seconds=stage_value(profiler,['pointcloud_reduction','pointcloud_filter']),
                           inference_seconds=stage_value(profiler,['grasp_generation']),
                           collision_seconds=stage_value(profiler,['collision_check']),
                           fixed_100_collision_seconds=stage_value(profiler,['fixed_collision_check']),
                           full_6_approach_seconds=stage_value(profiler,['full_approach_check']),
                           peak_cuda_allocated_mib=performance['peak_allocated_mib'],peak_cuda_reserved_mib=performance['peak_reserved_mib'])
                write_json(destination/'performance.json',performance)
                write_json(destination/'evaluation.json',row)
                np.savez_compressed(destination/'arrays.npz',**arrays)
                rows.append(row)
                save_csv(output/'summary.csv',rows)
                print(f"Scale {scale} {method} seed={seed}: {row['status']}, N={row['input_points']}, post={row.get('sampled_post_points')}/{row['raw_post_points']}, lost={row.get('sampling_lost_observed_obstacle')}, preprocess={row['preprocess_seconds']}",flush=True)
        del target,background,observed_post,view
        gc.collect()
    write_json(output/'summary.json',{'status':'complete','runs':len(rows),'successful_runs':sum(r['status']=='ok' for r in rows),
                                    'max_input_points':max(r['input_points'] for r in rows),
                                    'sampling_obstacle_losses':sum(r.get('sampling_lost_observed_obstacle',False) for r in rows),
                                    'failure_examples':[r for r in rows if r['status']!='ok' or r.get('sampling_lost_observed_obstacle')]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/visibility_fusion/prehardware_v2_20261005')
    args=parser.parse_args()
    output=(args.output or ROOT/'runs/pointcloud_stress'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if output.exists() and any(output.iterdir()): parser.error('output must be empty')
    output.mkdir(parents=True,exist_ok=True)
    try:
        run(output,args.source.resolve())
    except Exception as exc:
        import traceback
        write_json(output/'failure.json',{'error':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
