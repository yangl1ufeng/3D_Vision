"""Independently audit preparation 5/6 saved inputs, outcomes and resource records."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from offline_core.config import ROOT
from offline_core.scene_io import file_hash,load_scene
from offline_core.export import write_json
from experiments.input_degradation import mask_metrics


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))


def check_protocol(root):
    protocol=read(root/'protocol.json')
    for name,value in protocol['source_sha256'].items(): assert file_hash(ROOT/name)==value,name
    return protocol


def audit_robustness(root):
    protocol=check_protocol(root)
    table=rows(root/'summary.csv')
    assert len(table)==protocol['planned_synthetic_runs']+protocol['planned_prompt_runs']==159
    clean_masks={layout:np.load(root/'synthetic'/layout/'clean_0/1000/input/mask.npy') for layout in ['A','B']}
    clean_masks['real_prompt']=np.load(root/'prompts/box_clean_0/sam2_mask.npy')
    failures=0
    seen=set()
    for row in table:
        folder=root/row['folder']
        result=read(folder/'result.json')
        evaluation=read(folder/'evaluation.json')
        key=(row['layout'],row['family'],row['level'],row['seed'])
        assert key not in seen
        seen.add(key)
        assert result['status']==row['status']
        assert evaluation['material_degradation']==(row['material_degradation']=='True')
        if row['layout']!='real_prompt':
            scene=load_scene(folder/'input/scene.json')
            if row['family'] in ['dropout','hole']:
                baseline=load_scene(root/'synthetic'/row['layout']/f"clean_0/{row['seed']}/input/scene.json")
                keep=scene.depth_view.depth_m>0
                np.testing.assert_array_equal(scene.depth_view.depth_m[keep],baseline.depth_view.depth_m[keep])
        with np.load(folder/'arrays.npz') as data:
            metrics=mask_metrics(data['segmentation_mask'],clean_masks[row['layout']])
            assert abs(metrics['mask_iou']-float(row['mask_iou']))<1e-12
            assert int(np.sum(data['segmentation_mask']&data['valid_mask']))==int(row['target_raw_points'])
            if result['status']=='failed':
                failures+=1
                assert result['error'] and 'pipeline_failure' in row['degradation_reasons']
                assert not (folder/'candidates.json').exists()
                continue
            candidates=read(folder/'candidates.json')['candidates']
            assert len(data['poses_world_m'])==len(data['scores'])==len(candidates)==100
            assert int(data['accepted'].sum())==int(row['accepted'])
            np.testing.assert_allclose(data['poses_world_m'],[d['T_world_graspgen_gripper'] for d in candidates],atol=0,rtol=0)
            for candidate,accepted in zip(candidates,data['accepted']):
                assert bool(accepted)==candidate['accepted']
                if accepted: assert not candidate['reasons']
            if 'reference_collision' in data:
                ref=data['reference_collision']
                assert np.sum(data['geometric_accepted']&(ref==1))==int(row['false_accept_geometry'])
                assert np.sum(data['accepted']&(ref==1))==int(row['false_accept_final'])
                assert np.sum(ref>=0)==int(row['eligible'])
            if 'fixed_poses_world_m' in data:
                expected=np.mean(data['fixed_geometric_clear']!=data['fixed_clean_clear'])
                assert abs(expected-float(row['fixed_decision_flip_fraction']))<1e-12
        stages=result['performance']['stages']
        assert all(s['status']=='ok' and s['seconds']>=0 for s in stages.values())
    assert failures==read(root/'summary.json')['pipeline_failures']
    return {'status':'passed','runs':len(table),'preserved_pipeline_failures':failures}


def audit_stress(root):
    protocol=check_protocol(root)
    table=rows(root/'summary.csv')
    assert len(table)==protocol['planned_runs']==48
    lower,upper=np.asarray(protocol['thin_post']['lower']),np.asarray(protocol['thin_post']['upper'])
    counts={}
    for scale in protocol['scales']:
        with np.load(root/f'scale_{scale}/input.npz') as data:
            n=len(data['points_world_m'])
            assert n==sum(np.sum(data[f'depth_{i}_m']>0) for i in range(3))
            assert np.sum(data['labels']==4)==sum(np.sum(data[f'labels_{i}']==4) for i in range(3))
            counts[scale]=(n,int(np.sum(data['labels']==4)))
    assert all(counts[b][0]>counts[a][0] for a,b in zip(protocol['scales'][:-1],protocol['scales'][1:]))
    losses=failures=0
    for row in table:
        folder=root/row['folder']
        n,post=counts[int(row['scale'])]
        assert n==int(row['input_points']) and post==int(row['raw_post_points'])
        with np.load(folder/'arrays.npz') as data:
            background=data['background_collision_world_m']
            assert len(background)<=int(row['background_budget'])
            assert len(data['target_bounded_world_m'])<=protocol['target_budget']
            measured=int(np.sum(np.all((background>=lower-2e-6)&(background<=upper+2e-6),axis=1)))
            assert measured==int(row['sampled_post_points'])
            assert len(data['fixed_final_clear'])==100 and len(data['fixed_full_approach_clear'])==6
            if row['status']=='ok':
                assert len(data['poses_world_m'])==len(data['scores'])==100
            else:
                failures+=1
                assert row['error'] and not (folder/'final_only_candidates.json').exists()
        assert row['reference_post_collision']=='True' and row['reference_other_collision']=='False'
        loss=row['full_geometry_probe_detected']=='True' and row['sampled_geometry_probe_detected']=='False'
        assert loss==(row['sampling_lost_observed_obstacle']=='True')
        losses+=loss
        perf=read(folder/'performance.json')
        assert all(s['seconds']>=0 for s in perf['stages'].values())
        for name in ['pointcloud_reduction','fixed_collision_check']:
            stage=perf['stages'][name]
            assert stage['peak_allocated_mib']>0 and stage['status']=='ok'
        assert abs(float(row['peak_cuda_allocated_mib'])-perf['peak_allocated_mib'])<1e-9
    summary=read(root/'summary.json')
    assert losses==summary['sampling_obstacle_losses']
    assert len(table)-failures==summary['successful_runs']
    return {'status':'passed','runs':len(table),'preserved_pipeline_failures':failures,'observed_obstacle_losses':losses,'input_counts':counts}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness',type=Path)
    parser.add_argument('--stress',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    results={}
    if args.robustness: results['robustness']=audit_robustness(args.robustness)
    if args.stress: results['stress']=audit_stress(args.stress)
    write_json(args.report,results)
    print(json.dumps(results,indent=2))
