"""Aggregate saved stress trials without rerunning inference or changing outcomes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from offline_core.export import write_json
from experiments.degradation_runner import save_csv


def summarize(root):
    with (root/'summary.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    groups=[]
    for scale,method in dict.fromkeys((int(r['scale']),r['method']) for r in rows):
        selected=[r for r in rows if int(r['scale'])==scale and r['method']==method]
        group={'scale':scale,'method':method,'input_points':int(selected[0]['input_points']),
               'successful_runs':sum(r['status']=='ok' for r in selected),'runs':len(selected),
               'sampling_obstacle_losses':sum(r['sampling_lost_observed_obstacle']=='True' for r in selected),
               'post_points_min':min(int(r['sampled_post_points']) for r in selected),
               'post_points_max':max(int(r['sampled_post_points']) for r in selected)}
        for field in ['preprocess_seconds','inference_seconds','collision_seconds','fixed_100_collision_seconds','full_6_approach_seconds']:
            values=[float(r[field]) for r in selected if r[field]]
            for name,fn in [('median',np.median),('min',min),('max',max)]:
                group[f'{field}_{name}']=float(fn(values)) if values else None
        perf=[json.loads((root/r['folder']/'performance.json').read_text()) for r in selected]
        group['max_peak_cuda_allocated_mib']=max(p['peak_allocated_mib'] for p in perf)
        for name,stages in [('preprocess',['pointcloud_reduction','pointcloud_filter']),
                            ('inference',['grasp_generation']),('collision',['collision_check']),
                            ('fixed_collision',['fixed_collision_check'])]:
            values=[p['stages'][s]['peak_allocated_mib'] for p in perf for s in stages if s in p['stages']]
            group[name+'_peak_cuda_allocated_mib']=max(values) if values else None
        groups.append(group)
    write_json(root/'resource_summary.json',groups)
    save_csv(root/'resource_summary.csv',groups)
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    summarize(parser.parse_args().root)
