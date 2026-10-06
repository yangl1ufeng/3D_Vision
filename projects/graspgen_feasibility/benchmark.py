"""30 synchronized GPU inference trials on one official real-object sample."""
from pathlib import Path
import csv
import gc
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'third_party/GraspGen'))
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
import numpy as np
import torch
import pynvml

class DeviceMonitor:
    """Whole-device usage, including desktop/other processes, sampled at 20 ms."""
    def __enter__(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self
    def poll(self):
        while not self.stop.is_set():
            try:
                self.samples.append(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20)
            except Exception: pass
            self.stop.wait(.02)
    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
    @property
    def peak(self): return max(self.samples, default=0.)

def seed(n):
    random.seed(n); np.random.seed(n); torch.manual_seed(n); torch.cuda.manual_seed_all(n)

def measure(function):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()/2**20
    with DeviceMonitor() as monitor:
        started = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
    stats = {'seconds':seconds,
             'allocated_before_mib':before,
             'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
             'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
             'device_peak_used_mib':monitor.peak}
    return value, stats

def build_sampler():
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    return GraspGenSampler(load_grasp_cfg(str(ROOT/'checkpoints/graspgen_robotiq_2f_140.yml')))

def sample_all(sampler, pc, count):
    # Exactly one diffusion call: no automatic retry, no top-k=100 truncation.
    output = sampler.sample(pc, threshold=0., num_grasps=count, remove_outliers=True)
    poses, scores = output[:2]
    with torch.inference_mode():
        poses[:,3,3] = 1.  # Also present in the author's public run_inference wrapper.
    return poses, scores

def main():
    print('Loading official Robotiq model', flush=True)
    sampler, load_stats = measure(build_sampler)
    asset = ROOT/'data/sample_data/real_object_pc/1740787710_319213.json'
    content = json.loads(asset.read_text())
    pc = torch.as_tensor(np.asarray(content['pc'],dtype=np.float32),device='cuda')
    rows = []
    warmups = []
    for count in [50,100,200]:
        seed(9999)
        output, warm_stats = measure(lambda:sample_all(sampler,pc,count))
        warm_stats.update(candidates=count,raw_output=len(output[0]))
        warmups.append(warm_stats)
        del output
        for repeat in range(10):
            seed(1000+repeat)
            output, stats = measure(lambda:sample_all(sampler,pc,count))
            poses, scores = output
            row = dict(candidates=count,repeat=repeat+1,seed=1000+repeat,
                       raw_output=int(len(poses)),
                       above_08=int((scores>=.8).sum().item()),
                       finite_output=bool(torch.isfinite(poses).all().item() and torch.isfinite(scores).all().item()),
                       min_score=float(scores.min()),max_score=float(scores.max()),**stats)
            rows.append(row)
            print(json.dumps(row),flush=True)
            np.savez_compressed(ROOT/f'runs/grasps_{count}_{repeat+1:02}.npz',
                                poses=poses.cpu().numpy(),scores=scores.cpu().numpy())
            del poses,scores,output
            with (ROOT/'runs/inference_trials.csv').open('w',newline='') as out:
                writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[]
    for count in [50,100,200]:
        subset=[r for r in rows if r['candidates']==count]
        groups.append({'candidates':count,'runs':len(subset),
                       'mean_seconds':statistics.mean(r['seconds'] for r in subset),
                       'median_seconds':statistics.median(r['seconds'] for r in subset),
                       'min_seconds':min(r['seconds'] for r in subset),
                       'max_seconds':max(r['seconds'] for r in subset),
                       'max_allocated_mib':max(r['peak_allocated_mib'] for r in subset),
                       'max_reserved_mib':max(r['peak_reserved_mib'] for r in subset),
                       'max_device_used_mib':max(r['device_peak_used_mib'] for r in subset),
                       'mean_above_08':statistics.mean(r['above_08'] for r in subset),
                       'min_above_08':min(r['above_08'] for r in subset),
                       'max_above_08':max(r['above_08'] for r in subset)})
    report={'device':torch.cuda.get_device_name(0),'total_memory_mib':torch.cuda.get_device_properties(0).total_memory/2**20,
            'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,
            'upstream_commit':subprocess.check_output(['git','-C',str(ROOT/'third_party/GraspGen'),'rev-parse','HEAD'],text=True).strip(),
            'sample':str(asset.relative_to(ROOT)),'input_points':len(pc),'model':'official Robotiq-2F-140 PointNet++',
            'diffusion_steps':sampler.cfg.diffusion.num_diffusion_iters_eval,
            'protocol':'One warm-up per budget excluded; ten seeded calls each; FP32; outlier removal included; no retries; no top-k truncation; no visualization/collision/segmentation in inference-only timings.',
            'memory_note':'PyTorch peaks include resident model; NVML is whole-device sampled usage including desktop, not process memory.',
            'loading':load_stats,'warmups':warmups,'groups':groups,'trials':rows}
    (ROOT/'runs/inference_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(groups,indent=2),flush=True)

if __name__=='__main__': main()
