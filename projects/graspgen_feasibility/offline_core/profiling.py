"""Non-nested stage timing; CUDA synchronization and optional device-wide NVML."""
from contextlib import contextmanager
import threading
import time


class DeviceMonitor:
    def __enter__(self):
        self.samples = []
        self.error = None
        self.thread = None
        self.stop = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.thread = threading.Thread(target=self.poll, daemon=True)
            self.thread.start()
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        return self

    def poll(self):
        while not self.stop.is_set():
            try:
                self.samples.append(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20)
            except Exception as exc:
                self.error = str(exc)
            self.stop.wait(.02)

    def __exit__(self, *args):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
            self.nvml.nvmlShutdown()


class Profiler:
    def __init__(self):
        self.stages = {}
        self.active = False
        self.started = time.perf_counter()

    @contextmanager
    def stage(self, name, gpu=False):
        if self.active or name in self.stages:
            raise ValueError('profiling stages must be unique and non-nested')
        self.active = True
        torch = None
        monitor = None
        record = {}
        if gpu:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            record['allocated_before_mib'] = torch.cuda.memory_allocated() / 2**20
            monitor = DeviceMonitor().__enter__()
        started = time.perf_counter()
        try:
            yield
            if gpu:
                torch.cuda.synchronize()
            record['status'] = 'ok'
        except Exception:
            record['status'] = 'failed'
            raise
        finally:
            record['seconds'] = time.perf_counter() - started
            if gpu:
                record['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
                record['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
                monitor.__exit__()
                record['device_peak_used_mib'] = max(monitor.samples) if monitor.samples else None
                if monitor.error:
                    record['nvml_note'] = monitor.error
            self.stages[name] = record
            self.active = False

    def summary(self):
        gpu = [x for x in self.stages.values() if 'peak_allocated_mib' in x]
        device = [x['device_peak_used_mib'] for x in gpu if x['device_peak_used_mib'] is not None]
        compute = ['segmentation', 'pointcloud', 'grasp_generation', 'score_filter', 'collision_check', 'approach_check', 'observation_check']
        return {'stages': dict(self.stages),
                'compute_stage_seconds': sum(self.stages[n]['seconds'] for n in compute if n in self.stages),
                'measured_stage_seconds': sum(x['seconds'] for x in self.stages.values()),
                'peak_allocated_mib': max((x['peak_allocated_mib'] for x in gpu), default=None),
                'peak_reserved_mib': max((x['peak_reserved_mib'] for x in gpu), default=None),
                'device_peak_used_mib': max(device, default=None),
                'memory_note': 'CUDA peaks include resident models. NVML is whole-device usage including desktop, sampled every 20ms; unavailable values are null, not zero.',
                'timing_note': 'Stages do not nest. Compute excludes model loading, input conversion/IO, provenance hashing, and export. Wall times are reported separately.'}
