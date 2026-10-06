"""Runtime setup and exact model/source provenance."""
import importlib.metadata
import os
from pathlib import Path
import random
import subprocess
import sys
from .config import ROOT, project_path
from .scene_io import file_hash


def initialize(cfg):
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(ROOT / '.cache/torch_extensions'))
    sys.path.insert(0, str(project_path(cfg['grasp']['repo'])))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; official GraspGen inference needs the GPU environment')
    torch.cuda.set_device(0)
    return torch


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repo_info(repo):
    repo = Path(repo)
    command = ['git', '-c', f'safe.directory={repo.as_posix()}', '-c', 'core.excludesFile=', '-C', str(repo)]
    try:
        commit = subprocess.check_output(command + ['rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        import hashlib
        diff = subprocess.check_output(command + ['diff', '--no-ext-diff'], stderr=subprocess.DEVNULL)
        return {'commit': commit, 'modified': bool(diff), 'diff_sha256': hashlib.sha256(diff).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'modified': None, 'note': 'Git provenance unavailable'}


def provenance(cfg, grasp_backend, segmentation_backend):
    import torch
    versions = {}
    for name in ['torch', 'numpy', 'scipy', 'trimesh', 'diffusers', 'huggingface-hub', 'transformers', 'hydra-core', 'pointnet2_ops']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    weights = {}
    for name, path in grasp_backend.weight_paths.items():
        weights[name] = {'path': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}
    if segmentation_backend is not None:
        path = segmentation_backend.checkpoint
        weights['sam2'] = {'path': str(path), 'sha256': file_hash(path), 'bytes': path.stat().st_size}
    return {'python': sys.version, 'executable': sys.executable, 'packages': versions,
            'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'gpu_total_memory_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
            'graspgen_source': repo_info(project_path(cfg['grasp']['repo'])),
            'sam2_source': repo_info(project_path(cfg['segmentation']['sam2_repo'])) if segmentation_backend else None,
            'weights': weights,
            'core_source_sha256': {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
