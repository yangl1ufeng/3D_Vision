"""Convert the downloaded official scene without guessing camera calibration."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .config import ROOT
from .scene_io import file_hash


def prepare_official_sample():
    source = ROOT / 'data/sample_data/real_scene_pc/1745619598_467125.json'
    if not source.is_file():
        raise FileNotFoundError(f'Official sample must be downloaded first: {source}')
    dest = ROOT / 'data/offline_official'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest / 'scene.json'
    source_hash = file_hash(source)
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous.get('source', {}).get('sha256') == source_hash and all(
            (dest / name).is_file() for name in ['rgb.png', 'xyz_world_m.npy', 'valid.npy']
        ):
            return manifest
    original = json.loads(source.read_text(encoding='utf-8'))
    rgb = np.asarray(original['scene_info']['img_color'], dtype=np.uint8)
    # The author's field pc_color contains aligned world XYZ, not RGB colors.
    xyz = np.asarray(original['scene_info']['pc_color'], dtype=np.float32).reshape(*rgb.shape[:2], 3)
    depth = np.asarray(original['scene_info']['img_depth'])
    valid = np.isfinite(xyz).all(-1) & np.isfinite(depth) & (depth > 0)
    reference = np.asarray(original['object_info']['pc'], dtype=np.float32)
    finite_ids = np.flatnonzero(np.isfinite(xyz.reshape(-1, 3)).all(-1))
    distances, indices = cKDTree(xyz.reshape(-1, 3)[finite_ids]).query(reference)
    if distances.max() > 1e-5:
        raise ValueError('Official object points do not match scene pixels')
    rows, cols = np.unravel_index(finite_ids[indices], rgb.shape[:2])
    box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
    Image.fromarray(rgb).save(dest / 'rgb.png')
    np.save(dest / 'xyz_world_m.npy', xyz)
    np.save(dest / 'valid.npy', valid)
    data = {
        'schema_version': 1, 'scene_id': 'official_1745619598_467125', 'rgb': 'rgb.png',
        'geometry': {'kind': 'xyz', 'path': 'xyz_world_m.npy', 'unit': 'm',
                     'frame': 'world', 'aligned_to_rgb': True, 'valid_mask': 'valid.npy'},
        'frames': {'world': 'official_scene_world'},
        'target': {'id': 'blue_bowl', 'box_xyxy': box,
                   'prompt_origin': 'Box derived from exact pixel correspondence of official segmented object points; simulates user selection. Reference points are not used as inference input.'},
        'source': {'path': str(source), 'sha256': source_hash,
                   'note': 'Uses supplied world XYZ; does not infer unknown depth units, intrinsics, or extrinsics.'}}
    manifest.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return manifest
