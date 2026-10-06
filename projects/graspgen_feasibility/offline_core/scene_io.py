"""Load one aligned view from an explicit JSON manifest; normalize into meters/world."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import depth_to_xyz, meters_per_unit, transform_points, validate_transform
from .types import Scene
from .observation import DepthView


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_array(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        return np.load(path, allow_pickle=False)
    return np.asarray(Image.open(path))


def relative_asset(base, name):
    p = Path(name)
    p = p.resolve() if p.is_absolute() else (base / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f'input asset missing: {p}')
    return p


def load_scene(manifest_path):
    path = Path(manifest_path).resolve()
    m = json.loads(path.read_text(encoding='utf-8'))
    if m.get('schema_version') != 1:
        raise ValueError('unsupported input schema_version')
    for key in ['scene_id', 'rgb', 'geometry', 'frames', 'target']:
        if key not in m:
            raise ValueError(f'input manifest missing {key}')
    if not isinstance(m['scene_id'], str) or not m['scene_id'].strip():
        raise ValueError('scene_id must be a nonempty string')
    if not isinstance(m['target'].get('id'), str) or not m['target']['id'].strip():
        raise ValueError('target.id must be a nonempty string')
    if not isinstance(m['frames'].get('world'), str) or not m['frames']['world'].strip():
        raise ValueError('frames.world must name the output coordinate frame')
    base = path.parent
    rgb_path = relative_asset(base, m['rgb'])
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    H, W = rgb.shape[:2]
    geometry = m['geometry']
    if geometry.get('aligned_to_rgb') is not True:
        raise ValueError('geometry must explicitly be aligned to RGB pixels')
    factor = meters_per_unit(geometry.get('unit'), geometry.get('meters_per_unit'))
    data_path = relative_asset(base, geometry['path'])
    data = np.asarray(load_array(data_path), dtype=np.float32)
    assets = {'manifest': str(path), 'rgb': str(rgb_path), 'geometry': str(data_path)}
    T = m['frames'].get('T_world_camera')
    if T is not None:
        T = validate_transform(T)
    if geometry['kind'] == 'depth':
        if data.shape != (H, W):
            raise ValueError('depth and RGB resolution must match')
        if geometry.get('depth_convention') != 'z' or geometry.get('pinhole_rectified') is not True:
            raise ValueError('depth requires depth_convention=z and pinhole_rectified=true')
        if m['frames'].get('camera_convention') != 'opencv' or T is None:
            raise ValueError('depth requires opencv camera convention and explicit T_world_camera')
        depth_m = data * factor
        valid = np.isfinite(depth_m) & (depth_m > 0)
        xyz_camera = depth_to_xyz(depth_m, geometry['intrinsics'])
        xyz_world = transform_points(xyz_camera, T).astype(np.float32)
    elif geometry['kind'] == 'xyz':
        if data.shape != (H, W, 3):
            raise ValueError('XYZ must have shape HxWx3 aligned with RGB')
        xyz = data * factor
        valid = np.isfinite(xyz).all(-1)
        frame = geometry.get('frame')
        if frame == 'camera':
            if T is None or m['frames'].get('camera_convention') != 'opencv':
                raise ValueError('camera XYZ requires opencv convention and T_world_camera')
            valid &= xyz[..., 2] > 0
            xyz_world = transform_points(xyz, T).astype(np.float32)
        elif frame == 'world':
            xyz_world = xyz
        else:
            raise ValueError('XYZ frame must explicitly be camera or world')
    else:
        raise ValueError('geometry.kind must be depth or xyz')
    if geometry.get('valid_mask'):
        valid_path = relative_asset(base, geometry['valid_mask'])
        supplied = load_array(valid_path)
        if supplied.shape != (H, W):
            raise ValueError('geometry.valid_mask resolution must match RGB')
        if not np.isfinite(supplied).all() or not np.isin(supplied, [0, 1, 255]).all():
            raise ValueError('geometry.valid_mask must contain only 0/1 or 0/255')
        valid &= supplied.astype(bool)
        assets['valid_mask'] = str(valid_path)
    valid &= np.isfinite(xyz_world).all(-1)
    if not valid.any():
        raise ValueError('scene contains no valid geometry')
    target = dict(m['target'])
    if target.get('mask'):
        target['mask_path'] = str(relative_asset(base, target['mask']))
        assets['target_mask'] = target['mask_path']
    if 'box_xyxy' in target:
        box = np.asarray(target['box_xyxy'], dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('target.box_xyxy must contain four finite pixel coordinates')
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 < W and 0 <= y0 < y1 < H):
            raise ValueError('target.box_xyxy must be a nonempty box within the RGB image')
    if not target.get('mask') and 'box_xyxy' not in target:
        raise ValueError('target needs box_xyxy for SAM2, or an explicit provided mask')
    metadata = {'input_manifest': m, 'asset_paths': assets,
                'asset_sha256': {key: file_hash(value) for key, value in assets.items()},
                'geometry_source': geometry['kind'], 'input_unit': geometry['unit'],
                'meters_per_unit': factor, 'output_unit': 'm', 'world_frame': m['frames']['world'],
                'pose_convention': 'T_world_graspgen_gripper maps gripper-local column vectors to world',
                'timestamp': m.get('timestamp'), 'valid_geometry_pixels': int(valid.sum())}
    depth_view = None
    if geometry['kind'] == 'depth':
        depth_view = DepthView(m['scene_id'], np.where(valid, depth_m, 0), geometry['intrinsics'], T)
    metadata['calibrated_depth_available'] = depth_view is not None
    return Scene(m['scene_id'], target['id'], rgb, xyz_world, valid, target, metadata, path, depth_view)
