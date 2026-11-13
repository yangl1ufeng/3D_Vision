"""Validated, snapshot-able input for measured scene planning and execution."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT / 'configs/measured_table.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_rotation(item):
    a = float(item.get('yaw_table_rad', 0))
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def object_height(item):
    return item['dimensions_m'][2] if item['shape'] == 'box' else item['height_m']


def crop_mask(xyz, item):
    local = (xyz - np.asarray(item['center_table_m'])) @ object_rotation(item)
    h = object_height(item)
    if item['shape'] in ('box', 'voxel_compound_proxy'):
        keep = np.all(np.abs(local[:, :2]) < np.asarray(item['dimensions_m'][:2])/2 + .015, axis=1)
        floor = .02
    else:
        keep = np.linalg.norm(local[:, :2], axis=1) < item['radius_m'] + .02
        floor = .015
    return keep & (local[:, 2] > -h/2 + floor) & (local[:, 2] < h/2 + .025)


class SceneSpec:
    def __init__(self, data, base=PROJECT, target=None):
        d = copy.deepcopy(data)
        if d.get('version') != 1:
            raise ValueError('Scene version must be 1')
        for key in ('geometry_path', 'pointcloud_path'):
            value = Path(d[key])
            d[key] = str((base / value).resolve() if not value.is_absolute() else value.resolve())
        if 'geometry' not in d:
            d['geometry'] = json.loads(Path(d['geometry_path']).read_text(encoding='utf-8'))
        actual = digest(d['pointcloud_path'])
        if d.get('pointcloud_sha256', actual) != actual:
            raise ValueError('Scene point cloud changed since snapshot')
        d['pointcloud_sha256'] = actual
        if target is not None:
            d['target'] = target
        self.data = d
        self.geometry = d['geometry']
        self.target = d['target']
        objects = self.geometry['objects']
        if self.target not in objects:
            raise ValueError(f'Unknown target {self.target}; available: {list(objects)}')
        observation = d.get('target_observation')
        if observation is not None:
            if observation['target'] != self.target:
                raise ValueError('Clicked mask belongs to another target; select it again')
            for key, hash_key in [('points_table_path', 'points_sha256'), ('mask_path', 'mask_sha256'),
                                   ('provenance_path', 'provenance_sha256')]:
                value = Path(observation[key])
                value = (base/value).resolve() if not value.is_absolute() else value.resolve()
                observation[key] = str(value)
                if digest(value) != observation[hash_key]:
                    raise ValueError(f'Clicked target asset changed: {key}')
        if any(name in {'table', 'pedestal'} for name in objects):
            raise ValueError('Object IDs table and pedestal are reserved')
        def vector(value, n, label, positive=False):
            array = np.asarray(value, float)
            if array.shape != (n,) or not np.isfinite(array).all() or (positive and np.any(array <= 0)):
                raise ValueError(f'Invalid {label}')
            return array
        self.offset = vector(d['table_to_robot_translation_m'], 3, 'translation')
        self.drop_xy = vector(d['drop_xy_robot_m'], 2, 'drop zone')
        for name in ('table', 'pedestal'):
            vector(d[name]['center_robot_m'], 3, name)
            vector(d[name]['half_extents_m'], 3, name, True)
        self.table_top = d['table']['center_robot_m'][2] + d['table']['half_extents_m'][2]
        for name, item in objects.items():
            vector(item['center_table_m'], 3, name)
            if item['shape'] == 'box':
                vector(item['dimensions_m'], 3, name, True)
            elif item['shape'] == 'solid_cylinder_proxy':
                vector([item['radius_m'], item['height_m']], 2, name, True)
            elif item['shape'] == 'voxel_compound_proxy':
                vector(item['dimensions_m'], 3, name, True)
                vector([item['height_m']], 1, name, True)
                if not 1 <= len(item['parts']) <= 100:
                    raise ValueError('Invalid compound part count')
                for part in item['parts']:
                    vector(part['center_local_m'], 3, name)
                    vector(part['half_extents_m'], 3, name, True)
            else:
                raise ValueError(f'Unsupported shape for {name}: {item["shape"]}')
            if not np.isfinite(item.get('yaw_table_rad', 0)):
                raise ValueError('Invalid yaw')
        # Placement footprint must fit the configured support table.
        item = objects[self.target]
        radius = (max(item['dimensions_m'][:2])/2 if item['shape'] in ('box', 'voxel_compound_proxy') else item['radius_m'])
        table = d['table']
        if np.any(np.abs(self.drop_xy - table['center_robot_m'][:2]) + radius > np.asarray(table['half_extents_m'][:2])):
            raise ValueError('Drop zone does not fit on the configured table')
        physics = d.setdefault('target_physics', {'mass_kg': .09, 'lateral_friction': 1.2})
        vector([physics['mass_kg'], physics['lateral_friction']], 2, 'target physics', True)
        self.fingerprint = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding='utf-8')


def load_scene(path=DEFAULT_SCENE, target=None):
    path = Path(path).resolve()
    return SceneSpec(json.loads(path.read_text(encoding='utf-8')), path.parent, target)


def legacy_scene(geometry_path=None, geometry=None):
    d = load_scene().data
    if geometry_path is not None:
        d['geometry_path'] = str(Path(geometry_path).resolve())
        d['pointcloud_path'] = str(Path(geometry_path).resolve().parent/'desk_box_cup_measured.ply')
        d.pop('pointcloud_sha256', None)
        d.pop('geometry', None)
    if geometry is not None:
        d['geometry'] = geometry
    return SceneSpec(d)
