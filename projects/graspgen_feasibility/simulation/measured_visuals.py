"""Overlay measured colored PLY points onto a PyBullet camera image.

The measured points are visual evidence; separate simple collision bodies are
required for stable rigid-body dynamics. Unseen surfaces are never fabricated.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as p
from .scene_spec import crop_mask, object_rotation


PLY_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('view', 'u1')])


def load_measured_ply(path, include_labels=False):
    with Path(path).open('rb') as stream:
        header = bytearray()
        while not header.endswith(b'end_header\n'):
            line = stream.readline()
            if not line:
                raise ValueError('Incomplete PLY header')
            header.extend(line)
            if len(header) > 16384:
                raise ValueError('PLY header too long')
        header = header.decode('ascii')
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError('Expected measured binary little-endian PLY')
        count = int(next(line.split()[2] for line in header.splitlines()
                         if line.startswith('element vertex ')))
        data = np.frombuffer(stream.read(), dtype=PLY_DTYPE)
    if len(data) != count:
        raise ValueError('PLY vertex count does not match header')
    xyz = np.column_stack([data[name] for name in ('x', 'y', 'z')]).astype(np.float64)
    rgb = np.column_stack([data[name] for name in ('red', 'green', 'blue')]).astype(np.uint8)
    if include_labels:
        return xyz, rgb, data['view'].copy()
    return xyz, rgb


def project(xyz, view, projection, width, height):
    view = np.asarray(view, dtype=np.float64).reshape((4, 4), order='F')
    projection = np.asarray(projection, dtype=np.float64).reshape((4, 4), order='F')
    hom = np.column_stack((xyz, np.ones(len(xyz))))
    clip = hom @ view.T @ projection.T
    valid = clip[:, 3] > 0
    ndc = clip[:, :3] / np.where(valid[:, None], clip[:, 3:4], 1)
    pixel = np.column_stack(((ndc[:, 0] + 1) * width / 2,
                             (1 - ndc[:, 1]) * height / 2,
                             (ndc[:, 2] + 1) / 2))
    valid &= (pixel[:, 0] >= 1) & (pixel[:, 0] < width - 1)
    valid &= (pixel[:, 1] >= 1) & (pixel[:, 1] < height - 1)
    valid &= (pixel[:, 2] >= 0) & (pixel[:, 2] < 1)
    return pixel, valid


class MeasuredVisuals:
    def __init__(self, ply_path, geometry, offset, target='cup'):
        xyz, rgb, labels = load_measured_ply(ply_path, include_labels=True)
        objects = geometry['objects']
        item = objects[target]
        def visible_mask(obj):
            return (labels == obj['visual_label'] if 'visual_label' in obj
                    else crop_mask(xyz, obj))
        is_target = visible_mask(item)
        use_static = np.zeros(len(xyz), dtype=bool)
        self.object_point_counts = {target: int(is_target.sum())}
        for name, other in objects.items():
            if name != target:
                mask = visible_mask(other) & ~is_target
                use_static |= mask
                self.object_point_counts[name] = int(mask.sum())
        self.static_xyz = xyz[use_static] + offset
        self.static_rgb = rgb[use_static]
        self.target_world_xyz = xyz[is_target] + offset
        self.target_grasp_world_xyz = (xyz[is_target & crop_mask(xyz, item)] + offset
                                       if item.get('grasp_points_proxy_crop') else self.target_world_xyz)
        self.target_local_xyz = (xyz[is_target] - item['center_table_m']) @ object_rotation(item)
        self.target_rgb = rgb[is_target]
        # Compatibility for archived cup-only callers.
        self.cup_local_xyz = self.target_local_xyz
        self.cup_rgb = self.target_rgb
        self.source_points = len(xyz)
        self.target_points = int(is_target.sum())
        self.cup_points = self.target_points
        self.box_points = self.object_point_counts.get('box', 0)

    def draw(self, image, depth_buffer, view, projection, target_body):
        width, height = image.size
        base, quat = p.getBasePositionAndOrientation(target_body)
        rotation = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        target_xyz = self.target_local_xyz @ rotation.T + np.asarray(base)
        xyz = np.vstack((self.static_xyz, target_xyz))
        rgb = np.vstack((self.static_rgb, self.target_rgb))
        pixel, valid = project(xyz, view, projection, width, height)
        u = pixel[:, 0].astype(np.int32)
        v = pixel[:, 1].astype(np.int32)
        z = pixel[:, 2]
        scene_depth = np.asarray(depth_buffer).reshape(height, width)
        valid &= z <= scene_depth[np.clip(v, 0, height - 1), np.clip(u, 0, width - 1)] + .00015
        indices = np.flatnonzero(valid)
        # Draw far to near so the original RGB surface is visible without
        # turning the unmeasured backs into a textured solid object.
        indices = indices[np.argsort(z[indices])[::-1]]
        drawing = ImageDraw.Draw(image)
        static_count = len(self.static_xyz)
        for index in indices:
            x, y = int(u[index]), int(v[index])
            color = tuple(int(c) for c in rgb[index])
            if index >= static_count:
                drawing.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
            else:
                drawing.point((x, y), fill=color)
        return image
