"""Point-prompt SAM2 segmentation and measured RGB-D export for existing scenes."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from simulation.scene_spec import SceneSpec, PROJECT, crop_mask, digest

SAM_REPO = PROJECT/'third_party/sam2_click'
SAM_WEIGHTS = PROJECT/'checkpoints/sam2.1_hiera_tiny.pt'


def observation(spec, view_index=0):
    path = Path(spec.geometry['registration'])
    reg = json.loads(path.read_text(encoding='utf-8'))
    if not reg.get('passed'):
        raise ValueError('Saved view registration did not pass its checks')
    if not 0 <= view_index < len(reg['views']):
        raise ValueError('View index is outside the accepted views')
    folder = Path(reg['views'][view_index])
    rgb_path, depth_path = folder/'rgb.png', folder/'depth_m.npy'
    rgb = np.array(Image.open(rgb_path).convert('RGB'), copy=True)
    depth = np.load(depth_path, allow_pickle=False)
    if rgb.shape[:2] != depth.shape:
        raise ValueError('RGB and aligned depth resolution differ')
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    transform = np.asarray(reg['T_reference_table_view_table'][view_index]) @ np.linalg.inv(np.asarray(reg['T_view_camera_table'][view_index]))
    return {'rgb': rgb, 'depth': depth, 'intrinsic': report['aligned_depth_intrinsic'],
            'T_table_camera': transform, 'rgb_path': rgb_path, 'depth_path': depth_path,
            'registration_path': path, 'view_index': view_index}


def measured_points(obs, mask):
    if mask.shape != obs['depth'].shape:
        raise ValueError('Mask must match the original image size')
    depth = obs['depth']
    valid = mask & np.isfinite(depth) & (depth > .2) & (depth < 2.)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    k = obs['intrinsic']
    camera = np.column_stack(((u-k['cx'])*z/k['fx'], (v-k['cy'])*z/k['fy'], z))
    transform = obs['T_table_camera']
    return camera @ transform[:3, :3].T + transform[:3, 3], float(valid.sum()/max(mask.sum(), 1))


def associate_target(points, geometry):
    masks = {name: crop_mask(points, item) for name, item in geometry['objects'].items()}
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    ranked = sorted(counts, key=counts.get, reverse=True)
    if not ranked or counts[ranked[0]] < 100:
        raise ValueError('Selected region has too little depth on a modeled object')
    name = ranked[0]
    if counts[name] / max(len(points), 1) < .6:
        raise ValueError('Selected region contains too much background; refine the points')
    if len(ranked) > 1 and counts[ranked[1]] > .3 * counts[name]:
        raise ValueError('Selected region spans multiple modeled objects; refine the points')
    return name, masks[name], counts


class PointSegmenter:
    def __init__(self):
        if not SAM_WEIGHTS.is_file() or not (SAM_REPO/'sam2').is_dir():
            raise FileNotFoundError('SAM2 code or checkpoint missing')
        source = json.loads((PROJECT/'checkpoints/sam2.1_tiny_source.json').read_text())
        expected = next(f['lfs']['sha256'] for f in source['siblings'] if f['rfilename']==SAM_WEIGHTS.name)
        self.weight_hash = digest(SAM_WEIGHTS)
        if self.weight_hash != expected:
            raise ValueError('SAM2 checkpoint hash does not match the official model record')
        sys.path.insert(0, str(SAM_REPO))
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',
                                           str(SAM_WEIGHTS), device='cuda', apply_postprocessing=False))
        self.image_id = None

    def segment(self, rgb, positive, negative=()):
        torch = self.torch
        if not positive:
            raise ValueError('Click inside an object first')
        prompts = np.asarray(list(positive)+list(negative), dtype=np.float32)
        h, w = rgb.shape[:2]
        if not np.isfinite(prompts).all() or np.any(prompts < 0) or np.any(prompts >= [w, h]):
            raise ValueError('Click coordinates are outside the image')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if self.image_id != id(rgb):
                self.predictor.set_image(rgb)
                self.image_id = id(rgb)
            masks, scores, _ = self.predictor.predict(point_coords=prompts,
                point_labels=np.array([1]*len(positive)+[0]*len(negative)), multimask_output=True)
        for index in np.argsort(-scores):
            mask = masks[index].astype(bool)
            if not np.isfinite(scores[index]) or not 100 <= mask.sum() <= .6*h*w:
                continue
            if not all(mask[int(y), int(x)] for x,y in positive):
                continue
            if any(mask[int(y), int(x)] for x,y in negative):
                continue
            return mask, {'sam2_quality': float(scores[index]), 'mask_index': int(index),
                          'model': 'SAM2.1 Hiera Tiny', 'weights_sha256': self.weight_hash,
                          'positive_points_xy': [list(p) for p in positive],
                          'negative_points_xy': [list(p) for p in negative],
                          'source_commit': subprocess.check_output(['git','-C',str(SAM_REPO),'rev-parse','HEAD'],text=True).strip()}
        raise ValueError('No mask agrees with the clicks; refine the selection')


def preview(rgb, mask):
    out = rgb.copy()
    out[mask] = (.55*out[mask] + .45*np.array([30, 220, 110])).astype(np.uint8)
    return Image.fromarray(out)


def export_selection(spec, obs, mask, model_report, output, prompt_source):
    points, depth_fraction = measured_points(obs, mask)
    points = points[points[:, 2] > spec.geometry.get('table', {}).get('z_m', 0) + .015]
    target, keep, counts = associate_target(points, spec.geometry)
    visible, _ = measured_points(obs, np.ones(mask.shape, dtype=bool))
    visible_count = int(crop_mask(visible, spec.geometry['objects'][target]).sum())
    coverage = counts[target]/max(visible_count, 1)
    if coverage < .55:
        raise ValueError('Selected region covers only part of the object; add a positive point on its body')
    points = points[keep]
    # Sampling is only deduplication of measurements; no synthetic surfaces.
    _, indices = np.unique(np.floor(points/.002).astype(np.int32), axis=0, return_index=True)
    points = points[indices]
    if len(points) < 100:
        raise ValueError('Too few independent depth points after filtering')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(mask.astype(np.uint8)*255).save(output/'mask.png')
    preview(obs['rgb'], mask).save(output/'selection_preview.png')
    np.save(output/'target_points_table.npy', points)
    report = {**model_report, 'target': target, 'prompt_source': prompt_source,
              'view_index': obs['view_index'], 'rgb_path': str(obs['rgb_path']),
              'rgb_sha256': digest(obs['rgb_path']), 'depth_sha256': digest(obs['depth_path']),
              'registration_sha256': digest(obs['registration_path']),
              'mask_pixels': int(mask.sum()), 'valid_depth_fraction': depth_fraction,
              'visible_measured_target_coverage': coverage,
              'matched_depth_counts': counts, 'exported_points': len(points),
              'point_frame': 'reference table, metres',
              'scope': 'single-view SAM2 mask, measured aligned depth, existing object geometry association'}
    (output/'selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    data = dict(spec.data)
    data['target'] = target
    data['target_observation'] = {
        'target': target, 'points_table_path': str(output/'target_points_table.npy'),
        'points_sha256': digest(output/'target_points_table.npy'),
        'mask_path': str(output/'mask.png'), 'mask_sha256': digest(output/'mask.png'),
        'provenance_path': str(output/'selection.json'), 'provenance_sha256': digest(output/'selection.json')}
    SceneSpec(data).save(output/'scene.json')
    return report
