"""Auditable outputs with all candidates, including rejected candidates."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def rank_candidates(decisions, scores):
    accepted = [d['candidate_id'] for d in decisions if d['accepted']]
    accepted.sort(key=lambda i: (-float(scores[i]), i))
    for rank, i in enumerate(accepted, start=1):
        decisions[i]['rank'] = rank
    return accepted


def export_candidates(output, grasps, decisions, frame_name):
    records = [{**d, 'score': float(grasps.scores[d['candidate_id']]),
                'T_world_graspgen_gripper': grasps.poses_world_m[d['candidate_id']].tolist()}
               for d in decisions]
    write_json(output / 'candidates.json', {
        'schema_version': 1, 'world_frame': frame_name, 'translation_unit': 'm',
        'transform_convention': 'p_world = T_world_graspgen_gripper @ p_gripper (homogeneous column vectors)',
        'score_note': 'Ranking score, not measured success probability.', 'candidates': records})
    columns = ['candidate_id', 'score', 'accepted', 'rank', 'collision_checked', 'collision_state', 'reasons']
    approach_columns = ['approach_state', 'approach_first_hit_fraction', 'approach_first_hit_source', 'approach_reasons']
    observation_columns = ['observation_state', 'needs_more_views', 'unknown_fraction']
    pose_columns = [f'T{r}{c}' for r in range(4) for c in range(4)]
    with (output / 'candidates.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns + approach_columns + observation_columns + pose_columns)
        writer.writeheader()
        for record in records:
            row = {k: record[k] for k in columns}
            row['reasons'] = '|'.join(row['reasons'])
            approach = record.get('approach', {})
            observation = record.get('observation', {})
            row.update(observation_state=observation.get('state', 'not_checked'),
                       needs_more_views=observation.get('needs_more_views'), unknown_fraction=observation.get('unknown_fraction'))
            row.update(approach_state=approach.get('state', 'not_checked'),
                       approach_first_hit_fraction=approach.get('first_hit_fraction'),
                       approach_first_hit_source=approach.get('first_hit_source'),
                       approach_reasons='|'.join(approach.get('reasons', [])))
            row.update({f'T{r}{c}': record['T_world_graspgen_gripper'][r][c] for r in range(4) for c in range(4)})
            writer.writerow(json_safe(row))


def export_artifacts(output, scene, segmentation, clouds, grasps, decisions, checker, cfg, source):
    export_candidates(output, grasps, decisions, scene.metadata['world_frame'])
    write_json(output / 'input_manifest.json', scene.metadata)
    write_json(output / 'resolved_config.json', cfg)
    write_json(output / 'provenance.json', source)
    np.savez_compressed(output / 'scene_arrays.npz',
                       rgb=scene.rgb, segmentation_mask=segmentation.mask, valid_geometry_mask=scene.valid,
                       target_raw_world_m=clouds.target_raw_m, target_model_world_m=clouds.target_model_m,
                       outliers_world_m=clouds.outliers_m, background_full_world_m=clouds.background_full_m,
                       background_collision_world_m=clouds.background_collision_m,
                       poses_world_graspgen_gripper=grasps.poses_world_m, scores=grasps.scores,
                       accepted=np.asarray([d['accepted'] for d in decisions]),
                       collision_checked=np.asarray([d['collision_checked'] for d in decisions]),
                       collision_state=np.asarray([d['collision_state'] for d in decisions], dtype='U32'),
                       approach_state=np.asarray([d.get('approach', {}).get('state', 'not_checked') for d in decisions], dtype='U32'),
                       observation_state=np.asarray([d.get('observation', {}).get('state', 'not_checked') for d in decisions], dtype='U40'),
                       pregrasp_poses_world=np.asarray([d.get('approach', {}).get('pregrasp_pose_world', np.full((4, 4), np.nan)) for d in decisions]),
                       exclusion_reasons=np.asarray(['|'.join(d['reasons']) for d in decisions], dtype='U512'),
                       accepted_rank=np.asarray([d['rank'] or -1 for d in decisions], dtype=np.int32),
                       gripper_surface_local_m=checker.samples_local_m)
    overlay = scene.rgb.copy()
    overlay[segmentation.mask] = (overlay[segmentation.mask] * .45 + np.array([0, 255, 0]) * .55).astype(np.uint8)
    image = Image.fromarray(overlay)
    if 'box_xyxy' in scene.target:
        ImageDraw.Draw(image).rectangle(scene.target['box_xyxy'], outline='red', width=3)
    image.save(output / 'segmentation_overlay.png')
