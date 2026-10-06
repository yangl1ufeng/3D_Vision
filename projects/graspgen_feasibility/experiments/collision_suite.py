"""Frozen synthetic scenes in meters, using actual open Robotiq geometry."""
import numpy as np
import trimesh


def make_box(name, center, extents):
    center, half = np.asarray(center), np.asarray(extents) / 2
    return {'name': name, 'lower': (center - half).tolist(), 'upper': (center + half).tolist()}


def scenes():
    table = make_box('table', [0, 0, -.17], [.8, .8, .02])
    target = make_box('target_in_open_gap', [0, 0, .185], [.03, .015, .025])
    specifications = [
        ('clear', None, target),
        ('finger_penetration', make_box('obstacle', [.068, 0, .18], [.012, .012, .02]), target),
        ('palm_penetration', make_box('obstacle', [0, 0, .04], [.025, .025, .025]), target),
        ('thin_between_endpoints', make_box('thin_plate', [0, 0, -.006], [.02, .02, .0005]), target),
        ('near_finger_3mm', make_box('near_obstacle', [.077, 0, .18], [.004, .02, .025]), target),
        ('near_finger_12mm', make_box('near_obstacle', [.086, 0, .18], [.004, .02, .025]), target),
        ('side_clear', make_box('obstacle', [.16, 0, .10], [.04, .04, .10]), target),
        ('thin_post', make_box('thin_post', [.068, 0, .19], [.001, .001, .025]), target),
        ('table_endpoint', make_box('raised_table', [0, 0, -.006], [.4, .4, .012]), target),
        ('enclosing_obstacle', make_box('enclosing_solid', [0, 0, .10], [.40, .30, .40]), target),
        ('target_in_path', None, make_box('target', [0, 0, -.05], [.01, .01, .01])),
        ('target_in_palm', None, make_box('target', [0, 0, .04], [.01, .01, .01])),
    ]
    result = []
    for i, (name, obstacle, obj) in enumerate(specifications):
        # Primary candidate plus lateral/depth perturbations. Identical for all
        # algorithms, budgets and seeds. No generation/filter-dependent selection.
        offsets = [[0, 0, 0], [-.012, 0, 0], [.012, 0, 0], [0, .02, 0], [0, 0, -.01], [0, 0, .01]]
        poses = np.repeat(np.eye(4)[None], len(offsets), axis=0)
        poses[:, :3, 3] = offsets
        result.append({'scene_id': name, 'split': 'development' if i % 2 == 0 else 'held_out',
                       'background': [table] + ([obstacle] if obstacle else []), 'target': [obj],
                       'poses': poses})
    return result


def boxes_mesh(boxes):
    parts = []
    for box in boxes:
        lower, upper = np.asarray(box['lower']), np.asarray(box['upper'])
        mesh = trimesh.creation.box(extents=upper - lower)
        mesh.apply_translation((lower + upper) / 2)
        parts.append(mesh)
    return trimesh.util.concatenate(parts)


def sample_boxes(boxes, count, seed):
    return trimesh.sample.sample_surface(boxes_mesh(boxes), count, seed=seed)[0].astype(np.float32)
