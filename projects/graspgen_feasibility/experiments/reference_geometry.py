"""Independent solid-box reference against the ORIGINAL triangle mesh.

Continuous triangle/AABB separating-axis test: intersect time intervals on all
13 separating axes, without point clouds, convexifying, or sampling time.
Containment at the start is checked with solid-angle winding per closed mesh
component. Numerical boundary contact counts as collision.
"""
import numpy as np


def contains_point(triangles, point):
    a, b, c = np.moveaxis(np.asarray(triangles) - point, 1, 0)
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    numerator = np.einsum('ij,ij->i', a, np.cross(b, c))
    denominator = la * lb * lc + np.einsum('ij,ij->i', a, b) * lc + np.einsum('ij,ij->i', b, c) * la + np.einsum('ij,ij->i', c, a) * lb
    angle = np.sum(2 * np.arctan2(numerator, denominator))
    return abs(angle) > 2 * np.pi


def moving_triangles_box(triangles_start, translation, lower, upper):
    center = (np.asarray(lower) + upper) / 2
    half = (np.asarray(upper) - lower) / 2
    triangles = np.asarray(triangles_start, dtype=np.float64) - center
    edges = np.roll(triangles, -1, axis=1) - triangles
    axes = np.concatenate([np.broadcast_to(np.eye(3), (len(triangles), 3, 3)),
                           np.cross(edges[:, 0], edges[:, 1])[:, None, :],
                           np.cross(edges[:, :, None, :], np.eye(3)[None, None]).reshape(-1, 9, 3)], axis=1)
    norms = np.linalg.norm(axes, axis=2)
    axes = axes / np.where(norms > 1e-14, norms, 1)[:, :, None]
    projection = np.einsum('tvc,tac->tav', triangles, axes)
    minimum, maximum = projection.min(2), projection.max(2)
    radius = np.einsum('tac,c->ta', np.abs(axes), half)
    speed = np.einsum('tac,c->ta', axes, translation)
    stationary = np.abs(speed) < 1e-12
    possible = np.all(~stationary | ((minimum <= radius + 1e-10) & (maximum >= -radius - 1e-10)), axis=1)
    safe_speed = np.where(stationary, 1, speed)
    t1 = (-radius - maximum) / safe_speed
    t2 = (radius - minimum) / safe_speed
    enter = np.max(np.where(stationary, -np.inf, np.minimum(t1, t2)), axis=1)
    leave = np.min(np.where(stationary, np.inf, np.maximum(t1, t2)), axis=1)
    hit = possible & (np.maximum(enter, 0) <= np.minimum(leave, 1) + 1e-10)
    return bool(hit.any())


def mesh_boxes_collision(parts, pose_final, boxes, approach_distance=0, direction=(0, 0, 1)):
    rotation, final = pose_final[:3, :3], pose_final[:3, 3]
    translation = rotation @ (np.asarray(direction) * approach_distance)
    start = final - translation
    for part in parts:
        world_triangles = np.asarray(part.triangles) @ rotation.T + start
        for box in boxes:
            lower, upper = np.asarray(box['lower']), np.asarray(box['upper'])
            if moving_triangles_box(world_triangles, translation, lower, upper):
                return True
            # If the box starts wholly inside the gripper, surface SAT alone
            # has no intersection. Disconnected gripper parts are checked alone.
            if contains_point(world_triangles, (lower + upper) / 2):
                return True
    return False
