"""Place measured desk proxies beside a virtual UR5 and check a cup approach."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pybullet as p

from .planning import CollisionScene, PlanningError, plan_rrt, validate_path
from .ur5_robotiq import UR5Robotiq140


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY = PROJECT / 'runs/gemini_multiview/20261112_170625_237128/geometry.json'
# A virtual placement for this desk snapshot, not a measured camera-to-robot transform.
DEFAULT_OFFSET = np.array([.65, -.40, 0.])


def add_box(center, half, color, orientation=(0, 0, 0, 1)):
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    return p.createMultiBody(0, collision, visual, center, orientation)


def add_proxy(item, offset, color):
    center = (np.asarray(item['center_table_m']) + offset).tolist()
    if item['shape'] == 'box':
        return add_box(center, (np.asarray(item['dimensions_m']) / 2).tolist(), color,
                       p.getQuaternionFromEuler([0, 0, item['yaw_table_rad']]))
    if item['shape'] == 'solid_cylinder_proxy':
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=item['radius_m'], height=item['height_m'])
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=item['radius_m'],
                                     length=item['height_m'], rgbaColor=color)
        return p.createMultiBody(0, collision, visual, center)
    raise ValueError(f'Unsupported shape: {item["shape"]}')


def render(path, width=1100, height=760):
    view = p.computeViewMatrixFromYawPitchRoll([.4, 0, .13], 1.35, 48, -37, 0, 2)
    projection = p.computeProjectionMatrixFOV(55, width / height, .02, 5)
    _, _, rgba, _, _ = p.getCameraImage(width, height, view, projection,
                                        renderer=p.ER_TINY_RENDERER)
    Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]).save(path)


def check_approach(scene, start, center, top, orientation):
    """Plan to above the cup; inspect a downward approach without moving physics."""
    record = {'target': 'cup', 'method': 'top_down_heuristic', 'execution_started': False}
    scene.configure('pregrasp')
    pre = [float(center[0]), float(center[1]), float(top + .12)]
    record['pregrasp_tcp_m'] = pre
    record['start_collision'] = scene.collision(start)
    if record['start_collision']:
        record['status'] = 'start_rejected'
        return record, None
    try:
        goal = scene.ik(pre, orientation, start)
        record['pregrasp_collision'] = scene.collision(goal)
        if record['pregrasp_collision']:
            record['status'] = 'pregrasp_rejected'
            return record, None
        path, info = plan_rrt(scene, start, goal, timeout=20)
        validate_path(scene, path)
        record.update(status='pregrasp_path_validated', planner=info,
                      path_rad=[q.tolist() for q in path], collision_queries=scene.calls)
    except PlanningError as exc:
        record.update(status='planning_rejected', reason=str(exc), collision=scene.last_reason)
        return record, None

    # The cup is tall. At this stage, only test whether the open gripper can
    # descend toward its upper side. This is not a grasp stability test.
    scene.configure('approach')
    approach = []
    previous = path[-1]
    last_safe_z = pre[2]
    for z in np.linspace(pre[2], top - .055, 31)[1:]:
        xyz = [pre[0], pre[1], float(z)]
        try:
            q = scene.ik(xyz, orientation, previous)
        except PlanningError as exc:
            record['approach_stop'] = {'tcp_m': xyz, 'reason': str(exc)}
            break
        reason = scene.collision(q)
        if reason or not scene.edge(previous, q, .005):
            record['approach_stop'] = {'tcp_m': xyz, 'reason': reason or scene.last_reason}
            break
        approach.append(q.tolist())
        previous = q
        last_safe_z = float(z)
    record['safe_approach_path_rad'] = approach
    record['safe_approach_end_tcp_m'] = ([pre[0], pre[1], last_safe_z] if approach else None)
    if 'approach_stop' not in record:
        record['approach_status'] = 'sampled_path_clear'
    else:
        record['approach_status'] = 'stopped_by_collision_or_ik'
    return record, path


def run(geometry_path, output, offset=DEFAULT_OFFSET):
    geometry = json.loads(Path(geometry_path).read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=True)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError('PyBullet connection failed')
    scene = None
    try:
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        add_box([.55, 0, -.04], [.45, .45, .04], [.68, .72, .75, 1])
        add_box([0, 0, -.04], [.14, .16, .04], [.34, .38, .43, 1])
        robot = UR5Robotiq140()
        bodies = {}
        colors = {'box': [.85, .49, .20, 1], 'cup': [.19, .45, .74, 1]}
        for name, item in geometry['objects'].items():
            bodies[name] = add_proxy(item, offset, colors.get(name, [.4, .6, .4, 1]))
        demo = SimpleNamespace(robot=robot.body, arm=robot.arm, tcp=robot.tcp,
                               adapter=robot, cube=bodies['cup'], cube_size=.05,
                               client=client, fingers=robot.pads)
        scene = CollisionScene(demo, [])
        scene.use_reconstructed_scene(geometry, 'cup', offset)
        render(output / 'scene_initial.png')
        cup = geometry['objects']['cup']
        center = np.asarray(cup['center_table_m']) + offset
        top = float(center[2] + cup['height_m'] / 2)
        start = np.array([p.getJointState(robot.body, j)[0] for j in robot.arm])
        orientation = p.getQuaternionFromEuler([math.pi, 0, math.pi])
        cup_width = 2 * cup['radius_m']
        box_width = min(geometry['objects']['box']['dimensions_m'][:2])
        result = {
            'geometry_source': str(Path(geometry_path).resolve()),
            'scene_frame': 'virtual UR5 base; table z=0; no measured camera-to-robot calibration',
            'table_to_virtual_robot_translation_m': offset.tolist(),
            'objects': {name: {'center_robot_m': (np.asarray(item['center_table_m']) + offset).tolist(),
                               'shape': item['shape']} for name, item in geometry['objects'].items()},
            'gripper_max_opening_m': robot.max_width,
            'cup_width_m': cup_width,
            'cup_opening_with_20mm_clearance_m': cup_width + .02,
            'box_short_side_m': box_width,
            'box_opening_with_20mm_clearance_m': box_width + .02,
            'box_robust_pinch_fit': box_width + .02 <= robot.max_width,
            'cup_opening_fit': cup_width + .02 <= robot.max_width,
            'limitations': ['Cup is approximated as a solid cylinder; hollow interior and handle absent.',
                            'Virtual robot placement is a chosen layout, not calibration.',
                            'Top-down TCP pose is a heuristic, not a GraspGen candidate.',
                            'No physics grasp or transfer executed; collision checks are sampled.'],
        }
        result['cup_plan'], path = check_approach(scene, start, center, top, orientation)
        if path:
            for joint, q in zip(robot.arm, path[-1]):
                p.resetJointState(robot.body, joint, float(q))
            render(output / 'scene_pregrasp.png')
            if result['cup_plan']['safe_approach_path_rad']:
                for joint, q in zip(robot.arm, result['cup_plan']['safe_approach_path_rad'][-1]):
                    p.resetJointState(robot.body, joint, float(q))
                render(output / 'scene_approach.png')
        (output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        return result
    finally:
        if scene is not None:
            scene.close()
        p.disconnect(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument('--output', type=Path, default=PROJECT / 'runs/reconstructed_ur5' /
                        datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    result = run(args.geometry, args.output)
    print(json.dumps({'output': str(args.output.resolve()),
                      'status': result['cup_plan']['status'],
                      'approach': result['cup_plan'].get('approach_status'),
                      'reason': result['cup_plan'].get('reason')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
