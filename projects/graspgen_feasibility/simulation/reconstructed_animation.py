"""Run and record contact-driven pick/place for a selected reconstructed object."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

from .planned_demo import PlannedDemo
from .measured_visuals import MeasuredVisuals
from .reconstructed_scene import DEFAULT_GEOMETRY, DEFAULT_OFFSET, PROJECT
from .scene_spec import legacy_scene, load_scene, object_height


STAGE_LABELS = {
    'pregrasp': 'Move to pregrasp', 'approach': 'Approach target',
    'close': 'Close gripper', 'lift': 'Lift target', 'hold': 'Hold target',
    'transfer': 'Move to drop zone', 'lower': 'Lower cup',
    'release': 'Release', 'retreat': 'Retreat', 'settle': 'Settle',
}


class RecordedDemo(PlannedDemo):
    def __init__(self, output):
        self.frames = []
        self.record_frames = False
        self.last_frame_step = -48
        self.measured_visuals = None
        super().__init__(output, robot_model='ur5_robotiq140', scenario='free')

    def tick(self, count):
        super().tick(count)
        if self.record_frames and self.step - self.last_frame_step >= 48:
            self.capture_frame()

    def capture_frame(self):
        width, height = 720, 512
        view = p.computeViewMatrixFromYawPitchRoll([.43, .01, .15], 1.20, 48, -36, 0, 2)
        projection = p.computeProjectionMatrixFOV(55, width / height, .02, 5)
        camera = p.getCameraImage(width, height, view, projection,
                                  renderer=p.ER_TINY_RENDERER)
        rgba = camera[2]
        image = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3])
        if self.measured_visuals is not None:
            image = self.measured_visuals.draw(image, camera[3], view, projection, self.cube)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, width, 27], fill=(255, 255, 255))
        label = STAGE_LABELS.get(self.stage, self.stage)
        if self.measured_visuals is not None:
            label = 'Measured scene  |  ' + label
        draw.text((8, 7), label, fill=(15, 25, 35))
        self.frames.append(image)
        self.last_frame_step = self.step


def add_measured_bodies(demo, geometry, visual_mode, scene_spec=None):
    spec = scene_spec or legacy_scene(geometry=geometry)
    # Replace all synthetic support/target/marker bodies; keep the robot.
    for body in [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]:
        if body != demo.robot:
            p.removeBody(body)
    for name, color in [('table', [.68, .72, .75, 1]), ('pedestal', [.34, .38, .43, 1])]:
        item = spec.data[name]
        body = demo.box(item['center_robot_m'], item['half_extents_m'], 0, color)
        if name == 'table':
            demo.table = body
    demo.table_top = spec.table_top
    marker = p.createVisualShape(p.GEOM_BOX, halfExtents=[.065, .065, .0003], rgbaColor=[.2, .7, .35, 1])
    p.createMultiBody(0, -1, marker, [*spec.drop_xy, spec.table_top + .0003])
    demo.object_bodies = {}
    for name, item in geometry['objects'].items():
        center = (np.asarray(item['center_table_m']) + spec.offset).tolist()
        orientation = p.getQuaternionFromEuler([0, 0, item.get('yaw_table_rad', 0)])
        color = [.19, .45, .74, 1] if name == spec.target else [.85, .49, .20, 1]
        if item['shape'] == 'voxel_compound_proxy':
            collision = p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*len(item['parts']),
                halfExtents=[part['half_extents_m'] for part in item['parts']],
                collisionFramePositions=[part['center_local_m'] for part in item['parts']])
            visual = -1
        elif item['shape'] == 'box':
            half = (np.asarray(item['dimensions_m'])/2).tolist()
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color) if visual_mode=='proxy' else -1
        else:
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=item['radius_m'], height=item['height_m'])
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=item['radius_m'], length=item['height_m'], rgbaColor=color) if visual_mode=='proxy' else -1
        mass = spec.data['target_physics']['mass_kg'] if name == spec.target else 0
        demo.object_bodies[name] = p.createMultiBody(mass, collision, visual, center, orientation)
    demo.cube = demo.object_bodies[spec.target]  # Existing controller's target-body attribute.
    p.changeDynamics(demo.cube, -1, lateralFriction=spec.data['target_physics']['lateral_friction'], restitution=0)
    demo.initial_z = p.getBasePositionAndOrientation(demo.cube)[0][2]
    target = geometry['objects'][spec.target]
    width = min(target['dimensions_m'][:2]) if target['shape'] in ('box', 'voxel_compound_proxy') else 2*target['radius_m']
    demo.open_width = demo.width = min(demo.adapter.max_width, width + .02)
    demo.command([p.getJointState(demo.robot, j)[0] for j in demo.arm])
    demo.scene.use_reconstructed_scene(geometry, spec.target, spec.offset, spec)
    return (np.asarray(target['center_table_m']) + spec.offset).tolist(), target


def save_gif(frames, output):
    if not frames:
        return
    # Keep recording independent from optional video codecs on the user's PC.
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=125,
                   loop=0, optimize=True)


def run(output, geometry_path=DEFAULT_GEOMETRY, grasp_below_top=.03, visual_mode='measured', candidate=None, scene_spec=None):
    spec = scene_spec or legacy_scene(geometry_path)
    geometry = spec.geometry
    output.mkdir(parents=True, exist_ok=True)
    demo = None
    result = {'geometry_source': spec.data['geometry_path'],
              'target': spec.target, 'scene_fingerprint': spec.fingerprint,
              'object': geometry['objects'][spec.target]['shape'],
              'virtual_table_offset_m': spec.offset.tolist(),
              'target_physics': spec.data['target_physics'],
              'heuristic_grasp_below_top_m': grasp_below_top if candidate is None else None,
              'visual_mode': visual_mode,
              'contact_driven': True, 'object_attachment': False,
              'grasp_success': False, 'task_success': False}
    try:
        demo = RecordedDemo(output)
        center, target = add_measured_bodies(demo, geometry, visual_mode, spec)
        spec.save(output/'scene_snapshot.json')
        if candidate is not None:
            from .graspgen_bridge import quaternion, verify_mapping
            pose = np.asarray(candidate['tcp_pose_robot_m'])
            demo.orientation = quaternion(pose[:3, :3])
            demo.adapter.reset_fingers(0)
            verified = verify_mapping(demo.adapter)
            demo.adapter.manifest['graspgen_frame_mapping_verified'] = verified['passed']
            demo.adapter.manifest['graspgen_mapping'] = verified
            demo.adapter.write_manifest(output)
            demo.open_width = demo.width = demo.adapter.max_width
            demo.command([p.getJointState(demo.robot, j)[0] for j in demo.arm])
            result.update(grasp_generator_used=True, candidate_id=candidate['candidate_id'],
                          candidate_score=candidate['score'], tcp_pose_robot_m=pose.tolist(),
                          candidate_pose_modified=False)
        if visual_mode == 'measured':
            ply_path = Path(spec.data['pointcloud_path'])
            demo.measured_visuals = MeasuredVisuals(ply_path, geometry, spec.offset, spec.target)
            result['measured_ply_source'] = str(ply_path.resolve())
            result['measured_ply_points'] = demo.measured_visuals.source_points
            result['animated_target_points'] = demo.measured_visuals.target_points
            result['object_point_counts'] = demo.measured_visuals.object_point_counts
            result['rendering_note'] = ('Original measured RGB-D points are projected into animation frames; '
                                        'reconstructed objects are visualized from measured points; table remains a flat plane. '
                                        'Hidden box/cylinder collision proxies remain for dynamics. '
                                        'Missing surfaces are not filled in.')
        x, y = center[:2]
        top = center[2] + object_height(target) / 2
        demo.record_frames = True
        demo.capture_frame()
        grasp = np.array([x, y, top - grasp_below_top]) if candidate is None else pose[:3, 3]
        pregrasp = np.array([x, y, top + .12]) if candidate is None else grasp - .12 * pose[:3, 2]
        lift = np.array([x, y, top + .09]) if candidate is None else grasp + [0, 0, .14]
        demo.move('pregrasp', pregrasp.tolist())
        demo.move('approach', grasp.tolist())
        demo.stage = 'close'
        demo.width = 0
        demo.command([p.getJointState(demo.robot, j)[0] for j in demo.arm])
        demo.tick(240)
        demo.move('lift', lift.tolist())
        demo.stage = 'hold'
        demo.tick(720)
        result['hold_min_lift_m'] = demo.hold_min_height
        result['hold_both_fingers_contact_fraction'] = demo.hold_contact_steps / 720
        result['grasp_success'] = bool(demo.hold_min_height >= .10 and
                                       demo.hold_contact_steps >= 684)
        if result['grasp_success']:
            dest = spec.drop_xy.tolist()
            transfer = np.array([*dest, top + .09])
            lower = np.array([*dest, top - grasp_below_top])
            if candidate is not None:
                tcp_now = np.array(p.getLinkState(demo.robot, demo.tcp, computeForwardKinematics=True)[4])
                cup_now = np.array(p.getBasePositionAndOrientation(demo.cube)[0])
                shift = np.array([dest[0] - cup_now[0], dest[1] - cup_now[1], 0])
                transfer = tcp_now + shift
                lower = transfer - [0, 0, .14]
            demo.move('transfer', transfer.tolist())
            demo.move('lower', lower.tolist())
            demo.stage = 'release'
            demo.width = demo.open_width
            demo.command([p.getJointState(demo.robot, j)[0] for j in demo.arm])
            demo.tick(240)
            retreat = np.array([*dest, top + .09])
            if candidate is not None:
                retreat = np.array(p.getLinkState(demo.robot, demo.tcp, computeForwardKinematics=True)[4]) - .12 * pose[:3, 2]
            demo.move('retreat', retreat.tolist())
            demo.stage = 'settle'
            demo.tick(480)
            position = p.getBasePositionAndOrientation(demo.cube)[0]
            linear, angular = p.getBaseVelocity(demo.cube)
            result['final_target_position_m'] = list(position)
            result['final_target_speed_m_s'] = float(np.linalg.norm(linear))
            result['final_target_angular_speed_rad_s'] = float(np.linalg.norm(angular))
            result['task_success'] = bool(np.linalg.norm(np.asarray(position[:2]) - dest) < .05
                                          and abs(position[2] - (spec.table_top + object_height(target)/2)) < .01
                                          and result['final_target_speed_m_s'] < .01
                                          and result['final_target_angular_speed_rad_s'] < .1
                                          and not demo.robot_table_collision)
            if not result['task_success']:
                result['failure_stage'] = 'placement_verification'
        else:
            result['failure_stage'] = 'hold'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['failure_stage'] = demo.stage if demo is not None else 'setup'
    finally:
        if demo is not None:
            result['final_target_position_m'] = list(p.getBasePositionAndOrientation(demo.cube)[0])
            demo.capture_frame()
            result['animation'] = str((output/'pick_place.gif').resolve())
            save_gif(demo.frames, output / 'pick_place.gif')
            result.update(frame_count=len(demo.frames),
                          robot_table_collision=demo.robot_table_collision,
                          execution_collision=demo.execution_collision,
                          stages=demo.stages,
                          planning_records=demo.plan_records)
            demo.scene.close()
            p.disconnect(demo.client)
        (output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'runs/reconstructed_ur5' /
                        datetime.now().strftime('animation_%Y%m%d_%H%M%S'))
    parser.add_argument('--geometry', type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument('--grasp-below-top', type=float, default=.03)
    parser.add_argument('--visual-mode', choices=['measured', 'proxy'], default='measured')
    parser.add_argument('--screening', type=Path)
    parser.add_argument('--candidate-id', type=int)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--target')
    args = parser.parse_args()
    candidate = None
    spec = load_scene(args.scene, args.target) if args.scene else None
    if args.screening:
        screening = json.loads(args.screening.read_text(encoding='utf-8'))
        spec = spec or load_scene(screening['scene_snapshot'], args.target)
        if spec.fingerprint != screening['scene_fingerprint']:
            parser.error('Selected scene/target differs from screening')
        candidate = next((r for r in screening['records'] if r['candidate_id'] == args.candidate_id), None)
        if candidate is None or candidate['status'] != 'validated':
            parser.error('Candidate must have passed screening')
    elif args.candidate_id is not None:
        parser.error('--candidate-id requires --screening')
    elif args.target and spec is None:
        spec = load_scene(target=args.target)
    result = run(args.output, args.geometry, args.grasp_below_top, args.visual_mode, candidate, spec)
    if args.screening:
        result['screening_source'] = str(args.screening.resolve())
        result['mapping_verification'] = screening['mapping']
        (args.output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'grasp_success': result['grasp_success'],
                      'task_success': result['task_success'], 'error': result.get('error'),
                      'frames': result.get('frame_count')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
