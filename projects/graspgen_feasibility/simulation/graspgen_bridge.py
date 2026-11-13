"""Verify GraspGen mapping and screen candidates bound to a scene and target."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pybullet as p

from .reconstructed_animation import RecordedDemo, add_measured_bodies
from .reconstructed_scene import DEFAULT_GEOMETRY, PROJECT
from .planning import PlanningError, plan_rrt, validate_path
from .scene_spec import load_scene, digest


T_G_TCP = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                    [0., 0., 1., .2], [0., 0., 0., 1.]])


def quaternion(rotation):
    r = np.asarray(rotation, float)
    q = np.zeros(4)
    if np.trace(r) > 0:
        s = math.sqrt(1 + np.trace(r)) * 2
        q[:] = [(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
                (r[1, 0]-r[0, 1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) * 2
        q[i], q[j], q[k], q[3] = s/4, (r[i, j]+r[j, i])/s, (r[i, k]+r[k, i])/s, (r[k, j]-r[j, k])/s
    return (q / np.linalg.norm(q)).tolist()


def verify_mapping(adapter):
    base = p.getLinkState(adapter.body, adapter.links['robotiq_arg2f_base_link'], computeForwardKinematics=True)
    inv = p.invertTransform(base[4], base[5])
    pads = np.array([p.multiplyTransforms(*inv, *p.getLinkState(adapter.body, i,
                     computeForwardKinematics=True)[4:6])[0] for i in adapter.pads])
    pads_g = pads @ T_G_TCP[:3, :3].T
    pads_g = pads_g[np.argsort(pads_g[:, 0])]
    asset = PROJECT / 'third_party/GraspGen/assets/robotiq/robotiq_140_collision.obj'
    groups = {'geometry_0': [], 'geometry_1': []}
    name = None
    for line in asset.read_text().splitlines():
        if line.startswith('o '):
            name = line[2:].strip()
        elif line.startswith('v ') and name in groups:
            groups[name].append([float(x) for x in line.split()[1:4]])
    if any(len(v) != 8 for v in groups.values()):
        raise ValueError('Official mesh pad geometry changed')
    official = np.array([(np.min(v, axis=0) + np.max(v, axis=0))/2 for v in groups.values()])
    official = official[np.argsort(official[:, 0])]
    errors = np.linalg.norm(pads_g - official, axis=1)
    tcp = np.asarray(p.multiplyTransforms(*inv, *p.getLinkState(adapter.body, adapter.tcp,
                                          computeForwardKinematics=True)[4:6])[0])
    passed = bool(errors.max() < .0002 and np.linalg.norm(tcp - [0, 0, .2]) < 1e-6)
    if not passed:
        raise ValueError('Gripper frame landmark validation failed')
    return {'passed': passed, 'T_graspgen_base_simulation_tcp': T_G_TCP.tolist(),
            'official_collision_mesh_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
            'transformed_simulation_open_pad_centers_m': pads_g.tolist(),
            'official_mesh_open_pad_centers_m': official.tolist(),
            'pad_center_errors_m': errors.tolist(), 'simulation_tcp_in_base_m': tcp.tolist(),
            'convention': 'T_world_tcp = T_world_graspgen_base @ T_graspgen_base_simulation_tcp',
            'limits': 'Frame and open-pad landmarks verified; not proof of identical dynamics or all joint geometry.'}


def validate_candidate_source(candidate_path, spec):
    report = json.loads(Path(candidate_path).with_name('result.json').read_text(encoding='utf-8'))
    if report.get('scene_fingerprint') != spec.fingerprint or report.get('target') != spec.target:
        raise ValueError('Candidate scene/target mismatch; regenerate for the selected scene')
    if not report.get('passed') or report.get('output_frame') != 'virtual robot base, metres':
        raise ValueError('Invalid inference result or coordinate frame')
    if report.get('candidate_sha256') != digest(candidate_path):
        raise ValueError('Candidate file changed since inference')
    return report


def screen(candidate_path, output, scene_spec=None):
    spec = scene_spec or load_scene()
    validate_candidate_source(candidate_path, spec)
    output.mkdir(parents=True, exist_ok=True)
    geometry = spec.geometry
    data = np.load(candidate_path, allow_pickle=False)
    records = []
    demo = RecordedDemo(output)
    try:
        center, target = add_measured_bodies(demo, geometry, 'proxy', spec)
        demo.adapter.reset_fingers(0)
        mapping = verify_mapping(demo.adapter)
        scene = demo.scene
        start = np.array([p.getJointState(demo.robot, j)[0] for j in demo.arm])
        for index in np.argsort(-data['scores']):
            pose = data['poses_world_m'][index].astype(float) @ T_G_TCP
            qrot = quaternion(pose[:3, :3])
            goal_xyz = pose[:3, 3]
            pre_xyz = goal_xyz - .12 * pose[:3, 2]
            record = {'candidate_id': int(index), 'score': float(data['scores'][index]),
                      'tcp_pose_robot_m': pose.tolist(), 'status': 'rejected'}
            records.append(record)
            try:
                scene.configure('pregrasp')
                pre_q = scene.ik(pre_xyz.tolist(), qrot, start)
                if scene.collision(pre_q):
                    raise PlanningError(f'pregrasp_collision: {scene.last_reason}')
                previous = pre_q
                approach = [pre_q]
                for t in np.linspace(0, 1, 25)[1:]:
                    q = scene.ik((pre_xyz * (1-t) + goal_xyz * t).tolist(), qrot, previous)
                    if not scene.edge(previous, q, .005):
                        raise PlanningError(f'approach_collision: {scene.last_reason}')
                    approach.append(q)
                    previous = q
                path, info = plan_rrt(scene, start, pre_q, timeout=3)
                validate_path(scene, path)
                record.update(status='validated', planner=info,
                              pregrasp_path_rad=[q.tolist() for q in path],
                              approach_path_rad=[q.tolist() for q in approach])
            except PlanningError as exc:
                record['reason'] = str(exc)
            print(f'candidate {index}: {record["status"]}', flush=True)
        spec.save(output/'scene_snapshot.json')
        result = {'source_candidates': str(Path(candidate_path).resolve()), 'mapping': mapping,
                  'scene_fingerprint': spec.fingerprint, 'target': spec.target,
                  'scene_snapshot': str((output/'scene_snapshot.json').resolve()),
                  'candidate_count': len(records), 'validated_count': sum(r['status']=='validated' for r in records),
                  'records': records, 'collision_margin_m': .004, 'validation_joint_step_rad': .005,
                  'execution_started': False}
        (output/'screening.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    finally:
        demo.scene.close()
        p.disconnect(demo.client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--target')
    args = parser.parse_args()
    result = screen(args.candidates, args.output, load_scene(args.scene, args.target) if args.scene else load_scene(target=args.target))
    print('Validated:', result['validated_count'])
