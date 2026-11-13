"""Generate, screen and execute grasps for a configured scene and target."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from simulation.scene_spec import DEFAULT_SCENE, load_scene

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/measured_grasp'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--target')
    parser.add_argument('--candidates', type=int, default=64)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--list-targets', action='store_true')
    args = parser.parse_args()
    spec = load_scene(args.scene, args.target)
    if args.list_targets:
        print(json.dumps({'scene_id': spec.data.get('scene_id'), 'targets': list(spec.geometry['objects']),
                          'selected_target': spec.target}, ensure_ascii=False))
        return 0
    if not 1 <= args.candidates <= 200:
        parser.error('--candidates must be 1..200')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sim = ROOT/'.venv-sim/Scripts/python.exe'
    vision = ROOT/'.venv-vision/Scripts/python.exe'
    commands = [
        [sim, ROOT/'prepare_measured_grasps.py', '--scene', args.scene.resolve(), '--target', spec.target, '--output', output/'input'],
        [vision, ROOT/'smoke_graspgen.py', '--points', output/'input/target_robot_points.npy',
         '--provenance', output/'input/input_provenance.json', '--candidates', str(args.candidates), '--output', output/'candidates'],
        [sim, '-m', 'simulation.graspgen_bridge', output/'candidates/candidates.npz',
         '--scene', output/'input/scene_snapshot.json', '--output', output/'planning'],
        [sim, '-m', 'simulation.graspgen_task', output/'planning/screening.json', '--output', output/'execution'],
    ]
    status = {'scene_source': str(args.scene.resolve()), 'scene_fingerprint': spec.fingerprint,
              'target': spec.target, 'stages_completed': [], 'status': 'running'}
    names = ['prepare', 'inference', 'planning', 'execution']
    try:
        for name, command in zip(names, commands[:1] if args.prepare_only else commands):
            status['current_stage'] = name
            (output/'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
            subprocess.run([str(arg) for arg in command], cwd=ROOT, check=True)
            status['stages_completed'].append(name)
    except Exception as exc:
        status.update(status='failed', error=str(exc))
        (output/'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        raise
    if args.prepare_only:
        status['status'] = 'prepared'
        (output/'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print('Prepared:', output)
        return 0
    task = json.loads((output/'execution/task.json').read_text())
    status['status'] = task['status']
    (output/'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print('Results:', output)
    return 0 if task['task_success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
