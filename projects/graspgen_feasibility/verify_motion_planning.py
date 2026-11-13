"""End-to-end acceptance, including refusals before any planned arm motion."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent


def main():
    output = PROJECT/'runs/planning_validation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    records = []
    for name, expected in [('free', True), ('barrier', True), ('blocked_goal', False), ('unreachable', False)]:
        run = output/name
        cmd = [sys.executable, str(PROJECT/'run_simulation.py'), '--robot', 'ur5_robotiq140',
               '--planning', '--scenario', name, '--output', str(run)]
        process = subprocess.run(cmd, capture_output=True, cwd=PROJECT)
        (output/f'{name}.log').write_bytes(process.stdout+process.stderr)
        result = json.loads((run/'result.json').read_text())
        plans = json.loads((run/'planning.json').read_text())
        passed = result['task_success'] is expected and process.returncode == (0 if expected else 1)
        if expected:
            passed = passed and plans['execution_collision'] is None and all(r['status']=='executed' for r in plans['plans'])
        else:
            reason = 'invalid_goal' if name=='blocked_goal' else 'ik_unreachable'
            passed = passed and reason in result.get('error', '') and not any(r['execution_started'] for r in plans['plans'])
        if name == 'barrier':
            transfer = next(r for r in plans['plans'] if r['stage']=='transfer')
            passed = passed and transfer['method']=='rrt_connect' and transfer['direct_blocked'] and transfer['payload_included']
        records.append({'case': name, 'passed': bool(passed), 'result': result})
        print(f'{name}: {"PASS" if passed else "FAIL"}', flush=True)
    files = ['simulation/planning.py', 'simulation/planned_demo.py', 'simulation/pick_place.py',
             'simulation/ur5_robotiq.py', 'verify_motion_planning.py']
    summary = {'all_passed': all(r['passed'] for r in records), 'cases': records,
               'source_sha256': {f: hashlib.sha256((PROJECT/f).read_bytes()).hexdigest() for f in files}}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(output)
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
