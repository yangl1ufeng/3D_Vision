"""Run physical acceptance cases sequentially and retain successes and failures."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent


def main():
    output = PROJECT / 'runs' / 'simulation_validation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    cases = [
        ('ur5_30mm', ['--robot', 'ur5_robotiq140', '--cube-size', '.03'], True),
        ('ur5_50mm', ['--robot', 'ur5_robotiq140', '--cube-size', '.05'], True),
        ('ur5_90mm', ['--robot', 'ur5_robotiq140', '--cube-size', '.09'], True),
        ('ur5_open_fingers', ['--robot', 'ur5_robotiq140', '--no-grasp'], False),
        ('panda_regression', [], True),
    ]
    records = []
    for name, arguments, expected_success in cases:
        run = output / name
        command = [sys.executable, str(PROJECT / 'run_simulation.py'), *arguments, '--output', str(run)]
        completed = subprocess.run(command, cwd=PROJECT, capture_output=True)
        (output / f'{name}.log').write_bytes(completed.stdout + completed.stderr)
        result_file = run / 'result.json'
        result = json.loads(result_file.read_text(encoding='utf-8')) if result_file.is_file() else {}
        expected_exit = 0 if expected_success else 1
        passed = result.get('task_success') is expected_success and completed.returncode == expected_exit
        if not expected_success:
            passed = passed and result.get('failure_stage') == 'hold' and result.get('grasp_success') is False
        records.append({'case': name, 'passed': passed, 'exit_code': completed.returncode,
                        'expected_success': expected_success, 'result': result})
        print(f'{name}: {"PASS" if passed else "FAIL"}', flush=True)
    files = ['simulation/pick_place.py', 'simulation/ur5_robotiq.py', 'verify_simulation_baselines.py']
    summary = {'all_passed': all(r['passed'] for r in records), 'cases': records,
               'source_sha256': {name: hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in files},
               'note': 'Fixed synthetic scenes; not an estimate of real-world success rate.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(output)
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
