"""Execute validated GraspGen candidates in score order, retaining every attempt."""
import argparse
import hashlib
import json
from pathlib import Path
from .reconstructed_animation import run
from .scene_spec import load_scene


def execute(screening_path, output):
    screening = json.loads(screening_path.read_text(encoding='utf-8'))
    spec = load_scene(screening['scene_snapshot'])
    if spec.fingerprint != screening['scene_fingerprint']:
        raise ValueError('Scene snapshot differs from screened scene')
    if not screening['mapping']['passed']:
        raise ValueError('Mapping not verified')
    output.mkdir(parents=True, exist_ok=True)
    task = {'screening_source': str(screening_path.resolve()),
            'screening_sha256': hashlib.sha256(screening_path.read_bytes()).hexdigest(),
            'selection': 'descending model score among validated candidates; stop on full task success',
            'candidate_count': screening['candidate_count'],
            'validated_count': screening['validated_count'],
            'scene_fingerprint': spec.fingerprint, 'target': spec.target,
            'task_success': False, 'attempts': []}
    for candidate in sorted(screening['records'], key=lambda r: -r['score']):
        if candidate['status'] != 'validated':
            continue
        cid = candidate['candidate_id']
        print(f'Execute model candidate {cid}, score {candidate["score"]:.4f}', flush=True)
        directory = output / f'candidate_{cid}'
        result = run(directory, candidate=candidate, scene_spec=spec)
        result.update(screening_source=str(screening_path.resolve()), mapping_verification=screening['mapping'])
        (directory/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        task['attempts'].append({'candidate_id': cid, 'score': candidate['score'],
                                 'grasp_success': result['grasp_success'],
                                 'task_success': result['task_success'],
                                 'failure_stage': result.get('failure_stage'),
                                 'error': result.get('error'), 'result': str((directory/'result.json').resolve())})
        if result['task_success']:
            task.update(task_success=True, selected_candidate_id=cid,
                        animation=result['animation'])
        (output/'task.json').write_text(json.dumps(task, indent=2), encoding='utf-8')
        if task['task_success']:
            break
    task['status'] = 'success' if task['task_success'] else ('no_valid_candidate' if not task['attempts'] else 'attempts_exhausted')
    (output/'task.json').write_text(json.dumps(task, indent=2), encoding='utf-8')
    return task


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('screening', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.screening, args.output), indent=2), flush=True)
