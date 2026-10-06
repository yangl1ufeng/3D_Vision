"""Audit saved artifacts and compare independent same-seed runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from offline_core.export import write_json
from offline_core.geometry import valid_pose_mask


def audit(folder):
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    records = json.loads((folder / 'candidates.json').read_text(encoding='utf-8'))['candidates']
    with (folder / 'candidates.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(folder / 'scene_arrays.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    poses, scores = arrays['poses_world_graspgen_gripper'], arrays['scores']
    n = summary['counts']['raw']
    assert n == len(records) == len(rows) == len(poses) == len(scores)
    assert valid_pose_mask(poses).all() and np.isfinite(scores).all()
    assert [r['candidate_id'] for r in records] == list(range(n))
    np.testing.assert_allclose(poses, [r['T_world_graspgen_gripper'] for r in records], rtol=0, atol=0)
    np.testing.assert_allclose(scores, [r['score'] for r in records], rtol=0, atol=0)
    accepted = sum(r['accepted'] for r in records)
    assert accepted == summary['counts']['accepted'] == int(arrays['accepted'].sum())
    assert accepted + summary['counts']['rejected'] == n
    ranked = sorted([r for r in records if r['accepted']], key=lambda r: r['rank'])
    assert [r['rank'] for r in ranked] == list(range(1, accepted + 1))
    assert [r['score'] for r in ranked] == sorted([r['score'] for r in ranked], reverse=True)
    reason_counts = {}
    for record, row in zip(records, rows):
        assert int(row['candidate_id']) == record['candidate_id']
        assert row['reasons'] == '|'.join(record['reasons'])
        assert record['accepted'] == (record['collision_checked'] and record['collision_state'] == 'clear' and not record['reasons'])
        assert record['accepted'] or record['reasons']
        for reason in record['reasons']:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    assert reason_counts == summary['counts']['exclusion_reasons']
    perf = json.loads((folder / 'performance.json').read_text(encoding='utf-8'))
    assert perf == summary['performance']
    assert all(stage['status'] == 'ok' and stage['seconds'] >= 0 for stage in perf['stages'].values())
    return arrays, records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    a, ra, sa = audit(args.first)
    b, rb, sb = audit(args.second)
    exact = {}
    for key in ['segmentation_mask', 'target_model_world_m', 'background_collision_world_m',
                'gripper_surface_local_m', 'poses_world_graspgen_gripper', 'scores', 'accepted', 'exclusion_reasons']:
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-6) if np.issubdtype(a[key].dtype, np.number) else np.testing.assert_array_equal(a[key], b[key])
        exact[key] = bool(np.array_equal(a[key], b[key]))
    assert [r['reasons'] for r in ra] == [r['reasons'] for r in rb]
    assert sa['counts'] == sb['counts']
    result = {'status': 'passed', 'first': str(args.first.resolve()), 'second': str(args.second.resolve()),
              'counts': sa['counts'], 'bitwise_equal': exact,
              'max_pose_abs_difference': float(np.max(np.abs(a['poses_world_graspgen_gripper'] - b['poses_world_graspgen_gripper']))),
              'max_score_abs_difference': float(np.max(np.abs(a['scores'] - b['scores']))),
              'compute_seconds': [sa['performance']['compute_stage_seconds'], sb['performance']['compute_stage_seconds']]}
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
