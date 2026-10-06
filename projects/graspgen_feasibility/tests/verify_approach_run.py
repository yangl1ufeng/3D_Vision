"""Compare an approach-enabled run with an existing final-only baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
from offline_core.export import write_json
from tests.verify_saved_runs import audit


def verify(baseline, current):
    old_arrays, old, old_summary = audit(baseline)
    arrays, records, summary = audit(current)
    equal = []
    for key in ['segmentation_mask', 'target_model_world_m', 'background_full_world_m',
                'background_collision_world_m', 'gripper_surface_local_m', 'poses_world_graspgen_gripper', 'scores']:
        np.testing.assert_array_equal(arrays[key], old_arrays[key])
        equal.append(key)
    removed = []
    distance = summary['approach']['distance_m']
    direction = np.asarray(summary['approach']['direction_local'])
    for previous, record in zip(old, records):
        before = previous['reasons']
        after = [reason for reason in record['reasons'] if not reason.startswith(('approach_', 'final_'))]
        assert before == after
        approach = record.get('approach')
        if approach:
            T = np.asarray(record['T_world_graspgen_gripper'])
            pre = np.asarray(approach['pregrasp_pose_world'])
            np.testing.assert_allclose(pre[:3, 3], T[:3, 3] - T[:3, :3] @ direction * distance)
            np.testing.assert_allclose(pre[:3, :3], T[:3, :3])
            assert record['accepted'] == (previous['accepted'] and approach['state'] == 'clear')
            np.testing.assert_allclose(arrays['pregrasp_poses_world'][record['candidate_id']], pre)
            assert arrays['approach_state'][record['candidate_id']] == approach['state']
        else:
            assert not previous['accepted']
        if previous['accepted'] and not record['accepted']:
            removed.append({'candidate_id': record['candidate_id'], 'approach': approach})
    return {'status': 'passed', 'bitwise_equal_baseline_arrays': equal,
            'before_accepted': old_summary['counts']['accepted'], 'after_accepted': summary['counts']['accepted'],
            'remaining_candidate_ids': summary['accepted_candidate_ids_by_rank'], 'newly_rejected': removed,
            'approach_seconds': summary['performance']['stages']['approach_check']['seconds'],
            'compute_seconds': summary['performance']['compute_stage_seconds'],
            'note': 'Integration/regression check only; official sample lacks independent complete reference geometry.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('current', type=Path)
    args = parser.parse_args()
    result = verify(args.baseline, args.current)
    write_json(args.current/'approach_verification.json', result)
    print(json.dumps(result, indent=2))
