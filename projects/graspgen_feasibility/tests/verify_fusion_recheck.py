"""Audit fixed-candidate recheck artifacts and exercise rejected-fusion CLI isolation."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from offline_core.config import ROOT
from offline_core.scene_io import file_hash
from offline_core.export import write_json
from tests.verify_saved_runs import audit as audit_source


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(folder):
    report = read(folder/'recheck_summary.json')
    original = Path(report['source_run'])
    fusion = Path(report['fusion_run'])
    audit_source(original)
    provenance = read(folder/'provenance.json')
    for name, value in provenance['source_run_sha256'].items():
        assert file_hash(original/name) == value
    for name, value in provenance['fusion_artifacts_sha256'].items():
        assert file_hash(fusion/name) == value
    for name, value in provenance['source_sha256'].items():
        assert file_hash(ROOT/name) == value
    records = read(folder/'candidates.json')['candidates']
    with (folder/'candidates.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with np.load(folder/'rechecked_arrays.npz') as data, np.load(original/'scene_arrays.npz') as source:
        np.testing.assert_array_equal(data['scores'], source['scores'])
        np.testing.assert_array_equal(data['original_poses_world_m'], source['poses_world_graspgen_gripper'])
        np.testing.assert_array_equal(data['gripper_surface_local_m'], source['gripper_surface_local_m'])
        np.testing.assert_allclose(data['poses_world_m'], data['source_camera_correction_world'] @ source['poses_world_graspgen_gripper'], atol=1e-12, rtol=0)
        assert len(records) == len(rows) == len(data['scores']) == report['candidate_count']
        for i, (record, row) in enumerate(zip(records, rows)):
            assert record['candidate_id'] == int(row['candidate_id']) == i
            assert record['accepted'] == bool(data['accepted'][i]) == (row['accepted'] == 'True')
            assert record['score'] == float(data['scores'][i])
            np.testing.assert_allclose(record['T_world_graspgen_gripper'], data['poses_world_m'][i], atol=1e-12, rtol=0)
            if 'observation' in record:
                o = record['observation']
                assert o['state'] == row['observation_state']
                assert o['cells'] == sum(o[k] for k in ['free_cells', 'unknown_cells', 'occupied_cells', 'conflict_cells'])
            if record['accepted']:
                assert not record['reasons']
                assert record['approach']['state'] == 'clear'
                assert record['observation']['state'] == 'observed_free'
        assert sum(r['accepted'] for r in records) == report['after']['accepted']
        assert sum(r.get('observation', {}).get('unknown_cells', 0) for r in records) == report['after']['unknown_cells']
    return {'path': str(folder), 'candidates': len(records), 'status': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    parser.add_argument('--rejected-fusion', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    results = [audit(folder) for folder in args.folders]
    source = read(args.folders[0]/'recheck_summary.json')['source_run']
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)/'rejected'
        attempt = subprocess.run([sys.executable, str(ROOT/'recheck_fused_scene.py'), '--run', source,
                                  '--fusion', str(args.rejected_fusion.resolve()), '--output', str(output)], capture_output=True, text=True)
        assert attempt.returncode != 0
        assert 'fusion was rejected' in read(output/'failure.json')['error']
        assert not (output/'candidates.json').exists() and not (output/'rechecked_arrays.npz').exists()
    result = {'status': 'passed', 'runs': results, 'rejected_fusion_cli': 'refused before candidate export',
              'checks': ['original pipeline artifacts', 'source and fusion file hashes', 'candidate IDs, scores and samples preserved',
                         'source-camera correction applied to every candidate', 'JSON/CSV/NPZ consistency', 'no acceptance with unresolved observations']}
    write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
