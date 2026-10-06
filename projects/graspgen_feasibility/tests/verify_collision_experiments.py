"""Recompute metrics and audit all saved candidate decisions independently."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from offline_core.export import write_json


def read(folder, name):
    with (folder/name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def audit(folder):
    reference = read(folder, 'reference.csv')
    assert len(reference) == 72
    refs = {(r['scene_id'], r['candidate']): r for r in reference}
    assert len(refs) == 72
    for r in reference:
        assert (r['full_approach_collision'] == 'True') >= (r['final_all_collision'] == 'True') >= (r['final_background_collision'] == 'True')
    checks = [('baseline_candidates.csv', 'baseline_summary.csv', ['background_points', 'surface_points', 'margin_m'], 'final_background_collision', 5832),
              ('approach_candidates.csv', 'approach_summary.csv', ['margin_m'], 'full_approach_collision', 648),
              ('path_comparison.csv', 'path_summary.csv', ['method'], 'full_approach_collision', 648)]
    for raw_name, summary_name, keys, reference_key, expected in checks:
        rows = read(folder, raw_name)
        assert len(rows) == expected
        for r in rows:
            assert r['reference_collision'] == refs[(r['scene_id'], r['candidate'])][reference_key]
        unique = [(r['scene_id'], r['candidate'], r['seed'], *(r[k] for k in keys)) for r in rows]
        assert len(set(unique)) == len(rows)
        for summary in read(folder, summary_name):
            subset = [r for r in rows if all(r[k] == summary[k] for k in keys) and (summary['split'] == 'all' or r['split'] == summary['split'])]
            confusion = Counter((r['predicted_collision'] == 'True', r['reference_collision'] == 'True') for r in subset)
            accepted = confusion[False, False] + confusion[False, True]
            assert int(summary['total']) == len(subset)
            assert int(summary['accepted']) == accepted
            assert int(summary['false_accept']) == confusion[False, True]
            assert int(summary['false_reject']) == confusion[True, False]
            assert int(summary['retained_reference_clear']) == confusion[False, False]
            if accepted:
                assert abs(float(summary['false_accept_among_accepted']) - confusion[False, True]/accepted) < 1e-12
            else:
                assert summary['false_accept_among_accepted'] == ''
    paths = {(r['scene_id'], r['candidate'], r['seed'], r['method']): r['predicted_collision'] == 'True' for r in read(folder, 'path_comparison.csv')}
    for scene, candidate in refs:
        for seed in ['1000', '1001', '1002']:
            assert paths[scene, candidate, seed, 'combined_pipeline'] == (paths[scene, candidate, seed, 'final_only_default'] or paths[scene, candidate, seed, 'continuous_approach_only'])
    thin = json.loads((folder/'thin_obstacle.json').read_text())
    assert thin['reference_continuous_collision'] and not thin['reference_final_collision']
    assert thin['continuous']['state'] == 'collision'
    assert thin['discrete'][0]['detected'] is False
    assert all(r['detected'] for r in thin['discrete'][1:])
    return {'status': 'passed', 'reference_candidates': 72, 'baseline_evaluations': 5832,
            'approach_evaluations': 648, 'path_comparison_evaluations': 648,
            'checks': ['reference consistency', 'unique candidate identities', 'all summary confusion counts and denominators',
                       'combined filter semantics', 'thin obstacle endpoint miss and continuous detection']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    write_json(args.folder/'verification.json', result)
    print(json.dumps(result, indent=2))
