"""Replay saved planning snapshots at finer spacing; do not rerun physical motion."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pybullet as p

from simulation.planned_demo import PlannedDemo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    saved = json.loads((run/'planning.json').read_text())
    result = json.loads((run/'result.json').read_text())
    reports = []
    with tempfile.TemporaryDirectory(dir=run.parent) as temporary:
        demo = PlannedDemo(Path(temporary), cube_size=result['cube_size_m'], scenario=saved['scenario'])
        scene = demo.scene
        try:
            for entry in saved['plans']:
                if 'joint_path' not in entry:
                    continue
                for index, value in enumerate(entry['start_joint_positions']):
                    scene.b.resetJointState(scene.robot, index, value)
                scene.b.resetBasePositionAndOrientation(scene.target, *entry['target_pose_at_plan_start'])
                scene.attached = entry['payload_included']
                scene.attachment = entry['attachment_tcp']
                scene.allow_pads = entry['allow_pad_contact']
                scene.allow_support = entry['allow_table_support']
                path = [np.asarray(q) for q in entry['joint_path']]
                valid = all(scene.edge(a, b, resolution=.0025) for a, b in zip(path, path[1:]))
                reason = None
                if entry.get('direct_blocked'):
                    # Counterfactual direct motion, never executed in the live world.
                    n = max(1, int(np.ceil(np.max(np.abs(path[-1]-path[0]))/.0025)))
                    for t in np.linspace(0, 1, n+1):
                        reason = scene.collision(path[0]*(1-t)+path[-1]*t, margin=0)
                        if reason:
                            break
                reports.append({'stage': entry['stage'], 'dense_path_valid': valid,
                                'counterfactual_direct_penetration': reason})
        finally:
            scene.close()
            p.disconnect(demo.client)
    report = {'all_valid': all(r['dense_path_valid'] for r in reports) and bool(reports),
              'resolution_rad': .0025, 'same_collision_geometry': True, 'stages': reports}
    destination = run/'dense_recheck.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['all_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
