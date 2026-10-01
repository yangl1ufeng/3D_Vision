"""Verify controls share candidates and collect completed results transparently."""
import hashlib
import json
from pathlib import Path
import sys
import argparse
import time
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_rc3d_revision_stages import audit
from scripts.evaluate_absence_evidence import evaluate as evaluate_scores

NAMES = ('paper_main', 'paper_appendix', 'verified_main', 'verified_appendix')


def candidate_fingerprint(folder, direction):
    trace = json.loads((folder/f'{direction}_tracking.json').read_text())
    digest = hashlib.sha256()
    with np.load(folder/f'{direction}_tracking.npz') as masks:
        for row in trace:
            mask = masks[f"source_{row['id']:04d}"]
            digest.update(str(mask.shape).encode())
            digest.update(np.packbits(mask).tobytes())
            digest.update(json.dumps([row['predicted_iou'],row['proposal_origin']]).encode())
    return {'count':len(trace), 'sha256':digest.hexdigest()}


def collect():
    out = ROOT/'audit/fix_round2'
    rows, reference_plan, reference_fingerprints = [], None, None
    for name in NAMES:
        run = ROOT/f'runs/rc3d_{name}_pilot'
        progress_path = run/'progress.json'
        if not progress_path.exists():
            rows.append({'profile':name, 'status':'not_started'})
            continue
        progress = json.loads(progress_path.read_text())
        row = {'profile':name, 'status':progress['status'],
               'completed':len(progress['completed']), 'requested':progress['requested']}
        rows.append(row)
        if progress['status'] != 'complete':
            continue
        plan = json.loads((run/'plan.json').read_text())
        if reference_plan is None:
            reference_plan = plan
        for key in ('pair_ids','manifest_sha256','source_sha256','checkpoint_sha256','geometry_source'):
            if plan[key] != reference_plan[key]:
                raise ValueError(f'{name}: control mismatch for {key}')
        fingerprints = {}
        for pid in plan['pair_ids']:
            meta = json.loads((run/pid/'run.json').read_text())
            fingerprints[pid] = {'input_sha256':meta['input_sha256'], 'geometry_sha256':meta['geometry_sha256']}
            for view in (0,1):
                for direction in ('reference_to_target','target_to_reference'):
                    key = f'view_{view}/{direction}'
                    fingerprints[pid][key] = candidate_fingerprint(run/pid/f'view_{view}',direction)
        if reference_fingerprints is None:
            reference_fingerprints = fingerprints
        if fingerprints != reference_fingerprints:
            raise ValueError(f'{name}: actual candidate/input/geometry mismatch')
        row['identical_candidate_pool_verified'] = True
        metrics = json.loads((run/'metrics.json').read_text())
        row['primary_metrics'] = metrics['metrics']
        row['supplemental_object_box_metrics'] = metrics['object_box_metrics']
        stages = audit(run)
        (out/f'{name}_stages.json').write_text(json.dumps(stages,indent=2),encoding='utf-8')
        row['stage_counts'] = stages['counts']
        try:
            scores = evaluate_scores(run)
            (out/f'{name}_absence_scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
            row['supplemental_absence_evidence_metrics'] = scores['supplemental_metrics']
        except (KeyError, ValueError) as exc:
            # Keep a valid primary experiment even if optional evidence is missing.
            row['supplemental_absence_evidence_error'] = str(exc)
    complete = all(r['status']=='complete' for r in rows)
    report = {'scope':'four controls on eight fixed posthoc diagnostic pairs; not independent validation or full main reproduction',
        'updated_at_utc':datetime.now(timezone.utc).isoformat(), 'complete':complete,
        'official_author_evaluation_verified':False, 'profiles':rows}
    if reference_fingerprints is not None:
        (out/'candidate_fingerprints.json').write_text(json.dumps(reference_fingerprints,indent=2),encoding='utf-8')
    (out/'control_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines = ['# 第二轮修复：运行结果', '', '固定 8 对、16 个视角。此集合用于错误诊断，不能作为独立验证集或原论文全量成绩。', '',
             '| 配置 | 进度 | 源 AP50 | 目标 AP50 | 合并 AP50 | TP / FP |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        m = row.get('primary_metrics')
        if m:
            lines.append(f"| {row['profile']} | 完成 | {m['source']['ap']:.4f} | {m['target']['ap']:.4f} | {m['pooled']['ap']:.4f} | {m['pooled']['true_positives']} / {m['pooled']['false_positives']} |")
        else:
            lines.append(f"| {row['profile']} | {row['status']} {row.get('completed',0)}/8 | — | — | — | — |")
    lines += ['', '主指标继续使用类别优先级裁剪后的框与原始 SAM2 分割质量分数。完整对象框和固定存在性 logit 评分仅作为补充指标，保存在 JSON 中。', '',
              '候选一致性逐掩码、逐分数、逐来源校验；同时核对输入、几何、源码、权重与清单哈希。']
    (out/'CONTROL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status_path = ROOT/'reproduction_status.json'
    status = json.loads(status_path.read_text())
    status['updated_at_utc'] = report['updated_at_utc']
    status['revision']['fix_round2']['control_summary'] = 'audit/fix_round2/control_summary.json'
    if complete:
        status['revision']['status'] = 'fix_round2_subset_controls_complete_full_reproduction_unresolved'
        status['revision']['active_progress'] = None
        status['revision']['fix_round2']['status'] = 'four_fixed_subset_controls_complete_full_reproduction_unresolved'
        status['revision']['fix_round2']['completed_profiles'] = list(NAMES)
        status['revision']['fix_round2']['active_progress'] = None
    status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps({'complete':complete, 'profiles':rows},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    collect()
    if args.watch:
        previous = []
        while True:
            state = json.loads((ROOT/'audit/fix_round2/grid_progress.json').read_text())
            if state['completed'] != previous or state['status'] in ('complete','failed'):
                collect()
                previous = state['completed']
            if state['status']=='failed':
                raise RuntimeError(state.get('error','Control grid failed'))
            if state['status']=='complete':
                break
            time.sleep(20)
