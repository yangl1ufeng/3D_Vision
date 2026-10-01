"""Sequential, resumable RC-3D prototype inference. No metrics claimed here."""
import argparse
import json
import subprocess
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from goldilocs_local.reconstruct import sha256

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=1, help='first N fixed manifest entries; 100 for full dataset')
    p.add_argument('--out', type=Path, default=ROOT/'runs/rc3d_prototype')
    p.add_argument('--config', type=Path, default=ROOT/'configs/pairwise_prototype.json')
    args = p.parse_args()
    args.out = args.out.resolve()
    args.config = args.config.resolve()
    records = json.loads((ROOT/'data/rc3d_manifest.json').read_text())['pairs']
    if not 1 <= args.limit <= len(records):
        raise ValueError('limit must be between 1 and 100')
    args.out.mkdir(parents=True, exist_ok=True)
    def now():
        return datetime.now(timezone.utc).isoformat()
    sources = sorted((ROOT/'goldilocs_local').glob('*.py')) + [ROOT/'run_pair.py', ROOT/'scripts/evaluate_rc3d_diagnostic.py']
    signature = {'config': json.loads(args.config.read_text()),
                 'code_sha256': {str(f.relative_to(ROOT)): sha256(f) for f in sources},
                 'manifest_sha256': sha256(ROOT/'data/rc3d_manifest.json')}
    signature_path = args.out/'batch_signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError('Batch code, configuration or dataset manifest changed. Use a new output directory.')
    signature_path.write_text(json.dumps(signature, indent=2), encoding='utf-8')
    progress = {'kind': 'independent_inference', 'status': 'running', 'pid': os.getpid(),
                'started_at_utc': now(), 'requested_pairs': args.limit, 'completed': [], 'failed': []}
    def save_progress():
        progress['updated_at_utc'] = now()
        temporary = args.out/'progress.tmp'
        temporary.write_text(json.dumps(progress, indent=2), encoding='utf-8')
        # Windows readers can briefly hold the destination without delete sharing.
        # Retry the atomic replacement instead of aborting an otherwise valid run.
        for attempt in range(20):
            try:
                temporary.replace(args.out/'progress.json')
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.25)
    save_progress()
    for pair in records[:args.limit]:
        if any(sha256(f) != signature['code_sha256'][str(f.relative_to(ROOT))] for f in sources):
            progress['status'] = 'stopped_code_changed'
            save_progress()
            raise RuntimeError('Inference or evaluation code changed during the batch')
        progress['current_pair'] = pair['id']
        save_progress()
        folder = args.out/pair['id']
        folder.mkdir(parents=True, exist_ok=True)
        done = folder/'run.json'
        if done.exists():
            old_settings = json.loads((folder/'effective_settings.json').read_text())
            new_settings = json.loads(args.config.read_text())
            old = json.loads(done.read_text())
            if old_settings != new_settings or old['status'] != 'pairwise_inference_complete':
                raise ValueError('Existing run has different settings or incomplete inference; choose another output.')
        else:
            with (folder/'console.log').open('w', encoding='utf-8') as log:
                run = subprocess.run([sys.executable, '-u', str(ROOT/'run_pair.py'),
                    '--before', str(ROOT/pair['before']), '--after', str(ROOT/pair['after']),
                    '--out', str(folder), '--config', str(args.config)], stdout=log, stderr=subprocess.STDOUT)
            if run.returncode:
                progress['failed'].append(pair['id'])
                progress['status'] = 'failed'
                save_progress()
                raise RuntimeError(f"Inference failed: {pair['id']}. Inspect console.log")
        progress['completed'].append(pair['id'])
        save_progress()
        print(f"Completed {len(progress['completed'])}/{args.limit}: {pair['id']}", flush=True)
    progress['status'] = 'inference_complete'
    progress['current_pair'] = None
    save_progress()
    if args.limit == len(records):
        evaluator = ROOT/'scripts/evaluate_rc3d_author_order.py'
        progress['evaluator_sha256'] = sha256(evaluator)
        evaluated = subprocess.run([sys.executable, str(evaluator),
            '--runs', str(args.out), '--output', str(args.out/'diagnostic_metrics.json')])
        progress['status'] = 'complete' if evaluated.returncode == 0 else 'evaluation_failed'
        save_progress()
        if evaluated.returncode == 0:
            metrics = json.loads((args.out/'diagnostic_metrics.json').read_text(encoding='utf-8'))
            if metrics['completed_pairs'] != len(records) or metrics['evaluator_status'] != 'corrected_author_order':
                raise ValueError('Unexpected final evaluation scope or convention')
            (args.out/'postprocess_status.json').write_text(json.dumps({
                'status': 'corrected_metric_complete', 'completed_at_utc': now(),
                'evaluator': 'scripts/evaluate_rc3d_author_order.py',
                'evaluator_sha256': sha256(evaluator), 'completed_pairs': len(records),
                'ap50': metrics['ap50']['ap']}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
