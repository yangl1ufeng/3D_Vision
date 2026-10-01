"""After the active batch, replace the invalid ID-grouped metric with author-order AP.

Separate file to avoid changing the inference batch's frozen code signature.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'runs/rc3d_native'
deadline = time.monotonic()+8*60*60
while time.monotonic() < deadline:
    progress = json.loads((RUNS/'progress.json').read_text(encoding='utf-8'))
    if progress['status'] in ('complete', 'evaluation_failed', 'failed', 'stopped_code_changed'):
        break
    time.sleep(10)
else:
    raise TimeoutError('Batch did not reach a terminal state within eight hours')

if progress['status'] not in ('complete', 'evaluation_failed') or len(progress['completed']) != 100:
    (RUNS/'postprocess_status.json').write_text(json.dumps({
        'status': 'batch_incomplete', 'batch_status': progress['status'],
        'completed_pairs': len(progress['completed'])}, indent=2))
    raise RuntimeError('Batch incomplete; no full metric produced')

legacy = RUNS/'diagnostic_metrics.json'
if legacy.exists():
    quarantined = RUNS/'diagnostic_metrics_INVALID_image_id_grouping.json'
    if quarantined.exists():
        raise FileExistsError('Legacy evaluator quarantine file already exists')
    legacy.rename(quarantined)

corrected = RUNS/'diagnostic_metrics.json'
command = [sys.executable, str(ROOT/'scripts/evaluate_rc3d_author_order.py'),
           '--runs', str(RUNS), '--output', str(corrected)]
with (RUNS/'postprocess.log').open('w', encoding='utf-8') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
report = json.loads(corrected.read_text(encoding='utf-8'))
if report['completed_pairs'] != 100 or report['ap50']['gt_objects'] != 100:
    raise ValueError('Full author-order result has unexpected counts')
(RUNS/'postprocess_status.json').write_text(json.dumps({
    'status': 'corrected_metric_complete',
    'completed_at_utc': datetime.now(timezone.utc).isoformat(),
    'evaluator': 'scripts/evaluate_rc3d_author_order.py',
    'output': 'runs/rc3d_native/diagnostic_metrics.json',
    'completed_pairs': 100,
    'ap50': report['ap50']['ap']}, indent=2), encoding='utf-8')
print(json.dumps({'status': 'corrected_metric_complete', 'ap50': report['ap50']['ap']}))
