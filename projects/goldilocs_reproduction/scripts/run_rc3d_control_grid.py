"""Sequential alpha x identity controls on an identical unfiltered candidate pool."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    first=ROOT/'runs/rc3d_paper_main_pilot'
    out=ROOT/'audit/fix_round2'
    out.mkdir(parents=True,exist_ok=True)
    state={'scope':'same eight posthoc diagnostic pairs, not an independent benchmark',
           'status':'waiting_for_fresh_reference','completed':[],'current':first.name}
    def save():
        state['updated_at_utc']=datetime.now(timezone.utc).isoformat()
        (out/'grid_progress.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    try:
        while True:
            p=json.loads((first/'progress.json').read_text())
            if p['status']=='complete':break
            if p['status']!='running':raise RuntimeError(f"Reference ended: {p['status']}")
            save();time.sleep(20)
        ids=json.loads((first/'plan.json').read_text())['pair_ids']
        state.update(status='running',completed=[first.name],pair_ids=ids)
        from scripts.collect_fix_round2 import collect
        collect()
        for name in ('paper_appendix','verified_main','verified_appendix'):
            target=ROOT/f'runs/rc3d_{name}_pilot'
            state['current']=target.name;save()
            print('START '+target.name,flush=True)
            cmd=[sys.executable,'-u',str(ROOT/'run_rc3d_revision.py'),'--out',str(target),
                 '--config',str(ROOT/f'configs/rc3d_{name}.json'),'--paired-cache',str(first),
                 '--ids',*ids]
            with (out/f'{name}_console.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{target.name} failed; inspect its console log')
            progress=json.loads((target/'progress.json').read_text())
            if progress['status']!='complete':raise RuntimeError('Child did not complete')
            state['completed'].append(target.name);save()
            collect()
            print('DONE '+target.name,flush=True)
        state.update(status='complete',current=None);save()
    except Exception as exc:
        state.update(status='failed',error=str(exc));save()
        raise


if __name__=='__main__':
    main()
