"""Download only the official assets needed for the feasibility benchmark."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import requests

ROOT = Path(__file__).resolve().parent
FILES = [
    'checkpoints/graspgen_robotiq_2f_140.yml',
    'checkpoints/graspgen_robotiq_2f_140_gen.pth',
    'checkpoints/graspgen_robotiq_2f_140_dis.pth',
    'sample_data/real_object_pc/1740787710_319213.json',
    'sample_data/real_scene_pc/1745619598_467125.json',
]

def download(name):
    entry = next(x for x in json.loads((ROOT/'data/model_files.json').read_text()) if x['path'] == name)
    dest = ROOT/name if name.startswith('checkpoints/') else ROOT/'data'/name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix+'.part')
    if not (dest.exists() and dest.stat().st_size == entry['size']):
        for attempt in range(5):
            try:
                offset = part.stat().st_size if part.exists() else 0
                url = 'https://huggingface.co/adithyamurali/GraspGenModels/resolve/main/'+name
                with requests.get(url, params={'download':'true','t':str(time.time())},
                                  headers={'Range':f'bytes={offset}-'} if offset else {},
                                  stream=True, timeout=(30,90)) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    with part.open('ab' if append else 'wb') as out:
                        for block in response.iter_content(1024*1024):
                            if block: out.write(block)
                if part.stat().st_size != entry['size']:
                    raise RuntimeError(f'Incorrect size: {part.stat().st_size}, expected {entry["size"]}')
                part.replace(dest)
                break
            except Exception as exc:
                print(json.dumps({'file':name,'attempt':attempt,'error':str(exc)}), flush=True)
                if attempt == 4: raise
                time.sleep(2)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    expected = entry.get('lfs', {}).get('oid')
    if expected and digest != expected: raise RuntimeError('SHA256 mismatch: '+name)
    result = {'source':name,'path':str(dest.relative_to(ROOT)),'bytes':dest.stat().st_size,'sha256':digest}
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, FILES))
    (ROOT/'data/download_manifest.json').write_text(json.dumps(results,indent=2))
