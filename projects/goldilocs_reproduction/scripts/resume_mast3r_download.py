"""Resume official MASt3R download with four verified HTTP byte ranges.

Retains the original prefix until every part is complete. Only replaces the
checkpoint after a full-length assembled file is available.
"""
import concurrent.futures
import hashlib
import json
import shutil
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
URL = 'https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
TOTAL = 2754910614
PARTS = TARGET.parent/'mast3r_parts'
PARTS.mkdir(exist_ok=True)
manifest_path = PARTS/'ranges.json'
if TARGET.stat().st_size == TOTAL:
    print('Download already complete')
    raise SystemExit(0)
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    if TARGET.stat().st_size != manifest['prefix_bytes']:
        raise ValueError('Prefix changed since range manifest was created')
else:
    prefix = TARGET.stat().st_size
    remaining = TOTAL-prefix
    if remaining <= 0:
        raise ValueError('Invalid checkpoint size')
    size = (remaining+3)//4
    manifest = {'url': URL, 'prefix_bytes': prefix, 'total_bytes': TOTAL,
        'ranges': [[start, min(start+size, TOTAL)-1] for start in range(prefix, TOTAL, size)]}
    manifest_path.write_text(json.dumps(manifest, indent=2))


def download(i, start, end):
    part = PARTS/f'part_{i}.bin'
    expected = end-start+1
    for attempt in range(4):
        have = part.stat().st_size if part.exists() else 0
        if have == expected:
            return
        if have > expected:
            raise ValueError('Oversized partial range')
        try:
            first = start+have
            with requests.get(URL, headers={'Range': f'bytes={first}-{end}', 'Accept-Encoding': 'identity'},
                              stream=True, timeout=(20, 30)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {first}-{end}/{TOTAL}':
                    raise ValueError('Server did not honor exact byte range')
                with part.open('ab') as out:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:
                            out.write(chunk)
            if part.stat().st_size == expected:
                return
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2)
    raise RuntimeError(f'Incomplete part {i}')


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    tasks = [pool.submit(download, i, start, end) for i, (start, end) in enumerate(manifest['ranges'])]
    while not all(t.done() for t in tasks):
        have = manifest['prefix_bytes'] + sum(p.stat().st_size for p in PARTS.glob('part_*.bin'))
        progress = {'downloaded_bytes': have, 'total_bytes': TOTAL, 'percent': round(have/TOTAL*100, 2)}
        (PARTS/'progress.json').write_text(json.dumps(progress), encoding='utf-8')
        print(json.dumps(progress), flush=True)
        concurrent.futures.wait(tasks, timeout=10)
    for task in tasks:
        task.result()

assembled = TARGET.with_suffix('.assembling')
with assembled.open('wb') as out:
    with TARGET.open('rb') as first:
        shutil.copyfileobj(first, out, 8*1024*1024)
    for i in range(len(manifest['ranges'])):
        with (PARTS/f'part_{i}.bin').open('rb') as part:
            shutil.copyfileobj(part, out, 8*1024*1024)
if assembled.stat().st_size != TOTAL:
    raise ValueError('Assembled size mismatch')
assembled.replace(TARGET)
h = hashlib.sha256()
with TARGET.open('rb') as f:
    for block in iter(lambda: f.read(8*1024*1024), b''):
        h.update(block)
(PARTS/'complete.json').write_text(json.dumps({'bytes': TOTAL, 'sha256': h.hexdigest()}))
print('DOWNLOAD_COMPLETE', h.hexdigest(), flush=True)
