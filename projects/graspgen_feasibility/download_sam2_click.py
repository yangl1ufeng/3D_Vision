"""Download the official SAM2.1 Tiny checkpoint in retryable verified chunks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parent
URL = 'https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt'
SIZE = 156008466
SHA = '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'


def main():
    dest = ROOT/'checkpoints/sam2.1_hiera_tiny.pt'
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == SHA:
        print('Checkpoint already verified', flush=True)
        return
    cache = ROOT/'.cache/sam2_download'
    cache.mkdir(parents=True, exist_ok=True)
    step = 4*1024*1024
    prefix = dest.read_bytes() if dest.exists() else b''
    for start in range(0, len(prefix)//step*step, step):
        (cache/f'{start}.part').write_bytes(prefix[start:start+step])
    def fetch(start):
        end = min(start+step, SIZE)-1
        path = cache/f'{start}.part'
        if path.exists() and path.stat().st_size == end-start+1:
            return path
        for attempt in range(5):
            try:
                with requests.get(URL+f'?range_start={start}&attempt={attempt}',
                                  headers={'Range': f'bytes={start}-{end}'}, timeout=(20, 45)) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                        raise ValueError('Server did not honor requested byte range')
                    content = r.content
                    if len(content) != end-start+1:
                        raise ValueError('Truncated chunk')
                    path.write_bytes(content)
                    print(f'Chunk {start//step+1}/38 received', flush=True)
                    return path
            except Exception as exc:
                print(f'Retry chunk {start//step+1}: {type(exc).__name__}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, range(0, SIZE, step)))
    temp = dest.with_suffix('.verified_download')
    with temp.open('wb') as f:
        for path in paths:
            f.write(path.read_bytes())
    if hashlib.sha256(temp.read_bytes()).hexdigest() != SHA:
        raise ValueError('Final checkpoint SHA256 mismatch')
    temp.replace(dest)
    print('Official SAM2.1 Tiny checkpoint SHA256 verified', flush=True)


if __name__ == '__main__':
    main()
