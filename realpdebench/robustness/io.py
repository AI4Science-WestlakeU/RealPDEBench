"""Pinned assets and bounded, verified reads of released cylinder windows."""
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.error
import time
import numpy as np

MODEL_REVISION = '58898901ae1c557f810767b03b7bf5004f6ead63'
DATA_REVISION = '0ae4426085d250787f96d5456bc0d30ce5f277d2'
ASSETS = {
    'unet_real.pth': (f'https://huggingface.co/AI4Science-WestlakeU/RealPDEBench-models/resolve/{MODEL_REVISION}/cylinder/unet/real.pth', '4d6b4342a3a4bc9d03a40c99c1fc562136f4c1f259c601682ccde580db6cfc97'),
    'unet_config.yaml': (f'https://huggingface.co/AI4Science-WestlakeU/RealPDEBench-models/resolve/{MODEL_REVISION}/configs/cylinder/unet.yaml', '0d355bb50a2153d8e04fbc08c1fc885c0342ec848d5a1bd3c7dc6ac03fb5d5d4'),
    'mean_std.pt': (f'https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench/resolve/{DATA_REVISION}/cylinder/mean_std.pt', '2dff4af77180413434c2292c5c05e6cd7561f89cfecaa6b88d9c41f48da04efa'),
}


def retry_read(action):
    """Retry bounded transient public-download errors, never checksum errors."""
    for attempt in range(4):
        try: return action()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {408,429,500,502,503,504}:
                raise
            if attempt == 3: raise
            time.sleep(2**attempt)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def prepare_assets(directory, download=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in ASSETS.items():
        path = directory / name
        if not path.exists():
            if not download: raise FileNotFoundError(f'{path}; use --download to fetch pinned public assets')
            temporary = path.with_suffix(path.suffix + '.part')
            def acquire():
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as f:
                    while chunk := response.read(1024 * 1024): f.write(chunk)
            retry_read(acquire)
            if sha256(temporary) != expected: raise ValueError(f'Asset checksum mismatch: {name}')
            temporary.replace(path)
        if sha256(path) != expected: raise ValueError(f'Asset checksum mismatch: {name}')


def load_velocity(record, directory, download=False):
    name = record['window_file']
    if Path(name).name != name: raise ValueError('Window filenames must be local basenames')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    ranges = record.get('range_hashes', [])
    if not path.exists():
        if not download: raise FileNotFoundError(f'{path}; use --download to fetch pinned window ranges')
        if record.get('dataset_revision') != DATA_REVISION or len(ranges) != 2:
            raise ValueError('Bounded download requires the pinned u/v range manifest')
        shard = record['shard']
        if not shard.startswith('cylinder/hf_dataset/real/') or '..' in Path(shard).parts:
            raise ValueError('Unexpected shard path')
        columns = {}
        for field in ranges:
            if field['column'] not in {'u', 'v'}: raise ValueError('Expected measured velocity channels')
            start, end = int(field['start']), int(field['end'])
            url = f'https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench/resolve/{DATA_REVISION}/{shard}'
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            def acquire():
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                        raise ValueError('HTTP range response did not match requested boundaries')
                    return response.read(end-start+2)
            raw = retry_read(acquire)
            if len(raw) != end-start+1 or hashlib.sha256(raw).hexdigest() != field['sha256']:
                raise ValueError('Velocity range checksum mismatch')
            columns[field['column']] = np.frombuffer(raw, dtype='<f4').reshape(-1, 64, 128)
        velocity = np.stack([columns['u'], columns['v']], axis=-1)
        np.savez_compressed(path, velocity=velocity)
    with np.load(path, allow_pickle=False) as archive: velocity = archive['velocity'].copy()
    if velocity.ndim != 4 or velocity.shape[1:] != (64, 128, 2) or not np.isfinite(velocity).all():
        raise ValueError('Expected finite cylinder velocity [time,64,128,2]')
    for field in ranges:
        channel = {'u': 0, 'v': 1}[field['column']]
        raw = np.ascontiguousarray(velocity[..., channel], dtype='<f4').tobytes()
        if hashlib.sha256(raw).hexdigest() != field['sha256']: raise ValueError('Cached velocity differs from pinned range')
    return velocity
