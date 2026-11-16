"""Standard-library-only checksum check; no inference or writing."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Missing/unsafe bundle path: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != expected['bytes'] or actual != expected['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    print(f'PASS: {len(manifest["files"])} bundle files match size and SHA256')


if __name__ == '__main__':
    verify()
