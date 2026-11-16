"""Copy only manifest files to a temporary folder and smoke-test there."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    approved = Path(r'C:\Users\Admin\AppData\Local\Temp\opencode')
    with tempfile.TemporaryDirectory(prefix='iot-handover-', dir=approved if approved.is_dir() else None) as folder:
        moved = Path(folder)
        for name in [*manifest['files'], 'MANIFEST.json']:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe manifest entry')
            target = moved / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        for arguments in (['tools/verify_manifest.py'],
                          ['runtime/predict_depth.py', '--input', 'examples/raw_synthetic.json', '--station', 'synthetic-demo']):
            completed = subprocess.run([sys.executable, '-B', *arguments], cwd=moved,
                capture_output=True, text=True, encoding='utf-8', errors='replace')
            if completed.returncode:
                raise RuntimeError(completed.stderr)
        predicted = json.loads(completed.stdout)
        expected = json.loads((moved / 'examples/expected_synthetic.json').read_text(encoding='utf-8'))
        if predicted['status'] != 'ok' or abs(predicted['predicted_depth_cm'] - expected['predicted_depth_cm']) > 1e-6:
            raise ValueError('Moved bundle did not reproduce synthetic inference')
    print('PASS: manifest-only copy runs from a different folder without original project/data paths')


if __name__ == '__main__':
    main()
