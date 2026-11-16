"""Run handover checks. --record writes only KIEM_TRA_BAN_GIAO.json."""
import argparse
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    tasks = [('bundle_checksums', ['tools/verify_manifest.py']),
             ('unit_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v']),
             ('synthetic_prediction', ['runtime/predict_depth.py', '--input', 'examples/raw_synthetic.json', '--station', 'synthetic-demo']),
             ('relocated_folder_smoke', ['tools/verify_portability.py'])]
    if args.training_root:
        tasks.append(('project_validation_parity', ['tools/verify_against_project.py', '--training-root', str(args.training_root.resolve())]))
    results = []
    for name, command in tasks:
        completed = subprocess.run([sys.executable, '-B', *command], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
        results.append({'check': name, 'exit_code': completed.returncode,
                        'stdout': completed.stdout, 'stderr': completed.stderr})
        print(f'{name}: {"PASS" if completed.returncode == 0 else "FAIL"}', flush=True)
    forbidden = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and
                 p.suffix.lower() in ('.csv', '.parquet') and '.venv' not in p.parts]
    passed = all(r['exit_code'] == 0 for r in results) and not forbidden
    result = {'status': 'PASS' if passed else 'FAIL', 'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope': 'preparer machine only; recipient must rerun; not deployment approval or permission to redistribute',
        'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in ('pandas', 'numpy', 'xgboost')},
        'packaged_csv_or_parquet': forbidden, 'full_download_training_cli_executed': False,
        'checks': results}
    if args.record:
        (ROOT / 'KIEM_TRA_BAN_GIAO.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
