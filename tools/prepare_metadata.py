"""Maintainer tool: generate local inventory, synthetic expectation and checksums.

Only run when deliberately preparing a NEW handover snapshot. This does not
authorize distribution, refresh source snapshots, or alter original artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY = PureWindowsPath(r'D:\CODE\IOT training')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def manifest():
    files = {}
    for folder, directories, names in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in ('.venv', '__pycache__', '.pytest_cache', '.git'))
        for name in sorted(names):
            path = Path(folder) / name
            relative = path.relative_to(ROOT).as_posix()
            if relative in ('MANIFEST.json', 'KIEM_TRA_BAN_GIAO.json'):
                continue
            if name == '.env':
                raise ValueError('Secret file detected; do not package .env')
            files[relative] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    write(ROOT / 'MANIFEST.json', {'schema_version': 1, 'status': 'LOCAL_HANDOVER_CHECKSUMS_NOT_A_SIGNATURE',
        'excluded': ['MANIFEST.json (self)', 'KIEM_TRA_BAN_GIAO.json (test execution record)',
                     '.venv/', '__pycache__/', '.pytest_cache/', '.git/'], 'files': files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.training_root.resolve()
    origin = source / 'analysis/training_runs/20261110T043429_532355Z_raw_delta'
    import pandas as pd
    metrics = pd.read_csv(origin / 'metrics.csv').astype(object)
    write(ROOT / 'model/metrics.json', metrics.where(pd.notna(metrics), None).to_dict('records'))
    windows = json.loads((ROOT / 'model/window_provenance.json').read_text(encoding='utf-8'))
    cache = json.loads((ROOT / 'model/source_provenance.json').read_text(encoding='utf-8'))['cache_files']
    resources = []
    for row in windows:
        relative = PureWindowsPath(row['path']).relative_to(LEGACY).as_posix()
        resources.append({'relative_path': relative, 'purpose': row['split'], 'station': row['station'],
            'rows': row['rows'], 'recorded_sha256': row['sha256'], 'included_in_bundle': False})
    caches = []
    for row in cache:
        caches.append({'relative_path': PureWindowsPath(row['path']).relative_to(LEGACY).as_posix(),
            'station': row['station'], 'rows': row['rows'], 'recorded_sha256': row['sha256'], 'included_in_bundle': False})
    csv = [{'relative_path': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
            'integrity': 'size/mtime fingerprint only; full CSV SHA256 not computed', 'included_in_bundle': False}
           for p in sorted(source.glob('floodnet-flood-sensor-*.csv'))]
    write(ROOT / 'DATA_INVENTORY.json', {'scope': 'Selected three-core raw run only; not a complete inventory of all experiments',
        'model_run_relative_path': origin.relative_to(source).as_posix(),
        'source_url': 'https://zenodo.org/records/10211443',
        'metrics_conversion': {'source': (origin / 'metrics.csv').relative_to(source).as_posix(),
                               'source_sha256': digest(origin / 'metrics.csv'), 'bundle_target': 'model/metrics.json'},
        'distribution': 'NOT bundled; obtain independently or transfer only with required permission',
        'csv_source_files': csv, 'raw_window_files': resources, 'raw_source_cache_files': caches,
        'additional_project_resources': ['analysis/five_minute/selection_manifest.json',
            'analysis/five_minute/provisional_windows/', 'analysis/five_minute/review_18/',
            'analysis/training_runs/20261110T043429_532355Z_raw_delta/validation_predictions_local.parquet',
            'Other runs/windows/provenance are needed to reproduce other experiments; see reference docs.']})
    manifest()
    sys.path.insert(0, str(ROOT / 'runtime'))
    from predict_depth import predict, read_input
    frame, station = read_input(ROOT / 'examples/raw_synthetic.json')
    write(ROOT / 'examples/expected_synthetic.json', predict(frame, frame.index.max(), station))
    manifest()
    print(f'Prepared manifest/inventory/synthetic expectation: {ROOT}')


if __name__ == '__main__':
    main()
