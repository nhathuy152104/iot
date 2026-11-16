"""Verify a completed part-A diagnostic artifact set without writing or fitting."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from delta_experiment import read_json
from train_baseline_xgboost import digest


def validate(run):
    manifest = read_json(run / 'artifact_validation.json')
    summary = read_json(run / 'summary.json')
    assert manifest['status'] == summary['status'] == 'PASS'
    assert summary['fit_called'] is False and summary['test_loaded'] is False
    for name, checksum in manifest['output_sha256'].items():
        assert digest(run / name) == checksum, name
    for name, checksum in manifest['input_sha256'].items():
        assert digest(Path(name)) == checksum, name
    rows = pd.read_csv(run / 'paired_error_decomposition.csv')
    for _, part in rows.groupby('dimension'):
        assert part.windows.sum() == 92215
        np.testing.assert_allclose(part.balanced_change_contribution_cm.sum(), summary['balanced_change_cm'], atol=1e-10)
        for key, count in summary['threshold_transitions'].items():
            assert part[key].sum() == count, key
    print(f'PASS: {len(manifest["output_sha256"])} output checksums; '
          f'{len(manifest["input_sha256"])} input checksums; all paired partitions reconcile')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    validate(parser.parse_args().run_directory.resolve())
