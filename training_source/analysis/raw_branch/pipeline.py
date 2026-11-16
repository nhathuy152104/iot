"""Past-only raw binning, with processed depth used ONLY as a future label.

The decision clock is the archive observation clock, NOT known arrival times.
No centered filters, interpolation, carry-forward, station ranking or outlier cuts.
"""
import numpy as np
import pandas as pd

from five_minute.pipeline import split_masks


POLICY = {
    'bin': '5min UTC (t-5min,t], finalized at t; no unfinished bins',
    'aggregation': 'median of finite raw depths in the current completed bin',
    'null_policy': 'empty bin or nonfinite raw on LAST physical row -> missing; do not fall back to an earlier valid row',
    'negative_raw': 'max(raw_mm,0) before median; retain counts/flags, not a declaration of sensor failure',
    'minimum_samples': 1,
    'high_values_and_jumps': 'retain; no upper clipping or hard gradient/outlier rejection',
    'missing': 'no interpolation, forward/backward fill or synthetic observations',
    'history': 'three raw-derived bins t-10,t-5,t; full support (t-15min,t]',
    'label': 'last physical processed-depth observation in (t+5min,t+10min], finite and nonnegative',
    'proc_usage': 'future supervised label and retrospective comparison ONLY; never an input or history eligibility rule',
    'arrival_assumption': 'raw observations assumed available at archive timestamps; transport/calibration latency not reconstructed',
    'filter_scope': 'reduces some within-bin transients when multiple samples exist; cannot reliably remove sustained boxes or sparse blips',
}


def make_raw_grid(observations, as_of=None):
    """Finalize only bins <= as_of; last-row nulls remain visible and invalidate raw.

    The median can be computed from the samples already seen in each bin, so no
    state from a subsequent bin can change a finalized raw input.
    """
    if observations.empty:
        raise ValueError('No observations')
    if observations.index.tz is None:
        raise ValueError('Timezone-aware observation timestamps required')
    cutoff = observations.index.max() if as_of is None else pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError('Timezone-aware as_of required')
    frame = observations.loc[observations.index <= cutoff].sort_index(kind='stable').copy()
    if frame.empty:
        raise ValueError('No observations available at as_of')
    frame = frame.loc[~frame.index.duplicated(keep='first')]
    for name in ('depth_raw_mm', 'depth_proc_mm'):
        frame[name] = pd.to_numeric(frame[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
    frame['bin_end_utc'] = frame.index.ceil('5min')
    frame['source_time_utc'] = frame.index
    frame['nonnegative_raw_mm'] = frame.depth_raw_mm.clip(lower=0)
    frame['negative_raw_count'] = frame.depth_raw_mm.lt(0).astype('int64')
    frame['high_raw_count'] = frame.depth_raw_mm.ge(1000).astype('int64')
    grouped = frame.groupby('bin_end_utc', sort=True)
    last = frame.drop_duplicates('bin_end_utc', keep='last').set_index('bin_end_utc')
    last['raw_median_cm'] = grouped.nonnegative_raw_mm.median() / 10
    # Do not conceal a null on the newest physical sample by using older values.
    last.loc[last.depth_raw_mm.isna(), 'raw_median_cm'] = np.nan
    last['audit_source_rows'] = grouped.size()
    last['audit_finite_raw_rows'] = grouped.depth_raw_mm.count()
    last['audit_negative_raw_rows'] = grouped.negative_raw_count.sum()
    last['audit_high_raw_ge_100cm_rows'] = grouped.high_raw_count.sum()
    last['audit_last_raw_missing'] = last.depth_raw_mm.isna()
    index = pd.date_range(last.index.min(), cutoff.floor('5min'), freq='5min', name='decision_time_utc')
    grid = last.reindex(index)
    grid['audit_bin_observed'] = grid.source_time_utc.notna()
    for name in ('audit_source_rows', 'audit_finite_raw_rows', 'audit_negative_raw_rows', 'audit_high_raw_ge_100cm_rows'):
        grid[name] = grid[name].fillna(0).astype('int64')
    grid['audit_last_raw_missing'] = grid.audit_last_raw_missing.fillna(False).astype(bool)
    grid['source_age_seconds'] = (grid.index.to_series() - grid.source_time_utc).dt.total_seconds()
    return grid


def make_raw_features(grid, station, label):
    depth = grid.raw_median_cm
    rolling = depth.rolling(3, min_periods=3)
    result = pd.DataFrame(index=grid.index)
    for steps in (0, 1, 2):
        result[f'depth_lag_{steps * 5}_cm'] = depth.shift(steps)
        result[f'source_age_lag_{steps * 5}_seconds'] = grid.source_age_seconds.shift(steps)
    result['slope_5min_cm_per_min'] = (depth - depth.shift(1)) / 5
    result['slope_10min_cm_per_min'] = (depth - depth.shift(2)) / 10
    for name in ('mean', 'min', 'max', 'std'):
        result[f'{name}_three_bins_cm'] = getattr(rolling, name)()
    result[label] = grid.depth_proc_mm.shift(-2) / 10
    result['audit_target_source_time_utc'] = grid.source_time_utc.shift(-2)
    result['audit_realized_target_lead_seconds'] = (
        result.audit_target_source_time_utc - grid.index.to_series()).dt.total_seconds()
    result['deployment_id'] = station
    result['audit_negative_raw_history'] = grid.audit_negative_raw_rows.rolling(3, min_periods=1).sum().gt(0)
    result['audit_sparse_raw_history'] = grid.audit_finite_raw_rows.rolling(3, min_periods=1).min().lt(3)
    result['audit_source_rows_current_bin'] = grid.audit_source_rows
    result['audit_negative_raw_rows_current_bin'] = grid.audit_negative_raw_rows
    result['audit_high_raw_ge_100cm_rows_current_bin'] = grid.audit_high_raw_ge_100cm_rows
    return result


def eligibility_reasons(features, label, feature_names):
    """One disjoint reason per decision; no quality/large-error pruning."""
    reason = pd.Series('kept', index=features.index, name='reason')
    checks = [
        ('missing_or_invalid_current_raw_bin', features.depth_lag_0_cm.isna()),
        ('missing_or_invalid_lag5_raw_bin', features.depth_lag_5_cm.isna()),
        ('missing_or_invalid_lag10_raw_bin', features.depth_lag_10_cm.isna()),
        ('missing_or_invalid_future_proc_label', ~np.isfinite(features[label]) | features[label].lt(0)),
        ('nonfinite_derived_feature', ~np.isfinite(features[feature_names]).all(axis=1)),
    ]
    for name, bad in checks:
        reason.loc[reason.eq('kept') & bad] = name
    return reason


def keyed(frame):
    return frame.reset_index(names='decision_time_utc').set_index(
        ['deployment_id', 'decision_time_utc']).sort_index()


def membership_summary(station, split, raw_windows, old_windows, reasons, label):
    common = raw_windows.index.intersection(old_windows.index)
    added = raw_windows.index.difference(old_windows.index)
    lost = old_windows.index.difference(raw_windows.index)
    np.testing.assert_allclose(raw_windows.loc[common, label], old_windows.loc[common, label], rtol=0, atol=0)
    return {'station': station, 'split': split, 'proc_windows': len(old_windows),
        'raw_windows': len(raw_windows), 'common_windows': len(common),
        'raw_only_windows': len(added), 'proc_only_windows': len(lost),
        'proc_wet_ge_10cm': int(old_windows[label].ge(10).sum()),
        'raw_wet_ge_10cm': int(raw_windows[label].ge(10).sum()),
        'common_wet_ge_10cm': int(raw_windows.loc[common, label].ge(10).sum()),
        'raw_only_wet_ge_10cm': int(raw_windows.loc[added, label].ge(10).sum()),
        'proc_only_wet_ge_10cm': int(old_windows.loc[lost, label].ge(10).sum()),
        'negative_raw_history_windows': int(raw_windows.audit_negative_raw_history.sum()),
        'sparse_raw_history_windows': int(raw_windows.audit_sparse_raw_history.sum()),
        **{f'proc_only_reason_{name}': int(count) for name, count in reasons.reindex(lost).value_counts().items()}}
