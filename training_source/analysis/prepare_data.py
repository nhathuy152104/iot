"""Local exploratory checks and PROVISIONAL processed-depth training windows.

Archive filters' causality is unverified: these outputs are NOT approved for
real-time model evaluation. Original CSVs are read-only; outputs stay local.
"""
import calendar
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
STATIONS = {
    'daily_new_falcon': 'Long history / relatively few audit flags',
    'tired_red_monkey': 'Many threshold runs / fragmentation to inspect',
    'simply_pet_joey': 'Large jumps and short history / diagnostic comparison',
}
DEPTHS = ['depth_raw_mm', 'depth_filt_mm', 'depth_proc_mm']
VALIDATION_START = pd.Timestamp('2022-07-01', tz='UTC')
TEST_START = pd.Timestamp('2023-01-01', tz='UTC')


def split_masks(times):
    """Include the full earliest 1-minute bin, not only its right-edge label."""
    history_start = times - pd.Timedelta(minutes=11)
    target_time = times + pd.Timedelta(minutes=10)
    return {
        'train': target_time < VALIDATION_START,
        'validation': (history_start >= VALIDATION_START) & (target_time < TEST_START),
        'test': history_start >= TEST_START,
    }


def minute_grid(frame):
    # A bin labeled 10:01 uses only observations in (10:00, 10:01].
    # It never uses an observation later than its decision timestamp.
    # No interpolation or forward/backward filling.
    return frame[DEPTHS].resample('1min', label='right', closed='right').last()


def extract():
    cache = OUT / 'local_cache'
    cache.mkdir(exist_ok=True)
    if all((cache / f'{s}.parquet').exists() for s in STATIONS):
        return
    parts = {s: [] for s in STATIONS}
    paths = sorted(ROOT.glob('*.csv'), key=lambda p: (int(p.stem.split('-')[-1]),
        list(calendar.month_abbr).index(p.stem.split('-')[-2].title())))
    for path in paths:
        for chunk in pd.read_csv(path, usecols=['time', 'deployment_id', *DEPTHS], chunksize=200_000):
            for station in STATIONS:
                selected = chunk.loc[chunk['deployment_id'].eq(station)].copy()
                if not selected.empty:
                    parts[station].append(selected)
        print(f'Read {path.name}', flush=True)
    for station, batches in parts.items():
        frame = pd.concat(batches, ignore_index=True)
        frame['time'] = pd.to_datetime(frame['time'], format='mixed', utc=True)
        frame = frame.sort_values('time', kind='stable').drop_duplicates('time', keep='first')
        for col in DEPTHS:
            frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        frame.set_index('time').to_parquet(cache / f'{station}.parquet')


def plot_cases(station, grid):
    plot_dir = OUT / 'plots'
    plot_dir.mkdir(exist_ok=True)
    daily = grid.resample('1D').agg({'depth_proc_mm': 'max', 'depth_raw_mm': 'max'})
    cases = []
    peaks = daily['depth_proc_mm'].dropna().sort_values(ascending=False)
    for day in peaks.index:
        if not any(abs((day - prior).days) < 2 for _, prior in cases):
            cases.append(('high_processed_depth', day))
        if len(cases) >= 2:
            break
    disagreement = (grid['depth_raw_mm'] - grid['depth_proc_mm']).abs().resample('1D').max().dropna()
    if not disagreement.empty:
        cases.append(('raw_proc_disagreement', disagreement.idxmax()))
    dry = daily.loc[daily['depth_proc_mm'].eq(0)].index
    if len(dry):
        cases.append(('dry_comparison', dry[len(dry) // 2]))
    result = []
    for i, (kind, day) in enumerate(cases, 1):
        day_frame = grid.loc[day: day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)]
        if kind == 'raw_proc_disagreement':
            center = (day_frame['depth_raw_mm'] - day_frame['depth_proc_mm']).abs().idxmax()
        elif kind == 'dry_comparison':
            center = day + pd.Timedelta(hours=12)
        else:
            center = day_frame['depth_proc_mm'].idxmax()
        view = grid.loc[center - pd.Timedelta(hours=2): center + pd.Timedelta(hours=2)]
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, layout='constrained')
        for col, color in zip(DEPTHS, ['#777777', '#e69f00', '#0072b2']):
            axes[0].plot(view.index, view[col] / 10, label=col, color=color, marker='.', markersize=2, linewidth=.8)
        axes[0].axhline(10, color='red', linestyle='--', linewidth=.8, label='10 cm research reference')
        axes[0].set_ylabel('Depth (cm)')
        axes[0].legend(loc='upper left', fontsize=8)
        axes[0].grid(alpha=.2)
        axes[1].plot(view.index, (view['depth_raw_mm'] - view['depth_proc_mm']) / 10,
                     color='#8e44ad', marker='.', markersize=2)
        axes[1].set_ylabel('raw - proc (cm)')
        axes[1].set_xlabel('UTC; missing bins remain gaps (no interpolation)')
        axes[1].grid(alpha=.2)
        fig.suptitle(f'{station} | {kind} | {center:%Y-%m-%d %H:%M} UTC\nExploratory only: sensor trace does not verify a flood')
        path = plot_dir / f'{station}_{i}_{kind}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        result.append({'station': station, 'kind': kind, 'center_utc': str(center), 'plot': str(path),
            'proc_max_cm': float(view['depth_proc_mm'].max() / 10),
            'raw_max_cm': float(view['depth_raw_mm'].max() / 10),
            'max_abs_raw_proc_difference_cm': float((view['depth_raw_mm'] - view['depth_proc_mm']).abs().max() / 10),
            'missing_proc_bins': int(view['depth_proc_mm'].isna().sum()), 'total_bins': len(view)})
    return result


def build_windows(station, grid):
    depth = grid['depth_proc_mm'] / 10
    hist = depth.rolling(11, min_periods=11)
    valid_history = hist.count().eq(11)
    target = depth.shift(-10)
    nonnegative = hist.min().ge(0) & target.ge(0)
    eligible = valid_history & target.notna() & nonnegative
    features = pd.DataFrame(index=grid.index)
    for lag in (0, 1, 2, 3, 5, 10):
        features[f'depth_lag_{lag}_cm'] = depth.shift(lag)
    features['slope_5min_cm_per_min'] = (depth - depth.shift(5)) / 5
    features['slope_10min_cm_per_min'] = (depth - depth.shift(10)) / 10
    features['mean_10min_cm'] = hist.mean()
    features['min_10min_cm'] = hist.min()
    features['max_10min_cm'] = hist.max()
    features['std_10min_cm'] = hist.std()
    features['target_depth_10min_cm'] = target
    features['deployment_id'] = station
    # Diagnostic flags are NOT model features; raw can be very noisy.
    diff = (grid['depth_raw_mm'] - grid['depth_proc_mm']).abs()
    features['review_raw_proc_diff_ge_10cm_history'] = diff.rolling(11, min_periods=1).max().ge(100)
    features['review_proc_jump_gt_25_4cm_per_min_history'] = depth.diff().abs().gt(25.4).rolling(11, min_periods=1).max().astype(bool)
    features.index.name = 'decision_time_utc'
    masks = split_masks(features.index)
    outdir = OUT / 'provisional_windows'
    outdir.mkdir(exist_ok=True)
    stats = []
    used = 0
    for split, mask in masks.items():
        selected = features.loc[eligible & mask]
        used += len(selected)
        selected.to_parquet(outdir / f'{station}_{split}.parquet')
        wet = selected['target_depth_10min_cm'].ge(10)
        # Persistent current state is a baseline prediction, not a trained model.
        persistence_error = (selected['depth_lag_0_cm'] - selected['target_depth_10min_cm']).abs()
        stats.append({'station': station, 'split': split, 'windows': len(selected),
            'first_decision_utc': str(selected.index.min()) if len(selected) else None,
            'last_decision_utc': str(selected.index.max()) if len(selected) else None,
            'target_ge_10cm': int(wet.sum()),
            'review_raw_proc_diff_ge_10cm': int(selected['review_raw_proc_diff_ge_10cm_history'].sum()),
            'review_proc_jump': int(selected['review_proc_jump_gt_25_4cm_per_min_history'].sum()),
            'persistence_mae_cm': float(persistence_error.mean()) if len(selected) else None,
            'persistence_mae_target_ge_10cm_cm': float(persistence_error.loc[wet].mean()) if wet.any() else None})
    return {'station': station, 'grid_minutes': len(grid), 'missing_proc_bins': int(depth.isna().sum()),
        'complete_history_bins': int(valid_history.sum()), 'eligible_before_split_purge': int(eligible.sum()),
        'purged_boundary_windows': int(eligible.sum()) - used, 'splits': stats}


def main():
    extract()
    results, cases = [], []
    for station in STATIONS:
        frame = pd.read_parquet(OUT / 'local_cache' / f'{station}.parquet')
        grid = minute_grid(frame)
        cases += plot_cases(station, grid)
        results.append(build_windows(station, grid))
        print(f'Prepared {station}', flush=True)
    pd.DataFrame(cases).to_csv(OUT / 'plot_cases.csv', index=False, encoding='utf-8-sig')
    summary = {'status': 'PROVISIONAL: archive-filter causality unverified; not field-validated',
        'stations': STATIONS, 'time_grid': '1min right-closed/right-labeled; last valid value per field; no filling',
        'history': '11 valid nonnegative proc bins covering t-10min through t inclusive',
        'target': 'valid nonnegative proc bin at t+10min; intermediate FUTURE bins not required',
        'splits': {'train': 'all support before 2022-07-01 UTC; earliest history bin starts at t-11min',
            'validation': 'all support >=2022-07-01 and <2023-01-01 UTC',
            'test': 'all support >=2023-01-01 UTC'},
        'limits': ['No XGBoost trained or hyperparameters selected.',
            'Only three exploratory stations, not all 58 pluvial deployments.',
            'Temporal split prevents overlapping supports; cross-boundary weather events still need review.',
            'Processed archive may contain noncausal filtering; retrospective metrics are exploratory only.',
            'Short-lived between-bin extremes are not the prediction target.',
            'Do not publish or redistribute local cache/windows without checking FloodNet agreement.'],
        'results': results, 'plot_cases': cases}
    (OUT / 'preparation_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    rows = [s for r in results for s in r['splits']]
    pd.DataFrame(rows).to_csv(OUT / 'window_counts.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'window_counts': rows, 'plots': len(cases)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
