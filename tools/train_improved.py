"""Experimental training pipeline with Parquet caching, L1 loss, and richer kinematics."""
import argparse
import calendar
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
from predict_depth import FEATURES as BASE_FEATURES, checked_time, sha256

EXTRA_FEATURES = [
    'accel_cm_per_min2',       # Gia tốc dâng/rút mực nước (đạo hàm bậc 2)
    'range_three_bins_cm',     # Biên độ dao động trong 15 phút (max - min)
    'slope_30min_cm_per_min',  # Xu hướng dài hơn (30 phút)
    'mean_six_bins_cm',        # Mực nước trung bình 30 phút
    'std_six_bins_cm',         # Độ biến động 30 phút
]
ALL_FEATURES = BASE_FEATURES + EXTRA_FEATURES


def month_key(name):
    parts = Path(name).stem.split('-')
    return int(parts[-1]), list(calendar.month_abbr).index(parts[-2].title())


def build_rich_features(observations, as_of):
    """Mở rộng từ build_features gốc, giữ nguyên tính nhân quả (past-only)."""
    cutoff = checked_time(as_of).floor('5min')
    source = observations.loc[observations.index <= cutoff].sort_index(kind='stable').copy()
    if source.empty:
        return pd.DataFrame(columns=ALL_FEATURES, dtype=float)
    source = source.loc[~source.index.duplicated(keep='first')]
    source['raw'] = pd.to_numeric(source.depth_raw_mm, errors='coerce').replace([np.inf, -np.inf], np.nan)
    source['bin'] = source.index.ceil('5min')
    source['source_time'] = source.index
    source['nonnegative_raw'] = source.raw.clip(lower=0)

    grouped = source.groupby('bin', sort=True)
    last = source.drop_duplicates('bin', keep='last').set_index('bin')
    last['depth'] = grouped.nonnegative_raw.median() / 10.0
    last.loc[last.raw.isna(), 'depth'] = np.nan

    grid = last.reindex(pd.date_range(last.index.min(), cutoff, freq='5min', name='decision_time_utc'))
    depth = grid.depth
    age = (grid.index.to_series() - grid.source_time).dt.total_seconds()

    frame = pd.DataFrame(index=grid.index)
    for steps in (0, 1, 2):
        frame[f'depth_lag_{5 * steps}_cm'] = depth.shift(steps)
        frame[f'source_age_lag_{5 * steps}_seconds'] = age.shift(steps)

    slope_5 = (depth - depth.shift(1)) / 5.0
    slope_prev_5 = (depth.shift(1) - depth.shift(2)) / 5.0
    frame['slope_5min_cm_per_min'] = slope_5
    frame['slope_10min_cm_per_min'] = (depth - depth.shift(2)) / 10.0

    rolling_3 = depth.rolling(3, min_periods=3)
    for name in ('mean', 'min', 'max', 'std'):
        frame[f'{name}_three_bins_cm'] = getattr(rolling_3, name)()

    # Các đặc trưng cải tiến mới
    frame['accel_cm_per_min2'] = (slope_5 - slope_prev_5) / 5.0
    frame['range_three_bins_cm'] = frame['max_three_bins_cm'] - frame['min_three_bins_cm']
    frame['slope_30min_cm_per_min'] = (depth - depth.shift(6)) / 30.0
    rolling_6 = depth.rolling(6, min_periods=4)  # Cho phép thiếu tối đa 2 bin trong 30 phút
    frame['mean_six_bins_cm'] = rolling_6.mean()
    frame['std_six_bins_cm'] = rolling_6.std()

    return frame[ALL_FEATURES]


def prepare_or_load_cache(data_dir, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / 'train_rich.parquet'
    val_cache = cache_dir / 'val_rich.parquet'
    if train_cache.exists() and val_cache.exists():
        print('Đang nạp dữ liệu đã tiền xử lý từ cache Parquet...')
        return pd.read_parquet(train_cache), pd.read_parquet(val_cache)

    inventory = json.loads((ROOT / 'DATA_INVENTORY.json').read_text(encoding='utf-8'))
    names = sorted(
        [r['relative_path'] for r in inventory['csv_source_files'] if month_key(r['relative_path']) < (2023, 7)],
        key=month_key,
    )
    stations = ['daily_new_falcon', 'ugliest_cyan_elephant', 'widely_proud_lizard']
    parts = {s: [] for s in stations}

    for name in names:
        path = data_dir / name
        with pd.read_csv(path, usecols=['time', 'deployment_id', 'depth_raw_mm', 'depth_proc_mm'], chunksize=200000) as reader:
            for chunk in reader:
                chunk = chunk.loc[chunk.deployment_id.isin(stations)].copy()
                if chunk.empty:
                    continue
                chunk.index = pd.to_datetime(chunk.time, format='mixed', utc=True, errors='raise')
                for s in stations:
                    sub = chunk.loc[chunk.deployment_id.eq(s), ['depth_raw_mm', 'depth_proc_mm']]
                    if not sub.empty:
                        parts[s].append(sub)
        print(f'Đã đọc: {name}', flush=True)

    train_end = pd.Timestamp('2023-01-01', tz='UTC')
    val_end = pd.Timestamp('2023-07-01', tz='UTC')
    embargo = pd.Timedelta(hours=24)

    splits = {'train': [], 'validation': []}
    for s, sources in parts.items():
        source = pd.concat(sources).sort_index(kind='stable')
        source = source.loc[~source.index.duplicated(keep='first')].copy()
        features = build_rich_features(source[['depth_raw_mm']], source.index.max())

        source['proc'] = pd.to_numeric(source.depth_proc_mm, errors='coerce').replace([np.inf, -np.inf], np.nan)
        source['bin'] = source.index.ceil('5min')
        proc = source.drop_duplicates('bin', keep='last').set_index('bin').proc.reindex(features.index)
        label = proc.shift(-2) / 10.0

        valid = np.isfinite(features).all(axis=1) & np.isfinite(label) & label.ge(0)
        idx = features.index
        # Lùi lookback 30 phút để tránh rò rỉ qua ranh giới embargo
        masks = {
            'train': (idx + pd.Timedelta(minutes=10)) < (train_end - embargo),
            'validation': ((idx - pd.Timedelta(minutes=30)) >= (train_end + embargo))
                          & ((idx + pd.Timedelta(minutes=10)) < (val_end - embargo)),
        }
        for split, mask in masks.items():
            kept = valid & mask
            df = features.loc[kept].copy()
            df['label'] = label.loc[kept]
            df['station'] = s
            splits[split].append(df)

    train_df = pd.concat(splits['train']).sort_index(kind='stable')
    val_df = pd.concat(splits['validation']).sort_index(kind='stable')
    train_df.to_parquet(train_cache)
    val_df.to_parquet(val_cache)
    return train_df, val_df


def calc_metrics(y, p):
    err = np.abs(y - p)
    wet = y >= 10.0
    mae_wet = float(err[wet].mean()) if wet.any() else np.nan
    mae_nonwet = float(err[~wet].mean()) if (~wet).any() else np.nan
    return {
        'windows': int(len(y)),
        'wet_windows': int(wet.sum()),
        'mae_cm': float(err.mean()),
        'mae_wet_cm': mae_wet,
        'mae_nonwet_cm': mae_nonwet,
        'balanced_mae_cm': float((mae_wet + mae_nonwet) / 2.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('./cache_features'))
    parser.add_argument('--drop-lizard', action='store_true', help='Loại bỏ trạm nhiễu widely_proud_lizard')
    parser.add_argument('--objective', default='reg:absoluteerror', choices=['reg:absoluteerror', 'reg:squarederror', 'reg:pseudohubererror', 'reg:quantileerror'])
    parser.add_argument('--wet-weight', type=float, default=20.0)
    args = parser.parse_args()

    train_df, val_df = prepare_or_load_cache(args.data_dir.resolve(), args.cache_dir.resolve())

    if args.drop_lizard:
        train_df = train_df[train_df['station'] != 'widely_proud_lizard'].copy()
        print('Đã loại bỏ trạm widely_proud_lizard khỏi tập train.')

    model = XGBRegressor(
        objective=args.objective,
        quantile_alpha=0.65 if args.objective == 'reg:quantileerror' else None,
        eval_metric='mae',
        tree_method='hist',
        learning_rate=0.04,
        max_depth=4,
        n_estimators=350,
        min_child_weight=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=15.0,
        random_state=42,
        n_jobs=4,
    )

    y_train_delta = train_df['label'] - train_df['depth_lag_0_cm']
    weights = np.where(train_df['label'] >= 10.0, args.wet_weight, 1.0)
    model.fit(train_df[ALL_FEATURES], y_train_delta, sample_weight=weights, verbose=False)

    y_val = val_df['label'].to_numpy()
    cur_val = val_df['depth_lag_0_cm'].to_numpy()
    pred_val = np.maximum(0.0, cur_val + model.predict(val_df[ALL_FEATURES]))

    report = {
        'config': {
            'objective': args.objective,
            'drop_lizard': args.drop_lizard,
            'wet_weight': args.wet_weight,
            'num_features': len(ALL_FEATURES),
        },
        'overall_validation': {
            'persistence': calc_metrics(y_val, cur_val),
            'improved_model': calc_metrics(y_val, pred_val),
        },
        'per_station_validation': {},
    }

    for st in val_df['station'].unique():
        mask = (val_df['station'] == st).to_numpy()
        report['per_station_validation'][st] = {
            'persistence': calc_metrics(y_val[mask], cur_val[mask]),
            'improved_model': calc_metrics(y_val[mask], pred_val[mask]),
        }

    out_model = Path('./runs/improved_candidate.ubj')
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out_model)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()