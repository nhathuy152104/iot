"""Đánh giá nghiệm thu cuối cùng kiến trúc Two-Stage trên 3 tệp Test (Jul, Aug, Oct 2023)."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
sys.path.insert(0, str(ROOT / 'tools'))

from predict_depth import FEATURES as BASE_FEATURES, load_checked_model
from train_improved import ALL_FEATURES, build_rich_features
from benchmark_approaches import add_tidal_features
from full_metrics import classification_metrics, event_level_metrics, regression_and_tail_metrics

TEST_FILES = [
    'floodnet-flood-sensor-jul-2023.csv',
    'floodnet-flood-sensor-aug-2023.csv',
    'floodnet-flood-sensor-oct-2023.csv',
]
STATIONS = ['daily_new_falcon', 'ugliest_cyan_elephant']


def load_test_windows(data_dir: Path) -> pd.DataFrame:
    cache_path = ROOT / 'cache_features' / 'test_rich.parquet'
    if cache_path.is_file():
        print(f'Đang nạp dữ liệu Test từ cache: {cache_path}')
        return pd.read_parquet(cache_path)

    print('Đang trích xuất đặc trưng từ 3 tệp CSV tập Test (Jul, Aug, Oct 2023)...')
    parts = {s: [] for s in STATIONS}
    for name in TEST_FILES:
        path = data_dir / name
        with pd.read_csv(
            path,
            usecols=['time', 'deployment_id', 'depth_raw_mm', 'depth_proc_mm'],
            chunksize=200000,
        ) as reader:
            for chunk in reader:
                chunk = chunk.loc[chunk.deployment_id.isin(STATIONS)].copy()
                if chunk.empty:
                    continue
                chunk.index = pd.to_datetime(chunk.time, format='mixed', utc=True, errors='raise')
                for s in STATIONS:
                    sub = chunk.loc[chunk.deployment_id.eq(s), ['depth_raw_mm', 'depth_proc_mm']]
                    if not sub.empty:
                        parts[s].append(sub)
        print(f'Đã đọc: {name}', flush=True)

    test_start = pd.Timestamp('2023-07-01', tz='UTC') + pd.Timedelta(hours=24)
    frames = []
    for s, sources in parts.items():
        if not sources:
            continue
        source = pd.concat(sources).sort_index(kind='stable')
        source = source.loc[~source.index.duplicated(keep='first')].copy()
        features = build_rich_features(source[['depth_raw_mm']], source.index.max())

        source['proc'] = pd.to_numeric(source.depth_proc_mm, errors='coerce').replace([np.inf, -np.inf], np.nan)
        source['bin'] = source.index.ceil('5min')
        proc = source.drop_duplicates('bin', keep='last').set_index('bin').proc.reindex(features.index)
        label = proc.shift(-2) / 10.0

        df_s = features.copy()
        df_s['label'] = label
        df_s['station'] = s
        df_s = df_s.loc[df_s.index >= test_start]
        df_s = df_s.dropna(subset=ALL_FEATURES + ['label'])
        df_s = df_s.loc[df_s['label'] >= 0.0]
        frames.append(df_s)

    test_df = pd.concat(frames).sort_index(kind='stable')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(cache_path)
    return test_df


def main():
    data_dir = ROOT / 'data_floodnet'
    test_df = load_test_windows(data_dir)
    test_df, tidal_cols = add_tidal_features(test_df)

    y_true = test_df['label'].to_numpy()
    cur_depth = test_df['depth_lag_0_cm'].to_numpy()

    # 1. Persistence
    test_df['pred_persistence'] = cur_depth

    # 2. Mô hình gốc (raw_delta_d3_weight20)
    try:
        frozen_model = load_checked_model()
    except FileNotFoundError:
        cands = sorted(ROOT.glob('runs/*_downloaded_raw_candidate/candidate.ubj'))
        frozen_model = XGBRegressor(n_jobs=4)
        frozen_model.load_model(cands[-1])
    test_df['pred_frozen'] = np.maximum(0.0, cur_depth + frozen_model.predict(test_df[BASE_FEATURES]))

    # 3. Kiến trúc chốt Two-Stage (Tidal Classifier + L1 Regressor)
    art_dir = ROOT / 'runs' / 'twostage_final'
    clf = XGBClassifier(n_jobs=4)
    clf.load_model(art_dir / 'flood_classifier_tidal.ubj')

    reg = XGBRegressor(n_jobs=4)
    reg.load_model(art_dir / 'depth_regressor_l1.ubj')

    pred_depth = np.maximum(0.0, cur_depth + reg.predict(test_df[ALL_FEATURES]))
    prob_flood = clf.predict_proba(test_df[tidal_cols])[:, 1]

    test_df['pred_twostage_depth'] = pred_depth
    test_df['pred_twostage_prealert'] = np.where(prob_flood >= 0.60, 10.5, 0.0)
    test_df['pred_twostage_alert'] = np.where((prob_flood >= 0.80) | (pred_depth >= 10.0), 10.5, 0.0)

    ev_pers = event_level_metrics(test_df, 'pred_persistence', threshold=10.0)
    ev_froz = event_level_metrics(test_df, 'pred_frozen', threshold=10.0)
    ev_pre = event_level_metrics(test_df, 'pred_twostage_prealert', threshold=10.0)
    ev_alrt = event_level_metrics(test_df, 'pred_twostage_alert', threshold=10.0)
    ev_dep = event_level_metrics(test_df, 'pred_twostage_depth', threshold=10.0)

    report = {
        'test_split': 'Jul, Aug, Oct 2023 (Out-of-sample)',
        'total_test_windows': int(len(test_df)),
        'wet_windows_ge_10cm': int((y_true >= 10.0).sum()),
        '1_Persistence': {
            'regression': regression_and_tail_metrics(y_true, test_df['pred_persistence'].to_numpy()),
            'classification_10cm': classification_metrics(y_true, test_df['pred_persistence'].to_numpy(), 10.0),
            'events': {k: v for k, v in ev_pers.items() if k != 'events_detail'},
        },
        '2_Frozen_Original_Model': {
            'regression': regression_and_tail_metrics(y_true, test_df['pred_frozen'].to_numpy()),
            'classification_10cm': classification_metrics(y_true, test_df['pred_frozen'].to_numpy(), 10.0),
            'events': {k: v for k, v in ev_froz.items() if k != 'events_detail'},
        },
        '3_TwoStage_Final_Architecture': {
            'depth_regression_L1': regression_and_tail_metrics(y_true, pred_depth),
            'depth_peak_error_cm': ev_dep['mean_abs_peak_error_cm'],
            'prealert_level_prob_0_60': {
                **classification_metrics(y_true, test_df['pred_twostage_prealert'].to_numpy(), 10.0),
                'events': {k: v for k, v in ev_pre.items() if k != 'events_detail'},
            },
            'official_alert_prob_0_80_or_10cm': {
                **classification_metrics(y_true, test_df['pred_twostage_alert'].to_numpy(), 10.0),
                'events': {k: v for k, v in ev_alrt.items() if k != 'events_detail'},
            },
        },
    }

    out_path = art_dir / 'FINAL_TEST_REPORT.json'
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
