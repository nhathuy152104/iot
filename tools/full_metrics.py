import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
sys.path.insert(0, str(ROOT / 'tools'))

from predict_depth import FEATURES as BASE_FEATURES, load_checked_model
from train_improved import ALL_FEATURES


def regression_and_tail_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    diff = p - y
    abs_err = np.abs(diff)
    wet = y >= 10.0
    nonwet = ~wet

    def stats(mask):
        if not mask.any():
            return {'n': 0, 'mae_cm': None, 'rmse_cm': None, 'bias_cm': None, 'p95_err_cm': None, 'p99_err_cm': None}
        e = abs_err[mask]
        d = diff[mask]
        return {
            'n': int(mask.sum()),
            'mae_cm': round(float(e.mean()), 4),
            'rmse_cm': round(float(np.sqrt(np.mean(d ** 2))), 4),
            'bias_cm': round(float(d.mean()), 4),
            'p95_err_cm': round(float(np.percentile(e, 95)), 4),
            'p99_err_cm': round(float(np.percentile(e, 99)), 4),
        }

    overall = stats(np.ones_like(wet, dtype=bool))
    wet_s = stats(wet)
    nonwet_s = stats(nonwet)
    bal_mae = round((wet_s['mae_cm'] + nonwet_s['mae_cm']) / 2.0, 4) if wet_s['n'] and nonwet_s['n'] else None

    return {
        'balanced_mae_cm': bal_mae,
        'overall': overall,
        'wet_ge_10cm': wet_s,
        'nonwet_lt_10cm': nonwet_s,
    }


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 10.0) -> dict:
    y_bin = y >= threshold
    p_bin = p >= threshold

    tp = int(np.sum(y_bin & p_bin))
    fp = int(np.sum((~y_bin) & p_bin))
    fn = int(np.sum(y_bin & (~p_bin)))
    tn = int(np.sum((~y_bin) & (~p_bin)))

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = 1.0 - far if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * pod / (precision + pod)) if (precision + pod) > 0 else 0.0

    return {
        'threshold_cm': threshold,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'POD_recall': round(pod, 4),
        'Precision': round(precision, 4),
        'FAR_false_alarm': round(far, 4),
        'CSI_threat_score': round(csi, 4),
        'F1_score': round(f1, 4),
    }


def event_level_metrics(df: pd.DataFrame, pred_col: str, threshold: float = 10.0) -> dict:
    """Phân tích theo từng đợt ngập liên tục (các cụm >= 10cm cách nhau > 30 phút)."""
    events = []
    for st, grp in df.groupby('station'):
        grp = grp.sort_index()
        wet_times = grp.index[grp['label'] >= threshold]
        if len(wet_times) == 0:
            continue

        # Nhóm các mốc ngập cách nhau <= 30 phút thành 1 đợt ngập
        gap = wet_times.to_series().diff() > pd.Timedelta(minutes=30)
        event_ids = gap.cumsum()

        for _, times in wet_times.to_series().groupby(event_ids):
            t_onset = times.index.min()
            t_end = times.index.max()
            # Mở rộng vùng quan sát trước 30 phút và sau 15 phút
            win = grp.loc[t_onset - pd.Timedelta(minutes=30) : t_end + pd.Timedelta(minutes=15)]
            if win.empty:
                continue

            peak_true = float(win['label'].max())
            peak_pred = float(win[pred_col].max())
            peak_err = peak_pred - peak_true

            # Tìm thời điểm đầu tiên mô hình cảnh báo >= threshold trong đợt ngập
            warn_hits = win.index[win[pred_col] >= threshold]
            if len(warn_hits) > 0:
                t_warn = warn_hits.min()
                # Positive lead_time = cảnh báo sớm hơn thực tế
                lead_min = (t_onset - t_warn).total_seconds() / 60.0
                detected = True
            else:
                lead_min = None
                detected = False

            events.append({
                'station': st,
                'onset_utc': str(t_onset),
                'duration_bins': int(len(times)),
                'detected': detected,
                'lead_time_min': lead_min,
                'peak_true_cm': round(peak_true, 2),
                'peak_pred_cm': round(peak_pred, 2),
                'peak_bias_cm': round(peak_err, 2),
            })

    det_events = [e for e in events if e['detected']]
    leads = [e['lead_time_min'] for e in det_events]
    peak_abs_errs = [abs(e['peak_bias_cm']) for e in events]

    return {
        'total_flood_events': len(events),
        'detected_events': len(det_events),
        'event_detection_rate': round(len(det_events) / len(events), 4) if events else None,
        'mean_lead_time_min': round(float(np.mean(leads)), 2) if leads else None,
        'mean_abs_peak_error_cm': round(float(np.mean(peak_abs_errs)), 2) if peak_abs_errs else None,
        'events_detail': events,
    }


def benchmark_latency(model: XGBRegressor, sample_row: pd.DataFrame, runs: int = 200) -> dict:
    # Warmup
    for _ in range(10):
        model.predict(sample_row)
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(sample_row)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return {
        'single_row_mean_ms': round(float(np.mean(latencies)), 4),
        'single_row_p95_ms': round(float(np.percentile(latencies, 95)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, default=ROOT / 'cache_features' / 'val_rich.parquet')
    parser.add_argument('--improved-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'full_metrics_report.json')
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet.resolve())
    y = df['label'].to_numpy()
    df['pred_persistence'] = df['depth_lag_0_cm']

    # Load model gốc (ưu tiên MANIFEST, fallback sang bản tái lập chuẩn trong runs/)
    try:
        frozen_model = load_checked_model()
    except FileNotFoundError:
        candidates = sorted(ROOT.glob('runs/*_downloaded_raw_candidate/candidate.ubj'))
        if not candidates:
            raise
        frozen_model = XGBRegressor(n_jobs=1)
        frozen_model.load_model(candidates[-1])
    df['pred_frozen'] = np.maximum(0.0, df['depth_lag_0_cm'].to_numpy() + frozen_model.predict(df[BASE_FEATURES]))

    # Load model cải tiến
    improved_model = XGBRegressor(n_jobs=1)
    improved_model.load_model(args.improved_model.resolve())
    df['pred_improved'] = np.maximum(0.0, df['depth_lag_0_cm'].to_numpy() + improved_model.predict(df[ALL_FEATURES]))

    models = {
        'persistence': 'pred_persistence',
        'frozen_original_model': 'pred_frozen',
        'improved_model': 'pred_improved',
    }

    report = {
        'dataset': str(args.parquet),
        'total_windows': int(len(df)),
        'regression_and_tail': {},
        'classification_at_10cm': {},
        'flood_events_analysis': {},
        'inference_latency_ram': {
            'frozen_original_model': benchmark_latency(frozen_model, df[BASE_FEATURES].iloc[[0]]),
            'improved_model': benchmark_latency(improved_model, df[ALL_FEATURES].iloc[[0]]),
        },
    }

    for name, col in models.items():
        p = df[col].to_numpy()
        report['regression_and_tail'][name] = regression_and_tail_metrics(y, p)
        report['classification_at_10cm'][name] = classification_metrics(y, p, threshold=10.0)
        report['flood_events_analysis'][name] = event_level_metrics(df, col, threshold=10.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()