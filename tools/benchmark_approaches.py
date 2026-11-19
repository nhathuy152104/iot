import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from train_improved import ALL_FEATURES
from full_metrics import classification_metrics, event_level_metrics, regression_and_tail_metrics

LAG_COLS = sorted([c for c in ALL_FEATURES if c.startswith('depth_lag_') and c.endswith('_cm')], key=lambda x: int(x.split('_')[2]))


def add_tidal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    # Số giờ tính từ mốc chuẩn Unix epoch
    hours = out.index.astype('int64') / 1e9 / 3600.0
    # Chu kỳ bán nhật triều chính (M2 = 12.4206 giờ) và chu kỳ nhật triều (24.0 giờ)
    out['tide_m2_sin'] = np.sin(2 * np.pi * hours / 12.4206)
    out['tide_m2_cos'] = np.cos(2 * np.pi * hours / 12.4206)
    out['tide_24h_sin'] = np.sin(2 * np.pi * hours / 24.0)
    out['tide_24h_cos'] = np.cos(2 * np.pi * hours / 24.0)
    tidal_cols = ALL_FEATURES + ['tide_m2_sin', 'tide_m2_cos', 'tide_24h_sin', 'tide_24h_cos']
    return out, tidal_cols


def fit_predict_dlinear(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """DLinear: phân rã 6 điểm lag (30 phút) thành Trend (trung bình trượt) và Seasonal (phần dư)."""
    def build_dlinear_matrix(df: pd.DataFrame) -> np.ndarray:
        lags = df[LAG_COLS].to_numpy()  # (N, 3): t, t-5, t-10
        trend = np.column_stack([
            lags[:, 0:2].mean(axis=1),
            lags[:, 1:3].mean(axis=1),
            lags.mean(axis=1),
        ])
        seasonal = lags - trend
        vel = (lags[:, 0] - lags[:, 1]).reshape(-1, 1)
        acc = (lags[:, 0] - 2.0 * lags[:, 1] + lags[:, 2]).reshape(-1, 1)
        return np.hstack([trend, seasonal, vel, acc])

    X_tr = build_dlinear_matrix(train_df)
    y_tr = (train_df['label'] - train_df['depth_lag_0_cm']).to_numpy()
    w_tr = np.where(train_df['label'] >= 10.0, 15.0, 1.0)

    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    X_va = build_dlinear_matrix(val_df)
    delta_pred = model.predict(X_va)
    return np.maximum(0.0, val_df['depth_lag_0_cm'].to_numpy() + delta_pred)


def main():
    train_path = ROOT / 'cache_features' / 'train_rich.parquet'
    val_path = ROOT / 'cache_features' / 'val_rich.parquet'
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError('Thiếu tệp cache_features/train_rich.parquet hoặc val_rich.parquet')

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    y_val = val_df['label'].to_numpy()
    cur_val = val_df['depth_lag_0_cm'].to_numpy()

    # Nạp 2 mô hình đã có trên máy để làm nền và trộn (Blending)
    m_l1 = XGBRegressor(n_jobs=4)
    m_l1.load_model(ROOT / 'runs' / 'improved_l1_backup.ubj')
    pred_l1 = np.maximum(0.0, cur_val + m_l1.predict(val_df[ALL_FEATURES]))

    m_q65 = XGBRegressor(n_jobs=4)
    m_q65.load_model(ROOT / 'runs' / 'improved_candidate.ubj')
    pred_q65 = np.maximum(0.0, cur_val + m_q65.predict(val_df[ALL_FEATURES]))

    # --- HƯỚNG 1: DLinear ---
    pred_dlinear = fit_predict_dlinear(train_df, val_df)

    # --- HƯỚNG 2A: Adaptive Blending (L1 khi ổn định + Quantile 0.65 khi nước dâng nhanh) ---
    rising = val_df['slope_5min_cm_per_min'].to_numpy() >= 0.15
    pred_blend = np.where(rising, 0.3 * pred_l1 + 0.7 * pred_q65, 0.8 * pred_l1 + 0.2 * pred_q65)

    # --- HƯỚNG 2B: Two-Stage (Bộ phân loại chuyên săn ngập >= 10cm + Hồi quy L1) ---
    y_tr_bin = (train_df['label'] >= 10.0).astype(int)
    pos_weight = float((y_tr_bin == 0).sum() / max(1, (y_tr_bin == 1).sum()))
    clf = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight * 0.4,  # Cân bằng giữa POD và FAR
        eval_metric='logloss',
        random_state=42,
        n_jobs=4,
    )
    clf.fit(train_df[ALL_FEATURES], y_tr_bin)
    prob_flood = clf.predict_proba(val_df[ALL_FEATURES])[:, 1]

    # --- HƯỚNG 3: XGBoost + Tidal Harmonics (12.42h & 24h) ---
    tr_tide, tidal_cols = add_tidal_features(train_df)
    va_tide, _ = add_tidal_features(val_df)
    y_tr_delta = (tr_tide['label'] - tr_tide['depth_lag_0_cm']).to_numpy()
    w_tr = np.where(tr_tide['label'] >= 10.0, 20.0, 1.0)

    m_tide = XGBRegressor(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=4,
    )
    m_tide.fit(tr_tide[tidal_cols], y_tr_delta, sample_weight=w_tr)
    pred_tide = np.maximum(0.0, cur_val + m_tide.predict(va_tide[tidal_cols]))

    # Tổng hợp bảng so sánh
    candidates = {
        '1_DLinear_extrapolation': pred_dlinear,
        '2A_Adaptive_Blend_L1_Q65': pred_blend,
        '3_XGB_Tidal_Harmonics': pred_tide,
    }

    summary = {}
    for name, preds in candidates.items():
        val_df[name] = preds
        reg = regression_and_tail_metrics(y_val, preds)
        cls = classification_metrics(y_val, preds, threshold=10.0)
        ev = event_level_metrics(val_df, name, threshold=10.0)
        summary[name] = {
            'balanced_mae_cm': reg['balanced_mae_cm'],
            'mae_wet_cm': reg['wet_ge_10cm']['mae_cm'],
            'wet_bias_cm': reg['wet_ge_10cm']['bias_cm'],
            'POD_recall': cls['POD_recall'],
            'FAR_false_alarm': cls['FAR_false_alarm'],
            'CSI_threat_score': cls['CSI_threat_score'],
            'detected_events': f"{ev['detected_events']}/{ev['total_flood_events']}",
            'mean_lead_time_min': ev['mean_lead_time_min'],
            'mean_abs_peak_error_cm': ev['mean_abs_peak_error_cm'],
        }

    # Đánh giá riêng bộ phân loại Two-Stage (ngưỡng xác suất >= 0.5)
    y_bin = y_val >= 10.0
    p_bin = prob_flood >= 0.5
    tp = int(np.sum(y_bin & p_bin))
    fp = int(np.sum((~y_bin) & p_bin))
    fn = int(np.sum(y_bin & (~p_bin)))
    val_df['2B_TwoStage_alert_depth'] = np.where(p_bin, np.maximum(pred_l1, 10.0), np.minimum(pred_l1, 9.9))
    ev_2b = event_level_metrics(val_df, '2B_TwoStage_alert_depth', threshold=10.0)

    summary['2B_TwoStage_Classifier_plus_L1'] = {
        'balanced_mae_cm': regression_and_tail_metrics(y_val, pred_l1)['balanced_mae_cm'],
        'POD_recall': round(tp / (tp + fn), 4),
        'FAR_false_alarm': round(fp / (tp + fp), 4),
        'CSI_threat_score': round(tp / (tp + fp + fn), 4),
        'detected_events': f"{ev_2b['detected_events']}/{ev_2b['total_flood_events']}",
        'mean_lead_time_min': ev_2b['mean_lead_time_min'],
        'mean_abs_peak_error_cm': regression_and_tail_metrics(y_val, pred_l1)['wet_ge_10cm']['mae_cm'],
    }

    out_path = ROOT / 'runs' / 'benchmark_3_approaches.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()