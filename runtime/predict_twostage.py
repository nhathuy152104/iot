"""Production inference runtime for Two-Stage Flood Prediction (Tidal Classifier + L1 Regressor)."""
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from train_improved import ALL_FEATURES

TIDAL_COLS = ALL_FEATURES + ['tide_m2_sin', 'tide_m2_cos', 'tide_24h_sin', 'tide_24h_cos']
ARTIFACT_DIR = ROOT / 'runs' / 'twostage_final'


class TwoStageFloodPredictor:
    def __init__(
        self,
        clf_path: Path = ARTIFACT_DIR / 'flood_classifier_tidal.ubj',
        reg_path: Path = ARTIFACT_DIR / 'depth_regressor_l1.ubj',
        prealert_prob: float = 0.60,
        alert_prob: float = 0.80,
        flood_threshold_cm: float = 10.0,
    ) -> None:
        self.prealert_prob = prealert_prob
        self.alert_prob = alert_prob
        self.flood_threshold_cm = flood_threshold_cm

        self.clf = XGBClassifier(n_jobs=1)
        self.clf.load_model(clf_path)

        self.reg = XGBRegressor(n_jobs=1)
        self.reg.load_model(reg_path)

    @staticmethod
    def _build_features(
        timestamp_utc: datetime,
        d0: float,
        d5: float,
        d10: float,
        age0_s: float = 0.0,
        age5_s: float = 0.0,
        age10_s: float = 0.0,
    ) -> pd.DataFrame:
        """Tính 17 đặc trưng động học (khớp 100% ALL_FEATURES) + 4 đặc trưng chu kỳ thủy triều."""
        arr3 = np.array([d0, d5, d10], dtype=float)
        mean3 = float(arr3.mean())
        std3 = float(arr3.std(ddof=1)) if len(arr3) > 1 else 0.0
        min3 = float(arr3.min())
        max3 = float(arr3.max())

        hours = timestamp_utc.astimezone(timezone.utc).timestamp() / 3600.0
        row: dict[str, float] = {
            'depth_lag_0_cm': d0,
            'depth_lag_5_cm': d5,
            'depth_lag_10_cm': d10,
            'source_age_lag_0_seconds': age0_s,
            'source_age_lag_5_seconds': age5_s,
            'source_age_lag_10_seconds': age10_s,
            'delta_5min_cm': d0 - d5,
            'delta_10min_cm': d0 - d10,
            'slope_5min_cm_per_min': (d0 - d5) / 5.0,
            'slope_10min_cm_per_min': (d0 - d10) / 10.0,
            'accel_cm_per_min2': (d0 - 2.0 * d5 + d10) / 25.0,
            'mean_three_bins_cm': mean3,
            'min_three_bins_cm': min3,
            'max_three_bins_cm': max3,
            'std_three_bins_cm': std3,
            'range_three_bins_cm': max3 - min3,
            'slope_30min_cm_per_min': (d0 - d10) / 10.0,
            'mean_six_bins_cm': mean3,
            'std_six_bins_cm': std3,
            'tide_m2_sin': float(np.sin(2.0 * np.pi * hours / 12.4206)),
            'tide_m2_cos': float(np.cos(2.0 * np.pi * hours / 12.4206)),
            'tide_24h_sin': float(np.sin(2.0 * np.pi * hours / 24.0)),
            'tide_24h_cos': float(np.cos(2.0 * np.pi * hours / 24.0)),
        }
        return pd.DataFrame([row])

    def predict(
        self,
        timestamp_utc: datetime,
        depth_t0_cm: float,
        depth_t5_cm: float,
        depth_t10_cm: float,
    ) -> dict[str, Any]:
        if any(np.isnan(v) or v < 0.0 for v in (depth_t0_cm, depth_t5_cm, depth_t10_cm)):
            return {'status': 'unavailable', 'alert_level': 'UNKNOWN'}

        # Lọc nhiễu vật cản phần cứng: nước ngập đô thị không thể dâng > 20 cm trong 5 phút
        if abs(depth_t0_cm - depth_t5_cm) > 20.0 or depth_t0_cm > 200.0:
            return {
                'status': 'sensor_fault',
                'timestamp_utc': timestamp_utc.astimezone(timezone.utc).isoformat(),
                'current_depth_cm': round(depth_t0_cm, 2),
                'alert_level': 'SENSOR_FAULT',
            }

        # Lọc nhiễu vật cản phần cứng: nước ngập đô thị không thể dâng > 20 cm trong 5 phút
        if abs(depth_t0_cm - depth_t5_cm) > 20.0 or depth_t0_cm > 200.0 or (depth_t0_cm >= 10.0 and depth_t0_cm == depth_t5_cm == depth_t10_cm):
            return {
                'status': 'sensor_fault',
                'timestamp_utc': timestamp_utc.astimezone(timezone.utc).isoformat(),
                'current_depth_cm': round(depth_t0_cm, 2),
                'alert_level': 'SENSOR_FAULT',
            }

        feats = self._build_features(timestamp_utc, depth_t0_cm, depth_t5_cm, depth_t10_cm)

        prob = float(self.clf.predict_proba(feats[TIDAL_COLS])[0, 1])
        delta_pred = float(self.reg.predict(feats[ALL_FEATURES])[0])
        pred_depth = max(0.0, round(depth_t0_cm + delta_pred, 2))

        if prob >= self.alert_prob or pred_depth >= self.flood_threshold_cm:
            alert_level = 'FLOOD_ALERT'
        elif prob >= self.prealert_prob:
            alert_level = 'PRE_ALERT'
        else:
            alert_level = 'NORMAL'

        return {
            'status': 'ok',
            'timestamp_utc': timestamp_utc.astimezone(timezone.utc).isoformat(),
            'current_depth_cm': round(depth_t0_cm, 2),
            'predicted_depth_10min_cm': pred_depth,
            'flood_probability': round(prob, 4),
            'alert_level': alert_level,
        }


if __name__ == '__main__':
    predictor = TwoStageFloodPredictor()
    sample = predictor.predict(
        timestamp_utc=datetime(2023, 6, 13, 0, 25, tzinfo=timezone.utc),
        depth_t0_cm=9.4,
        depth_t5_cm=8.6,
        depth_t10_cm=7.9,
    )
    print(sample)
