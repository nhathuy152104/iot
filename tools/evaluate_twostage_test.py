import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from train_improved import ALL_FEATURES
from benchmark_approaches import add_tidal_features
from full_metrics import classification_metrics, event_level_metrics, regression_and_tail_metrics

def main():
    train_df, tidal_cols = add_tidal_features(pd.read_parquet(ROOT / 'cache_features' / 'train_rich.parquet'))
    val_df, _ = add_tidal_features(pd.read_parquet(ROOT / 'cache_features' / 'val_rich.parquet'))

    y_tr = (train_df['label'] >= 10.0).astype(int)
    pos_w = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    clf = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w * 0.4,
        eval_metric='logloss', random_state=42, n_jobs=4,
    )
    clf.fit(train_df[tidal_cols], y_tr)

    out_dir = ROOT / 'runs' / 'twostage_final'
    out_dir.mkdir(parents=True, exist_ok=True)
    clf.save_model(out_dir / 'flood_classifier_tidal.ubj')
    (out_dir / 'depth_regressor_l1.ubj').write_bytes((ROOT / 'runs' / 'improved_l1_backup.ubj').read_bytes())
    print('Saved artifacts to:', out_dir)

if __name__ == '__main__':
    main()
