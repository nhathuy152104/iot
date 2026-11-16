"""Synthetic tests of baselines, metrics and validation-only model choice."""
import unittest

import numpy as np
import pandas as pd

from train_baseline_xgboost import baseline_predictions, choose, evaluate


class TrainingTests(unittest.TestCase):
    def test_baselines_use_only_past_and_clip_negative(self):
        x=pd.DataFrame({'depth_lag_0_cm':[5.,5.], 'slope_5min_cm_per_min':[1.,-1.]})
        p=baseline_predictions(x)
        np.testing.assert_allclose(p['persistence'],[5,5])
        np.testing.assert_allclose(p['linear_trend_5min'],[15,0])

    def test_metrics_include_high_depth_and_new_crossings(self):
        r=evaluate([0,20,20,0],[0,0,20,20],[0,0,20,0])
        self.assertEqual(r['tp'],1)
        self.assertEqual(r['fn'],1)
        self.assertEqual(r['fp'],1)
        self.assertEqual(r['tn'],1)
        self.assertEqual(r['mae_wet_cm'],10)
        self.assertEqual(r['balanced_wet_nonwet_mae_cm'],10)
        self.assertEqual(r['new_crossing_windows'],1)
        self.assertEqual(r['new_crossing_recall'],0)

    def test_no_positive_targets_is_explicit_not_perfect_recall(self):
        r=evaluate([0,0],[0,0],[0,0])
        self.assertIsNone(r['recall_wet'])
        self.assertIsNone(r['mae_wet_cm'])
        self.assertIsNone(r['balanced_wet_nonwet_mae_cm'])

    def test_choose_uses_declared_balanced_metric(self):
        rows=[{'model':'dry_best','balanced_wet_nonwet_mae_cm':5,'mae_cm':.01},
              {'model':'balanced_best','balanced_wet_nonwet_mae_cm':2,'mae_cm':.2}]
        self.assertEqual(choose(rows),'balanced_best')
        with self.assertRaises(ValueError):
            choose([{'model':'none','balanced_wet_nonwet_mae_cm':None,'mae_cm':0}])


if __name__=='__main__':
    unittest.main()
