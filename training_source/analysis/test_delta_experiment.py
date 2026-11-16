"""Synthetic regression checks; improving predictive accuracy is not a unit test."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from delta_experiment import (
    METRIC, DEFAULT_ORIGIN, delta_target, depth_from_delta, make_model,
    original_config, select_forecaster, verified_split,
)


class DeltaTests(unittest.TestCase):
    def test_negative_delta_is_not_clipped_before_reconstructing_depth(self):
        delta = delta_target([2., 0., 30.], [5., 8., 20.])
        np.testing.assert_allclose(delta, [-3., -8., 10.])
        np.testing.assert_allclose(depth_from_delta([5., 8., 20.], delta), [2., 0., 30.])

    def test_zero_delta_is_exactly_persistence_even_at_unseen_high_depth(self):
        np.testing.assert_allclose(depth_from_delta([0., 10., 200.], [0., 0., 0.]), [0., 10., 200.])

    def test_clip_only_final_depth_no_upper_cap(self):
        np.testing.assert_allclose(depth_from_delta([5., 200.], [-8., 20.]), [0., 220.])

    def test_reject_bad_shapes_or_nonfinite_values(self):
        for function in (delta_target, depth_from_delta):
            for a, b in (([1], [1, 2]), ([1], [np.nan]), ([np.inf], [1])):
                with self.assertRaises(ValueError):
                    function(a, b)

    def test_test_split_is_blocked_before_any_file_access(self):
        with patch('delta_experiment.read_json') as read:
            with self.assertRaises(ValueError):
                verified_split('test', DEFAULT_ORIGIN, {}, [])
            read.assert_not_called()

    def test_baseline_can_win_and_is_preferred_on_exact_tie(self):
        def row(name, score):
            return {'model': name, METRIC: score, 'mae_cm': .1}
        self.assertEqual(select_forecaster([row('delta_a', 2), row('persistence', 1)]), 'persistence')
        self.assertEqual(select_forecaster([row('delta_a', 1), row('persistence', 1)]), 'persistence')
        self.assertEqual(select_forecaster([row('delta_a', .9), row('persistence', 1)]), 'delta_a')
        with self.assertRaises(ValueError):
            select_forecaster([row('bad', None)])

    def test_actual_fit_uses_changes_and_reconstruction_not_absolute_predictions(self):
        # A constant +2cm delta must also produce +2 at an unseen 200cm level.
        config = {'common_xgboost_params': {'objective': 'reg:squarederror', 'tree_method': 'hist', 'n_jobs': 1},
                  'random_seed': 42, 'device': 'cpu'}
        model = make_model(config, {'max_depth': 1, 'n_estimators': 2})
        x = pd.DataFrame({'depth_lag_0_cm': np.arange(10.)})
        model.fit(x, delta_target(x.depth_lag_0_cm + 2, x.depth_lag_0_cm))
        predicted = depth_from_delta([200.], model.predict(pd.DataFrame({'depth_lag_0_cm': [200.]})))
        np.testing.assert_allclose(predicted, [202.], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
