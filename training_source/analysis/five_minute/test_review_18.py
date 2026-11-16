"""Synthetic checks only; no source observations embedded."""
import unittest

import numpy as np
import pandas as pd

from pipeline import DEPTHS, make_features
from review_18 import episode_review, proposed_role, runs, subset_mask


def grid(values):
    index=pd.date_range('2022-01-01',periods=len(values),freq='5min',tz='UTC')
    g=pd.DataFrame({c:values for c in DEPTHS},index=index,dtype=float)
    g['source_age_seconds']=0.
    g['source_time_utc']=index
    return g


class ReviewTests(unittest.TestCase):
    def test_contiguous_runs_do_not_bridge_false(self):
        self.assertEqual(runs([False,True,True,False,True]),[(1,3),(4,5)])
        self.assertEqual(runs([]),[])

    def test_bracketed_short_pulse_flag_is_not_truth(self):
        episodes,shape=episode_review(grid([0,200,0]))
        self.assertTrue(episodes[0]['possible_short_pulse'])
        self.assertEqual(int(shape.sum()),1)

    def test_missing_context_is_not_filled(self):
        episodes,shape=episode_review(grid([np.nan,200,0]))
        self.assertFalse(episodes[0]['possible_short_pulse'])
        self.assertFalse(shape.any())

    def test_box_requires_stable_plateau_and_both_edges(self):
        episodes,shape=episode_review(grid([0,200,201,200,0]))
        self.assertTrue(episodes[0]['possible_box'])
        self.assertEqual(int(shape.sum()),3)
        episodes,shape=episode_review(grid([0,200,201,200,np.nan]))
        self.assertFalse(episodes[0]['possible_box'])

    def test_rise_fall_not_marked_as_box(self):
        episodes,shape=episode_review(grid([0,100,200,400,300,200,100,0]))
        self.assertFalse(episodes[0]['shape_ambiguous'])
        self.assertFalse(shape.any())

    def test_offline_label_mismatch_is_metadata_not_input(self):
        g=grid([0,0,0,0,200,0,0])
        g.iloc[4,g.columns.get_loc('depth_raw_mm')]=500
        windows,eligible=make_features(g,'synthetic')
        windows=windows.loc[eligible]
        episodes,shape=episode_review(g)
        flags=subset_mask(windows,g,shape)
        self.assertTrue(flags.iloc[0]['offline_target_raw_proc_disagreement'])
        self.assertFalse(flags.iloc[0]['history_raw_proc_disagreement'])
        self.assertFalse(flags.iloc[0]['proposed_low_flag_subset'])

    def test_role_requires_wet_diversity_not_just_many_rows(self):
        stats={'windows':2000,'decision_days':50,'wet_windows_ge_10cm':100,
               'nonwet_windows_lt_10cm':1900,'wet_days':4,'wet_groups_6h':4}
        self.assertEqual(proposed_role(stats),'candidate_event_segments')
        stats['wet_groups_6h']=1
        self.assertEqual(proposed_role(stats),'background_or_limited_wet')


if __name__=='__main__':
    unittest.main()
