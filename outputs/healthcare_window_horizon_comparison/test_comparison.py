"""Chronology, cohort and result checks for the Healthcare research experiment."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from run_comparison import HERE, LM, CONTROLS, offset, outer_train, prior_calls
from research_metrics import paired_statistics, summarize


class Chronology(unittest.TestCase):
    def test_day_offsets_enforce_source_coverage(self):
        cal=pd.date_range('2025-01-06',periods=5,freq='B')
        d=pd.Series(pd.to_datetime(['2025-01-06','2025-01-10','2025-01-11']))
        actual=offset(d,cal,2)
        self.assertEqual(actual.iloc[0],pd.Timestamp('2025-01-08'))
        self.assertTrue(actual.iloc[1:].isna().all())

    def test_strict_maturity_and_no_current_year(self):
        cutoff=pd.Timestamp('2025-01-17T21:00Z')
        d=pd.DataFrame({'year':[2024,2024,2024,2025], 'eligible_20':[True]*4,
                        'mature_20':[cutoff-pd.Timedelta(minutes=1),cutoff,cutoff+pd.Timedelta(minutes=1),cutoff]})
        train,purged=outer_train(d,2025,20,2020,cutoff)
        self.assertEqual(train.index.tolist(),[0])
        self.assertEqual(purged,2)

    def test_previous_call_counter_has_no_target_dependency(self):
        d=pd.DataFrame({'transcriptid':[1,2,3,4], 'permno':[100,100,100,100],
                        'clock':pd.to_datetime(['2023-01-01','2023-02-01','2023-02-01','2023-03-01'],utc=True),
                        'car_2_20':[.02,np.nan,.3,-.01]})
        result=prior_calls(d)
        self.assertEqual(result.tolist(),[0,1,1,3])
        self.assertTrue(result.equals(prior_calls(d.drop(columns='car_2_20'))))
        self.assertEqual(len(LM),8)
        self.assertNotIn('hedging_answers',LM)
        self.assertTrue({'car_2_5','car_2_20','vol_post_2_20'}.isdisjoint(CONTROLS))

    def test_compares_correlations_with_correct_targets(self):
        x=np.arange(100)
        z=np.column_stack([x,-x,-x,x])
        result=paired_statistics(z,[(0,1),(2,3),(0,3)])
        np.testing.assert_allclose(result,[-1.,-1.,1.],rtol=0,atol=1e-14)
        q,spread,_=summarize(x,x,100)
        self.assertEqual(sum(row['n'] for row in q),100)
        self.assertEqual(spread,80)


@unittest.skipUnless((HERE/'paired_inference.csv').is_file(),'Complete the evaluation to check the saved tables')
class SavedResults(unittest.TestCase):
    @staticmethod
    def read(name):
        return pd.read_csv(HERE/f'{name}.csv',float_precision='round_trip')

    def test_all_saved_cv_and_outer_labels_are_mature(self):
        inner=self.read('cv_folds')
        outer=self.read('training_cohorts')
        self.assertEqual(len(outer),24)
        self.assertEqual(len(inner),72)
        self.assertTrue(pd.to_datetime(inner.max_fit_maturity,utc=True).lt(pd.to_datetime(inner.inner_cutoff,utc=True)).all())
        self.assertTrue(pd.to_datetime(outer.max_training_maturity,utc=True).lt(pd.to_datetime(outer.refit_cutoff,utc=True)).all())
        keys=['year','horizon','benchmark','window']
        checked=inner.merge(outer,validate='many_to_one',on=keys)
        self.assertTrue(pd.to_datetime(checked.max_validation_maturity,utc=True).lt(pd.to_datetime(checked.refit_cutoff,utc=True)).all())
        self.assertTrue(checked.validation_year.lt(checked.year).all())

    def test_annual_matches_and_differences(self):
        p=self.read('annual_model_performance')
        w=self.read('window_comparison')
        h=self.read('horizon_comparison')
        self.assertEqual(set(p.year),{2024,2025,2026})
        self.assertEqual(len(p),144)
        np.testing.assert_allclose(w.rolling_minus_expanding,w.rolling_rank_ic-w.expanding_rank_ic,atol=1e-13)
        np.testing.assert_allclose(h.long_minus_short,h.long_rank_ic-h.short_rank_ic,atol=1e-13)
        groups=p.groupby(['year','benchmark','horizon','sample'])
        self.assertTrue(groups.events.nunique().eq(1).all())
        self.assertTrue(groups.issuers.nunique().eq(1).all())
        c=self.read('horizon_cohorts')
        self.assertTrue(c.full_events.ge(c.matched_horizons_events).all())


if __name__=='__main__':
    unittest.main()
