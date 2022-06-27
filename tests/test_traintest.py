"""Tests for the s2s.traintest module.
"""
import numpy as np
import pandas as pd
import pytest
import s2s.traintest
from s2s.time import AdventCalendar

class TestTrainTest:
    # Define all required inputs as fixtures:
    @pytest.fixture(autouse=True)
    def dummy_intervals_df(self):
        freq = "180d"
        n_intervals = pd.Timedelta("365days") // pd.to_timedelta(freq)
        # create calendar object
        cal = AdventCalendar(anchor_date=(10, 15), freq=freq)
        intervals = cal.map_years(2019, 2021, flat=True)
        # infer anchor years from give intervals
        anchor_years = [timestamp.right.year for timestamp in intervals[::n_intervals]]
        # Create DataFrame from intervals Series to store the train/test groups
        traintest_base = pd.DataFrame(data = {
        'anchor_year': np.repeat(anchor_years, n_intervals),
        'i_intervals': np.tile(range(n_intervals), len(anchor_years)),
        'intervals': intervals,
        })        

        return traintest_base

    def test_kfold(self, dummy_intervals_df):
        traintest_group = s2s.traintest.kfold(dummy_intervals_df, n_splits=2)
        # check the first fold
        expected_group = ['test', 'test', 'train', 'train', 'train', 'train']
        assert np.array_equal(traintest_group["fold_1"].values, expected_group)

    def test_random_strat(self):
        with pytest.raises(NotImplementedError):
            s2s.traintest.random_strat()

    def test_timeseries_split(self):
        with pytest.raises(NotImplementedError):
            s2s.traintest.timeseries_split()

    