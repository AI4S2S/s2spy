"""Tests for the s2s.traintest module.
"""
import numpy as np
import pytest
import s2spy.traintest
from s2spy.time import AdventCalendar


class TestTrainTest:
    # Define all required inputs as fixtures:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = AdventCalendar(anchor_date=(10, 15), freq="180d")
        return cal.map_years(2019, 2021)

    def test_kfold(self, dummy_calendar):
        traintest_group = s2spy.traintest.kfold(dummy_calendar.intervals, n_splits=2)
        # check the first fold
        expected_group = ['train', 'train', 'test']
        assert np.array_equal(traintest_group["fold_1"].values, expected_group)

    def test_random_strat(self):
        with pytest.raises(NotImplementedError):
            s2spy.traintest.random_strat()

    def test_timeseries_split(self):
        with pytest.raises(NotImplementedError):
            s2spy.traintest.timeseries_split()

    def test_iter_traintest(self):
        with pytest.raises(NotImplementedError):
            s2spy.traintest.iter_traintest('traintest_group', 'data')
