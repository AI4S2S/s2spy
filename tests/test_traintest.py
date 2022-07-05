"""Tests for the s2s.traintest module.
"""
import numpy as np
import pytest
import s2s.traintest
from s2s.time import AdventCalendar

class TestTrainTest:
    # Define all required inputs as fixtures:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = AdventCalendar(anchor_date=(10, 15), freq="180d")
        return cal.map_years(2019, 2021)

    def test_kfold(self, dummy_calendar):
        traintest_group = s2s.traintest.kfold(dummy_calendar._intervals, n_splits=2) # pylint: disable=protected-access
        # check the first fold
        expected_group = ['train', 'train', 'test']
        assert np.array_equal(traintest_group["fold_1"].values, expected_group)

    def test_random_strat(self):
        with pytest.raises(NotImplementedError):
            s2s.traintest.random_strat()

    def test_timeseries_split(self):
        with pytest.raises(NotImplementedError):
            s2s.traintest.timeseries_split()
