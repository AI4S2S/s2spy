"""Tests for the s2s.time module.
"""
import numpy as np
import pandas as pd
import pytest
from s2s.time import AdventCalendar


def interval(start, end):
    """Shorthand for more readable tests."""
    return pd.Interval(pd.Timestamp(start), pd.Timestamp(end))


class TestAdventCalendar:
    def test_init(self):
        cal = AdventCalendar()
        assert isinstance(cal, AdventCalendar)

    def test_repr(self):
        cal = AdventCalendar()
        assert repr(cal) == "AdventCalendar(month=11, day=30, freq=7d, n=52)"

    def test_str(self):
        cal = AdventCalendar()
        assert str(cal) == "52 periods of 7d leading up to 11/30."

    def test_discard(self):
        cal = AdventCalendar()

        with pytest.raises(NotImplementedError):
            cal.discard(max_lag=5)

    def test_map_year(self):
        cal = AdventCalendar(anchor_date=(12, 31), freq="180d")
        year = cal.map_year(2020)
        expected = np.array(
            [
                interval("2020-07-04", "2020-12-31"),
                interval("2020-01-06", "2020-07-04"),
            ]
        )
        assert np.array_equal(year, expected)

    def test_map_years(self):
        cal = AdventCalendar(anchor_date=(12, 31), freq="180d")
        years = cal.map_years(2020, 2021)
        expected = np.array(
            [
                [
                    interval("2021-07-04", "2021-12-31"),
                    interval("2021-01-05", "2021-07-04"),
                ],
                [
                    interval("2020-07-04", "2020-12-31"),
                    interval("2020-01-06", "2020-07-04"),  # notice the leap day
                ],
            ]
        )
        assert np.array_equal(years, expected)

    def test_mark_target_period(self):
        cal = AdventCalendar()

        with pytest.raises(NotImplementedError):
            cal.mark_target_period(end="20200101", periods=5)

        with pytest.raises(NotImplementedError):
            cal.mark_target_period(start="20200101", periods=5)

        with pytest.raises(NotImplementedError):
            cal.mark_target_period(start="20190101", end="20200101")

        with pytest.raises(ValueError):
            cal.mark_target_period(end="20200101")

    def test_get_lagged_indices(self):
        cal = AdventCalendar()

        with pytest.raises(NotImplementedError):
            cal.get_lagged_indices()

    def test_get_train_indices(self):
        cal = AdventCalendar()

        with pytest.raises(NotImplementedError):
            cal.get_train_indices("leave_n_out", {"n": 5})

    def test_get_test_indices(self):
        cal = AdventCalendar()

        with pytest.raises(NotImplementedError):
            cal.get_test_indices("leave_n_out", {"n": 5})

    def test_get_train_test_indices(self):
        cal = AdventCalendar()

        with pytest.raises(NotImplementedError):
            cal.get_train_test_indices("leave_n_out", {"n": 5})
