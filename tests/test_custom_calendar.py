"""Tests for the s2spy.time.CustomCalendar module.
"""
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import pytest
from s2spy.time import CustomCalendar
from s2spy.time import PrecursorPeriod
from s2spy.time import TargetPeriod


def interval(start, end, closed="right"):
    """Shorthand for more readable tests."""
    return pd.Interval(pd.Timestamp(start), pd.Timestamp(end), closed = closed)


class TestPeriod:
    """Test Period objects."""

    def test_target_period(self):
        target = TargetPeriod("20d", "10d")
        assert isinstance(target, TargetPeriod)
        assert target.length == DateOffset(days=20)
        assert target.gap == DateOffset(days=10)

    def test_precursor_period(self):
        precursor = PrecursorPeriod("10d", "-5d")
        assert isinstance(precursor, PrecursorPeriod)
        assert precursor.length == DateOffset(days=10)
        assert precursor.gap == DateOffset(days=-5)

    def test_period_months(self):
        target = TargetPeriod("2M", "1M")
        assert target.length == DateOffset(months=2)
        assert target.gap == DateOffset(months=1)

    def test_period_weeks(self):
        target = TargetPeriod("3W", "2W")
        assert target.length == DateOffset(weeks=3)
        assert target.gap == DateOffset(weeks=2)


class TestCustomCalendar:
    """Test CustomCalendar methods."""

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = CustomCalendar(anchor="12-31")
        # append building blocks
        cal.add_interval("target", "20d")
        cal.add_interval("precursor", "10d")
        # map years
        cal = cal.map_years(2021, 2021)
        return cal

    def test_init(self):
        cal = CustomCalendar(anchor="12-31")
        assert isinstance(cal, CustomCalendar)

    def test_repr(self):
        cal = CustomCalendar(anchor="12-31")
        assert repr(cal) == ("CustomCalendar(n_targets=0)")

    def test_show(self, dummy_calendar):
        expected_calendar_repr = (
            "i_interval -1 1\n anchor_year \n 2021"
            + "[2021-12-21, 2021-12-31) [2021-12-31, 2022-01-20)"
        )
        expected_calendar_repr = expected_calendar_repr.replace(" ", "")
        assert repr(dummy_calendar.show()).replace(" ", "") == expected_calendar_repr

    def test_no_intervals(self):
        cal = CustomCalendar(anchor="12-31")
        with pytest.raises(ValueError):
            cal.get_intervals()

    def test_flat(self, dummy_calendar):
        expected = np.array(
            [interval("2021-12-21", "2021-12-31", closed="left"),
             interval("2021-12-31", "2022-01-20", closed="left")]
        )
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_append(self, dummy_calendar):
        dummy_calendar.add_interval("target", "30d")
        dummy_calendar = dummy_calendar.map_years(2021, 2021)
        expected = np.array(
            [interval("2021-12-21", "2021-12-31", closed="left"),
             interval("2021-12-31", "2022-01-20", closed="left"),
             interval("2022-01-20", "2022-02-19", closed="left"),]
        )
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_gap_intervals(self, dummy_calendar):
        dummy_calendar.add_interval("target", "20d", gap="10d")
        dummy_calendar = dummy_calendar.map_years(2021, 2021)
        expected = np.array(
            [interval("2021-12-21", "2021-12-31", closed="left"),
             interval("2021-12-31", "2022-01-20", closed="left"),
             interval("2022-01-30", "2022-02-19", closed="left"),]
        )
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_overlap_intervals(self, dummy_calendar):
        dummy_calendar.add_interval("precursor", "10d", gap="-5d")
        dummy_calendar = dummy_calendar.map_years(2021, 2021)
        expected = np.array(
            [interval("2021-12-16", "2021-12-26", closed="left"),
             interval("2021-12-21", "2021-12-31", closed="left"),
             interval("2021-12-31", "2022-01-20", closed="left"),]
        )
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_map_to_data(self, dummy_calendar):
        # create dummy data for testing
        time_index = pd.date_range('20201110', '20211211', freq='10d')
        var = np.random.random(len(time_index))
        # generate input data
        test_data = pd.Series(var, index=time_index)
        # map year to data
        calendar = dummy_calendar.map_to_data(test_data)
        # expected intervals
        expected = np.array(
            [interval("2020-12-21", "2020-12-31", closed="left"),
             interval("2020-12-31", "2021-01-20", closed="left"),]
        )
        assert np.array_equal(calendar.flat, expected)

    def test_non_day_interval_length(self, dummy_calendar):
        cal = CustomCalendar(anchor="December")
        cal.add_interval("target", "1M")
        cal.add_interval("precursor", "10M")
        cal.map_years(2020, 2020)
        expected = np.array(
            [interval("2020-02-01", "2020-12-01", closed="left"),
             interval("2020-12-01", "2021-01-01", closed="left")]
        )
        assert np.array_equal(cal.flat, expected)

    # The following tests only check if the plotter completely fails,
    # visuals are not checked.
    def test_visualize(self, dummy_calendar):
        dummy_calendar.visualize(relative_dates=False)

    def test_visualize_relative_dates(self, dummy_calendar):
        dummy_calendar.visualize(relative_dates=True)

    def test_visualize_with_text(self, dummy_calendar):
        dummy_calendar.visualize(add_length=True)
