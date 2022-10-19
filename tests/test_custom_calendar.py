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

class TestCustomCalendar:
    """Test CustomCalendar methods."""

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = CustomCalendar(anchor="12-31")
        # create target periods
        target_1 = TargetPeriod("20d")
        # create precursor periods
        precursor_1 = PrecursorPeriod("10d")
        # append building blocks
        cal.append(target_1)
        cal.append(precursor_1)
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
        target_2 = TargetPeriod("30d")
        dummy_calendar.append(target_2)
        dummy_calendar = dummy_calendar.map_years(2021, 2021)
        expected = np.array(
            [interval("2021-12-21", "2021-12-31", closed="left"),
             interval("2021-12-31", "2022-01-20", closed="left"),
             interval("2022-01-20", "2022-02-19", closed="left"),]
        )
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_gap_intervals(self, dummy_calendar):
        target_2 = TargetPeriod("20d", "10d")
        dummy_calendar.append(target_2)
        dummy_calendar = dummy_calendar.map_years(2021, 2021)
        expected = np.array(
            [interval("2021-12-21", "2021-12-31", closed="left"),
             interval("2021-12-31", "2022-01-20", closed="left"),
             interval("2022-01-30", "2022-02-19", closed="left"),]
        )
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_overlap_intervals(self, dummy_calendar):
        precursor_2 = PrecursorPeriod("10d", "-5d")
        dummy_calendar.append(precursor_2)
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

    # The following tests only check if the plotter completely fails,
    # visuals are not checked.
    def test_visualize(self, dummy_calendar):
        dummy_calendar.visualize()

    def test_visualize_with_text(self, dummy_calendar):
        dummy_calendar.visualize(add_freq=True)