"""Tests for the s2spy.time module.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from s2spy.time import AdventCalendar
from s2spy.time import MonthlyCalendar
from s2spy.time import WeeklyCalendar


def interval(start, end, closed="left"):
    """Shorthand for more readable tests."""
    return pd.Interval(pd.Timestamp(start), pd.Timestamp(end), closed = closed)


class TestAdventCalendar:
    """Test AdventCalendar methods."""

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        """Test AdventCalendar methods."""
        cal = AdventCalendar(anchor="12-31", freq="240d")
        cal.map_years(2021, 2021)
        return cal

    def test_init(self):
        cal = AdventCalendar(anchor="12-31")
        assert isinstance(cal, AdventCalendar)

    def test_repr(self):
        cal = AdventCalendar(anchor="12-31")
        assert repr(cal) == ("AdventCalendar(freq=7d, n_targets=1)")

    def test_show(self, dummy_calendar):
        expected_calendar_repr = (
            "i_interval 1\n anchor_year \n 2021[2021-12-31,2022-08-28)"
        )
        expected_calendar_repr = expected_calendar_repr.replace(" ", "")
        assert repr(dummy_calendar.show()).replace(" ", "") == expected_calendar_repr

    def test_flat(self):
        expected = np.array([interval("2021-12-31", "2022-08-28")])
        cal = AdventCalendar(anchor="12-31", freq="240d")
        cal.map_years(2021, 2021)
        assert np.array_equal(cal.flat, expected)

    def test_no_intervals(self):
        cal = AdventCalendar(anchor="12-31")
        with pytest.raises(ValueError):
            cal.get_intervals()

    def test_incorrect_freq(self):
        with pytest.raises(ValueError):
            AdventCalendar(anchor="12-31", freq="2W")

    def test_set_max_lag(self):
        cal = AdventCalendar(anchor="12-31")
        cal.set_max_lag(max_lag=5)

    def test_set_max_lag_incorrect_val(self):
        cal = AdventCalendar(anchor="12-31")
        with pytest.raises(ValueError):
            cal.set_max_lag(-1)

    def test_visualize(self, dummy_calendar):
        dummy_calendar.visualize()

    def test_visualize_with_text(self, dummy_calendar):
        dummy_calendar.visualize(add_freq=True)


class TestMonthlyCalendar:
    """Test MonthlyCalendar methods."""

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = MonthlyCalendar(anchor="Dec", freq="8M")
        cal.map_years(2021, 2021)
        return cal

    def test_init(self):
        cal = MonthlyCalendar(anchor="Dec", freq="2M")
        assert isinstance(cal, MonthlyCalendar)

    def test_repr(self):
        cal = MonthlyCalendar(anchor="Dec", freq="2M")
        assert repr(cal) == ("MonthlyCalendar(freq=2M, n_targets=1)")

    def test_show(self, dummy_calendar):
        expected_calendar_repr = (
            "i_interval 1\n anchor_year \n 2021 (2021 Dec, 2022 Aug]"
        )
        expected_calendar_repr = expected_calendar_repr.replace(" ", "")
        assert repr(dummy_calendar.show()).replace(" ", "") == expected_calendar_repr

    def test_flat(self, dummy_calendar):
        expected = np.array([interval("2021-12-01", "2022-08-01")])
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_no_intervals(self):
        cal = MonthlyCalendar()
        with pytest.raises(ValueError):
            cal.get_intervals()

    def test_incorrect_freq(self):
        with pytest.raises(ValueError):
            MonthlyCalendar(freq="2d")

    def test_visualize(self, dummy_calendar):
        dummy_calendar.visualize()

    def test_visualize_with_text(self, dummy_calendar):
        dummy_calendar.visualize(add_freq=True)


class TestWeeklyCalendar:
    """Test WeeklyCalendar methods."""

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = WeeklyCalendar(anchor="W48", freq="30W")
        cal.map_years(2021, 2021)
        return cal

    def test_init(self):
        cal = WeeklyCalendar(anchor="W48", freq="30W")
        assert isinstance(cal, WeeklyCalendar)

    def test_repr(self):
        cal = WeeklyCalendar(anchor="W48", freq="30W")
        assert repr(cal) == ("WeeklyCalendar(freq=30W, n_targets=1)")

    def test_show(self, dummy_calendar):
        expected_calendar_repr = (
            "i_interval 1\n anchor_year \n 2021 (2021-W48, 2022-W26]"
        )
        expected_calendar_repr = expected_calendar_repr.replace(" ", "")
        assert repr(dummy_calendar.show()).replace(" ", "") == expected_calendar_repr

    def test_flat(self, dummy_calendar):
        expected = np.array([interval("2021-11-29", "2022-06-27")])
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_no_intervals(self):
        cal = WeeklyCalendar(anchor="W40")
        with pytest.raises(ValueError):
            cal.get_intervals()

    def test_incorrect_freq(self):
        with pytest.raises(ValueError):
            WeeklyCalendar(anchor="W40", freq="2d")

    def test_visualize(self, dummy_calendar):
        dummy_calendar.visualize()

    def test_visualize_with_text(self, dummy_calendar):
        dummy_calendar.visualize(add_freq=True)


class TestMap:
    """Test map to year(s)/data methods"""

    def test_map_years(self):
        cal = AdventCalendar(anchor="12-31", freq="180d")
        cal.map_years(2020, 2021)
        expected = np.array(
            [
                [
                    interval("2021-07-04", "2021-12-31"),
                    interval("2021-12-31", "2022-06-29"),
                ],
                [
                    interval("2020-07-04", "2020-12-31"),
                    interval("2020-12-31", "2021-06-29"),  # notice the leap day
                ],
            ]
        )
        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_years_single(self):
        cal = AdventCalendar(anchor="12-31", freq="180d")
        cal.map_years(2020, 2020)
        expected = np.array(
            [
                [
                    interval("2020-07-04", "2020-12-31"),
                    interval("2020-12-31", "2021-06-29"),
                ]
            ]
        )
        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_to_data_edge_case_last_year(self):
        # test the edge value when the input could not cover the anchor date
        cal = AdventCalendar(anchor="10-15", freq="180d")
        # single year covered
        time_index = pd.date_range("20191020", "20211001", freq="60d")
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        cal.map_to_data(timeseries)
        expected = np.array(
            [
                [
                    interval("2020-04-18", "2020-10-15"),
                    interval("2020-10-15", "2021-04-13"),
                ]
            ]
        )
        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_to_data_single_year_coverage(self):
        # test the single year coverage
        cal = AdventCalendar(anchor="6-30", freq="180d")
        # multiple years covered
        time_index = pd.date_range("20210101", "20211231", freq="7d")
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        cal.map_to_data(timeseries)

        expected = np.array(
            [
                [
                    interval("2021-01-01", "2021-06-30"),
                    interval("2021-06-30", "2021-12-27"),
                ]
            ]
        )

        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_to_data_edge_case_first_year(self):
        # test the edge value when the input covers the anchor date
        cal = AdventCalendar(anchor="10-15", freq="180d")
        # multiple years covered
        time_index = pd.date_range("20191010", "20211225", freq="60d")
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        cal.map_to_data(timeseries)

        expected = np.array(
            [
                [
                    interval("2020-04-18", "2020-10-15"),
                    interval("2020-10-15", "2021-04-13"),
                ],
                [
                    interval("2019-04-18", "2019-10-15"),
                    interval("2019-10-15", "2020-04-12"),  # notice the leap day
                ],
            ]
        )

        assert np.array_equal(cal.get_intervals(), expected)

    # def test_map_to_data_value_error(self):
    #     # test when the input data is not sufficient to cover one year
    #     cal = AdventCalendar(anchor="10-15", freq='180d')
    #     with pytest.raises(ValueError):
    #         time_index = pd.date_range('20201020', '20211001', freq='60d')
    #         test_data = np.random.random(len(time_index))
    #         timeseries = pd.Series(test_data, index=time_index)
    #         cal.map_to_data(timeseries)

    def test_map_to_data_input_time_backward(self):
        # test when the input data has reverse order time index
        cal = AdventCalendar(anchor="10-15", freq="180d")
        time_index = pd.date_range("20201010", "20211225", freq="60d")
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index[::-1])
        cal.map_to_data(timeseries)

        expected = np.array(
            [
                [
                    interval("2020-04-18", "2020-10-15"),
                    interval("2020-10-15", "2021-04-13"),
                ]
            ]
        )

        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_to_data_xarray_input(self):
        # test when the input data has reverse order time index
        cal = AdventCalendar(anchor="10-15", freq="180d")
        time_index = pd.date_range("20201010", "20211225", freq="60d")
        test_data = np.random.random(len(time_index))
        dataarray = xr.DataArray(data=test_data, coords={"time": time_index})
        cal.map_to_data(dataarray)

        expected = np.array(
            [
                interval("2020-04-18", "2020-10-15"),
                interval("2020-10-15", "2021-04-13"),
            ]
        )

        assert np.all(cal.get_intervals() == expected)

    def test_missing_time_dim(self):
        cal = AdventCalendar(anchor="10-15", freq="180d")
        time_index = pd.date_range("20191020", "20211001", freq="60d")
        test_data = np.random.random(len(time_index))
        dataframe = pd.DataFrame(test_data, index=time_index)
        dataset = dataframe.to_xarray()
        with pytest.raises(ValueError):
            cal.map_to_data(dataset)

    def test_non_time_dim(self):
        cal = AdventCalendar(anchor="10-15", freq="180d")
        time_index = pd.date_range("20191020", "20211001", freq="60d")
        test_data = np.random.random(len(time_index))
        dataframe = pd.DataFrame(test_data, index=time_index)
        dataset = dataframe.to_xarray().rename({"index": "time"})
        dataset["time"] = np.arange(dataset["time"].size)
        with pytest.raises(ValueError):
            cal.map_to_data(dataset)

    # Note: add more test cases for different number of target periods!
    max_lag_edge_cases = [(73, [2019], 74), (72, [2019, 2018], 73)]
    # Test the edge cases of max_lag; where the max_lag just fits in exactly 365 days,
    # and where the max_lag just causes the calendar to skip a year
    @pytest.mark.parametrize("max_lag,expected_index,expected_size", max_lag_edge_cases)
    def test_max_lag_skip_years(self, max_lag, expected_index, expected_size):
        calendar = AdventCalendar(anchor="12-31", freq="5d")
        calendar.set_max_lag(max_lag)
        calendar = calendar.map_years(2018, 2019)

        np.testing.assert_array_equal(
            calendar.get_intervals().index.values, expected_index
        )
        assert calendar.get_intervals().iloc[0].size == expected_size
