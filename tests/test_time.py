"""Tests for the s2s.time module.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
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
        assert repr(cal) == "AdventCalendar(month=11, day=30, freq=7d)"

    def test_str(self):
        cal = AdventCalendar()
        assert str(cal) == "52 periods of 7d leading up to 11/30."

    def test_discard(self):
        cal = AdventCalendar()

        with pytest.raises(NotImplementedError):
            cal.discard(max_lag=5)

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

class TestMap:
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

    def test_map_years_single(self):
        cal = AdventCalendar(anchor_date=(12, 31), freq="180d")
        year = cal.map_years(2020, 2020)
        expected = np.array([
            [
                interval("2020-07-04", "2020-12-31"),
                interval("2020-01-06", "2020-07-04"),
            ]
        ])
        assert np.array_equal(year, expected)

    def test_map_to_data_edge_case_last_year(self):
        # test the edge value when the input could not cover the anchor date
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
        # single year covered
        time_index = pd.date_range('20191020', '20211001', freq='60d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        year = cal.map_to_data(timeseries)
        expected = np.array([
            [
                interval("2020-04-18", "2020-10-15"),
                interval("2019-10-21", "2020-04-18"),
            ]
        ])
        assert np.array_equal(year, expected)

    def test_map_to_data_single_year_coverage(self):
        # test the single year coverage
        cal = AdventCalendar(anchor_date=(12, 31), freq='180d')
        # multiple years covered
        time_index = pd.date_range('20210101', '20211231', freq='7d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        year = cal.map_to_data(timeseries)

        expected = np.array([
            [
                interval("2021-07-04", "2021-12-31"),
                interval("2021-01-05", "2021-07-04"),
            ]
        ])

        assert np.array_equal(year, expected)

    def test_map_to_data_edge_case_first_year(self):
        # test the edge value when the input covers the anchor date
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
        # multiple years covered
        time_index = pd.date_range('20191010', '20211225', freq='60d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        year = cal.map_to_data(timeseries)

        expected = np.array(
            [
                [
                    interval("2021-04-18", "2021-10-15"),
                    interval("2020-10-20", "2021-04-18"),
                ],
                [
                    interval("2020-04-18", "2020-10-15"),
                    interval("2019-10-21", "2020-04-18"),  # notice the leap day
                ],
            ]
        )

        assert np.array_equal(year, expected)

    def test_map_to_data_value_error(self):
        # test when the input data is not sufficient to cover one year
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
        with pytest.raises(ValueError):
            time_index = pd.date_range('20201020', '20211001', freq='60d')
            test_data = np.random.random(len(time_index))
            timeseries = pd.Series(test_data, index=time_index)
            cal.map_to_data(timeseries)

    def test_map_to_data_input_time_backward(self):
        # test when the input data has reverse order time index
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
        time_index = pd.date_range('20201010', '20211225', freq='60d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index[::-1])
        year = cal.map_to_data(timeseries)

        expected = np.array([
            [
                interval("2021-04-18", "2021-10-15"),
                interval("2020-10-20", "2021-04-18"),
            ]
        ])

        assert np.array_equal(year, expected)

    def test_map_to_data_xarray_input(self):
        # test when the input data has reverse order time index
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
        time_index = pd.date_range('20201010', '20211225', freq='60d')
        test_data = np.random.random(len(time_index))
        da = xr.DataArray(
            data=test_data,
            coords={'time': time_index})
        year = cal.map_to_data(da)

        expected = np.array(
            [
                interval("2021-04-18", "2021-10-15"),
                interval("2020-10-20", "2021-04-18"),
            ]
        )

        assert np.all(year.values == expected)

class TestResample:
    # Define all required inputs as fixtures:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        return AdventCalendar(anchor_date=(10, 15), freq="180d")

    @pytest.fixture(params=["20151020", "20191020"])
    def dummy_series(self, request):
        time_index = pd.date_range(request.param, "20211001", freq="60d")
        test_data = np.random.random(len(time_index))
        expected = np.array([test_data[4:7].mean(), test_data[1:4].mean()])
        series = pd.Series(test_data, index=time_index, name='data1')
        return series, expected

    @pytest.fixture
    def dummy_dataframe(self, dummy_series):
        series, expected = dummy_series
        return pd.DataFrame(series), expected

    @pytest.fixture
    def dummy_dataarray(self, dummy_series):
        series, expected = dummy_series
        da = series.to_xarray()
        da = da.rename({'index': 'time'})
        return da, expected

    @pytest.fixture
    def dummy_dataset(self, dummy_dataframe):
        dataframe, expected = dummy_dataframe
        ds = dataframe.to_xarray().rename({'index': 'time'})
        return ds, expected

    # Tests start here:
    def test_nontime_index(self, dummy_calendar, dummy_series):
        series, _ = dummy_series
        series = series.reset_index()
        with pytest.raises(ValueError):
            dummy_calendar.resample(series)

    def test_series(self, dummy_calendar, dummy_series):
        series, expected = dummy_series
        resampled_data = dummy_calendar.resample(series)
        np.testing.assert_allclose(
            resampled_data['data1'].iloc[:2], expected)

    def test_unnamed_series(self, dummy_calendar, dummy_series):
        series, expected = dummy_series
        series.name = None
        resampled_data = dummy_calendar.resample(series)
        np.testing.assert_allclose(
            resampled_data["mean_data"].iloc[:2], expected)

    def test_dataframe(self, dummy_calendar, dummy_dataframe):
        dataframe, expected = dummy_dataframe
        resampled_data = dummy_calendar.resample(dataframe)
        np.testing.assert_allclose(resampled_data["data1"].iloc[:2], expected)

    def test_dataarray(self, dummy_calendar, dummy_dataarray):
        da, expected = dummy_dataarray
        resampled_data = dummy_calendar.resample(da)
        testing_vals = resampled_data["data1"].isel(anchor_year=0)
        np.testing.assert_allclose(testing_vals, expected)

    def test_dataset(self, dummy_calendar, dummy_dataset):
        ds, expected = dummy_dataset
        resampled_data = dummy_calendar.resample(ds)
        testing_vals = resampled_data["data1"].isel(anchor_year=0)
        np.testing.assert_allclose(testing_vals, expected)

    def test_missing_time_dim(self, dummy_calendar, dummy_dataset):
        ds, _ = dummy_dataset
        with pytest.raises(ValueError):
            dummy_calendar.resample(ds.rename({'time': 'index'}))

    def test_non_time_dim(self, dummy_calendar, dummy_dataset):
        ds, _ = dummy_dataset
        ds['time'] = np.arange(ds['time'].size)
        with pytest.raises(ValueError):
            dummy_calendar.resample(ds)
