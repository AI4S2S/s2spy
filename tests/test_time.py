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

    def test_map_to_data(self):
        # test the edge value when the input could not cover the anchor date
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
        # single year covered
        time_index = pd.date_range('20191020', '20211001', freq='60d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        year = cal.map_to_data(timeseries)
        expected = np.array(
            [
                interval("2020-04-18", "2020-10-15"),
                interval("2019-10-21", "2020-04-18"),
            ]
        )

        assert np.array_equal(year, expected)

        # test the edge value when the input covers the anchor date
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

        # test the input with time index in backward order
        timeseries = pd.Series(test_data, index=time_index[::-1])
        year = cal.map_to_data(timeseries)

        assert np.array_equal(year, expected)
        
        # test when the input data is not sufficient to cover one year 
        with pytest.raises(ValueError):
            time_index = pd.date_range('20201020', '20211001', freq='60d')
            test_data = np.random.random(len(time_index))
            timeseries = pd.Series(test_data, index=time_index)
            year = cal.map_to_data(timeseries)

    def test_resample(self):
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
        time_index = pd.date_range('20191020', '20211001', freq='60d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        
        # test pandas Series without name
        resampled_data = cal.resample(timeseries)
        expected = np.array([test_data[4:7].mean(), test_data[1:4].mean()])

        assert np.array_equal(resampled_data['mean_data'].values, expected)

        # test pandas Series with name
        timeseries = timeseries.rename('data1')
        resampled_data = cal.resample(timeseries)

        assert np.array_equal(resampled_data['data1'].values, expected)

        # test pandas DataFrame
        dataframe = pd.DataFrame(timeseries)
        dataframe['data2'] = dataframe['data1']
        resampled_data = cal.resample(dataframe)

        assert np.array_equal(resampled_data['data1'].values, expected)

        # test for multi-year pandas input
        time_index = pd.date_range('20151020', '20211001', freq='60d')
        test_data = np.random.random(len(time_index))
        expected_my = np.array([test_data[4:7].mean(), test_data[1:4].mean()])
        timeseries_my = pd.Series(test_data, index=time_index).rename('data1')
        resampled_data = cal.resample(timeseries_my)

        assert np.array_equal(resampled_data['data1'].iloc[:2].values, expected_my)

        # test for non-time index failure
        series = pd.Series(test_data)
        with pytest.raises(ValueError):
            cal.resample(series)
        
        # test xarray DataArray
        data_array = timeseries.to_xarray()
        data_array = data_array.rename({'index':'time'})
        resampled_data = cal.resample(data_array)

        assert np.array_equal(resampled_data['data1'].values, expected)

        # test xarray Dataset
        dataset = dataframe.to_xarray()
        dataset = dataset.rename({'index':'time'})
        resampled_data = cal.resample(dataset)

        assert np.array_equal(resampled_data['data1'].values, expected)

        # test for missing time dimension failure
        dataset = dataframe.to_xarray()
        with pytest.raises(ValueError):
            resampled_data = cal.resample(dataset)

        # test for time dimension without time data failure
        data_array = series.to_xarray()
        data_array = data_array.rename({'index': 'time'})
        with pytest.raises(ValueError):
            resampled_data = cal.resample(data_array)
        
        # test multi-year xr input
        data_array_my = timeseries_my.to_xarray()
        data_array_my = data_array_my.rename({'index': 'time'})
        resampled_data = cal.resample(data_array_my)

        assert np.array_equal(resampled_data['data1'].isel(index=slice(None, 2)).values, expected_my)

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
