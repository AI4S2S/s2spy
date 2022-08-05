"""Tests for the s2spy.time module.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from s2spy.time import AdventCalendar
from s2spy.time import resample


def interval(start, end):
    """Shorthand for more readable tests."""
    return pd.Interval(pd.Timestamp(start), pd.Timestamp(end))


class TestAdventCalendar:
    """Test AdventCalendar methods."""
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = AdventCalendar(anchor=(12, 31), freq="240d")
        cal.map_years(2021, 2021)
        return cal

    def test_init(self):
        cal = AdventCalendar()
        assert isinstance(cal, AdventCalendar)

    def test_repr(self):
        cal = AdventCalendar()
        assert repr(cal) == (
            "AdventCalendar(month=11, day=30, freq=7d, n_targets=1, max_lag=None)"
        )

    def test_show(self, dummy_calendar):
        expected_calendar_repr = ('i_interval (target) 0\n'
                                  'anchor_year \n'
                                  '2021 (2021-05-05, 2021-12-31]')
        expected_calendar_repr = expected_calendar_repr.replace(" ", "")
        assert repr(dummy_calendar.show()).replace(" ", "") == expected_calendar_repr

    def test_flat(self):
        expected = np.array([interval("2021-05-05", "2021-12-31")])
        cal = AdventCalendar(anchor=(12, 31), freq="240d")
        cal.map_years(2021, 2021)
        assert np.array_equal(cal.flat, expected)

    def test_flat_no_intervals(self):
        cal = AdventCalendar()
        with pytest.raises(ValueError):
            cal.get_intervals()


class TestMap:
    """Test map to year(s)/data methods"""
    def test_map_years(self):
        cal = AdventCalendar(anchor=(12, 31), freq="180d")
        cal.map_years(2020, 2021)
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
        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_years_single(self):
        cal = AdventCalendar(anchor=(12, 31), freq="180d")
        cal.map_years(2020, 2020)
        expected = np.array([
            [
                interval("2020-07-04", "2020-12-31"),
                interval("2020-01-06", "2020-07-04"),
            ]
        ])
        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_to_data_edge_case_last_year(self):
        # test the edge value when the input could not cover the anchor date
        cal = AdventCalendar(anchor=(10, 15), freq='180d')
        # single year covered
        time_index = pd.date_range('20191020', '20211001', freq='60d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        cal.map_to_data(timeseries)
        expected = np.array([
            [
                interval("2020-04-18", "2020-10-15"),
                interval("2019-10-21", "2020-04-18"),
            ]
        ])
        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_to_data_single_year_coverage(self):
        # test the single year coverage
        cal = AdventCalendar(anchor=(12, 31), freq='180d')
        # multiple years covered
        time_index = pd.date_range('20210101', '20211231', freq='7d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        cal.map_to_data(timeseries)

        expected = np.array([
            [
                interval("2021-07-04", "2021-12-31"),
                interval("2021-01-05", "2021-07-04"),
            ]
        ])

        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_to_data_edge_case_first_year(self):
        # test the edge value when the input covers the anchor date
        cal = AdventCalendar(anchor=(10, 15), freq='180d')
        # multiple years covered
        time_index = pd.date_range('20191010', '20211225', freq='60d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index)
        cal.map_to_data(timeseries)

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

        assert np.array_equal(cal.get_intervals(), expected)

    # def test_map_to_data_value_error(self):
    #     # test when the input data is not sufficient to cover one year
    #     cal = AdventCalendar(anchor=(10, 15), freq='180d')
    #     with pytest.raises(ValueError):
    #         time_index = pd.date_range('20201020', '20211001', freq='60d')
    #         test_data = np.random.random(len(time_index))
    #         timeseries = pd.Series(test_data, index=time_index)
    #         cal.map_to_data(timeseries)

    def test_map_to_data_input_time_backward(self):
        # test when the input data has reverse order time index
        cal = AdventCalendar(anchor=(10, 15), freq='180d')
        time_index = pd.date_range('20201010', '20211225', freq='60d')
        test_data = np.random.random(len(time_index))
        timeseries = pd.Series(test_data, index=time_index[::-1])
        cal.map_to_data(timeseries)

        expected = np.array([
            [
                interval("2021-04-18", "2021-10-15"),
                interval("2020-10-20", "2021-04-18"),
            ]
        ])

        assert np.array_equal(cal.get_intervals(), expected)

    def test_map_to_data_xarray_input(self):
        # test when the input data has reverse order time index
        cal = AdventCalendar(anchor=(10, 15), freq='180d')
        time_index = pd.date_range('20201010', '20211225', freq='60d')
        test_data = np.random.random(len(time_index))
        dataarray = xr.DataArray(
            data=test_data,
            coords={'time': time_index})
        cal.map_to_data(dataarray)

        expected = np.array(
            [
                interval("2021-04-18", "2021-10-15"),
                interval("2020-10-20", "2021-04-18"),
            ]
        )

        assert np.all(cal.get_intervals() == expected)

    def test_missing_time_dim(self):
        cal = AdventCalendar(anchor=(10, 15), freq='180d')
        time_index = pd.date_range('20191020', '20211001', freq='60d')
        test_data = np.random.random(len(time_index))
        dataframe = pd.DataFrame(test_data, index=time_index)
        dataset = dataframe.to_xarray()
        with pytest.raises(ValueError):
            cal.map_to_data(dataset)

    def test_non_time_dim(self):
        cal = AdventCalendar(anchor=(10, 15), freq='180d')
        time_index = pd.date_range('20191020', '20211001', freq='60d')
        test_data = np.random.random(len(time_index))
        dataframe = pd.DataFrame(test_data, index=time_index)
        dataset = dataframe.to_xarray().rename({'index':'time'})
        dataset['time'] = np.arange(dataset['time'].size)
        with pytest.raises(ValueError):
            cal.map_to_data(dataset)

    # Note: add more test cases for different number of target periods!
    max_lag_edge_cases = [(73, ['2019'], 74),
                          (72, ['2019', '2018'], 73)]
    # Test the edge cases of max_lag; where the max_lag just fits in exactly 365 days,
    # and where the max_lag just causes the calendar to skip a year
    @pytest.mark.parametrize("max_lag,expected_index,expected_size", max_lag_edge_cases)
    def test_max_lag_skip_years(self, max_lag, expected_index, expected_size):
        calendar = AdventCalendar(anchor=(12, 31), freq="5d", max_lag=max_lag)
        calendar = calendar.map_years(2018, 2019)

        np.testing.assert_array_equal(calendar.get_intervals().index.values, expected_index)
        assert calendar.get_intervals().iloc[0].size == expected_size


class TestResample:
    """Test resample methods."""
    # Define all required inputs as fixtures:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        return AdventCalendar(anchor=(10, 15), freq="180d")

    @pytest.fixture(autouse=True, params=[1,2,3])
    def dummy_calendar_targets(self, request):
        return AdventCalendar(anchor=(5, 10), freq="100D", n_targets=request.param)

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
        dataarray = series.to_xarray()
        dataarray = dataarray.rename({'index': 'time'})
        return dataarray, expected

    @pytest.fixture
    def dummy_dataset(self, dummy_dataframe):
        dataframe, expected = dummy_dataframe
        dataset = dataframe.to_xarray().rename({'index': 'time'})
        return dataset, expected

    # Tests start here:
    def test_non_mapped_calendar(self, dummy_calendar):
        with pytest.raises(ValueError):
            resample(dummy_calendar, None)

    def test_nontime_index(self, dummy_calendar, dummy_series):
        series, _ = dummy_series
        cal = dummy_calendar.map_to_data(series)
        series = series.reset_index()
        with pytest.raises(ValueError):
            resample(cal, series)

    def test_series(self, dummy_calendar, dummy_series):
        series, expected = dummy_series
        cal = dummy_calendar.map_to_data(series)
        resampled_data = resample(cal, series)
        np.testing.assert_allclose(
            resampled_data['data1'].iloc[:2], expected)

    def test_unnamed_series(self, dummy_calendar, dummy_series):
        series, expected = dummy_series
        series.name = None
        cal = dummy_calendar.map_to_data(series)
        resampled_data = resample(cal, series)
        np.testing.assert_allclose(
            resampled_data["mean_data"].iloc[:2], expected)

    def test_dataframe(self, dummy_calendar, dummy_dataframe):
        dataframe, expected = dummy_dataframe
        cal = dummy_calendar.map_to_data(dataframe)
        resampled_data = resample(cal, dataframe)
        np.testing.assert_allclose(resampled_data["data1"].iloc[:2], expected)

    def test_dataarray(self, dummy_calendar, dummy_dataarray):
        dataarray, expected = dummy_dataarray
        cal = dummy_calendar.map_to_data(dataarray)
        resampled_data = resample(cal, dataarray)
        testing_vals = resampled_data["data1"].isel(anchor_year=0)
        np.testing.assert_allclose(testing_vals, expected)

    def test_dataset(self, dummy_calendar, dummy_dataset):
        dataset, expected = dummy_dataset
        cal = dummy_calendar.map_to_data(dataset)
        resampled_data = resample(cal, dataset)
        testing_vals = resampled_data["data1"].isel(anchor_year=0)
        np.testing.assert_allclose(testing_vals, expected)

    def test_target_period_dataframe(self, dummy_calendar_targets, dummy_dataframe):
        df, _ = dummy_dataframe
        calendar = dummy_calendar_targets.map_to_data(df)
        resampled_data = resample(calendar, df)
        expected = np.zeros(resampled_data.index.size, dtype=bool)
        for i in range(calendar.n_targets):
            expected[i::3] = True
        np.testing.assert_array_equal(resampled_data['target'].values, expected)

    def test_target_period_dataset(self, dummy_calendar_targets, dummy_dataset):
        ds, _ = dummy_dataset
        calendar = dummy_calendar_targets.map_to_data(ds)
        resampled_data = resample(calendar, ds)
        expected = np.zeros(3, dtype=bool)
        expected[:dummy_calendar_targets.n_targets] = True
        np.testing.assert_array_equal(resampled_data['target'].values, expected)
