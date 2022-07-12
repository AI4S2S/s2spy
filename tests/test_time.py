"""Tests for the s2spy.time module.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from s2spy.time import AdventCalendar


# pylint: disable=protected-access,missing-function-docstring


def interval(start, end):
    """Shorthand for more readable tests."""
    return pd.Interval(pd.Timestamp(start), pd.Timestamp(end))


class TestAdventCalendar:
    """Test AdventCalendar methods."""
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = AdventCalendar(anchor_date=(12, 31), freq="240d")
        cal.map_years(2021, 2021)
        return cal

    def test_init(self):
        cal = AdventCalendar()
        assert isinstance(cal, AdventCalendar)

    def test_repr(self):
        cal = AdventCalendar()
        assert repr(cal) == "AdventCalendar(month=11, day=30, freq=7d)"

    def test_repr_with_intervals(self, dummy_calendar):
        expected_calendar_repr = \
            'i_interval 0\nanchor_year \n2021 (2021-05-05, 2021-12-31]'.replace(" ", "")
        assert repr(dummy_calendar).replace(" ", "") == expected_calendar_repr

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
            cal.mark_target_period(start="20200101", periods=5)

    def test_get_lagged_indices(self):
        cal = AdventCalendar()

        with pytest.raises(NotImplementedError):
            cal.get_lagged_indices()

    def test_flat(self, dummy_calendar):
        expected = np.array([interval("2021-05-05", "2021-12-31")])
        cal = AdventCalendar(anchor_date=(12, 31), freq="240d")
        cal.map_years(2021, 2021)
        assert np.array_equal(dummy_calendar.flat, expected)

    def test_flat_no_intervals(self):
        cal = AdventCalendar()
        with pytest.raises(ValueError):
            cal.flat # pylint: disable=pointless-statement

class TestMap:
    """Test map to year(s)/data methods"""
    def test_map_years(self):
        cal = AdventCalendar(anchor_date=(12, 31), freq="180d")
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
        assert np.array_equal(cal._intervals, expected)

    def test_map_years_single(self):
        cal = AdventCalendar(anchor_date=(12, 31), freq="180d")
        cal.map_years(2020, 2020)
        expected = np.array([
            [
                interval("2020-07-04", "2020-12-31"),
                interval("2020-01-06", "2020-07-04"),
            ]
        ])
        assert np.array_equal(cal._intervals, expected)

    def test_map_to_data_edge_case_last_year(self):
        # test the edge value when the input could not cover the anchor date
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
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
        assert np.array_equal(cal._intervals, expected)

    def test_map_to_data_single_year_coverage(self):
        # test the single year coverage
        cal = AdventCalendar(anchor_date=(12, 31), freq='180d')
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

        assert np.array_equal(cal._intervals, expected)

    def test_map_to_data_edge_case_first_year(self):
        # test the edge value when the input covers the anchor date
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
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

        assert np.array_equal(cal._intervals, expected)

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
        cal.map_to_data(timeseries)

        expected = np.array([
            [
                interval("2021-04-18", "2021-10-15"),
                interval("2020-10-20", "2021-04-18"),
            ]
        ])

        assert np.array_equal(cal._intervals, expected)

    def test_map_to_data_xarray_input(self):
        # test when the input data has reverse order time index
        cal = AdventCalendar(anchor_date=(10, 15), freq='180d')
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

        assert np.all(cal._intervals == expected)

    # Note: add more test cases for different number of target periods!
    max_lag_edge_cases = [(73, ['2019'], 74),
                          (72, ['2019', '2018'], 73)]
    # Test the edge cases of max_lag; where the max_lag just fits in exactly 365 days,
    # and where the max_lag just causes the calendar to skip a year
    @pytest.mark.parametrize("max_lag,expected_index,expected_size", max_lag_edge_cases)
    def test_max_lag_skip_years(self, max_lag, expected_index, expected_size):
        calendar = AdventCalendar(anchor_date=(12, 31), freq="5d", max_lag=max_lag)
        calendar = calendar.map_years(2018, 2019)

        np.testing.assert_array_equal(calendar._intervals.index.values, expected_index)
        assert calendar._intervals.iloc[0].size == expected_size


class TestResample:
    """Test resample methods."""
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
        dataarray = series.to_xarray()
        dataarray = dataarray.rename({'index': 'time'})
        return dataarray, expected

    @pytest.fixture
    def dummy_dataset(self, dummy_dataframe):
        dataframe, expected = dummy_dataframe
        dataset = dataframe.to_xarray().rename({'index': 'time'})
        return dataset, expected

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
        dataarray, expected = dummy_dataarray
        resampled_data = dummy_calendar.resample(dataarray)
        testing_vals = resampled_data["data1"].isel(anchor_year=0)
        np.testing.assert_allclose(testing_vals, expected)

    def test_dataset(self, dummy_calendar, dummy_dataset):
        dataset, expected = dummy_dataset
        resampled_data = dummy_calendar.resample(dataset)
        testing_vals = resampled_data["data1"].isel(anchor_year=0)
        np.testing.assert_allclose(testing_vals, expected)

    def test_missing_time_dim(self, dummy_calendar, dummy_dataset):
        dataset, _ = dummy_dataset
        with pytest.raises(ValueError):
            dummy_calendar.resample(dataset.rename({'time': 'index'}))

    def test_non_time_dim(self, dummy_calendar, dummy_dataset):
        dataset, _ = dummy_dataset
        dataset['time'] = np.arange(dataset['time'].size)
        with pytest.raises(ValueError):
            dummy_calendar.resample(dataset)

class TestTrainTest:
    """Test train test methods."""
    # Define all required inputs as fixtures:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = AdventCalendar(anchor_date=(10, 15), freq="180d")
        return cal.map_years(2019, 2021)

    def test_set_traintest_method(self, dummy_calendar):
        dummy_calendar.set_traintest_method("kfold", n_splits = 2, shuffle = False)
        # check the first fold
        expected_group = ['test', 'test', 'train']
        assert np.array_equal(dummy_calendar._traintest["fold_0"].values, expected_group)

    def test_set_traintest_method_not_support(self, dummy_calendar):
        # test when the given method is not supported
        with pytest.raises(ValueError):
            dummy_calendar.set_traintest_method("not_a_real_method")

    def test_traintest_method_not_set(self, dummy_calendar):
        with pytest.raises(RuntimeError):
            dummy_calendar.traintest # pylint: disable=pointless-statement

    def test_traintest_exist(self, dummy_calendar):
        dummy_calendar.set_traintest_method("kfold", n_splits = 2)
        # test when the train/test method has already been set
        # and overwrite is not flagged
        with pytest.raises(ValueError):
            dummy_calendar.set_traintest_method("kfold", n_splits = 3)

    def test_traintest_overwrite(self, dummy_calendar):
        dummy_calendar.set_traintest_method("kfold", n_splits = 2)
        dummy_calendar.set_traintest_method("kfold", n_splits = 3, overwrite = True)
        expected_group = ['train', 'test', 'train']
        assert np.array_equal(dummy_calendar._traintest["fold_1"].values, expected_group)

    def test_traintest(self, dummy_calendar):
        dummy_calendar.set_traintest_method("kfold", n_splits = 2, shuffle = False)
        traintest_group = dummy_calendar.traintest
        expected = np.array(
            [
                [
                    interval("2021-04-18", "2021-10-15"),
                    interval("2020-10-20", "2021-04-18"),
                ],
                [
                    interval("2020-04-18", "2020-10-15"),
                    interval("2019-10-21", "2020-04-18"),
                ],
                [
                    interval("2019-04-18", "2019-10-15"),
                    interval("2018-10-20", "2019-04-18"),
                ],
            ]
        )

        assert np.array_equal(traintest_group, expected)
