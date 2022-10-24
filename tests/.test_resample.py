"""Tests for s2spy.time's resample module.
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from s2spy.time import AdventCalendar
from s2spy.time import resample


class TestResample:
    """Test resample methods."""

    # Define all required inputs as fixtures:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        return AdventCalendar(anchor="10-15", freq="180d")

    @pytest.fixture(autouse=True, params=[1, 2, 3])
    def dummy_calendar_targets(self, request):
        return AdventCalendar(anchor="5-10", freq="100d", n_targets=request.param)

    @pytest.fixture(params=["20151020", "20191015"])
    def dummy_series(self, request):
        time_index = pd.date_range(request.param, "20211001", freq="60d")
        test_data = np.random.random(len(time_index))
        expected = np.array([test_data[4:7].mean(), test_data[1:4].mean()])
        series = pd.Series(test_data, index=time_index, name="data1")
        return series, expected

    @pytest.fixture
    def dummy_dataframe(self, dummy_series):
        series, expected = dummy_series
        return pd.DataFrame(series), expected

    @pytest.fixture
    def dummy_dataarray(self, dummy_series):
        series, expected = dummy_series
        dataarray = series.to_xarray()
        dataarray = dataarray.rename({"index": "time"})
        return dataarray, expected

    @pytest.fixture
    def dummy_dataset(self, dummy_dataframe):
        dataframe, expected = dummy_dataframe
        dataset = dataframe.to_xarray().rename({"index": "time"})
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
        np.testing.assert_allclose(resampled_data["data1"].iloc[:2], expected)

    def test_unnamed_series(self, dummy_calendar, dummy_series):
        series, expected = dummy_series
        series.name = None
        cal = dummy_calendar.map_to_data(series)
        resampled_data = resample(cal, series)
        np.testing.assert_allclose(resampled_data["mean_data"].iloc[:2], expected)

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
        np.testing.assert_array_equal(resampled_data["target"].values, expected)

    def test_target_period_dataset(self, dummy_calendar_targets, dummy_dataset):
        ds, _ = dummy_dataset
        calendar = dummy_calendar_targets.map_to_data(ds)
        resampled_data = resample(calendar, ds)
        expected = np.zeros(3, dtype=bool)
        expected[: dummy_calendar_targets.n_targets] = True
        np.testing.assert_array_equal(resampled_data["target"].values, expected)

    def test_allow_overlap_dataframe(self):
        calendar = AdventCalendar(anchor="10-15", freq="100d")
        calendar.set_max_lag(5, allow_overlap=True)
        time_index = pd.date_range("20151101", "20211101", freq="50d")
        test_data = np.random.random(len(time_index))
        series = pd.Series(test_data, index=time_index)
        calendar.map_to_data(series)
        intervals = calendar.get_intervals()
        # 4 anchor years are expected if overlap is allowed
        assert len(intervals.index) == 4

    # Test data for missing intervals, too low frequency.
    def test_missing_intervals_dataframe(self, dummy_calendar, dummy_dataframe):
        dataframe, _ = dummy_dataframe
        cal = dummy_calendar.map_years(2020, 2025)
        with pytest.warns(UserWarning):
            resample(cal, dataframe)

    def test_missing_intervals_dataset(self, dummy_calendar, dummy_dataset):
        dataset, _ = dummy_dataset
        cal = dummy_calendar.map_years(2020, 2025)
        with pytest.warns(UserWarning):
            resample(cal, dataset)

    def test_low_freq_dataframe(self, dummy_dataframe):
        cal = AdventCalendar(anchor="10-15", freq="1d")
        dataframe, _ = dummy_dataframe
        cal = cal.map_to_data(dataframe)
        with pytest.warns(UserWarning):
            resample(cal, dataframe)

    def test_low_freq_dataset(self, dummy_dataset):
        cal = AdventCalendar(anchor="10-15", freq="1d")
        dataset, _ = dummy_dataset
        cal = cal.map_to_data(dataset)
        with pytest.warns(UserWarning):
            resample(cal, dataset)

    def test_1day_freq_dataframe(self):
        # Will test the regular expression match and pre-pending of '1' in the
        # check_input_frequency utility function
        calendar = AdventCalendar(anchor="10-15", freq="1d")
        time_index = pd.date_range("20191101", "20211101", freq="1d")
        test_data = np.random.random(len(time_index))
        series = pd.Series(test_data, index=time_index, name="data1")
        calendar.map_to_data(series)
        calendar.get_intervals()

    def test_to_netcdf(self, dummy_calendar, dummy_dataset):
        # Test to ensure that xarray data resampled using the calendar can be written
        # to a netCDF file.
        dataset, _ = dummy_dataset
        cal = dummy_calendar.map_to_data(dataset)
        resampled_data = resample(cal, dataset)
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = Path(tmpdirname) / 'test.nc'
            resampled_data.to_netcdf(path)
