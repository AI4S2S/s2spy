"""Tests for the s2spy.preprocess module."""
import numpy as np
import pytest
import scipy.signal
import xarray as xr
from s2spy import preprocess
from s2spy.preprocess import Preprocessor


TEST_FILE_PATH = "./tests/test_rgdr/test_data"

# pylint: disable=protected-access
# ruff: noqa: B018
# functions are called without variable assignment for testing the error message
# therefore need to suppress flake-burger B018


class TestPreprocessMethods:
    """Test preprocess methods."""

    # Define inputs as fixtures
    @pytest.fixture
    def raw_field(self):
        return xr.open_dataset(
            f"{TEST_FILE_PATH}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc"
        ).sel(
            time=slice("2010-01-01", "2011-12-31"),
            latitude=slice(40, 30),
            longitude=slice(180, 190),
        )

    def test_check_input_data_incorrect_type(self):
        dummy_data = np.ones((3, 3))
        with pytest.raises(ValueError):
            preprocess._check_input_data(dummy_data)

    def test_check_input_data_incorrect_dims(self, raw_field):
        dummy_dataarray = raw_field["sst"][0, ...]  # drop time dimension
        with pytest.raises(ValueError):
            preprocess._check_input_data(dummy_dataarray)

    def test_unknown_method_get_trend(self, raw_field):
        with pytest.raises(ValueError):
            preprocess._get_trend(raw_field, "unknown")

    def test_get_and_subtract_linear_trend(self, raw_field):
        expected = xr.apply_ufunc(
            scipy.signal.detrend,
            raw_field,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
        ).transpose("time", ...)
        trend = preprocess._get_trend(raw_field, "linear")
        result = preprocess._subtract_trend(raw_field, "linear", trend)
        np.testing.assert_array_almost_equal(result["sst"], expected["sst"])

    def test_check_temporal_resolution(self):
        with pytest.raises(ValueError):
            preprocess._check_temporal_resolution("hourly")  # type: ignore

    def test_get_climatology_daily(self, raw_field):
        result = preprocess._get_climatology(raw_field, timescale="daily")
        expected = (
            raw_field["sst"].sel(time=slice("2010-01-01", "2010-12-31")).data
            + raw_field["sst"].sel(time=slice("2011-01-01", "2011-12-31")).data
        ) / 2
        np.testing.assert_array_almost_equal(result["sst"], expected)

    def test_get_climatology_weekly(self, raw_field):
        raw_field_weekly = raw_field.resample(time="W").mean()
        result = preprocess._get_climatology(raw_field_weekly, timescale="weekly")
        # need to consider the actual calendar week number for the expected climatology
        raw_field_weekly["time"] = raw_field_weekly["time"].dt.isocalendar().week
        expected = raw_field_weekly.groupby("time").mean()
        np.testing.assert_array_almost_equal(result["sst"], expected["sst"])

    def test_get_climatology_monthly(self, raw_field):
        raw_field_monthly = raw_field.resample(time="M").mean()
        result = preprocess._get_climatology(raw_field_monthly, timescale="monthly")
        expected = (
            raw_field_monthly["sst"].sel(time=slice("2010-01-01", "2010-12-31")).data
            + raw_field_monthly["sst"].sel(time=slice("2011-01-01", "2011-12-31")).data
        ) / 2
        np.testing.assert_array_almost_equal(result["sst"], expected)

    def test_get_climatology_wrong_timescale(self, raw_field):
        with pytest.raises(ValueError):
            preprocess._get_climatology(raw_field, timescale="hourly")  # type: ignore

    @pytest.mark.parametrize("timescale", ("weekly", "monthly"))
    def test_check_data_resolution_mismatch(self, raw_field, timescale):
        with pytest.warns(UserWarning):
            preprocess._check_data_resolution_match(raw_field, timescale)

    def test_check_data_resolution_match(self, raw_field):
        preprocess._check_data_resolution_match(raw_field, "daily")


class TestPreprocessor:
    """Test preprocessor."""

    @pytest.fixture
    def preprocessor(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            timescale="daily",
            detrend="linear",
            subtract_climatology=True,
        )
        return prep

    @pytest.fixture(params=[None, 1])
    def preprocessor_no_rolling(self, request):
        prep = preprocess.Preprocessor(
            rolling_window_size=request.param,
            timescale="monthly",
            detrend=None,
            subtract_climatology=True,
        )
        return prep

    @pytest.fixture
    def preprocessor_no_climatology(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            timescale="daily",
            detrend="linear",
            subtract_climatology=False,
        )
        return prep

    @pytest.fixture
    def preprocessor_no_detrend(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            timescale="daily",
            detrend=None,
            subtract_climatology=True,
        )
        return prep

    @pytest.fixture
    def raw_field(self):
        return xr.open_dataset(
            f"{TEST_FILE_PATH}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc"
        ).sel(
            time=slice("2010-01-01", "2014-12-31"),  # including leap year
            latitude=slice(40, 30),
            longitude=slice(180, 190),
        )

    def test_init(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            timescale="weekly",
            detrend="linear",
            subtract_climatology=True,
        )
        assert isinstance(prep, Preprocessor)

    def test_fit(self, preprocessor, raw_field):
        preprocessor.fit(raw_field)
        assert (
            preprocessor.climatology is not None
        )  # climatology with leap years are counted

    def test_fit_no_rolling(self, preprocessor_no_rolling, raw_field):
        preprocessor_no_rolling.fit(raw_field)
        assert preprocessor_no_rolling.climatology == preprocess._get_climatology(
            raw_field, timescale="daily"
        )

    def test_transform(self, preprocessor, raw_field):
        preprocessor.fit(raw_field)
        preprocessed_data = preprocessor.transform(raw_field)
        assert preprocessed_data is not None

    def test_transform_without_fit(self, preprocessor, raw_field):
        with pytest.raises(ValueError):
            preprocessor.transform(raw_field)

    def test_fit_and_transform_no_climatology(
        self, preprocessor_no_climatology, raw_field
    ):
        preprocessor_no_climatology.fit(raw_field)
        results = preprocessor_no_climatology.transform(raw_field)
        expected = preprocess._subtract_trend(
            raw_field, "linear", preprocessor_no_climatology.trend
        )
        assert results == expected

    def test_fit_and_transform_no_detrend(self, preprocessor_no_detrend, raw_field):
        preprocessor_no_detrend.fit(raw_field)
        results = preprocessor_no_detrend.transform(raw_field)
        expected = (
            raw_field.groupby("time.dayofyear") - preprocessor_no_detrend.climatology
        )
        assert results == expected

    def test_fit_and_transform_no_climatology_and_detrend(self, raw_field):
        prep = preprocess.Preprocessor(
            rolling_window_size=10,
            timescale="daily",
            detrend=None,
            subtract_climatology=False,
        )
        prep.fit(raw_field)
        results = prep.transform(raw_field)

        assert results == raw_field

    def test_fit_transform(self, preprocessor, raw_field):
        preprocessed_data = preprocessor.fit_transform(raw_field)
        assert preprocessed_data is not None

    def test_trend_property_not_fit(self, preprocessor):
        with pytest.raises(ValueError, match="The preprocessor has to be fit"):
            preprocessor.trend

    def test_trend_property_no_detrend(self, preprocessor_no_detrend, raw_field):
        preprocessor_no_detrend.fit(raw_field)
        with pytest.raises(ValueError, match="Detrending is set to `None`"):
            preprocessor_no_detrend.trend

    def test_climatology_property_not_fit(self, preprocessor):
        with pytest.raises(ValueError, match="The preprocessor has to be fit"):
            preprocessor.climatology

    def test_trend_property_no_climatology(self, preprocessor_no_climatology):
        with pytest.raises(ValueError, match="subtract_climatology is set to `False`"):
            preprocessor_no_climatology.climatology
