"""Tests for the s2spy.preprocess module.
"""
import numpy as np
import pytest
import scipy.signal
import xarray as xr
from s2spy import preprocess
from s2spy.preprocess import Preprocessor


TEST_FILE_PATH = "./tests/test_rgdr/test_data"

# pylint: disable=protected-access


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

    def test_get_and_apply_linear_trend(self, raw_field):
        expected = xr.apply_ufunc(
            scipy.signal.detrend,
            raw_field,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
        ).transpose("time", ...)
        trend = preprocess._get_trend(raw_field, "linear")
        result = preprocess._apply_trend(raw_field, "linear", trend)
        np.testing.assert_array_almost_equal(result["sst"], expected["sst"])

    def test_get_climatology(self, raw_field):
        result = preprocess._get_climatology(raw_field)
        expected = (
            raw_field["sst"].sel(time=slice("2010-01-01", "2010-12-31")).data
            + raw_field["sst"].sel(time=slice("2011-01-01", "2011-12-31")).data
        ) / 2
        np.testing.assert_array_almost_equal(result["sst"], expected)


class TestPreprocessor:
    """Test preprocessor."""

    @pytest.fixture
    def preprocessor(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            detrend="linear",
            remove_climatology=True,
        )
        return prep

    @pytest.fixture
    def raw_field(self):
        return xr.open_dataset(
            f"{TEST_FILE_PATH}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc"
        ).sel(
            time=slice("2010-01-01", "2014-12-31"), #including leap year
            latitude=slice(40, 30),
            longitude=slice(180, 190),
        )

    def test_init(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            detrend="linear",
            remove_climatology=True,
        )        
        assert isinstance(prep, Preprocessor)

    def test_fit(self, preprocessor, raw_field):
        preprocessor.fit(raw_field)
        assert preprocessor.climatology is not None # climatology with leap years are counted

    def test_rolling_mean(self, raw_field):
        prep = preprocess.Preprocessor(
            rolling_window_size=2,
            detrend=None,
            remove_climatology=False,
        )
        test_data = raw_field.sel(time=slice("2010-01-01", "2010-01-02"))
        data_rolling = prep.fit(test_data)
        test_data["sst"][1,:,:] = test_data["sst"].mean(axis=0)
        np.testing.assert_array_almost_equal(data_rolling["sst"], test_data["sst"])

    def test_transform(self, preprocessor, raw_field):
        preprocessor.fit(raw_field)
        preprocessed_data = preprocessor.transform(raw_field)
        assert preprocessed_data is not None

    def test_transform_without_fit(self, preprocessor, raw_field):
        with pytest.raises(ValueError):
            preprocessor.transform(raw_field)

    def test_fit_transform(self, preprocessor, raw_field):
        preprocessed_data = preprocessor.fit_transform(raw_field)
        assert preprocessed_data is not None

