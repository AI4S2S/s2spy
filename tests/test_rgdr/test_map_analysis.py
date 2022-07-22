"""Tests for the s2s._RGDR.map_analysis module."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from s2spy.rgdr import _map_analysis


# pylint: disable=protected-access


class TestMapAnalysis:
    @pytest.fixture(autouse=True)
    def dummy_dataarray(self):
        time = pd.date_range("20161001", "20211001", freq="60d")

        da = xr.DataArray(
            np.tile(np.arange(len(time)), (2, 2, 1)),
            dims=["lat", "lon", "time"],
            coords={
                "time": time,
                "lat": np.arange(0, 2),
                "lon": np.arange(0, 2),
            },
        )
        return da

    @pytest.fixture(autouse=True)
    def dummy_timeseries(self, dummy_dataarray):
        return dummy_dataarray.isel(lat=0, lon=0).drop_vars(["lat", "lon"])

    def test_pearsonr(self):
        result = _map_analysis._pearsonr_nan([0, 0, 1], [0, 1, 1])
        expected = (0.5, 0.66666667)
        np.testing.assert_array_almost_equal(result, expected)

    def test_pearsonr_nan(self):
        result = _map_analysis._pearsonr_nan([np.nan, 0, 1], [0, 1, 1])
        expected = (np.nan, np.nan)
        np.testing.assert_array_almost_equal(result, expected)

    def test_correlation(self, dummy_dataarray, dummy_timeseries):
        c_val, p_val = _map_analysis.correlation(dummy_dataarray, dummy_timeseries)

        np.testing.assert_equal(c_val.values, 1)
        np.testing.assert_equal(p_val.values, 0)

    def test_correlation_dim_name(self, dummy_dataarray, dummy_timeseries):
        da = dummy_dataarray.rename({"time": "i_interval"})

        c_val, p_val = _map_analysis.correlation(
            da, dummy_timeseries.values, time_dim="i_interval"
        )

        np.testing.assert_equal(c_val.values, 1)
        np.testing.assert_equal(p_val.values, 0)

    def test_correlation_wrong_target_dim_name(self, dummy_dataarray, dummy_timeseries):
        ts = dummy_timeseries.rename({"time": "dummy"})
        with pytest.raises(ValueError):
            _map_analysis.correlation(dummy_dataarray, ts)

    def test_correlation_wrong_field_dim_name(self, dummy_dataarray, dummy_timeseries):
        da = dummy_dataarray.rename({"time": "dummy"})
        with pytest.raises(ValueError):
            _map_analysis.correlation(da, dummy_timeseries)

    def test_correlation_wrong_dim_count(self, dummy_dataarray):
        with pytest.raises(ValueError):
            _map_analysis.correlation(dummy_dataarray, dummy_dataarray)

    def test_correlation_wrong_target_shape(self, dummy_dataarray):
        ts = np.zeros((100, 1, 1))
        with pytest.raises(ValueError):
            _map_analysis.correlation(dummy_dataarray, ts)

    def test_partial_correlation(self):
        with pytest.raises(NotImplementedError):
            _map_analysis.partial_correlation("field", "target", "z")

    def test_regression(self):
        with pytest.raises(NotImplementedError):
            _map_analysis.regression("field", "target")

    def test_save_map(self):
        with pytest.raises(NotImplementedError):
            _map_analysis.save_map()

    def test_load_map(self):
        with pytest.raises(NotImplementedError):
            _map_analysis.load_map()
