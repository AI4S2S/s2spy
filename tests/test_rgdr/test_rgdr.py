"""Tests for the s2s._RGDR.rgdr module."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from s2spy import RGDR
from s2spy.rgdr import _rgdr
from s2spy.time import AdventCalendar


# pylint: disable=protected-access


class TestRGDR:
    """Test RGDR methods."""

    @pytest.fixture(autouse=True)
    def dummy_rgdr(self):
        return RGDR("timeseries")

    def test_init(self):
        rgdr = RGDR("timeseries")
        assert isinstance(rgdr, RGDR)

    def test_fit(self, dummy_rgdr):
        with pytest.raises(NotImplementedError):
            dummy_rgdr.fit("data")

    def test_transform(self, dummy_rgdr):
        with pytest.raises(NotImplementedError):
            dummy_rgdr.transform("data")


class TestCorrelation:
    @pytest.fixture(autouse=True)
    def dummy_dataarray(self):
        time = pd.date_range("20161001", "20211001", freq="60d")

        return xr.DataArray(
            np.tile(np.arange(len(time)), (2, 2, 1)),
            dims=["lat", "lon", "time"],
            coords={
                "time": time,
                "lat": np.arange(0, 2),
                "lon": np.arange(0, 2),
            },
        )

    @pytest.fixture(autouse=True)
    def dummy_timeseries(self, dummy_dataarray):
        return dummy_dataarray.isel(lat=0, lon=0).drop_vars(["lat", "lon"])

    def test_pearsonr(self):
        result = _rgdr._pearsonr_nan([0, 0, 1], [0, 1, 1])
        expected = (0.5, 0.66666667)
        np.testing.assert_array_almost_equal(result, expected)

    def test_pearsonr_nan(self):
        result = _rgdr._pearsonr_nan([np.nan, 0, 1], [0, 1, 1])
        expected = (np.nan, np.nan)
        np.testing.assert_array_almost_equal(result, expected)

    def test_correlation(self, dummy_dataarray, dummy_timeseries):
        c_val, p_val = _rgdr.correlation(dummy_dataarray, dummy_timeseries)

        np.testing.assert_equal(c_val.values, 1)
        np.testing.assert_equal(p_val.values, 0)

    def test_correlation_dim_name(self, dummy_dataarray, dummy_timeseries):
        da = dummy_dataarray.rename({"time": "i_interval"})
        ts = dummy_timeseries.rename({"time": "i_interval"})
        c_val, p_val = _rgdr.correlation(da, ts, corr_dim="i_interval")

        np.testing.assert_equal(c_val.values, 1)
        np.testing.assert_equal(p_val.values, 0)

    def test_correlation_wrong_target_dim_name(self, dummy_dataarray, dummy_timeseries):
        ts = dummy_timeseries.rename({"time": "dummy"})
        with pytest.raises(AssertionError):
            _rgdr.correlation(dummy_dataarray, ts)

    def test_correlation_wrong_field_dim_name(self, dummy_dataarray, dummy_timeseries):
        da = dummy_dataarray.rename({"time": "dummy"})
        with pytest.raises(AssertionError):
            _rgdr.correlation(da, dummy_timeseries)

    def test_correlation_wrong_target_dims(self, dummy_dataarray):
        ts = dummy_dataarray.rename({"lat": "latitude"})
        with pytest.raises(AssertionError):
            _rgdr.correlation(dummy_dataarray, ts)

    def test_partial_correlation(self):
        with pytest.raises(NotImplementedError):
            _rgdr.partial_correlation("field", "target", "z")

    def test_regression(self):
        with pytest.raises(NotImplementedError):
            _rgdr.regression("field", "target")


test_file_path = "./tests/test_rgdr/test_data"


class TestMapRegions:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        return AdventCalendar(anchor_date=(8, 31), freq="30d")

    @pytest.fixture(autouse=True)
    def example_target(self):
        return xr.open_dataset(f"{test_file_path}/tf5_nc5_dendo_80d77.nc").sel(
            cluster=3
        )

    @pytest.fixture(autouse=True)
    def example_field(self):
        return xr.open_dataset(
            f"{test_file_path}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc"
        )

    @pytest.fixture(autouse=True)
    def example_corr(self, dummy_calendar, example_target, example_field):
        field = dummy_calendar.resample(example_field)
        target = dummy_calendar.resample(example_target)

        field["corr"], field["p_val"] = _rgdr.correlation(
            field.sst, target.ts.sel(i_interval=0), corr_dim="anchor_year"
        )

        return field

    @pytest.fixture(autouse=True)
    def expected_labels(self):
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0],
            ]
        )

    def test_dbscan(self, example_corr, expected_labels):
        clusters = _rgdr.masked_spherical_dbscan(example_corr.sel(i_interval=1))

        np.testing.assert_array_equal(clusters["cluster_labels"], expected_labels)

    def test_dbscan_min_area(self, example_corr, expected_labels):
        clusters = _rgdr.masked_spherical_dbscan(
            example_corr.sel(i_interval=1), min_area_km2=3000**2
        )
        expected_labels[expected_labels == -1] = 0  # Small -1 cluster is missing

        np.testing.assert_array_equal(clusters["cluster_labels"], expected_labels)

    def test_cluster(self, example_corr):
        clustered_data = _rgdr.reduce_to_clusters(
            example_corr.sel(i_interval=1), eps_km=600, alpha=0.04
        )

        np.testing.assert_array_equal(clustered_data["cluster_labels"], [-1, 0, 1])

        assert set(clustered_data.sst.dims) == {"anchor_year", "cluster_labels"}
