"""Tests for the s2s.rgdr module."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from s2spy import RGDR
from s2spy.rgdr import rgdr
from s2spy.time import AdventCalendar
from s2spy.time import resample


TEST_FILE_PATH = "./tests/test_rgdr/test_data"

# pylint: disable=protected-access


@pytest.fixture(autouse=True, scope="class")
def dummy_calendar():
    return AdventCalendar(anchor=(8, 31), freq="30d")


@pytest.fixture(autouse=True, scope="module")
def raw_target():
    return xr.open_dataset(f"{TEST_FILE_PATH}/tf5_nc5_dendo_80d77.nc").sel(cluster=3)


@pytest.fixture(autouse=True, scope="module")
def raw_field():
    return xr.open_dataset(
        f"{TEST_FILE_PATH}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc"
    )


@pytest.fixture(autouse=True, scope="class")
def example_field(raw_field, dummy_calendar):
    cal = dummy_calendar.map_to_data(raw_field)
    return resample(cal, raw_field).sst.isel(i_interval=1)


@pytest.fixture(autouse=True, scope="class")
def example_target(raw_target, raw_field, dummy_calendar):
    cal = dummy_calendar.map_to_data(raw_field)
    return resample(cal, raw_target).ts.isel(i_interval=0)


@pytest.fixture(scope="class")
def example_corr(example_target, example_field):
    return rgdr.correlation(example_field, example_target, corr_dim="anchor_year")


@pytest.fixture(autouse=True, scope="class")
def expected_labels():
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        ]
    )


class TestCorrelation:
    """Validates that the correlation ufunc works, as well as the input checks."""

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
        result = rgdr._pearsonr_nan([0, 0, 1], [0, 1, 1])
        expected = (0.5, 0.66666667)
        np.testing.assert_array_almost_equal(result, expected)

    def test_pearsonr_nan(self):
        result = rgdr._pearsonr_nan([np.nan, 0, 1], [0, 1, 1])
        expected = (np.nan, np.nan)
        np.testing.assert_array_almost_equal(result, expected)

    def test_correlation(self, dummy_dataarray, dummy_timeseries):
        c_val, p_val = rgdr.correlation(dummy_dataarray, dummy_timeseries)

        np.testing.assert_equal(c_val.values, 1)
        np.testing.assert_equal(p_val.values, 0)

    def test_correlation_dim_name(self, dummy_dataarray, dummy_timeseries):
        da = dummy_dataarray.rename({"time": "i_interval"})
        ts = dummy_timeseries.rename({"time": "i_interval"})
        c_val, p_val = rgdr.correlation(da, ts, corr_dim="i_interval")

        np.testing.assert_equal(c_val.values, 1)
        np.testing.assert_equal(p_val.values, 0)

    def test_correlation_wrong_target_dim_name(self, dummy_dataarray, dummy_timeseries):
        ts = dummy_timeseries.rename({"time": "dummy"})
        with pytest.raises(AssertionError):
            rgdr.correlation(dummy_dataarray, ts)

    def test_correlation_wrong_field_dim_name(self, dummy_dataarray, dummy_timeseries):
        da = dummy_dataarray.rename({"time": "dummy"})
        with pytest.raises(AssertionError):
            rgdr.correlation(da, dummy_timeseries)

    def test_correlation_wrong_target_dims(self, dummy_dataarray):
        ts = dummy_dataarray.rename({"lat": "latitude"})
        with pytest.raises(AssertionError):
            rgdr.correlation(dummy_dataarray, ts)

    def test_partial_correlation(self):
        with pytest.raises(NotImplementedError):
            rgdr.partial_correlation("field", "target", "z")

    def test_regression(self):
        with pytest.raises(NotImplementedError):
            rgdr.regression("field", "target")


class TestDBSCAN:
    """Separately validates that the DBSCAN-based clustering works as expected."""

    @pytest.fixture(autouse=True)
    def dummy_dbscan_params(self):
        return {"alpha": 0.05, "eps": 600, "min_area": 1000**2}

    def test_dbscan(
        self, example_corr, example_field, dummy_dbscan_params, expected_labels
    ):
        corr, p_val = example_corr

        clusters = rgdr.masked_spherical_dbscan(
            example_field, corr, p_val, dummy_dbscan_params
        )

        np.testing.assert_array_equal(clusters["cluster_labels"], expected_labels)

    def test_dbscan_min_area(
        self, example_corr, example_field, dummy_dbscan_params, expected_labels
    ):
        corr, p_val = example_corr
        dbscan_params = dummy_dbscan_params
        dbscan_params["min_area"] = 3000**2
        clusters = rgdr.masked_spherical_dbscan(
            example_field, corr, p_val, dbscan_params
        )
        expected_labels[expected_labels == -1] = 0  # Small -1 cluster is missing

        np.testing.assert_array_equal(clusters["cluster_labels"], expected_labels)


class TestRGDR:
    """Test RGDR and its methods."""

    @pytest.fixture(autouse=True)
    def dummy_rgdr(self, example_target):
        return RGDR(example_target, min_area_km2=1000**2)

    def test_init(self, example_target):
        rgdr = RGDR(example_target, min_area_km2=1000**2)
        assert isinstance(rgdr, RGDR)

    def test_transform_before_fit(self, dummy_rgdr, example_field):
        "Should fail as RGDR first has to be fit to (training) data."
        with pytest.raises(ValueError):
            dummy_rgdr.transform(example_field)

    def test_fit(self, dummy_rgdr, example_field):
        clustered_data = dummy_rgdr.fit(example_field)
        cluster_labels = np.array([-2.0, -1.0, 0.0, 1.0])
        np.testing.assert_array_equal(clustered_data["cluster_labels"], cluster_labels)

    def test_transform(self, dummy_rgdr, example_field):
        clustered_data = dummy_rgdr.fit(example_field)
        clustered_data = dummy_rgdr.transform(example_field)
        cluster_labels = np.array([-2.0, -1.0, 0.0, 1.0])
        np.testing.assert_array_equal(clustered_data["cluster_labels"], cluster_labels)

    def test_corr_plot(self, dummy_rgdr, example_field):
        dummy_rgdr.plot_correlation(example_field)

    def test_corr_plot_ax(self, dummy_rgdr, example_field):
        _, (ax1, ax2) = plt.subplots(ncols=2)
        dummy_rgdr.plot_correlation(example_field, ax1=ax1, ax2=ax2)

    def test_cluster_plot(self, dummy_rgdr, example_field):
        dummy_rgdr.plot_clusters(example_field)

    def test_cluster_plot_ax(self, dummy_rgdr, example_field):
        _, ax = plt.subplots()
        dummy_rgdr.plot_clusters(example_field, ax=ax)
