"""Tests for the s2s._RGDR.map_regions module."""
import numpy as np
import pytest
from s2spy.rgdr import _map_analysis
from s2spy.rgdr import _map_regions
from s2spy.time import AdventCalendar
import xarray as xr


test_file_path = './tests/test_rgdr/test_data'


class TestMapRegions:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        return AdventCalendar(anchor_date=(8, 31), freq="30d")

    @pytest.fixture(autouse=True)
    def example_target(self):
        return xr.open_dataset(f'{test_file_path}/tf5_nc5_dendo_80d77.nc').sel(cluster=3)

    @pytest.fixture(autouse=True)
    def example_field(self):
        return xr.open_dataset(
            f'{test_file_path}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc')

    @pytest.fixture(autouse=True)
    def example_corr(self, dummy_calendar, example_target, example_field):
        field = dummy_calendar.resample(example_field)
        target = dummy_calendar.resample(example_target)

        field['corr'], field['p_val'] = _map_analysis.correlation(
            field.sst,
            target.ts.sel(i_interval=0),
            corr_dim='anchor_year'
        )

        return field

    @pytest.fixture(autouse=True)
    def expected_labels(self):
        return np.array([
                [ 0.,  0.,  0.,  0., -1., -1.,  0., -2., -2., -2.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -2., -2., -2.,  0.,  0.],
                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -2., -2., -2.,  0.,  0.],
                [ 0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  0.],
                [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  0., -2., -2., -2., -2., -2.]
                ])

    def test_dbscan(self, example_corr, expected_labels):
        clusters = _map_regions.dbscan(example_corr.sel(i_interval=1))

        np.testing.assert_array_equal(clusters["cluster_labels"], expected_labels)

    def test_dbscan_min_area(self, example_corr, expected_labels):
        clusters = _map_regions.dbscan(example_corr.sel(i_interval=1),
                                       min_area_km2=3000**2)
        expected_labels[expected_labels == -1] = 0 # Small -1 cluster is missing

        np.testing.assert_array_equal(clusters["cluster_labels"], expected_labels)

    def test_cluster(self, example_corr):
        clustered_data = _map_regions.cluster(example_corr.sel(i_interval=1),
                                              eps_km=600, alpha=0.04)

        np.testing.assert_array_equal(clustered_data["cluster_labels"],
                                      [-1, 0, 1])

        assert set(clustered_data.sst.dims) == {'anchor_year', 'cluster_labels'}
