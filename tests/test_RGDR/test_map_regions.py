"""Tests for the s2s._RGDR.map_regions module."""
import pytest
from s2spy._RGDR import map_regions


class TestMapRegions:
    def test_spatial_mean(self):
        with pytest.raises(NotImplementedError):
            map_regions.spatial_mean()

    def test_cluster_dbscan(self):
        with pytest.raises(NotImplementedError):
            map_regions.cluster_dbscan()
