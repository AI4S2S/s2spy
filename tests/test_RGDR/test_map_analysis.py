"""Tests for the s2s._RGDR.map_analysis module."""
import pytest
from s2spy._RGDR import map_analysis


class TestMapAnalysis:
    def test_correlation(self):
        with pytest.raises(NotImplementedError):
            map_analysis.correlation('field', 'target')

    def test_partial_correlation(self):
        with pytest.raises(NotImplementedError):
            map_analysis.partial_correlation('field', 'target', 'z')

    def test_regression(self):
        with pytest.raises(NotImplementedError):
            map_analysis.regression('field', 'target')

    def test_save_map(self):
        with pytest.raises(NotImplementedError):
            map_analysis.save_map()

    def test_load_map(self):
        with pytest.raises(NotImplementedError):
            map_analysis.load_map()
