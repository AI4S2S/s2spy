"""Tests for the s2s.RGDR.map_analysis module.
"""
import pytest
from s2spy._RGDR import map_analysis


class TestMapAnalysis:
    def test_correlation(self):
        with pytest.raises(NotImplementedError):
            map_analysis.correlation('field', 'target')
