"""Tests for the s2s.dimensionality module.
"""
import pytest
import s2spy.dimensionality


class TestDimensionality:
    def test_RGDR(self):
        with pytest.raises(NotImplementedError):
            s2spy.dimensionality.RGDR('series')
    
    def test_PCA(self):
        with pytest.raises(NotImplementedError):
            s2spy.dimensionality.PCA()

    def test_MCA(self):
        with pytest.raises(NotImplementedError):
            s2spy.dimensionality.MCA()
