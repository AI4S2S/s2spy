"""Tests for the s2s._RGDR.rgdr module."""
import pytest
from s2spy import RGDR


# pylint: disable=protected-access


class TestRGDR:
    """Test RGDR methods."""
    @pytest.fixture(autouse=True)
    def dummy_rgdr(self):
        rgdr = RGDR('timeseries')
        return rgdr

    def test_init(self):
        rgdr = RGDR('timeseries')
        assert isinstance(rgdr, RGDR)

    def test_lag_shifting(self, dummy_rgdr):
        with pytest.raises(NotImplementedError):
            dummy_rgdr.lag_shifting()

    def test_fit(self, dummy_rgdr):
        with pytest.raises(NotImplementedError):
            dummy_rgdr.fit("data")

    
