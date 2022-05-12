"""Tests for the s2s.my_module module.
"""
import pytest

from s2s.time import TimeIndex


def test_timeindex_notimplemented():
    with pytest.raises(NotImplementedError):
        TimeIndex()
