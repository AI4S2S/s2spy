"""Tests for the s2s.my_module module.
"""
import pytest

from s2s.time import TimeIndex


class TestTimeIndex:

    def test_init(self):
        index = TimeIndex()
        # TODO fix this test
        assert index._index = Interval('2020-11-23', '2020-11-30', closed='right')
