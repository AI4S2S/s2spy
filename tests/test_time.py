"""Tests for the s2s.my_module module.
"""
import numpy as np
import pandas as pd
import pytest
from s2s.time import TimeIndex


class TestTimeIndex:
    def test_init(self):
        index = TimeIndex()
        expected = pd.interval_range(
            end=pd.Timestamp("2020-11-30"), periods=52, freq="7d"
        )
        # TODO make more useful tests
        np.array_equal(index._index, expected)  # noqa

    def test_str(self):
        index = TimeIndex()
        assert str(index) == str(index._index)  # noqa

    def test_discard(self):
        index = TimeIndex()

        with pytest.raises(NotImplementedError):
            index.discard(max_lag=5)

    def test_mark_target_period(self):
        index = TimeIndex()

        with pytest.raises(NotImplementedError):
            index.mark_target_period(end='20200101', periods=5)

    def test_resample_with_dataframe(self):
        index = TimeIndex()
        df = pd.DataFrame([1, 2, 3], index=pd.date_range("20200101", periods=3))

        with pytest.raises(NotImplementedError):
            index.resample(df)

    def test_resample_with_dataarray(self):
        index = TimeIndex()
        df = pd.DataFrame([1, 2, 3], index=pd.date_range("20200101", periods=3))
        da = df.to_xarray()

        with pytest.raises(NotImplementedError):
            index.resample(da)

    def test_get_lagged_indices(self):
        index = TimeIndex()

        with pytest.raises(NotImplementedError):
            index.get_lagged_indices()

    def test_get_train_indices(self):
        index = TimeIndex()

        with pytest.raises(NotImplementedError):
            index.get_train_indices("leave_n_out", {"n": 5})

    def test_get_test_indices(self):
        index = TimeIndex()

        with pytest.raises(NotImplementedError):
            index.get_test_indices("leave_n_out", {"n": 5})

    def get_train_test_indices(self):
        index = TimeIndex()

        with pytest.raises(NotImplementedError):
            index.get_train_test_indices("leave_n_out", {"n": 5})
