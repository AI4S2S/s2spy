"""Tests for the s2s.traintest module.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
import s2spy.traintest
from s2spy.time import AdventCalendar
from s2spy.time import resample


class TestTrainTest:
    # Define all required inputs as fixtures:
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        return AdventCalendar(anchor=(10, 15), freq="180d")

    @pytest.fixture(autouse=True)
    def dummy_dataframe(self):
        time_index = pd.date_range("20181020", "20211001", freq="60d")
        test_data = np.random.random(len(time_index))
        return pd.DataFrame(test_data, index=time_index, columns=["data1"])

    @pytest.fixture(autouse=True)
    def dummy_dataset(self, dummy_dataframe):
        return dummy_dataframe.to_xarray().rename({"index": "time"})

    @pytest.fixture(autouse=True)
    def dummy_dataframe_short(self):
        time_index = pd.date_range("20191020", "20211001", freq="60d")
        test_data = np.random.random(len(time_index))
        return pd.DataFrame(test_data, index=time_index, columns=["data1"])

    @pytest.fixture(autouse=True)
    def dummy_dataset_short(self, dummy_dataframe_short):
        return dummy_dataframe_short.to_xarray().rename({"index": "time"})

    def test_kfold_df(self, dummy_calendar, dummy_dataframe):
        mapped_calendar = dummy_calendar.map_to_data(dummy_dataframe)
        df = resample(mapped_calendar, dummy_dataframe)
        df = s2spy.traintest.split_groups(KFold(n_splits=2), df)
        expected_group = ["test", "test", "train", "train"]
        assert np.array_equal(df["split_0"].values, expected_group)

    def test_kfold_ds(self, dummy_calendar, dummy_dataset):
        mapped_calendar = dummy_calendar.map_to_data(dummy_dataset)
        ds = resample(mapped_calendar, dummy_dataset)
        ds = s2spy.traintest.split_groups(KFold(n_splits=2), ds)
        expected_group = ["test", "train"]
        assert np.array_equal(ds.traintest.values[0], expected_group)

    def test_kfold_df_short(self, dummy_calendar, dummy_dataframe_short):
        "Should fail as there is only a single anchor year: no splits can be made"
        mapped_calendar = dummy_calendar.map_to_data(dummy_dataframe_short)
        df = resample(mapped_calendar, dummy_dataframe_short)
        with pytest.raises(ValueError):
            df = s2spy.traintest.split_groups(KFold(n_splits=2), df)

    def test_kfold_ds_short(self, dummy_calendar, dummy_dataset_short):
        "Should fail as there is only a single anchor year: no splits can be made"
        mapped_calendar = dummy_calendar.map_to_data(dummy_dataset_short)
        ds = resample(mapped_calendar, dummy_dataset_short)
        with pytest.raises(ValueError):
            ds = s2spy.traintest.split_groups(KFold(n_splits=2), ds)

    def test_alternative_key(self, dummy_calendar, dummy_dataset):
        mapped_calendar = dummy_calendar.map_to_data(dummy_dataset)
        ds = resample(mapped_calendar, dummy_dataset)
        ds = s2spy.traintest.split_groups(KFold(n_splits=2), ds, key="i_interval")
        assert "i_interval" in ds.traintest.dims
