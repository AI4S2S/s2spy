"""s2spy train/test splitting methods.

A collection of train/test splitting approaches for cross-validation.
"""
from typing import Optional
from typing import Type
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection._split import BaseCrossValidator


def split_groups(
    splitter: Type[BaseCrossValidator],
    data: Union[xr.Dataset, pd.DataFrame],
    key: Optional[str] = "anchor_year",
):
    """Splits calendar resampled data into train/test groups, based on the input key.
    As splitter, a Splitter Class such as sklearn's KFold can be passed.

    For an overview of the sklearn Splitter Classes see:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

    Similarly to sklearn's GroupKFold, `split_groups` makes sure to split the data into
    non-overlapping groups. However, `split_groups` is designed to work directly with
    panda's DataFrames and xarray's Datasets, and determines the groups based on an input
    key.

    Args:
        splitter (SplitterClass): Initialized splitter class, much have a `fit(X)` method
            which splits up `X` into multiple folds of train/test data.
        data (xr.Dataset or pd.DataFrame): Dataset with `key` as dimension or
            DataFrame with `key` as column. E.g. data resampled using
            `s2spy.time.AdventCalendar`'s `resample` method will have the `key`
            'anchor_year'.
        key (str): Key by which to group the data. Defaults to 'anchor_year'

    Returns:
        The input dataset with extra coordinates or columns added for each fold,
            containing the labels 'train', 'test', and possibly 'skip'.
    """
    if isinstance(data, xr.Dataset):
        return _split_dataset(splitter, data, key)
    if isinstance(data, pd.DataFrame):
        return _split_dataframe(splitter, data, key)
    raise ValueError("Input data should be of type xr.Dataset or pd.DataFrame")


def _split_dataset(splitter, data, key):
    """Splitter implementation for xarray's Dataset."""
    if key not in data.dims:
        raise ValueError(f"'{key}' is not a dimension in the Dataset.")
    if data[key].size <= 1:
        raise ValueError(f"Input data must have more than 1 unique {key} to split data.")

    data = data.copy()  # otherwise the dataset is modified inplace

    splits = splitter.split(data[key])

    split_data = np.empty(
        (splitter.get_n_splits(data[key].size), data[key].size), dtype="<U6"
    )

    for i, (train_indices, test_indices) in enumerate(splits):
        split_data[i, :] = "skip"
        split_data[i, train_indices] = "train"
        split_data[i, test_indices] = "test"

    if 'split' not in data.dims:
        data.expand_dims("split")
        
    if 'traintest' in data.coords:
        identical = np.char.equal(data['traintest'].values, split_data).all()
        if not identical:
            print('Traintest split was already present in data and unequal to '
                  'newly generated split, traintest split has been updated.')
        
    data["split"] = np.arange(split_data.shape[0])
    data["traintest"] = (["split", key], split_data)
    data = data.set_coords("traintest")

    return data


def _split_dataframe(splitter, data, key):
    """Splitter implementation for Pandas DataFrame."""
    if key not in data.keys():
        raise ValueError(f"'{key}' is not a key in the input DataFrame.")

    group_labels = np.unique(data[key])
    if group_labels.size <= 1:
        raise ValueError(f"Input data must have more than 1 unique {key} to split data.")

    splits = splitter.split(group_labels)

    # label every row in the dataframe
    for i, (train_indices, test_indices) in enumerate(splits):
        # create column split_# filled with "skip" in dataframe
        col_name = f"split_{i}"
        data[col_name] = ""

        # create dictionary with key and train/test labels
        indices = np.empty(group_labels.size, dtype="<U6")
        indices[:] = "skip"
        indices[train_indices] = "train"
        indices[test_indices] = "test"
        train_test_dict = dict(zip(group_labels, indices))

        # map key row values with train/test
        data[col_name] = data[key].map(train_test_dict)

    return data
