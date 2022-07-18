"""AI4S2S train/test splitting methods.

A collection of train/test splitting approaches for cross-validation.
"""
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import KFold as _KFold


def kfold(data: Union[pd.DataFrame, xr.Dataset], n_splits: int = 2, shuffle=False,
          **kwargs):
    """
    K-Folds cross-validator, which splits provided intervals into k
    consecutive folds.

    For more information

    args:
        data: DataFrame to store the train/test groups. It must have a column named "anchor_years".
        n_splits: number of train/test splits.

    Other keyword arguments: see the documentation for sklearn.model_selection.KFold:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Returns:
        Pandas DataFrame with columns containing train/test label in each fold.
    """
    cv = _KFold(n_splits, shuffle=shuffle, **kwargs)

    if isinstance(data, pd.DataFrame):
        folds = cv.split(data)

        # Map folds to a new dataframe
        output = pd.DataFrame(index=data.index)
        for i, (train_index, test_index) in enumerate(folds):
            col_name = f"fold_{i}"
            output[col_name] = "skip"
            output[col_name].iloc[train_index] = "train"
            output[col_name].iloc[test_index] = "test"

        return output

    folds = cv.split(data.anchor_year)

    for i, (train_indices, test_indices) in enumerate(folds):
        fold_name = f"fold_{i}"
        labels = np.empty(data.anchor_year.size, dtype='<U6')
        labels[:] = "skip"
        labels[train_indices] = "train"
        labels[test_indices] = "test"
        data[fold_name] = ('anchor_year', labels)
        data = data.set_coords(fold_name)

    return data


def random_strat():
    """random_strat cross-validator."""
    raise NotImplementedError


def timeseries_split():
    """timeseries_split cross-validator."""
    raise NotImplementedError


ALL_METHODS = {
    "kfold": kfold,
    "random_strat": random_strat,
    "timeseries_split": timeseries_split,
    # "leave_n_out": leave_n_out,
    # "random": random,
    # "repeated_kfold": repeated_kfold,
    # "split": split,
}
