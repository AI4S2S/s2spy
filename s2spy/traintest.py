"""s2spy train/test splitting methods.

A collection of train/test splitting approaches for cross-validation.
"""
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import KFold


def fold_by_anchor(cv, data):
    """Splits calendar resampled data into train/test groups, based on the anchor year.
    As splitter, a Splitter Class such as sklearn's KFold can be passed.

    For an overview of the sklearn Splitter Classes see:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

    Args:
        cv (Splitter Class): Initialized splitter class, much have a `fit(X)` method
            which splits up `X` into multiple folds of train/test data.
        data (xr.Dataset): Dataset with `anchor_year` as dimension. E.g. data resampled
            using `s2spy.time.AdventCalendar`'s `resample` method.

    Returns:
        xr.Dataset: The input dataset with extra coordinates added for each fold,
            containing the labels 'train', 'test', and possibly 'skip'.
    """

    if isinstance(data, xr.Dataset):
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


def kfold(df: pd.DataFrame, n_splits: int = 2, shuffle=False, **kwargs):
    """
    K-Folds cross-validator, which splits provided intervals into k
    consecutive folds.

    For more information

    args:
        df: DataFrame to store the train/test groups. It must have a column named "anchor_years".
        n_splits: number of train/test splits.

    Other keyword arguments: see the documentation for sklearn.model_selection.KFold:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Returns:
        Pandas DataFrame with columns containing train/test label in each fold.
    """
    cv = KFold(n_splits, shuffle=shuffle, **kwargs)
    folds = cv.split(df)

    # Map folds to a new dataframe
    output = pd.DataFrame(index=df.index)
    for i, (train_index, test_index) in enumerate(folds):
        col_name = f"fold_{i}"
        output[col_name] = "skip"
        output[col_name].iloc[train_index] = "train"
        output[col_name].iloc[test_index] = "test"

    return output


def random_strat():
    """random_strat cross-validator."""
    raise NotImplementedError


def iter_traintest(traintest_group, data, dim_function = None):
    """Iterators for train/test data and executor of dim reudction.

    This iterator will loop through the training data based on the given
    train/test split group and pass the data through the pre-defined
    dimensionality reduction function.
    """
    # To do: check if dimensionality reduction function is given.
    # To do: loop through training data (data) and call dimensionality
    #  reduction function (dim_function) for each train data
    # To do: extract training data from resampled dataset based on
    #  the train/test split group (traintest_group)
    # To do: concatenate output from each iteration.
    raise NotImplementedError


ALL_METHODS = {
    "kfold": kfold,
    "random_strat": random_strat,
    # "leave_n_out": leave_n_out,
    # "random": random,
    # "repeated_kfold": repeated_kfold,
    # "split": split,
}
