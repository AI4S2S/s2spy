"""s2spy train/test splitting methods.

A collection of train/test splitting approaches for cross-validation.
"""
import pandas as pd
from sklearn.model_selection import KFold


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


def timeseries_split():
    """timeseries_split cross-validator."""
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
    "timeseries_split": timeseries_split,
    # "leave_n_out": leave_n_out,
    # "random": random,
    # "repeated_kfold": repeated_kfold,
    # "split": split,
}
