"""AI4S2S train/test splitting methods.

A collection of train/test splitting approaches used for cross-validation.
"""
import pandas as pd
from sklearn.model_selection import KFold


def kfold(df: pd.DataFrame, n_splits: int = 2, **kwargs):
    """
    K-Folds cross-validator, which splits provided intervals into k
    consecutive folds.

    For more information

    args:
        intervals_df: DataFrame to store the train/test groups.
        n_splits: number of train/test splits.

    Other keyword arguments: see the documentation for sklearn.model_selection.KFold:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Returns:
        Pandas DataFrame with columns containing train/test label in each fold.
    """
    cv = KFold(n_splits, **kwargs)
    folds = cv.split(df)

    # Map folds to a new dataframe
    output = pd.DataFrame(index=df.index)
    for i, (train_index, test_index) in enumerate(folds):
        col_name = f'fold_{i}'
        output[col_name] = 'skip'
        output[col_name].iloc[train_index] = 'train'
        output[col_name].iloc[test_index] = 'test'

    return output


def random_strat():
    raise NotImplementedError


def timeseries_split():
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
