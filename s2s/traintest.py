"""AI4S2S train/test splitting methods.

A collection of train/test splitting approaches used for cross-validation.
"""
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def kfold(intervals_df: pd.DataFrame, n_splits: int=2, **kwargs):
    """
    K-Folds cross-validator, which splits provided intervals into k
    consecutive folds.

    For more information

    args:
        intervals_df: DataFrame to store the train/test groups.
        n_splits: number of train/test splits.
        # To do
        Gap_prior: skip n groups/years prior to every test group/year
        Gap_after: skip n groups/years after every test group/year

    Other keyword arguments: see the documentation for sklearn.model_selection.KFold:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Returns:
        Pandas DataFrame with columns containing train/test label in each fold.
    """
    # instantiate kfold cross-validator
    cv = KFold(n_splits, **kwargs)
    anchor_years = np.unique(intervals_df.anchor_year.values)
    folds = cv.split(anchor_years)

    for i, (train_index, test_index) in enumerate(folds):
        # get train year index and test year index
        train_years = [anchor_years[index] for index in train_index]
        test_years = [anchor_years[index] for index in test_index]
        # label train/test group
        col_name = 'fold_' + str(i)
        intervals_df[col_name] = 'skip'
        intervals_df.loc[intervals_df['anchor_year'].isin(train_years), col_name] = 'train'
        intervals_df.loc[intervals_df['anchor_year'].isin(test_years), col_name] = 'test'

    return intervals_df

def random_strat():
    raise NotImplementedError

def timeseries_split():
    raise NotImplementedError

ALL_METHODS = {
  "kfold": kfold,
  "rand_strat": random_strat,
  "timeseries_split": timeseries_split,
  # "leave_n_out": leave_n_out,
  # "random": random,
  # "repeated_kfold": repeated_kfold,
  # "split": split,
}
