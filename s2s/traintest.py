"""AI4S2S train/test splitting methods.

A collection of train/test splitting approaches.
"""
#from sklearn.model_selection import KFold
import pandas as pd


def kfold(intervals_df: pd.DataFrame, **kwargs):
  """
  K-Folds cross-validator, which splits provided intervals into k
  consecutive folds.
  """
  return intervals_df

def random_strat(intervals_df: pd.DataFrame):
  raise NotImplementedError

def timeseries_split(intervals_df: pd.DataFrame):
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
