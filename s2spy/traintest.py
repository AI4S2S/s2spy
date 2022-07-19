"""AI4S2S train/test splitting methods.

A collection of train/test splitting approaches for cross-validation.
"""
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit


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


def random_strat(df: pd.Dataframe, n_splits: int = 2, q_threshold: float = .66, seed: int = 1):
    raise NotImplementedError
#     """Random stratified K-folds cross-validator which splits provided intervals into k
#     consecutive folds, while preserving the percentage of samples for each fold.
    
#     Args:

#     Returns:

#     """
#     # not sure about the groups with the new adventcalendar, for now:
#     groups = df['anchor_year'].values

#     # create binary dataset on data column of df with threshold q
#     df_binary = pd.zeros(df.index.size, dtype=int)
#     df_binary[(df['data'] > df['data'].quantile(q=q_threshold)).values] = 1
#     df_binary[(df['data'] < df['data'].quantile(q=1-q_threshold)).values] = -1
#     df_binary[np.logical_and(df_binary!=1, df_binary!=-1)] = 0

#     cv = StratifiedKFold(n_splits=kfold, shuffle=True,
#                                random_state=seed)
#     test_indices = []
#     train_indices = []
#     for train_index, test_index in cv.split(
#         X=df_binary.index, y=df_binary.values
#         ):
#         test_indices.append(test_index)
#         train_indices.append(train_index)

#     label_test = np.zeros(df.index.size , dtype=int)
#     for i, test_fold in enumerate(test_indices):
#         for j, year in enumerate(groups):
#             if j in list(test_fold):
#                 label_test[j] = i

#     cv = PredefinedSplit(label_test)
#     cv.uniqgroups = test_indices
    
#     return cv


def timeseries_split(df: pd.DataFrame, n_splits: int, **kwargs):
    """
    Timeseries split validator, which splits the timeseries data at fixed time intervals. In each
    split, the test indices are higher than before. 

    Args:
        df: DataFrame to store the train/test groups. It must have a column named "anchor_years".
        n_splits: number of train/test splits.
    
    Other keyword arguments: see the documentation for sklearn.model_selection.KFold:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

    Returns:
        Pandas DataFrame with columns containing train/test label in each fold.
    
    Example:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import s2spy.time
        >>> from s2spy.traintest import timeseries_split

        >>> #generate data and initiate calendar
        >>> time_index = pd.date_range('20171020', '20211001', freq='15d')
        >>> random_data = np.random.random(len(time_index))
        >>> example_series = pd.Series(random_data, index=time_index)
        >>> example_dataframe = pd.DataFrame(example_series.rename('data1'))
        >>> calendar = s2spy.time.AdventCalendar(anchor_date=(10, 15), freq='90d')
        
        >>> #resample data
        >>> resampled_dataframe = calendar.resample(example_dataframe)

        >>> #generate splits
        >>> timeseries_split(resampled_dataframe, n_splits=2)
                fold_0	fold_1
            0	train	train
            1	train	train
            2	train	train
            3	train	train
            4	test	train
            5	test	train
            6	test	train
            7	test	train
            8	skip	test
            9	skip	test
            10	skip	test
            11	skip	test
    """
    cv = TimeSeriesSplit(n_splits, **kwargs)
    folds = cv.split(df)

    # Map folds to a new dataframe
    output = pd.DataFrame(index=df.index)
    for i, (train_index, test_index) in enumerate(folds):
        col_name = f"fold_{i}"
        output[col_name] = "skip"
        output[col_name].iloc[train_index] = "train"
        output[col_name].iloc[test_index] = "test"

    return output


ALL_METHODS = {
    "kfold": kfold,
    # "random_strat": random_strat,
    "timeseries_split": timeseries_split,
    # "leave_n_out": leave_n_out,
    # "random": random,
    # "repeated_kfold": repeated_kfold,
    # "split": split,
}
