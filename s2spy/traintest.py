"""s2spy train/test splitting methods.

A collection of train/test splitting approaches for cross-validation.
"""
from typing import Optional
from typing import Type
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
from .time import AdventCalendar, MonthlyCalendar
from ._resample import resample_bins_constructor
from sklearn.model_selection._split import BaseCrossValidator


class traintest_splits:
    """ Train-test splitter build upon s2spy.time.calender"""

    def __init__(
        self, 
        splitter: Type[BaseCrossValidator], 
        calendar: Type[Union[AdventCalendar, MonthlyCalendar]],
        key: Optional[str] = "anchor_year"
    ) -> None:
        """Train-test splitter.
        
        Splits calendar DataFrame into train/test groups, based on the input key.
        As splitter, a Splitter Class such as sklearn's KFold can be passed.

        For an overview of the sklearn Splitter Classes see:
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

        Similarly to sklearn's GroupKFold, `split_groups` makes sure to split the data into
        non-overlapping groups. 

        Args:
            splitter (SplitterClass): Initialized splitter class, much have a `fit(X)` method
                which splits up `X` into multiple folds of train/test data.
            calendar (s2spy.time._base_calendar.BaseCalendar): instance of s2spy Calendar type 
                for which calendar.map_years() or calendar.map_to_data(data) has been executed. 
            key (str): Key by which to group the data. Defaults to 'anchor_year'

        Returns:
            Initialized traintest_splits
        """
        
        self.splitter = splitter
        df = resample_bins_constructor(calendar.get_intervals())
        self.traintest_df = _split_dataframe(splitter, df, key)
        
        # tranform to xr.DataArray
        splits = [key for key in self.traintest_df.keys() if 'split' in key]
        anchor_years = np.unique(self.traintest_df['anchor_year'].values)
        n_intervals = self.traintest_df['i_interval'].max()+1
        data = self.traintest_df[splits].values[::n_intervals].swapaxes(0,1)
        self.traintest_xr = xr.DataArray(data=data, 
                             dims=['split', 'anchor_year'], 
                             coords=[range(len(splits)), anchor_years])
        
    def split_iterate(self, *data_args, y=None):
        """Train/test iterator.

        Iterator of train/test splits from given data.

        Args:
            data (xr.Dataset or pd.DataFrame): Dataset with `split` as dimension or
                DataFrame with `split_no.` as column. E.g. data resampled using
                `s2spy.time.AdventCalendar`'s `resample` method will have the `key`
                'anchor_year'.
        return:
            Generator contains train data and test data seperately based on splits.
        """
        
        splits = [key for key in self.traintest_df.keys() if 'split' in key]
        if not splits:
            raise ValueError("Input data must contain train/test splits."
                             "Initialize traintest_splits with a splitter and calendar.")        
        
        # check the input data type
        train_Xs = [] ; test_Xs = []
        for data in data_args:
            for i, split in enumerate(splits):
                if isinstance(data, xr.Dataset):
                    train_X = data.sel(anchor_year = self.traintest_xr.sel(split=i) == "train")
                    test_X = data.sel(anchor_year = self.traintest_xr.sel(split=i) == "test")
                elif isinstance(data, pd.DataFrame):
                    train_X = data.loc[self.traintest_df[f'{split}'] == "train"]
                    test_X = data.loc[self.traintest_df[f'{split}'] == "test"]
                else:
                    raise ValueError("Input data_args should be (list of) type xr.Dataset or pd.DataFrame")
            train_Xs.append(train_X)
            test_Xs.append(test_X)

        # if user does not pas list of data_args, it will expect return of input dtype
        if len(train_Xs) == 1:
            train_Xs = train_Xs[0]
            test_Xs = test_Xs[0]

        # check the input y type
        if y is None:
            yield train_Xs, test_Xs
        else:
            for i, split in enumerate(splits):
                if isinstance(y, xr.Dataset):
                    train_y = y.sel(anchor_year = self.traintest_xr.sel(split=i) == "train")
                    test_y = y.sel(anchor_year = self.traintest_xr.sel(split=i) == "test")
                elif isinstance(y, pd.DataFrame):
                    train_y = y.loc[self.traintest_df[f'{split}'] == "train"]
                    test_y = y.loc[self.traintest_df[f'{split}'] == "test"]
                else:
                    raise ValueError("Input y should be of type xr.Dataset or pd.DataFrame")

            yield train_Xs, train_y, test_Xs, test_y
        
        
# def split_groups(
#     splitter: Type[BaseCrossValidator],
#     data: Union[xr.Dataset, pd.DataFrame],
#     key: Optional[str] = "anchor_year",
# ):
#     """Splits calendar resampled data into train/test groups, based on the input key.
#     As splitter, a Splitter Class such as sklearn's KFold can be passed.

#     For an overview of the sklearn Splitter Classes see:
#     https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

#     Similarly to sklearn's GroupKFold, `split_groups` makes sure to split the data into
#     non-overlapping groups. However, `split_groups` is designed to work directly with
#     panda's DataFrames and xarray's Datasets, and determines the groups based on an input
#     key.

#     Args:
#         splitter (SplitterClass): Initialized splitter class, much have a `fit(X)` method
#             which splits up `X` into multiple folds of train/test data.
#         data (xr.Dataset or pd.DataFrame): Dataset with `key` as dimension or
#             DataFrame with `key` as column. E.g. data resampled using
#             `s2spy.time.AdventCalendar`'s `resample` method will have the `key`
#             'anchor_year'.
#         key (str): Key by which to group the data. Defaults to 'anchor_year'

#     Returns:
#         The input dataset with extra coordinates or columns added for each fold,
#             containing the labels 'train', 'test', and possibly 'skip'.
#     """
#     if isinstance(data, xr.Dataset):
#         return _split_dataset(splitter, data, key)
#     if isinstance(data, pd.DataFrame):
#         return _split_dataframe(splitter, data, key)
#     raise ValueError("Input data should be of type xr.Dataset or pd.DataFrame")


# def _split_dataset(splitter, data, key):
#     """Splitter implementation for xarray's Dataset."""
#     if key not in data.dims:
#         raise ValueError(f"'{key}' is not a dimension in the Dataset.")
#     if data[key].size <= 1:
#         raise ValueError(f"Input data must have more than 1 unique {key} to split data.")

#     data = data.copy()  # otherwise the dataset is modified inplace

#     splits = splitter.split(data[key])

#     split_data = np.empty(
#         (splitter.get_n_splits(data[key].size), data[key].size), dtype="<U6"
#     )

#     for i, (train_indices, test_indices) in enumerate(splits):
#         split_data[i, :] = "skip"
#         split_data[i, train_indices] = "train"
#         split_data[i, test_indices] = "test"

#     data.expand_dims("split")
#     data["split"] = np.arange(split_data.shape[0])
#     data["traintest"] = (["split", key], split_data)
#     data = data.set_coords("traintest")

#     return data


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


    
