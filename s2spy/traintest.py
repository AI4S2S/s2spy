"""s2spy train/test splitting methods.

A collection of train/test splitting approaches for cross-validation.
"""
import numpy as np
import pandas as pd
from typing import Union
import xarray as xr


def fold_by_anchor(cv, data: Union[xr.Dataset, pd.DataFrame]):
    """Splits calendar resampled data into train/test groups, based on the anchor year.
    As splitter, a Splitter Class such as sklearn's KFold can be passed.

    For an overview of the sklearn Splitter Classes see:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

    Args:
        cv (Splitter Class): Initialized splitter class, much have a `fit(X)` method
            which splits up `X` into multiple folds of train/test data.
        data (xr.Dataset or pd.DataFrame): Dataset with `anchor_year` as dimension or
            DataFrame with `anchor_year` as column. E.g. data resampled using 
            `s2spy.time.AdventCalendar`'s `resample` method.
            
    Returns:
        xr.Dataset: The input dataset with extra coordinates added for each fold,
            containing the labels 'train', 'test', and possibly 'skip'.
        pd.DataFrame: The input dataframe with extra columns added for each fold,
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

    elif isinstance(data, pd.DataFrame):
        # split the anchor years
        anchor_years = np.unique(data['anchor_year'])
        folds = cv.split(anchor_years)

        # label every row in the dataframe
        for i, (train_indices, test_indices) in enumerate(folds):
            # create column fold_# filled with "skip" in dataframe
            col_name = f"fold_{i}"
            data[col_name] = ""

            # create dictionary with anchor_years and train/test
            indices = np.empty(anchor_years.size, dtype='<U6')
            indices[:] = "skip"
            indices[train_indices] = "train"
            indices[test_indices] = "test"
            train_test_dict = dict(zip(anchor_years, indices))

            # map anchor_year row values with train/test 
            data[col_name] = data['anchor_year'].map(train_test_dict)

    else:
        raise ValueError('Invalid input data')

    return data

def leave_n_out():
    '''cross validator that leaves out n anchor years, instead of cv 
    defined on number of splits
    '''
    # yet to be implemented
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
