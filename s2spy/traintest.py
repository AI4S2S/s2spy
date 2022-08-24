"""s2spy train/test splitting methods.

A collection of train/test splitting approaches for cross-validation.
"""
from typing import Iterable
from typing import Optional
from typing import Type
import numpy as np
import xarray as xr
from sklearn.model_selection._split import BaseCrossValidator


def _all_equal(arrays: Iterable[Iterable]):
    """Return true if all arrays are equal"""
    try:
        arrays = iter(arrays)
        first = next(arrays)
        return all(np.array_equal(first, rest) for rest in arrays)
    except StopIteration:
        return True


class TrainTestSplit():
    """Splitters (multiple) xr.DataArrays across a given dimension.

    Calling `split()` on this object returns an iterator that allows passing in
    multiple input arrays at once. They need to have matching coordinates along
    the given dimension.

    For an overview of the sklearn Splitter Classes see:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

    Args:
        splitter (SplitterClass): Initialized splitter class, much have a
            `fit(X)` method which splits up `X` into multiple folds of
            train/test data.
    """
    def __init__(self, splitter: Type[BaseCrossValidator]) -> None:
        self.splitter = splitter

    def split(self, *x_args: Iterable[xr.DataArray], y: Optional[xr.DataArray]=None, dim: str="anchor_year"):
        """Iterate over splits.

        Args:
            x_args: one or multiple xr.DataArray's that share the same coordinate along the given dimension
            y: (optional) xr.DataArray that shares the same coordinate along the given dimension
            dim: name of the dimension along which to split the data.

        Returns:
            Iterator over the splits
        """
        # Check that all inputs share the dim coordinate over which they will be split.
        split_dim_coords = []
        for x in x_args:
            try:
                split_dim_coords.append(x[dim])
            except KeyError as err:
                raise ValueError(
                    f"Not all input data arrays have the {dim} dimension."
                ) from err

        if not _all_equal(split_dim_coords):
            raise ValueError(
                f"Input arrays are not equal along {dim} dimension."
                )

        if y is not None and not np.array_equal(y[dim], x[dim]):
            raise ValueError(
                f"Input arrays are not equal along {dim} dimension."
            )

        # Now we know that all inputs are equal..
        for (train_indices, test_indices) in self.splitter.split(x[dim]):
            x_train = [da.isel({dim: train_indices}) for da in x_args]
            x_test = [da.isel({dim: test_indices}) for da in x_args]
            y_train = y.isel({dim: train_indices})
            y_test = y.isel({dim: test_indices})
            yield x_train, x_test, y_train, y_test


if __name__ == "__main__":
    # Just for testing
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    import s2spy.time
    import s2spy.traintest

    # Dummy data
    n = 50
    time_index = pd.date_range("20151020", periods=n, freq="60d")
    time_coord = {"time": time_index}
    x1 = xr.DataArray(np.random.randn(n), coords=time_coord, name="precursor1")
    x2 = xr.DataArray(np.random.randn(n), coords=time_coord, name="precursor2")
    y = xr.DataArray(np.random.randn(n), coords=time_coord, name="target")

    # Fit to calendar
    calendar = s2spy.time.AdventCalendar(anchor=(10, 15), freq="180d")
    calendar.map_to_data(x1)  # TODO: would be nice to pass in multiple at once.
    x1 = s2spy.time.resample(calendar, x1)
    x2 = s2spy.time.resample(calendar, x2)
    y = s2spy.time.resample(calendar, y)

    # Cross-validation
    kfold = KFold(n_splits=3)
    cv = s2spy.traintest.TrainTestSplit(kfold)
    for (x1_train, x2_train), (x1_test, x2_test), y_train, y_test in cv.split(x1, x2, y=y):
        print("Train:", x1_train.anchor_year.values)
        print("Test:", x1_test.anchor_year.values)

    # Shorthand notation
    x = [x1, x2]
    for x_train, x_test, y_train, y_test in cv.split(*x, y=y):
        x1_train, x2_train = x_train
        x1_test, x2_test = x_test
        print("Train:", x1_train.anchor_year.values)
        print("Test:", x1_test.anchor_year.values)
