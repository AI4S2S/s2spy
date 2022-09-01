"""s2spy train/test splitting methods.

Wrapper around sklearn splitters for working with (multiple) xarray dataarrays.
"""
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
import numpy as np
import xarray as xr
from sklearn.model_selection._split import BaseCrossValidator


# Mypy type aliases
XType = Union[xr.DataArray, List[xr.DataArray]]

# For output types, variables are split in 2
XOnly = Tuple[XType, XType]
XAndY = Tuple[XType, XType, xr.DataArray, xr.DataArray]
XMaybeY = Iterable[Union[XOnly, XAndY]]


class CoordinateMismatch(Exception):
    """Custom exception for unmatching coordinates"""


def _all_equal(arrays):
    """Return true if all arrays are equal"""
    try:
        arrays = iter(arrays)
        first = next(arrays)
        return all(np.array_equal(first, rest) for rest in arrays)
    except StopIteration:
        return True


class TrainTestSplit():
    """Split (multiple) xr.DataArrays across a given dimension.

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

    def split(
        self,
        *x_args: xr.DataArray,
        y: Optional[xr.DataArray] = None,
        dim: str = "anchor_year"
    ) -> XMaybeY:
        """Iterate over splits.

        Args:
            x_args: one or multiple xr.DataArray's that share the same
                coordinate along the given dimension
            y: (optional) xr.DataArray that shares the same coordinate along the
                given dimension
            dim: name of the dimension along which to split the data.

        Returns:
            Iterator over the splits
        """
        # Check that all inputs share the same dim coordinate
        coords = []
        x: xr.DataArray  # Initialize x to set scope outside loop
        for x in x_args:
            try:
                coords.append(x[dim])
            except KeyError as err:
                raise CoordinateMismatch(
                    f"Not all input data arrays have the {dim} dimension."
                ) from err

        if not _all_equal(coords):
            raise CoordinateMismatch(
                f"Input arrays are not equal along {dim} dimension."
                )

        if y is not None and not np.array_equal(y[dim], x[dim]):
            raise CoordinateMismatch(
                f"Input arrays are not equal along {dim} dimension."
            )

        if x[dim].size <=1:
            raise ValueError(
                f"Invalid input: need at least 2 values along dimension {dim}"
            )

        # Now we know that all inputs are equal.
        for (train_indices, test_indices) in self.splitter.split(x[dim]):
            if len(x_args) == 1:
                x_train: XType = x.isel({dim: train_indices})
                x_test: XType = x.isel({dim: test_indices})
            else:
                x_train = [da.isel({dim: train_indices}) for da in x_args]
                x_test = [da.isel({dim: test_indices}) for da in x_args]

            if y is None:
                yield x_train, x_test
            else:
                y_train = y.isel({dim: train_indices})
                y_test = y.isel({dim: test_indices})
                yield x_train, x_test, y_train, y_test
