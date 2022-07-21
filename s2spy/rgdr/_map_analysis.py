"""Spatial-temporal data analysis utils.

A toolbox for spatial-temporal data analysis, including regression,
correlation, auto-correlation and relevant utilities functions.
"""
from typing import Union
import numpy as np
import xarray as xr
from scipy.stats import pearsonr as _pearsonr


def _pearsonr_nan(x: np.ndarray, y: np.ndarray):
    """NaN friendly implementation of scipy.stats.pearsonr. Calculates the correlation
    coefficient between two arrays, as well as the p-value of this correlation.

    Args:
        x: 1-D array
        y: 1-D array
    Returns:
        r_coefficient
        p_value

    """
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return np.nan, np.nan
    return _pearsonr(x, y)


def correlation(
    field: xr.DataArray, target: Union[xr.DataArray, np.ndarray], time_dim: str = "time"
):
    """Calculate correlation maps.

    Args:
        field: Spatial data with a time dimension named `time_dim`, for which each
            location should have the Pearson correlation coefficient calculated with the
            target timeseries.
        target: Timeseries which has to be correlated with the spatial data. If it is
            a DataArray, it requires a dimension named `time_dim`

    Returns:
        r_coefficient: DataArray filled with the correlation coefficient for each
            non-time coordinate.
        p_value: DataArray filled with the two-tailed p-values for each computed
            correlation coefficient.
    """
    if not is_1d(target):
        raise ValueError("Target timeseries should be 1-dimensional")
    if isinstance(target, xr.DataArray) and (time_dim not in target.dims):
        raise ValueError(
            f"input target does not have contain the '{time_dim}' dimension"
        )
    if time_dim not in field.dims:
        raise ValueError(
            f"input field does not have contain the '{time_dim}' dimension"
        )

    return xr.apply_ufunc(
        _pearsonr_nan,
        field,
        target,
        input_core_dims=[[time_dim], [time_dim]],
        vectorize=True,
        output_core_dims=[[], []],
    )


def is_1d(timeseries: Union[xr.DataArray, np.ndarray]):
    if isinstance(timeseries, xr.DataArray) and timeseries.ndim > 1:
        return False
    elif isinstance(timeseries, np.ndarray) and len(timeseries.shape) > 1:
        return False
    return True


def partial_correlation(field, target, z):
    """Calculate partial correlation maps."""
    raise NotImplementedError


def regression(field, target):
    """Regression analysis on entire maps.

    Methods include Linear, Ridge, Lasso.
    """
    raise NotImplementedError


def save_map():
    """Save calculated coefficients.

    Store calculated coefficients and significance values, and
    save them as netcdf files.
    """
    raise NotImplementedError


def load_map():
    """Load coefficients from saved netcdf maps."""
    raise NotImplementedError
