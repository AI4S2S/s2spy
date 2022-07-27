"""Spatial-temporal data analysis utils.

A toolbox for spatial-temporal data analysis, including regression,
correlation, auto-correlation and relevant utilities functions.
"""
import numpy as np
import xarray as xr
from scipy.stats import pearsonr as _pearsonr


def _pearsonr_nan(x: np.ndarray, y: np.ndarray):
    """NaN friendly implementation of scipy.stats.pearsonr. Calculates the correlation
    coefficient between two arrays, as well as the p-value of this correlation. However,
    instead of raising an error when encountering NaN values, this function will return
    both the correlation coefficient and the p-value as NaN.

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
    field: xr.DataArray, target: xr.DataArray, corr_dim: str = "time"
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate correlation maps.

    Args:
        field: Spatial data with a dimension named `corr_dim`, over which each
            location should have the Pearson correlation coefficient calculated with the
            target data.
        target: Data which has to be correlated with the spatial data. Requires a
            dimension named `corr_dim`.
        corr_dim: Dimension over which the correlation coefficient should be calculated.

    Returns:
        r_coefficient: DataArray filled with the correlation coefficient for each
            non-`corr_dim` coordinate.
        p_value: DataArray filled with the two-tailed p-values for each computed
            correlation coefficient.
    """
    assert (
        corr_dim in target.dims
    ), f"input target does not have contain the '{corr_dim}' dimension"
    assert (
        corr_dim in field.dims
    ), f"input field does not have contain the '{corr_dim}' dimension"
    assert np.all(
        [dim in field.dims for dim in target.dims]
    ), "Field and target dims do not match"

    return xr.apply_ufunc(
        _pearsonr_nan,
        field,
        target,
        input_core_dims=[[corr_dim], [corr_dim]],
        vectorize=True,
        output_core_dims=[[], []],
    )


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
