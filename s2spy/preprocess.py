"""Preprocessor for s2spy workflow."""
from typing import Tuple
from typing import Union
import numpy as np
import scipy.stats
import xarray as xr


def _linregress(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Calculate the slope and intercept between two arrays using scipy's linregress.

    Used to make linregress more ufunc-friendly.

    Args:
        x: First array.
        y: Second array.

    Returns:
        slope, intercept
    """
    slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
    return slope, intercept


def _trend_linear(data: Union[xr.DataArray, xr.Dataset]) -> dict:
    """Calculate the linear trend over time.

    Args:
        data: The input data of which you want to know the trend.

    Returns:
        Dictionary containing the linear trend information (slope and intercept)
    """
    slope, intercept = xr.apply_ufunc(
        _linregress,
        data["time"].astype(float),
        data,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],
        vectorize=True,
    )
    return {"slope": slope, "intercept": intercept}


def _subtract_linear_trend(data: Union[xr.DataArray, xr.Dataset], trend: dict):
    """Subtract a previously calclulated linear trend from (new) data."""
    return data - trend["intercept"] - trend["slope"] * (data["time"].astype(float))


def _get_trend(data: Union[xr.DataArray, xr.Dataset], method: str):
    """Calculate the trend, with a certain method. Only linear is implemented."""
    if method == "linear":
        return _trend_linear(data)
    raise ValueError(f"Unkown detrending method '{method}'")


def _subtract_trend(data: Union[xr.DataArray, xr.Dataset], method: str, trend: dict):
    """Subtract the previously calculated trend from (new) data. Only linear is implemented."""
    if method == "linear":
        return _subtract_linear_trend(data, trend)
    raise NotImplementedError


def _get_climatology(data: Union[xr.Dataset, xr.DataArray]):
    """Calculate the climatology of timeseries data."""
    return data.groupby("time.dayofyear").mean("time")


def _check_input_data(data: Union[xr.DataArray, xr.Dataset]):
    """Check the input data for compatiblity with the preprocessor.

    Args:
        data: Data to validate.

    Raises:
        ValueError: If the input data is of the wrong type.
        ValueError: If the input data does not have a 'time' dimension.
    """
    if not any(isinstance(data, dtype) for dtype in (xr.DataArray, xr.Dataset)):
        raise ValueError(
            "Input data has to be an xarray-DataArray or xarray-Dataset, "
            f"not {type(data)}"
        )
    if "time" not in data.dims:
        raise ValueError(
            "Analysis is done of the 'time' dimension, but the input data"
            f" only has dims: {data.dims}"
        )


class Preprocessor:
    """Preprocessor for s2s data."""

    def __init__(
        self,
        rolling_window_size: Union[int, None],
        rolling_min_periods: int = 1,
        subtract_climatology: bool = True,
        detrend: Union[str, None] = "linear",
    ):
        """Preprocessor for s2s data. Can detrend as well as deseasonalize.

        On calling `.fit(data)`, the preprocessor will:
         - Calculate the rolling mean of the input data.
         - Calculate and store the climatology of the rolling mean.
         - Calculate and store the trend of the rolling mean.

        When calling `.transform(data)`, the preprocessor will:
         - Remove the climatology from a copy of the data.
         - Remove the (stored) trend from this deseasonalized data.
         - Return the detrended and deseasonalized data.

        Args:
            rolling_window_size: The size of the rolling window that will be applied
                before calculating the trend and climatology. Setting this to None will
                skip this step.
            rolling_min_periods: The minimum number of periods within a rolling window.
                If higher than 1 (the default), NaN values will be present at the start
                and end of the preprocessed data.
            subtract_climatology (optional): If you want to calculate and remove the
                climatology of the data. Defaults to True.
            detrend (optional): Which method to use for detrending. Currently the only method
                supported is "linear". If you want to skip detrending, set this to None.
        """
        self._window_size = rolling_window_size
        self._min_periods = rolling_min_periods
        self._detrend = detrend
        self._subtract_climatology = subtract_climatology

        self._climatology: Union[xr.DataArray, xr.Dataset]
        self._trend: dict
        self._is_fit = False

    def fit(self, data: Union[xr.DataArray, xr.Dataset]) -> None:
        """Fit this Preprocessor to input data.

        Args:
            data: Input data for fitting.
        """
        _check_input_data(data)
        if self._window_size not in [None, 1]:
            data_rolling = data.rolling(
                dim={"time": self._window_size},  # type: ignore
                min_periods=self._min_periods,
                center=True,
            ).mean()
        # TODO: give option to be a gaussian-like window, instead of a block.
        else:
            data_rolling = data

        if self._subtract_climatology:
            self._climatology = _get_climatology(data_rolling)

        if self._detrend is not None:
            self._trend = _get_trend(
                data_rolling.groupby("time.dayofyear") - self._climatology
                if self._subtract_climatology
                else data_rolling,
                self._detrend,
            )

        self._is_fit = True

    def transform(
        self, data: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Apply the preprocessing steps to the input data.

        Args:
            data: Input data to perform preprocessing.

        Returns:
            Preprocessed data.
        """
        if not self._is_fit:
            raise ValueError(
                "The preprocessor has to be fit to data before a transform"
                " can be applied"
            )

        if self._subtract_climatology:
            d = data.groupby("time.dayofyear") - self._climatology
        else:
            d = data

        if self._detrend is not None:
            return _subtract_trend(d, self._detrend, self.trend)

        return d

    def fit_transform(
        self, data: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        """Fit this Preprocessor to input data, and then apply the steps to the data.

        Args:
            data: Input data for fit and transform.

        Returns:
            Preprocessed data.
        """
        self.fit(data)
        return self.transform(data)

    @property
    def trend(self) -> dict:
        """Return the stored trend (dictionary)."""
        if not self._detrend:
            raise ValueError("Detrending is set to `None`, so no trend is available")
        if not self._is_fit:
            raise ValueError(
                "The preprocessor has to be fit to data before the trend"
                " can be requested."
            )
        return self._trend

    @property
    def climatology(self) -> Union[xr.DataArray, xr.Dataset]:
        """Return the stored climatology data."""
        if not self._subtract_climatology:
            raise ValueError(
                "`subtract_climatology is set to `False`, so no climatology "
                "data is available"
            )
        if not self._is_fit:
            raise ValueError(
                "The preprocessor has to be fit to data before the"
                " climatology can be requested."
            )
        return self._climatology
