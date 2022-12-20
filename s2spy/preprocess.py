from typing import Union
import scipy.stats
import xarray as xr


def _linregress(x, y):
    slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
    return slope, intercept


def trend_linear(data: Union[xr.DataArray, xr.Dataset]) -> dict:
    slope, intercept = xr.apply_ufunc(
        _linregress,
        data["time"].astype(float),
        data,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],
        vectorize=True,
    )
    return {"slope": slope, "intercept": intercept}


def apply_linear_trend(data: Union[xr.DataArray, xr.Dataset], trend: dict):
    return data - trend["intercept"] - trend["slope"] * (data["time"].astype(float))


def get_trend(data: Union[xr.DataArray, xr.Dataset], method: str):
    if method == "linear":
        return trend_linear(data)
    raise ValueError(f"Unkown detrending method '{method}'")


def apply_trend(data: Union[xr.DataArray, xr.Dataset], method: str, trend: dict):
    if method == "linear":
        return apply_linear_trend(data, trend)
    raise NotImplementedError


def get_climatology(data: Union[xr.Dataset, xr.DataArray]):
    return data.groupby("time.dayofyear").mean("time")


def check_input_data(data: Union[xr.DataArray, xr.Dataset]):
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
    def __init__(
        self,
        rolling_window_size: int,
        rolling_min_periods: int = 1,
        detrend: Union[str, None] = "linear",
        remove_climatology: bool = True,
    ):
        self._window_size = rolling_window_size
        self._min_periods = rolling_min_periods
        self._detrend = detrend
        self._remove_climatology = remove_climatology

        self._climatology: Union[xr.DataArray, xr.Dataset]
        self._trend: dict
        self._is_fit = False

    def fit(self, data: Union[xr.DataArray, xr.Dataset]) -> None:
        check_input_data(data)

        data_rolling = data.rolling(
            dim={"time": self._window_size}, min_periods=self._min_periods, center=True
        ).mean()
        # TODO: give option to be a gaussian-like window, instead of a block.

        if self._detrend is not None:
            self._trend = get_trend(data_rolling, self._detrend)

        if self._remove_climatology:
            self._climatology = get_climatology(
                (
                    apply_trend(data_rolling, self._detrend, self._trend)
                    if self._detrend is not None
                    else data_rolling
                )
            )
        self._is_fit = True

    def transform(
        self, data: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        if not self._is_fit:
            raise ValueError("The preprocessor has to be fit to data before a transform"
                             " can be applied")

        if self._detrend is not None:
            d = apply_trend(data, self._detrend, self.trend)
        else:
            d = data

        if self._remove_climatology:
            return d.groupby("time.dayofyear") - self.climatology
        return d

    def fit_transform(
        self, data: Union[xr.DataArray, xr.Dataset]
    ) -> Union[xr.DataArray, xr.Dataset]:
        self.fit(data)
        return self.transform(data)

    @property
    def trend(self) -> dict:
        if not self._is_fit:
            raise ValueError("The preprocessor has to be fit to data before the trend"
                             " can be requested.")
        if not self._detrend:
            raise ValueError("Detrending is set to `None`, so no trend is available")
        return self._trend

    @property
    def climatology(self) -> Union[xr.DataArray, xr.Dataset]:
        if not self._is_fit:
            raise ValueError("The preprocessor has to be fit to data before the"
                             " climatology can be requested.")
        if not self._remove_climatology:
            raise ValueError("`remove_climatology is set to `False`, so no climatology "
                             "data is available")
        return self._climatology