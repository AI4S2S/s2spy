import pandas as pd
import xarray as xr

PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)

def check_input_data(data) -> None:
    """Utility function to check input data.

    Checks if:
        Input data is pd.Dataframe/pd.Series/xr.Dataset/xr.DataArray.
        Input data has a time index (pd), or a dim named `time` containing datetime
            values
    """
    if not isinstance(data, PandasData + XArrayData):
        raise ValueError("The input data is neither a pandas or xarray object")
    if isinstance(data, PandasData):
        check_time_dim_pandas(data)
    elif isinstance(data, XArrayData):
        check_time_dim_xarray(data)


def check_time_dim_xarray(data) -> None:
    """Utility function to check if xarray data has a time dimensions with time data."""
    if "time" not in data.dims:
        raise ValueError(
            "The input DataArray/Dataset does not contain a `time` dimension"
        )
    if not xr.core.common.is_np_datetime_like(data["time"].dtype):
        raise ValueError("The `time` dimension is not of a datetime format")


def check_time_dim_pandas(data) -> None:
    """Utility function to check if pandas data has an index with time data."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The input data does not have a datetime index.")
