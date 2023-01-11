import re
import warnings
from typing import Dict
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr


PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)


def check_timeseries(data) -> None:
    """Utility function to check input data.

    Checks if:
     - Input data is pd.Dataframe/pd.Series/xr.Dataset/xr.DataArray.
     - Input data has a time index (pd), or a dim named `time` containing datetime
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
    if not xr.core.common.is_np_datetime_like(data["time"].dtype):  # type: ignore
        raise ValueError("The `time` dimension is not of a datetime format")


def check_time_dim_pandas(data) -> None:
    """Utility function to check if pandas data has an index with time data."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The input data does not have a datetime index.")


def check_empty_intervals(data: Union[pd.DataFrame, xr.Dataset]) -> None:
    """Utility to check for empty intervals in data.

    Note: For the Dataset, all values within a certain interval, anchor_year combination
    have to be NaN, to allow for, e.g., empty gridcells in a latitude/longitude grid.

    Args:
        data (Union[pd.DataFrame, xr.Dataset]): Data that should be checked for empty
            intervals. Should be done after resampling the data.

    Raises:
        UserWarning: If the data is insufficient.

    Returns:
        None
    """
    if isinstance(data, pd.DataFrame) and not np.any(np.isnan(data.iloc[:, 3:])):
        return None
    if isinstance(data, xr.Dataset):
        time_vars = [var for var in data.data_vars if "anchor_year" in data[var].dims]
        if not any(
            data[var].isnull().any(dim=["i_interval", "anchor_year"]).all()
            for var in time_vars
        ):
            return None

    warnings.warn(
        "The input data could not fully cover the calendar's intervals. "
        "Intervals without available data will contain NaN values."
    )
    return None


def check_input_frequency(calendar, data):
    """Checks the frequency of (input) data.

    Note: Pandas and xarray have the builtin function `infer_freq`, but this function is
    not robust enough for our purpose, so we have to manually infer the frequency if the
    builtin one fails.
    """
    if isinstance(data, PandasData):
        data_freq = pd.infer_freq(data.index)  # type: ignore
        if data_freq is None:  # Manually infer the frequency
            data_freq = np.min(data.index.values[1:] - data.index.values[:-1])
    else:
        data_freq = xr.infer_freq(data.time)
        if data_freq is None:  # Manually infer the frequency
            data_freq = (data.time.values[1:] - data.time.values[:-1]).min()

    if isinstance(data_freq, str):
        data_freq.replace("-", "")
        if not re.match(r"\d+\D", data_freq):
            data_freq = "1" + data_freq

    if pd.Timedelta(calendar.freq) < pd.Timedelta(data_freq):
        warnings.warn(
            """Target frequency is smaller than the original frequency.
            The resampled data will contain NaN values, as there is no data
            available within all intervals."""
        )


def convert_interval_to_bounds(data: xr.Dataset) -> xr.Dataset:
    """Converts pandas intervals to bounds in a xarray Dataset.

    pd.Interval objects cannot be written to netCDF. To allow writing the
    calendar-resampled data to netCDF these intervals have to be converted to bounds.
    This function adds a 'bounds' dimension, with 'left' and 'right' coordinates, and
    converts the 'interval' coordinates to this system.

    Args:
        data: Input data with intervals as pd.Interval objects.

    Returns:
        Input data with the intervals converted to bounds.
    """
    stacked = data.stack(coord=["anchor_year", "i_interval"])
    bounds = np.array([[val.left, val.right] for val in stacked.interval.values])
    stacked["interval"] = (("coord", "bounds"), bounds)
    return stacked.unstack("coord")


def assert_bokeh_available():
    """Util that attempts to load the optional module bokeh."""
    try:
        import bokeh as _  # pylint: disable=import-outside-toplevel

    except ImportError as e:
        raise ImportError(
            "Could not import the `bokeh` module.\nPlease install this"
            " before continuing, with either `pip` or `conda`."
        ) from e


def get_month_names() -> Dict:
    """Generates a dictionary with English lowercase month names and abbreviations.

    Returns:
        Dictionary containing the English names of the months, including their
            abbreviations, linked to the number of each month.
            E.g. {'december': 12, 'jan': 1}
    """
    return {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }


def check_month_day(month: int, day: int = 1):
    """Checks if the input day/month combination is valid.

    Months must be between 1 and 12, and days must be within 1 and 28/30/31 (depending
    on the month).

    Args:
        month: Month number
        day: Day number. Defaults to 1.
    """

    if month in {1, 3, 5, 7, 8, 10, 12}:
        if (day < 1) or (day > 31):
            raise ValueError(
                "Incorrect anchor input. "
                f"Day number {day} is not a valid day for month {month}"
            )
    elif month in {4, 6, 9, 11}:
        if (day < 1) or (day > 30):
            raise ValueError(
                "Incorrect anchor input. "
                f"Day number {day} is not a valid day for month {month}"
            )
    elif month == 2:
        if (day < 1) or (day > 28):
            raise ValueError(
                "Incorrect anchor input. "
                f"Day number {day} is not a valid day for month {month}"
            )
    else:
        raise ValueError(
            "Incorrect anchor input. Month number must be between 1 and 12."
        )


def check_week_day(week: int, day: int = 1):
    if week == 53:
        raise ValueError(
            "Incorrect anchor input. "
            "Week 53 is not a valid input, as not every year contains a 53rd week."
        )
    if (week < 1) or (week > 52):
        raise ValueError(
            "Incorrect anchor input. Week numbers must be between 1 and 52."
        )
    if (day < 1) or (day > 7):
        raise ValueError(
            "Incorrect anchor input. Weekday numbers must be between 1 and 7."
        )


def parse_freqstr_to_dateoffset(time_str):
    """Parses the user-input time strings.

    Args:
        time_str: Time length string in the right formatting.

    Returns:
        Dictionary as keyword argument for Pandas DateOffset.
    """
    if re.fullmatch(r"[+-]?\d*d", time_str):
        time_dict = {"days": int(time_str[:-1])}
    elif re.fullmatch(r"[+-]?\d*M", time_str):
        time_dict = {"months": int(time_str[:-1])}
    elif re.fullmatch(r"[+-]?\d*W", time_str):
        time_dict = {"weeks": int(time_str[:-1])}
    else:
        raise ValueError("Please input a time string in the correct format.")

    return time_dict
