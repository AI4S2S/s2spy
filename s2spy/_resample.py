from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
from . import utils


PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)


def mark_target_period(
    input_data: Union[pd.DataFrame, xr.Dataset]
) -> Union[pd.DataFrame, xr.Dataset]:
    """Mark interval periods that fall within the given number of target periods.

    Pass a pandas Series/DataFrame with an 'i_interval' column, or an xarray
    DataArray/Dataset with an 'i_interval' coordinate axis. It will return an
    object with an added column in the Series/DataFrame or an
    added coordinate axis in the DataSet called 'target'. This is a boolean
    indicating whether the index time interval is a target period or not. This is
    determined by the instance variable 'n_targets'.

    Args:
        input_data: Input data for resampling. For a Pandas object, one of its
        columns must be called 'i_interval'. An xarray object requires a coordinate
        axis named 'i_interval' containing an interval counter for every period.

    Returns:
        Input data with boolean marked target periods, similar data format as
            given inputs.
    """
    if isinstance(input_data, PandasData):
        input_data["target"] = np.ones(input_data.index.size, dtype=bool)
        input_data["target"] = input_data["target"].where(
            input_data["i_interval"] > 0, other=False
        )

    else:
        # input data is xr.Dataset
        target = input_data["i_interval"] > 0
        input_data = input_data.assign_coords(coords={"target": target})

    return input_data


def resample_bins_constructor(
    intervals: Union[pd.Series, pd.DataFrame]
) -> pd.DataFrame:
    """Restructures the interval object into a tidy DataFrame.

    Args:
        intervals: the output interval `pd.Series` or `pd.DataFrame` from the
            `map_to_data` function.

    Returns:
        Pandas DataFrame with 'anchor_year', 'i_interval', and 'interval' as
            columns.
    """
    # Make a tidy dataframe where the intervals are linked to the anchor year
    if isinstance(intervals, pd.DataFrame):
        bins = intervals.copy()
        bins.index.rename("anchor_year", inplace=True)
        bins = bins.melt(
            var_name="i_interval", value_name="interval", ignore_index=False
        )
        bins = bins.sort_values(by=["anchor_year", "i_interval"])
    else:
        # Massage the dataframe into the same tidy format for a single year
        bins = pd.DataFrame(intervals)
        bins = bins.melt(
            var_name="anchor_year", value_name="interval", ignore_index=False
        )
        bins.index.rename("i_interval", inplace=True)
    bins = bins.reset_index()

    return bins


def contains(interval_index: pd.IntervalIndex, timestamps) -> np.ndarray:
    """Checks elementwise if the intervals contain the timestamps.
    Will return a boolean array of the shape (n_timestamps, n_intervals).

    Args:
        interval_index: An IntervalIndex containing all intervals that should be checked.
        timestamps: A 1-D array containing

    Returns:
        np.ndarray: 2-D mask array
    """
    if interval_index.closed_left:
        a = np.greater_equal(timestamps, interval_index.left.values[np.newaxis].T)
    else:
        a = np.greater(timestamps, interval_index.left.values[np.newaxis].T)
    if interval_index.closed_right:
        b = np.less_equal(timestamps, interval_index.right.values[np.newaxis].T)
    else:
        b = np.less(timestamps, interval_index.right.values[np.newaxis].T)
    return a & b


def create_means_matrix(intervals, timestamps):
    """Creates a matrix to be used to compute the mean data value of each interval.

    E.g.: `means = np.dot(matrix, data)`.

    Args:
        intervals: A 1-D array-like containing the pd.Interval objects.
        timestamps: A 1-D array containing the timestamps of the input data.

    Returns:
        np.ndarray: 2-D array that can will compute the mean.
    """
    matrix = contains(pd.IntervalIndex(intervals), timestamps).astype(float)
    return matrix / matrix.sum(axis=1, keepdims=True)


def resample_pandas(
    calendar, input_data: Union[pd.Series, pd.DataFrame]
) -> pd.DataFrame:
    """Internal function to handle resampling of Pandas data.

    Args:
        input_data (pd.Series or pd.DataFrame): Data provided by the user to the
            `resample` function

    Returns:
        pd.DataFrame: DataFrame containing the intervals and data resampled to
            these intervals.
    """
    if isinstance(input_data, pd.Series):
        name = "data" if input_data.name is None else input_data.name
        input_data = pd.DataFrame(input_data.rename(name))

    data = resample_bins_constructor(calendar.get_intervals())
    means_matrix = create_means_matrix(data.interval.values, input_data.index.values)

    for colname in input_data.columns:
        data[colname] = np.dot(means_matrix, input_data[colname])

    return data


def resample_dataset(calendar, input_data: xr.Dataset) -> xr.Dataset:
    """Internal function to handle resampling of xarray data.

    Args:
        input_data (xr.DataArray or xr.Dataset): Data provided by the user to the
            `resample` function

    Returns:
        xr.Dataset: Dataset containing the intervals and data resampled to
            these intervals.
    """

    data = calendar.flat.to_xarray().rename("interval")
    data = data.to_dataset()
    data = data.stack(anch_int=("anchor_year", "i_interval"))

    means_matrix = xr.DataArray.from_dict(
        {
            "coords": {
                "time": {"dims": "time", "data": input_data["time"].values},
            },
            "dims": ("anch_int", "time"),
            "data": create_means_matrix(
                data["interval"].values, input_data["time"].values
            ),
        }
    )

    resampled_vars = [xr.DataArray] * len(input_data.data_vars)
    for i, var in enumerate(input_data.data_vars):
        resampled_vars[i] = xr.dot(input_data[var], means_matrix, dims=["time"]).rename(
            var
        )

    data = xr.merge([data] + resampled_vars)
    data = data.unstack().set_coords(["interval"])
    return utils.convert_interval_to_bounds(data)


def resample(
    mapped_calendar,
    input_data: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset],
) -> Union[pd.DataFrame, xr.Dataset]:
    """Resample input data to the calendar frequency.

    Pass a pandas Series/DataFrame with a datetime axis, or an
    xarray DataArray/Dataset with a datetime coordinate called 'time'.
    It will return the same object with the datetimes resampled onto
    the Calendar's Index by binning the data into the Calendar's intervals
    and calculating the mean of each bin.

    Note: this function is intended for upscaling operations, which means
    the calendar frequency is larger than the original frequency of input data (e.g.
    `freq` is "7days" and the input is daily data). It supports downscaling
    operations but the user need to be careful since the returned values may contain
    "NaN".

    Args:
        mapped_calendar: Calendar object with either a map_year or map_to_data mapping.
        input_data: Input data for resampling. For a Pandas object its index must be
            either a pandas.DatetimeIndex. An xarray object requires a dimension
            named 'time' containing datetime values.

    Raises:
        UserWarning: If the calendar frequency is smaller than the frequency of
            input data

    Returns:
        Input data resampled based on the calendar frequency, similar data format as
            given inputs.

    Example:
        Assuming the input data is pd.DataFrame containing random values with index
        from 2021-11-11 to 2021-11-01 at daily frequency.

        >>> import s2spy.time
        >>> import pandas as pd
        >>> import numpy as np
        >>> cal = s2spy.time.AdventCalendar(anchor="12-31", freq="180d")
        >>> time_index = pd.date_range("20191201", "20211231", freq="1d")
        >>> var = np.arange(len(time_index))
        >>> input_data = pd.Series(var, index=time_index)
        >>> cal = cal.map_to_data(input_data)
        >>> bins = s2spy.time.resample(cal, input_data)
        >>> bins # doctest: +NORMALIZE_WHITESPACE
            anchor_year  i_interval                  interval   data  target
        0          2019          -1  [2019-07-04, 2019-12-31)   14.5   False
        1          2019           1  [2019-12-31, 2020-06-28)  119.5    True
        2          2020          -1  [2020-07-04, 2020-12-31)  305.5   False
        3          2020           1  [2020-12-31, 2021-06-29)  485.5    True
    """
    intervals = mapped_calendar.get_intervals()

    if intervals is None:
        raise ValueError("Generate a calendar map before calling resample")

    utils.check_timeseries(input_data)
    # This check is still valid for all calendars with `freq`, but not for CustomCalendar
    # TO DO: add this check when all calendars are rebased on the CustomCalendar
    # utils.check_input_frequency(mapped_calendar, input_data)

    if isinstance(input_data, PandasData):
        resampled_data = resample_pandas(mapped_calendar, input_data)
    else:
        if isinstance(input_data, xr.DataArray):
            input_data.name = "data" if input_data.name is None else input_data.name
            input_data = input_data.to_dataset()
        resampled_data = resample_dataset(mapped_calendar, input_data)

    utils.check_empty_intervals(resampled_data)

    # mark target periods before returning the resampled data
    return mark_target_period(resampled_data)
