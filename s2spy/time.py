"""s2spy time utils.

Utilities designed to aid in seasonal-to-subseasonal prediction experiments in
which we search for skillful predictors preceding a certain event of interest.

Time indices are anchored to the target period of interest. By keeping
observations from the same cycle (typically 1 year) together and paying close
attention to the treatment of adjacent cycles, we avoid information leakage
between train and test sets.

Example:

    >>> import s2spy.time
    >>>
    >>> # Countdown the weeks until New Year's Eve
    >>> calendar = s2spy.time.AdventCalendar(anchor_date=(12, 31), freq="7d")
    >>> calendar
    AdventCalendar(month=12, day=31, freq=7d)
    >>> print(calendar)
    52 periods of 7d leading up to 12/31.

    >>> # Get the 180-day periods leading up to New Year's eve for the year 2020
    >>> calendar = s2spy.time.AdventCalendar(anchor_date=(12, 31), freq='180d')
    >>> calendar.map_years(2020, 2020) # doctest: +NORMALIZE_WHITESPACE
    i_interval                 (target) 0                         1
    anchor_year
    2020         (2020-07-04, 2020-12-31]  (2020-01-06, 2020-07-04]

    >>> # Get the 180-day periods leading up to New Year's eve for 2020 - 2022 inclusive.
    >>> calendar = s2spy.time.AdventCalendar(anchor_date=(12, 31), freq='180d')
    >>> # note the leap year:
    >>> calendar.map_years(2020, 2022) # doctest: +NORMALIZE_WHITESPACE
    i_interval                 (target) 0                         1
    anchor_year
    2022         (2022-07-04, 2022-12-31]  (2022-01-05, 2022-07-04]
    2021         (2021-07-04, 2021-12-31]  (2021-01-05, 2021-07-04]
    2020         (2020-07-04, 2020-12-31]  (2020-01-06, 2020-07-04]

    >>> # To get a stacked representation:
    >>> calendar.map_years(2020, 2022).flat
    anchor_year  i_interval
    2022         0             (2022-07-04, 2022-12-31]
                 1             (2022-01-05, 2022-07-04]
    2021         0             (2021-07-04, 2021-12-31]
                 1             (2021-01-05, 2021-07-04]
    2020         0             (2020-07-04, 2020-12-31]
                 1             (2020-01-06, 2020-07-04]
    dtype: interval

"""
import calendar as pycalendar
import warnings
from typing import Tuple
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
from . import _resample
from ._base_calendar import BaseCalendar


PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)

month_mapping_dict = {
    **{v.upper(): k for k, v in enumerate(pycalendar.month_abbr)},
    **{v.upper(): k for k, v in enumerate(pycalendar.month_name)},
}


class AdventCalendar(BaseCalendar):
    """Countdown time to anticipated anchor date or period of interest."""

    def __init__(
        self,
        anchor_date: Tuple[int, int] = (11, 30),
        freq: str = "7d",
        n_targets: int = 1,
        max_lag: int = None,
    ) -> None:
        """Instantiate a basic calendar with minimal configuration.

        Set up the calendar with given freq ending exactly on the anchor date.
        The index will extend back in time as many periods as fit within the
        cycle time of one year.

        Args:
            anchor_date: Tuple of the form (month, day). Effectively the origin
                of the calendar. It will countdown until this date.
            freq: Frequency of the calendar.
            n_targets: integer specifying the number of target intervals in a period.
            max_lag: Maximum number of lag periods after the target period. If `None`,
                the maximum lag will be determined by how many fit in each anchor year.
                If a maximum lag is provided, the intervals can either only cover part
                of the year, or extend over multiple years. In case of a large max_lag
                number where the intervals extend over multiple years, anchor years will
                be skipped to avoid overlapping intervals.

        Example:
            Instantiate a calendar counting down the weeks until new-year's
            eve.

            >>> import s2spy.time
            >>> calendar = s2spy.time.AdventCalendar(anchor_date=(12, 31), freq="7d")
            >>> calendar
            AdventCalendar(month=12, day=31, freq=7d)
            >>> print(calendar)
            "52 periods of 7d leading up to 12/31."

        """
        self.month = anchor_date[0]
        self.day = anchor_date[1]
        self.freq = freq
        self._nintervals = pd.Timedelta("365days") // pd.to_timedelta(freq)
        self.n_targets = n_targets
        self._traintest = None
        self.intervals = None

        periods_per_year = pd.Timedelta("365days") / pd.to_timedelta(freq)
        # Determine the amount of intervals, and number of anchor years to skip
        if max_lag:
            self._nintervals = max_lag + self.n_targets
            self._skip_years = (
                np.ceil(self._nintervals / periods_per_year).astype(int) - 1
            )
        else:
            self._nintervals = int(periods_per_year)
            self._skip_years = 0

    def _map_year(self, year: int) -> pd.Series:
        """Internal routine to return a concrete IntervalIndex for the given year.

        Since the AdventCalendar represents a periodic event, it is first
        instantiated without a specific year. This method adds a specific year
        to the calendar and returns an intervalindex, applying the
        AdvenctCalendar to this specific year.

        Args:
            year: The year for which the AdventCalendar will be realized

        Returns:
            Pandas Series filled with Intervals of the calendar's frequency, counting
            backwards from the calendar's anchor_date.
        """
        anchor = pd.Timestamp(year, self.month, self.day)
        intervals = pd.interval_range(
            end=anchor, periods=self._nintervals, freq=self.freq
        )
        intervals = pd.Series(intervals[::-1], name=str(year))
        intervals.index.name = "i_interval"
        return intervals


class MonthlyCalendar(BaseCalendar):
    def __init__(self, anchor_month: str, freq: str = "1M", n_targets: int = 1) -> None:
        self.month = month_mapping_dict[anchor_month.upper()]
        self.freq = freq
        self._nintervals = 12 // int(freq.replace("M", ""))
        self.n_targets = n_targets
        self._traintest = None
        self.intervals = None
        self._skip_years = 0

    def _map_year(self, year: int) -> pd.Series:
        """Internal routine to return a concrete IntervalIndex for the given year.

        Since the MonthlyCalendar represents a periodic event, it is first
        instantiated without a specific year. This method adds a specific year
        to the calendar and returns an intervalindex, applying the
        AdvenctCalendar to this specific year.

        Args:
            year: The year for which the MonthlyCalendar will be realized

        Returns:
            Pandas Series filled with Intervals of the calendar's frequency, counting
            backwards from the calendar's anchor_date.
        """
        anchor = pd.Timestamp(year, self.month, 1) + pd.tseries.offsets.MonthEnd(0)

        intervals = pd.interval_range(
            end=anchor, periods=self._nintervals, freq=self.freq
        )

        intervals = pd.Series(intervals[::-1], name=str(year))
        intervals.index.name = "i_interval"
        return intervals

    def _interval_as_month(self, interval):
        """Turns an interval with pandas Timestamp values to a string with the years and
        week numbers, for a more intuitive representation to the user.

        Args:
            interval (pd.Interval): Pandas interval.

        Returns:
            str: String in the form of '(2020 Jan, 2020 Feb]'
        """
        left = interval.left.strftime("%Y %b")
        right = interval.right.strftime("%Y %b")
        return f"({left}, {right}]"

    def __repr__(self):
        if self.intervals is not None:
            return repr(
                _resample.label_targets(self, self.intervals.applymap(self._interval_as_month))
            )

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"MonthlyCalendar({props})"

    def _repr_html_(self):
        """For jupyter notebook to load html compatiable version of __repr__."""
        if self.intervals is not None:
            # pylint: disable=protected-access
            return _resample.label_targets(
                self,
                self.intervals.applymap(self._interval_as_month)
            )._repr_html_()

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"MonthlyCalendar({props})"


class WeeklyCalendar(BaseCalendar):
    def __init__(
        self,
        anchor_week: int,
        freq: str = "1W",
        n_targets: int = 1,
    ) -> None:
        self.week = anchor_week
        self.freq = freq
        self._nintervals = 52 // int(freq.replace("W", ""))
        self.n_targets = n_targets
        self._traintest = None
        self.intervals = None
        self._skip_years = 0

    def _map_year(self, year: int) -> pd.Series:
        """Internal routine to return a concrete IntervalIndex for the given year.

        Since the WeeklyCalendar represents a periodic event, it is first
        instantiated without a specific year. This method adds a specific year
        to the calendar and returns an intervalindex, applying the
        AdvenctCalendar to this specific year.

        Args:
            year: The year for which the WeeklyCalendar will be realized

        Returns:
            Pandas Series filled with Intervals of the calendar's frequency, counting
            backwards from the calendar's anchor_date.
        """
        # Day 0 of a weeknumber is sunday. Weeks start on monday (strftime functionality)
        anchor = pd.to_datetime(f"{year}-{self.week}-0", format="%Y-%W-%w")

        intervals = pd.interval_range(
            end=anchor, periods=self._nintervals, freq=self.freq
        )

        intervals = pd.Series(intervals[::-1], name=str(year))
        intervals.index.name = "i_interval"
        return intervals

    def _interval_as_weeknr(self, interval):
        """Turns an interval with pandas Timestamp values to a string with the years and
        week numbers, for a more intuitive representation to the user.

        Args:
            interval (pd.Interval): Pandas interval.

        Returns:
            str: String in the form of '(2020-50, 2020-51]'
        """
        left = interval.left.strftime("%Y-%W")
        right = interval.right.strftime("%Y-%W")
        return f"({left}, {right}]"

    def __repr__(self):
        if self.intervals is not None:
            return repr(
                _resample.label_targets(self, self.intervals.applymap(self._interval_as_weeknr))
            )

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"WeeklyCalendar({props})"

    def _repr_html_(self):
        """For jupyter notebook to load html compatiable version of __repr__."""
        if self.intervals is not None:
            # pylint: disable=protected-access
            return _resample.label_targets(
                self,
                self.intervals.applymap(self._interval_as_weeknr)
            )._repr_html_()

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"WeeklyCalendar({props})"


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
        >>> cal = s2spy.time.AdventCalendar(freq='180d')
        >>> time_index = pd.date_range('20191201', '20211231', freq='1d')
        >>> var = np.arange(len(time_index))
        >>> input_data = pd.Series(var, index=time_index)
        >>> bins = cal.resample(input_data)
        >>> bins
            anchor_year  i_interval                  interval  mean_data  target
        0        2020           0  (2020-06-03, 2020-11-30]      275.5    True
        1        2020           1  (2019-12-06, 2020-06-03]       95.5   False
        2        2021           0  (2021-06-03, 2021-11-30]      640.5    True
        3        2021           1  (2020-12-05, 2021-06-03]      460.5   False

    """
    if mapped_calendar.intervals is None:
        raise ValueError("Generate a calendar map before calling resample")

    if not isinstance(input_data, PandasData + XArrayData):
        raise ValueError("The input data is neither a pandas or xarray object")

    if isinstance(input_data, PandasData):
        if not isinstance(input_data.index, pd.DatetimeIndex):
            raise ValueError("The input data does not have a datetime index.")

        # raise a warning for upscaling
        # target frequency must be larger than the (absolute) input frequency
        if input_data.index.freq:
            input_freq = input_data.index.freq
            input_freq = input_freq if input_freq.n > 0 else -input_freq
            if pd.Timedelta(mapped_calendar.freq) < input_freq:
                warnings.warn(
                    """Target frequency is smaller than the original frequency.
                    The resampled data will contain NaN values, as there is no data
                    available within all intervals."""
                )

        resampled_data = _resample.resample_pandas(mapped_calendar, input_data)

    # Data must be xarray
    else:
        if "time" not in input_data.dims:
            raise ValueError(
                "The input DataArray/Dataset does not contain a `time` dimension"
            )
        if not xr.core.common.is_np_datetime_like(input_data["time"].dtype):
            raise ValueError("The `time` dimension is not of a datetime format")

        resampled_data = _resample.resample_xarray(mapped_calendar, input_data)

    # mark target periods before returning the resampled data
    return _resample.mark_target_period(mapped_calendar, resampled_data)
