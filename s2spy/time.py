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
from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr
from ._base_calendar import BaseCalendar

PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)

month_mapping_dict = {v.upper(): k for k, v in enumerate(pycalendar.month_abbr)} | {
    v.upper(): k for k, v in enumerate(pycalendar.month_name)
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
        self._n_intervals = pd.Timedelta("365days") // pd.to_timedelta(freq)
        self._n_targets = n_targets
        self._traintest = None
        self._intervals = None

        periods_per_year = pd.Timedelta("365days") / pd.to_timedelta(freq)
        # Determine the amount of intervals, and number of anchor years to skip
        if max_lag:
            self._n_intervals = max_lag + self._n_targets
            self._skip_years = (
                np.ceil(self._n_intervals / periods_per_year).astype(int) - 1
            )
        else:
            self._n_intervals = int(periods_per_year)
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
            end=anchor, periods=self._n_intervals, freq=self.freq
        )
        intervals = pd.Series(intervals[::-1], name=str(year))
        intervals.index.name = "i_interval"
        return intervals


class MonthlyCalendar(BaseCalendar):
    def __init__(self, anchor_month: str, freq: str = "1M", n_targets: int = 1) -> None:
        self.month = month_mapping_dict[anchor_month.upper()]
        self.freq = freq
        self._n_intervals = 12 // int(freq.replace("M", ""))
        self._n_targets = n_targets
        self._traintest = None
        self._intervals = None
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
            end=anchor, periods=self._n_intervals, freq=self.freq
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
            str: String in the form of '(2020-50, 2020-51]'
        """
        left = interval.left.strftime('%Y %b')
        right = interval.right.strftime('%Y %b')
        return f"({left}, {right}]"

    def __repr__(self):
        if self._intervals is not None:
            return repr(self._label_targets(
                self._intervals.applymap(self._interval_as_month)
                ))

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"MonthlyCalendar({props})"

    def _repr_html_(self):
        """For jupyter notebook to load html compatiable version of __repr__."""
        if self._intervals is not None:
            # pylint: disable=protected-access
            return self._label_targets(
                self._intervals.applymap(self._interval_as_month)
                )._repr_html_()

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"MonthlyCalendar({props})"


class WeeklyCalendar(BaseCalendar):
    def __init__(self, anchor_week: int, freq: str = "1W", n_targets: int = 1,) -> None:
        self.week = anchor_week
        self.freq = freq
        self._n_intervals = 52 // int(freq.replace("W", ""))
        self._n_targets = n_targets
        self._traintest = None
        self._intervals = None
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
        anchor = pd.to_datetime(f'{year}-{self.week}-0', format='%Y-%W-%w')

        intervals = pd.interval_range(
            end=anchor, periods=self._n_intervals, freq=self.freq
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
        left = interval.left.strftime('%Y-%W')
        right = interval.right.strftime('%Y-%W')
        return f"({left}, {right}]"

    def __repr__(self):
        if self._intervals is not None:
            return repr(self._label_targets(
                self._intervals.applymap(self._interval_as_weeknr)
                ))

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"WeeklyCalendar({props})"

    def _repr_html_(self):
        """For jupyter notebook to load html compatiable version of __repr__."""
        if self._intervals is not None:
            # pylint: disable=protected-access
            return self._label_targets(
                self._intervals.applymap(self._interval_as_weeknr)
                )._repr_html_()

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"WeeklyCalendar({props})"
