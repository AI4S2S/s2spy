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
    >>> calendar = s2spy.time.AdventCalendar(anchor=(12, 31), freq="7d")
    >>> calendar
    AdventCalendar(month=12, day=31, freq=7d, n_targets=1)

    >>> # Get the 180-day periods leading up to New Year's eve for the year 2020
    >>> calendar = s2spy.time.AdventCalendar(anchor=(12, 31), freq='180d')
    >>> calendar = calendar.map_years(2020, 2020)
    >>> calendar.show() # doctest: +NORMALIZE_WHITESPACE
    i_interval                 (target) 0                         1
    anchor_year
    2020         (2020-07-04, 2020-12-31]  (2020-01-06, 2020-07-04]

    >>> # Get the 180-day periods leading up to New Year's eve for 2020 - 2022 inclusive.
    >>> calendar = s2spy.time.AdventCalendar(anchor=(12, 31), freq='180d')
    >>> calendar = calendar.map_years(2020, 2022)
    >>> # note the leap year:
    >>> calendar.show() # doctest: +NORMALIZE_WHITESPACE
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
import re
from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr
from ._base_calendar import BaseCalendar
from ._resample import resample  # pylint: disable=unused-import


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
        anchor: Tuple[int, int] = (11, 30),
        freq: str = "7d",
        n_targets: int = 1,
    ) -> None:
        """Instantiate a basic calendar with minimal configuration.

        Set up the calendar with given freq ending exactly on the anchor date.
        The index will extend back in time as many periods as fit within the
        cycle time of one year.

        Args:
            anchor: Tuple of the form (month, day). Effectively the origin
                of the calendar. It will countdown until this date.
            freq: Frequency of the calendar.
            n_targets: integer specifying the number of target intervals in a period.

        Example:
            Instantiate a calendar counting down the weeks until new-year's
            eve.

            >>> import s2spy.time
            >>> calendar = s2spy.time.AdventCalendar(anchor=(12, 31), freq="7d")
            >>> calendar
            AdventCalendar(month=12, day=31, freq=7d, n_targets=1)

        """
        if not re.fullmatch(r"\d*d", freq):
            raise ValueError("Please input a frequency in the form of '10d'")
        self.month = anchor[0]
        self.day = anchor[1]
        self.freq = freq
        self.n_targets = n_targets

        self._max_lag:int = 0
        self._allow_overlap: bool = False

    def _get_anchor(self, year: int) -> pd.Timestamp:
        """Generates a timestamp for the end of interval 0 in year.

        Args:
            year (int): anchor year for which the anchor timestamp should be generated

        Returns:
            pd.Timestamp: Timestamp at the end of the anchor_years interval 0.
        """
        return pd.Timestamp(year, self.month, self.day)


class MonthlyCalendar(BaseCalendar):
    """Countdown time to anticipated anchor month, in steps of whole months."""

    def __init__(
        self,
        anchor: str = "Dec",
        freq: str = "1M",
        n_targets: int = 1,
    ) -> None:
        """Instantiate a basic monthly calendar with minimal configuration.

        Set up the calendar with given freq ending exactly on the anchor month.
        The index will extend back in time as many periods as fit within the
        cycle time of one year.

        Args:
            anchor: Str in the form 'January' or 'Jan'. Effectively the origin
                of the calendar. It will countdown up to this month.
            freq: Frequency of the calendar, in the form '1M', '2M', etc.
            n_targets: integer specifying the number of target intervals in a period.

        Example:
            Instantiate a calendar counting down the quarters (3 month periods) until
            december.

            >>> import s2spy.time
            >>> calendar = s2spy.time.MonthlyCalendar(anchor='Dec', freq="3M")
            >>> calendar
            MonthlyCalendar(month=12, freq=3M, n_targets=1)

        """
        if not re.fullmatch(r"\d*M", freq):
            raise ValueError("Please input a frequency in the form of '2M'")

        self.month = month_mapping_dict[anchor.upper()]
        self.freq = freq
        self.n_targets = n_targets
        self._max_lag = 0
        self._allow_overlap = False

    def _get_anchor(self, year: int) -> pd.Timestamp:
        """Generates a timestamp for the end of interval 0 in year.

        Args:
            year (int): anchor year for which the anchor timestamp should be generated

        Returns:
            pd.Timestamp: Timestamp at the end of the anchor_years interval 0.
        """
        return pd.Timestamp(year, self.month, 1) + pd.tseries.offsets.MonthEnd(0)

    def _get_nintervals(self) -> int:
        """Calculates the number of intervals that should be generated by _map year.

        Returns:
            int: Number of intervals for one anchor year.
        """
        periods_per_year = 12 / int(self.freq.replace("M", ""))
        return (
            (self._max_lag + self.n_targets) if self._max_lag > 0 else int(periods_per_year)
        )

    def _get_skip_nyears(self) -> int:
        """Determine how many years need to be skipped to avoid overlapping data.

        Required to prevent information leakage between anchor years.

        Returns:
            int: Number of years that need to be skipped.
        """
        nmonths = int(self.freq.replace("M", ""))
        return (
            0
            if self._max_lag > 0 and self._allow_overlap
            else int(np.ceil(nmonths / 12) - 1)
        )

    def _interval_as_month(self, interval):
        """Turns an interval with pandas Timestamp values to a formatted string.

        The string will contain the years and months, for a more intuitive
        representation to the user.

        Args:
            interval (pd.Interval): Pandas interval.

        Returns:
            str: String in the form of '(2020 Jan, 2020 Feb]'
        """
        left = interval.left.strftime("%Y %b")
        right = interval.right.strftime("%Y %b")
        return f"({left}, {right}]"

    def show(self) -> pd.DataFrame:
        """Displays the intervals the Calendar will generate for the current setup.

        Returns:
            pd.Dataframe: Dataframe containing the calendar intervals, with the target
                periods labelled.
        """
        return self._label_targets(self.get_intervals()).applymap(
            self._interval_as_month
        )


class WeeklyCalendar(BaseCalendar):
    """Countdown time to anticipated anchor week number, in steps of calendar weeks."""

    def __init__(
        self,
        anchor: int,
        freq: str = "1W",
        n_targets: int = 1,
    ) -> None:
        """Instantiate a basic week number calendar with minimal configuration.

        Set up the calendar with given freq ending exactly on the anchor week.
        The index will extend back in time as many weeks as fit within the
        cycle time of one year (i.e. 52).
        Note that the difference between this calendar and the AdventCalendar revolves
        around the use of calendar weeks (Monday - Sunday), instead of 7-day periods.

        Args:
            anchor: Int denoting the week number. Effectively the origin of the calendar.
                It will countdown until this week.
            freq: Frequency of the calendar, e.g. '2W'.
            n_targets: integer specifying the number of target intervals in a period.

        Example:
            Instantiate a calendar counting down the weeks until week number 40.

            >>> import s2spy.time
            >>> calendar = s2spy.time.WeeklyCalendar(anchor=40, freq="1W")
            >>> calendar
            WeeklyCalendar(week=40, freq=1W, n_targets=1)

        """
        if not re.fullmatch(r"\d*W", freq):
            raise ValueError("Please input a frequency in the form of '4W'")

        self.week = anchor
        self.freq = freq
        self.n_targets = n_targets

        self._max_lag: int = 0
        self._allow_overlap: bool = False

    def _get_anchor(self, year: int) -> pd.Timestamp:
        """Generates a timestamp for the end of interval 0 in year.

        Args:
            year (int): anchor year for which the anchor timestamp should be generated

        Returns:
            pd.Timestamp: Timestamp at the end of the anchor_years interval 0.
        """
        # Day 0 of a weeknumber is sunday. Weeks start on monday (strftime functionality)
        return pd.to_datetime(f"{year}-{self.week}-0", format="%Y-%W-%w")

    def _interval_as_weeknr(self, interval: pd.Interval) -> str:
        """Turns an interval with pandas Timestamp values to a formatted string.

        The string will contain the years and week numbers, for a more intuitive
        representation to the user.

        Args:
            interval (pd.Interval): Pandas interval.

        Returns:
            str: String in the form of '(2020-50, 2020-51]'
        """
        left = interval.left.strftime("%Y-W%W")
        right = interval.right.strftime("%Y-W%W")
        return f"({left}, {right}]"

    def show(self) -> pd.DataFrame:
        """Displays the intervals the Calendar will generate for the current setup.

        Returns:
            pd.Dataframe: Dataframe containing the calendar intervals, with the target
                periods labelled.
        """
        return self._label_targets(self.get_intervals()).applymap(
            self._interval_as_weeknr
        )
