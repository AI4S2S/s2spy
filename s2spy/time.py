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
from abc import ABC
from typing import Tuple
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
from . import utils
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

        self._max_lag: int = 0
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
            (self._max_lag + self.n_targets)
            if self._max_lag > 0
            else int(periods_per_year)
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


def _parse_anchor(anchor_str: str) -> Tuple[str, str]:
    """Parses the user-input anchor.

    Args:
        anchor_str: Anchor string in the right formatting.

    Returns:
        Datetime formatter to parse the anchor into a date.
    """
    if re.fullmatch("\\d{1,2}-\\d{1,2}", anchor_str):
        fmt = "%m-%d"
    elif re.fullmatch("\\d{1,2}", anchor_str):
        fmt = "%m"
    elif re.fullmatch("W\\d{1,2}-\\d", anchor_str):
        fmt = "W%W-%w"
    elif re.fullmatch("W\\d{1,2}", anchor_str):
        fmt = "W%W-%w"
        anchor_str += "-1"
    elif anchor_str.lower() in utils.get_month_names():
        anchor_str = str(utils.get_month_names()[anchor_str.lower()])
        fmt = "%m"
    else:
        raise ValueError(f"Anchor input '{anchor_str}' does not match expected format")
    return anchor_str, fmt


class CustomCalendar(BaseCalendar):
    """Build a calendar from sratch with basic construction elements."""

    def __init__(self, anchor: str):
        """Instantiate a basic container for building calendar using basic blocks.

        This is a highly flexible calendar which allows the user to build their own
        calendar with the basic building blocks of target and precursor periods.

        Users have the freedom to create calendar with customized intervals, gap
        between intervals, and even overlapped intervals. They need to manage the
        calendar themselves.

        Args:
            anchor: String denoting the anchor date. The following inputs are valid:
                    - "MM-DD" for a month and day. E.g. "12-31".
                    - "MM" for only a month, e.g. "4" for March.
                    - English names and abbreviations of months. E.g. "December" or "jan".
                    - "Www" for a week number, e.g. "W05" for the fifth week of the year.
                    - "Www-D" for a week number plus day of week. E.g. "W01-4" for the
                        first thursday of the year.

        Attributes:
            n_targets (int): Number of targets that inferred from the appended
            `TargetPeriod` blocks.
        """
        self._anchor, self._anchor_fmt = _parse_anchor(anchor)
        self._targets: list[TargetPeriod] = []
        self._precursors: list[PrecursorPeriod] = []
        self._total_length_target = 0
        self._total_length_precursor = 0
        self.n_targets = 0

        self._allow_overlap: bool = False

    def _get_anchor(self, year: int) -> pd.Timestamp:
        """Generates a timestamp for the end of interval 0 in year.

        Args:
            year (int): anchor year for which the anchor timestamp should be generated

        Returns:
            pd.Timestamp: Timestamp at the end of the anchor_years interval 0.
        """
        return pd.to_datetime(
            f"{year}-" + self._anchor, format="%Y-" + self._anchor_fmt
        )

    def append(self, period_block):
        """Append target/precursor periods to the calendar."""
        if period_block.target:
            self._targets.append(period_block)
            # count length
            self._total_length_target += period_block.length + period_block.gap

        else:
            self._precursors.append(period_block)
            # count length
            self._total_length_precursor += period_block.length + period_block.gap

    def _map_year(self, year: int):
        """Replace old map_year function"""
        intervals_target = self._concatenate_periods(year, self._targets, True)
        intervals_precursor = self._concatenate_periods(year, self._precursors, False)

        self.n_targets = len(intervals_target)
        year_intervals = intervals_precursor[::-1] + intervals_target

        # turn the list of intervals into pandas series
        year_intervals = pd.Series(year_intervals[::-1], name=year)
        year_intervals.index.name = "i_interval"
        return year_intervals

    def _concatenate_periods(self, year, list_periods, is_target):
        # generate intervals based on the building blocks
        intervals = []
        if is_target:
            # build from left to right
            left_date = self._get_anchor(year)
            # loop through all the building blocks to
            for block in list_periods:
                left_date += pd.Timedelta(block.gap, unit="D")
                right_date = left_date + pd.Timedelta(block.length, unit="D")
                intervals.append(pd.Interval(left_date, right_date, closed="left"))
                # update left date
                left_date = right_date
        else:
            # build from right to left
            right_date = self._get_anchor(year)
            # loop through all the building blocks to
            for block in list_periods:
                right_date -= pd.Timedelta(block.gap, unit="D")
                left_date = right_date - pd.Timedelta(block.length, unit="D")
                intervals.append(pd.Interval(left_date, right_date, closed="left"))
                # update right date
                right_date = left_date

        return intervals

    def map_years(self, start: int, end: int):
        """Adds a start and end year mapping to the calendar.

        If the start and end years are the same, the intervals for only that single
        year are returned by calendar.get_intervals().

        Args:
            start: The first year for which the calendar will be realized
            end: The last year for which the calendar will be realized

        Returns:
            The calendar mapped to the input start and end year.
        """
        if start > end:
            raise ValueError("The start year cannot be greater than the end year")

        self._first_year = start
        self._last_year = end
        self._mapping = "years"
        return self

    def map_to_data(
        self,
        input_data: Union[pd.Series, pd.DataFrame, xr.Dataset, xr.DataArray],
    ):
        """Map the calendar to input data period.

        Stores the first and last intervals of the input data to the calendar, so that
        the intervals can cover the data to the greatest extent.

        Args:
            input_data: Input data for datetime mapping. Its index must be either
                pandas.DatetimeIndex, or an xarray `time` coordinate with datetime
                data.

        Returns:
            The calendar mapped to the input data period.
        """
        utils.check_timeseries(input_data)

        # check the datetime order of input data
        if isinstance(input_data, PandasData):
            self._first_timestamp = input_data.index.min()
            self._last_timestamp = input_data.index.max()
        else:
            self._first_timestamp = pd.Timestamp(input_data.time.min().values)
            self._last_timestamp = pd.Timestamp(input_data.time.max().values)

        self._mapping = "data"

        return self

    def _set_year_range_from_timestamps(self):
        min_year = self._first_timestamp.year
        max_year = self._last_timestamp.year

        # ensure that the input data could always cover the advent calendar
        # last date check
        if self._map_year(max_year).iloc[0].right > self._last_timestamp:
            max_year -= 1
        # first date check
        while self._map_year(min_year).iloc[-1].right <= self._first_timestamp:
            min_year += 1

        # map year(s) and generate year realized advent calendar
        if max_year >= min_year:
            self._first_year = min_year
            self._last_year = max_year
        else:
            raise ValueError(
                "The input data could not cover the target advent calendar."
            )

        return self

    def _rename_i_intervals(self, intervals: pd.DataFrame) -> pd.DataFrame:
        """Adds target labels to the header row of the intervals.

        Args:
            intervals (pd.Dataframe): Dataframe with intervals.

        Returns:
            pd.Dataframe: Dataframe with target periods labelled.
        """
        # rename preursors
        intervals = intervals.rename(
            columns={i: self.n_targets - i - 1 for i in range(self.n_targets,
             len(intervals.columns))}
        )

        # rename targets
        intervals = intervals.rename(
            columns={i: self.n_targets - i for i in range(self.n_targets)}
        )

        return intervals

    def get_intervals(self) -> pd.DataFrame:
        """Method to retrieve updated intervals from the Calendar object."""
        if self._mapping is None:
            raise ValueError(
                "Cannot retrieve intervals without map_years or "
                "map_to_data having configured the calendar."
            )
        if self._mapping == "data":
            self._set_year_range_from_timestamps()

        year_range = range(
            self._last_year, self._first_year - 1, -(self._get_skip_nyears() + 1)
        )

        intervals = pd.concat([self._map_year(year) for year in year_range], axis=1).T

        intervals.index.name = "anchor_year"
        return intervals.sort_index(ascending=False, axis=1)

    def _get_skip_nyears(self) -> int:
        """Determine how many years need to be skipped to avoid overlapping data.

        Required to prevent information leakage between anchor years.

        Returns:
            int: Number of years that need to be skipped.
        """
        years = pd.to_timedelta(
            self._total_length_target + self._total_length_precursor
        ) / pd.Timedelta("365days")

        return 0 if self._allow_overlap else int(np.ceil(years).astype(int) - 1)

    def show(self) -> pd.DataFrame:
        """Displays the intervals the Calendar will generate for the current setup.

        Returns:
            pd.Dataframe: Dataframe containing the calendar intervals, with the target
                periods labelled.
        """
        return self._rename_i_intervals(self.get_intervals())


class Period(ABC):
    """Basic construction element of calendar for defining target period."""

    def __init__(self, length: int, gap: int = 0, target: bool = False) -> None:
        self.length = length
        self.gap = gap
        self.target = target
        # TO DO: support lead_time
        # self.lead_time = lead_time


class TargetPeriod(Period):
    """Instantiate a build block as target period."""

    def __init__(self, length: int, gap: int = 0) -> None:
        self.length = length
        self.gap = gap
        self.target = True


class PrecursorPeriod(Period):
    """Instantiate a build block as precursor period."""

    def __init__(self, length: int, gap: int = 0) -> None:
        self.length = length
        self.gap = gap
        self.target = False
