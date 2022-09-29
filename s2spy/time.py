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
import warnings
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
        self, anchor: Tuple[int, int] = (11, 30), freq: str = "7d", n_targets: int = 1,
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
        self, anchor: str = "Dec", freq: str = "1M", n_targets: int = 1,
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

    def __init__(self, anchor: int, freq: str = "1W", n_targets: int = 1,) -> None:
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


class CustomCalendar(BaseCalendar):
    """Build a calendar from sratch with basic construction elements."""

    def __init__(self, anchor: Tuple[int, int] = (11, 30)):
        """Instantiate a basic container for building calendar using basic blocks.

        This is a highly flexible calendar which allows the user to build their own
        calendar with the basic building blocks of target and precursor periods.

        Note that two rules are applied to avoid information leakage and disorder:
            - period block next to the anchor date should not have negative gap
            - if a appended period block has negative gap, its absolute value must
             be smaller than the length of the precedent period block
        """
        self.month = anchor[0]
        self.day = anchor[1]
        self._list_target = []
        self._list_precursor = []
        self._total_length_target = 0
        self._total_length_precursor = 0
        self.n_targets = 0
        # identifier for completion of appending
        self._end_indentifier = False

        self._allow_overlap: bool = False

    def _get_anchor(self, year: int) -> pd.Timestamp:
        """Generates a timestamp for the end of interval 0 in year.

        Args:
            year (int): anchor year for which the anchor timestamp should be generated

        Returns:
            pd.Timestamp: Timestamp at the end of the anchor_years interval 0.
        """
        return pd.Timestamp(year, self.month, self.day)

    def append(self, period_block):
        """Append target/precursor periods to the calendar."""
        if self._end_indentifier:
            warnings.warn(
                "Calerdar already exists and this overwrites the old calendar."
            )
        if period_block.target:
            self._append_check(self._list_target, period_block)
            self._list_target.append(period_block)
            # count length
            self._total_length_target += period_block.length + period_block.gap

        else:
            self._append_check(self._list_precursor, period_block)
            self._list_precursor.append(period_block)
            # count length
            self._total_length_precursor += period_block.length + period_block.gap

    def _append_check(self, list_period, period_block):
        """Rules check for appending target precursor blocks."""
        if not list_period:
            if period_block.gap < 0:
                raise ValueError(
                    "First appended period block should not contain a negative gap."
                )
        else:
            if (period_block.gap + list_period[-1].length) < 0:
                raise ValueError(
                    "Appended period block has negative gap larger than"
                    + " the length of precedent period block."
                )

    def _map_year(self, year: int):
        """Replace old map_year function"""
        # flag the calendar as existed
        self._end_indentifier = True

        intervals_target = self._concatenate_periods(
            year, self._list_target, self._total_length_target, True
        )
        intervals_precursor = self._concatenate_periods(
            year, self._list_precursor, self._total_length_precursor, False
        )

        self.n_targets = len(intervals_target)
        year_intervals = intervals_precursor[::-1] + intervals_target

        # turn the list of intervals into pandas series
        year_intervals = pd.Series(year_intervals[::-1], name=str(year))
        year_intervals.index.name = "i_interval"
        return year_intervals

    def _concatenate_periods(self, year, list_periods, total_length, is_target):
        # calculate start and end date
        if is_target:
            # anchor date is included
            start_date = self._get_anchor(year)
            end_date = start_date + pd.Timedelta(total_length, unit="D")
            # generate date ranges
            date_range = pd.date_range(start=start_date, end=end_date)
        else:
            # anchor date is excluded
            start_date = self._get_anchor(year) - pd.Timedelta(
                total_length + 1, unit="D"
            )
            end_date = self._get_anchor(year) - pd.Timedelta(1, unit="D")
            # generate date ranges
            date_range = pd.date_range(start=start_date, end=end_date)
            date_range = date_range[::-1]
        # generate intervals based on the building blocks
        intervals = []
        # pointer to the index of date for right boundary of interval
        left_date_index = 0
        # loop through all the building blocks to
        for block in list_periods:
            left_date_index += block.gap
            # pointer to the index of date for left boundary of interval
            right_date_index = left_date_index + block.length - 1
            if is_target:
                intervals.append(
                    pd.Interval(
                        date_range[left_date_index] - pd.Timedelta(1, unit="D"), # open left close right
                        date_range[right_date_index],
                        closed="right",
                    )
                )
            else:
                intervals.append(
                    pd.Interval(
                        date_range[right_date_index] - pd.Timedelta(1, unit="D"),
                        date_range[left_date_index],
                        closed="right",
                    )
                )
            # move pointer to new start
            left_date_index = right_date_index + 1

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

    def _label_targets(self, intervals: pd.DataFrame) -> pd.DataFrame:
        """Adds target labels to the header row of the intervals.

        Args:
            intervals (pd.Dataframe): Dataframe with intervals.

        Returns:
            pd.Dataframe: Dataframe with target periods labelled.
        """
        return intervals.rename(
            columns={i: f"(target) {i}" for i in range(self.n_targets)}
        )

    def get_intervals(self) -> pd.DataFrame:
        """Method to retrieve updated intervals from the Calendar object."""
        if self._mapping is None:
            raise ValueError(
                "Cannot retrieve intervals without map_years or "
                "map_to_data having configured the calendar."
            )
        # if self._mapping == "data":
        #     self._set_year_range_from_timestamps()

        year_range = range(
            self._last_year, self._first_year - 1, -(self._get_skip_nyears() + 1)
        )

        intervals = pd.concat([self._map_year(year) for year in year_range], axis=1).T

        intervals.index.name = "anchor_year"
        return intervals

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
        return self._label_targets(self.get_intervals())


class Period:
    """Basic construction element of calendar for defining target period."""

    def __init__(self, length: int, gap: int = 0, target: bool = False) -> None:
        self.length = length
        self.gap = gap
        self.target = target
        # TO DO: support lead_time
        # self.lead_time = lead_time


def target_period(length, gap: int = 0):
    """Instantiate a build block as target period. """
    return Period(length=length, gap=gap, target=True)


def precursor_period(length, gap: int = 0):
    """Instantiate a build block as precursor period."""
    return Period(length=length, gap=gap, target=False)
