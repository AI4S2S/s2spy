"""BaseCalendar is a template for specific implementations of different calendars.

The BaseCalendar includes most methods required for all calendar operations, except for
a set of abstract methods (e.g., __init__, _get_anchor, ...). These will have to be
customized for each specific calendar.
"""
from abc import ABC
from abc import abstractmethod
import re
from typing import Tuple
from typing import Union
import numpy as np
import pandas as pd
import xarray as xr
from . import _plot
from . import utils


PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)


class BaseCalendar(ABC):
    """Base calendar class which serves as a template for specific implementations."""

    _mapping = None

    @abstractmethod
    def __init__(
        self,
        anchor,
    ):
        """For initializing calendars, the following five variables will be required."""
        self._anchor, self._anchor_fmt = self._parse_anchor(anchor)
        self._targets: list[TargetPeriod] = []
        self._precursors: list[PrecursorPeriod] = []
        self._total_length_target = pd.Timedelta("0d")
        self._total_length_precursor = pd.Timedelta("0d")

        self.n_targets = 0
        self._max_lag: int = 0
        self._allow_overlap: bool = False

    def _get_anchor(self, year: int) -> pd.Timestamp:
        """Method to generate an anchor timestamp for your specific calendar.

        The method should return the exact timestamp of the end of the anchor_year's
        0 interval, e.g., for the AdventCalendar:
        pd.Timestamp(year, self.month, self.day)

        Args:
            year (int): anchor year for which the anchor timestamp should be generated

        Returns:
            pd.Timestamp: Timestamp at the end of the anchor_years interval 0.
        """
        return pd.to_datetime(
            f"{year}-" + self._anchor, format="%Y-" + self._anchor_fmt
        )

    def _parse_anchor(self, anchor_str: str) -> Tuple[str, str]:
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

    def _append(self, period_block):
        """Append target/precursor periods to the calendar."""
        if period_block.target:
            self._targets.append(period_block)
            # count length
            self._total_length_target += period_block.length + period_block.gap

        else:
            self._precursors.append(period_block)
            # count length
            self._total_length_precursor += period_block.length + period_block.gap

    def _map_year(self, year: int) -> pd.Series:
        """Internal routine to return a concrete IntervalIndex for the given year.

        Since our calendars are used to study periodic events, they are first
        instantiated without specific year(s). This method adds a specific year
        to the calendar and returns an intervalindex, mapping the
        Calendar to the given year.

        Intended for internal use, in conjunction with map_years or map_to_data.

        Args:
            year: The year for which the Calendar will be realized

        Returns:
            Pandas Series filled with Intervals of the calendar's frequency, counting
            backwards from the calendar's achor.
        """
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
                left_date += block.gap
                right_date = left_date + block.length
                intervals.append(pd.Interval(left_date, right_date, closed="left"))
                # update left date
                left_date = right_date
        else:
            # build from right to left
            right_date = self._get_anchor(year)
            # loop through all the building blocks to
            for block in list_periods:
                right_date -= block.gap
                left_date = right_date - block.length
                intervals.append(pd.Interval(left_date, right_date, closed="left"))
                # update right date
                right_date = left_date

        return intervals

    def _get_skip_nyears(self) -> int:
        """Determine how many years need to be skipped to avoid overlapping data.

        Required to prevent information leakage between anchor years.

        Returns:
            int: Number of years that need to be skipped.
        """
        years = (
            self._total_length_target + self._total_length_precursor
        ) / pd.Timedelta("365days")

        return 0 if self._allow_overlap else int(np.ceil(years).astype(int) - 1)

    def set_max_lag(self, max_lag: int, allow_overlap: bool = False) -> None:
        """Set the maximum lag of a calendar.
        Sets the maximum number of lag periods after the target period. If `0`,
        the maximum lag will be determined by how many fit in each anchor year.
        If a maximum lag is provided, the intervals can either only cover part
        of the year, or extend over multiple years. In case of a large max_lag
        number where the intervals extend over multiple years, anchor years will
        be skipped to avoid overlapping intervals. To allow overlapping
        intervals, use the `allow_overlap` kwarg.

        Note that this method 
        Args:
            max_lag: Maximum number of lag periods after the target period.
            allow_overlap: Allows intervals to overlap between anchor years, if the
                max_lag is set to a high enough number that intervals extend over
                multiple years. `False` by default, to avoid train/test information
                leakage.
        """
        if (max_lag < 0) or (max_lag % 1 > 0):
            raise ValueError(
                "Max lag should be an integer with a value of 0 or greater"
                f", not {max_lag} of type {type(max_lag)}."
            )

        self._max_lag = max_lag
        self._allow_overlap = allow_overlap

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

    def _rename_intervals(self, intervals: pd.DataFrame) -> pd.DataFrame:
        """Adds target labels to the header row of the intervals.

        Args:
            intervals (pd.Dataframe): Dataframe with intervals.

        Returns:
            pd.Dataframe: Dataframe with target periods labelled, sorted by i_interval value.
        """

        # rename precursors
        intervals = intervals.rename(
            columns={
                i: self.n_targets - i - 1
                for i in range(self.n_targets, len(intervals.columns))
            }
        )

        # rename targets
        intervals = intervals.rename(
            columns={i: self.n_targets - i for i in range(self.n_targets)}
        )

        return intervals.sort_index(axis=1)

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

        intervals = self._rename_intervals(intervals)

        intervals.index.name = "anchor_year"
        return intervals.sort_index(axis=0, ascending=False)

    def show(self) -> pd.DataFrame:
        """Displays the intervals the Calendar will generate for the current setup.

        Returns:
            pd.Dataframe: Dataframe containing the calendar intervals, with the target
                periods labelled.
        """
        return self.get_intervals()

    def __repr__(self) -> str:
        """String representation of the Calendar."""
        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        calendar_name = self.__class__.__name__
        return f"{calendar_name}({props})"

    def visualize(
        self,
        n_years: int = 3,
        add_freq: bool = False,
    ) -> None:
        """Plots a visualization of the current calendar setup, to aid in user setup.

        Args:
            add_freq: Toggles if the frequency of the intervals should be displayed.
                      Defaults to False (Matplotlib plotter only)
            n_years: Sets the maximum number of anchor years that should be shown. By
                     default only the most recent 3 are visualized, to ensure that they
                     fit within the plot.
        """
        n_years = max(n_years, 1)
        n_years = min(n_years, len(self.get_intervals().index))
        _plot.matplotlib_visualization(self, n_years, add_freq)

    def visualize_interactive(
        self,
        relative_dates: bool,
        n_years: int = 3,
    ) -> None:
        """Plots a visualization of the current calendar setup using `bokeh`.

        Note: Requires the `bokeh` package to be installed in the active enviroment.

        Args:
            relative_dates: If False, absolute dates will be used. If True, each anchor
                            year is aligned by the anchor date, so that all anchor years
                            line up vertically.
            n_years: Sets the maximum number of anchor years that should be shown. By
                     default only the most recent 3 are visualized, to ensure that they
                     fit within the plot.

        Returns:
            None
        """
        if utils.bokeh_available():
            # pylint: disable=import-outside-toplevel
            from ._bokeh_plots import bokeh_visualization
            return bokeh_visualization(self, n_years, relative_dates)
        return None

    @property
    def flat(self) -> pd.DataFrame:
        """Returns the flattened intervals."""
        return self.get_intervals().stack()  # type: ignore


class Period(ABC):
    """Basic construction element of calendar for defining target period."""

    def __init__(self, length: str, gap: str = "0d", target: bool = False) -> None:
        self.length = pd.Timedelta(length)
        self.gap = pd.Timedelta(gap)
        self.target = target
        # TO DO: support lead_time
        # self.lead_time = lead_time


class TargetPeriod(Period):
    """Instantiate a build block as target period."""

    def __init__(self, length: str, gap: str = "0d") -> None:
        self.length = pd.Timedelta(length)
        self.gap = pd.Timedelta(gap)
        self.target = True


class PrecursorPeriod(Period):
    """Instantiate a build block as precursor period."""

    def __init__(self, length: str, gap: str = "0d") -> None:
        self.length = pd.Timedelta(length)
        self.gap = pd.Timedelta(gap)
        self.target = False
