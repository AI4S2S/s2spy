"""BaseCalendar is a template for specific implementations of different calendars.

The BaseCalendar includes most methods required for all calendar operations, except for
a set of abstract methods (e.g., __init__, _get_anchor, ...). These will have to be
customized for each specific calendar.
"""
from abc import ABC
from abc import abstractmethod
from calendar import month_abbr
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from . import utils


PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)


def plot_interval(
    anchor_date: pd.Timestamp, interval: pd.Interval, ax: plt.Axes, color: str
):
    """Utility for the calendar visualization.

    Plots a rectangle representing a single interval.

    Args:
        anchor_date: Pandas timestamp representing the anchor date.
        interval: Interval that should be added to the plot.
        ax: Axis to plot the interval in.
        color: (Matplotlib compatible) color that the rectangle should have.
    """
    right = (anchor_date - interval.right).days
    hwidth = (interval.right - interval.left).days

    ax.add_patch(
        Rectangle(
            (right, anchor_date.year - 0.4),
            hwidth,
            0.8,
            facecolor=color,
            alpha=1,
            edgecolor="k",
            linewidth=1.5,
        )
    )


class BaseCalendar(ABC):
    """Base calendar class which serves as a template for specific implementations."""

    _mapping = None

    @abstractmethod
    def __init__(
        self,
        anchor,
        freq,
        n_targets: int = 1,
    ):
        """For initializing calendars, the following five variables will be required."""
        self.n_targets = n_targets
        self.anchor = anchor
        self.freq = freq

        self._max_lag = 0
        self._allow_overlap = False

    @abstractmethod
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
        return pd.Timestamp()

    def set_max_lag(self, max_lag: int, allow_overlap: bool = False) -> None:
        """Set the maximum lag of a calendar.

        Sets the maximum number of lag periods after the target period. If `0`,
        the maximum lag will be determined by how many fit in each anchor year.
        If a maximum lag is provided, the intervals can either only cover part
        of the year, or extend over multiple years. In case of a large max_lag
        number where the intervals extend over multiple years, anchor years will
        be skipped to avoid overlapping intervals. To allow overlapping
        intervals, use the `allow_overlap` kwarg.

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
        year_intervals = pd.interval_range(
            end=self._get_anchor(year),
            periods=self._get_nintervals(),
            freq=self.freq,
        )

        year_intervals = pd.Series(year_intervals[::-1], name=str(year))
        year_intervals.index.name = "i_interval"
        return year_intervals

    def _get_nintervals(self) -> int:
        """Calculates the number of intervals that should be generated by _map year.

        Returns:
            int: Number of intervals for one anchor year.
        """
        periods_per_year = pd.Timedelta("365days") / pd.to_timedelta(self.freq)
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
        nintervals = self._get_nintervals()
        periods_per_year = pd.Timedelta("365days") / pd.to_timedelta(self.freq)

        return (
            0
            if self._max_lag > 0 and self._allow_overlap
            else int(np.ceil(nintervals / periods_per_year).astype(int) - 1)
        )

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
        if self._mapping == "data":
            self._set_year_range_from_timestamps()

        year_range = range(
            self._last_year, self._first_year - 1, -(self._get_skip_nyears() + 1)
        )

        intervals = pd.concat([self._map_year(year) for year in year_range], axis=1).T

        intervals.index.name = "anchor_year"
        return intervals

    def show(self) -> pd.DataFrame:
        """Displays the intervals the Calendar will generate for the current setup.

        Returns:
            pd.Dataframe: Dataframe containing the calendar intervals, with the target
                periods labelled.
        """
        return self._label_targets(self.get_intervals())

    def __repr__(self) -> str:
        """String representation of the Calendar."""
        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        calendar_name = self.__class__.__name__
        return f"{calendar_name}({props})"

    def visualize(self) -> None:
        """Plots a visualization of the current calendar setup, to aid in user setup."""
        intervals = self.get_intervals()

        _, ax = plt.subplots()

        for year_intervals in intervals.values:
            anchor_date = year_intervals[0].right

            # Plot the anchor intervals
            for interval in year_intervals[0 : self.n_targets : 2]:
                plot_interval(anchor_date, interval, ax=ax, color="tab:orange")
            for interval in year_intervals[1 : self.n_targets : 2]:
                plot_interval(anchor_date, interval, ax=ax, color="tab:red")

            # Plot the precursor intervals
            for interval in year_intervals[self.n_targets :: 2]:
                plot_interval(anchor_date, interval, ax=ax, color="tab:blue")
            for interval in year_intervals[self.n_targets + 1 :: 2]:
                plot_interval(anchor_date, interval, ax=ax, color="tab:cyan")

        left_bound = (anchor_date - intervals.values[-1][-1].left).days
        ax.set_xlim([left_bound + 5, -5])
        ax.set_xlabel(
            f"Days before anchor date ({anchor_date.day}"
            f" {month_abbr[anchor_date.month]})"
        )

        anchor_years = intervals.index.astype(int).values
        ax.set_ylim([anchor_years.min() - 0.5, anchor_years.max() + 0.5])
        ax.set_yticks(anchor_years)
        ax.set_ylabel("Anchor year")

        # Add a custom legend to explain to users what the colors mean
        legend_elements = [
            Patch(
                facecolor="tab:orange",
                edgecolor="tab:red",
                label="Target interval",
                hatch="//",
                linewidth=1.5,
            ),
            Patch(
                facecolor="tab:cyan",
                edgecolor="tab:blue",
                label="Precursor interval",
                hatch="//",
                linewidth=1.5,
            ),
        ]
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))

    @property
    def flat(self) -> pd.DataFrame:
        """Returns the flattened intervals."""
        return self.get_intervals().stack()
