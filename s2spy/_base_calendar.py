"""BaseCalendar is a template for specific implementations of different calendars.

The BaseCalendar includes most methods required for all calendar operations, except for
a set of abstract methods (e.g., __init__, _get_anchor, ...). These will have to be
customized for each specific calendar.
"""
import copy
import re
import warnings
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import DateOffset
from . import _plot
from . import utils


MappingYears = Tuple[Literal["years"], int, int]
MappingData = Tuple[Literal["data"], pd.Timestamp, pd.Timestamp]

PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)


class BaseCalendar(ABC):
    """Base calendar class which serves as a template for specific implementations."""

    _mapping = None

    @abstractmethod
    def __init__(
        self,
        anchor: str,
    ) -> None:
        """For initializing calendars, the following five variables will be required."""
        self._anchor, self._anchor_fmt = self._parse_anchor(anchor)
        self.targets: List[Interval] = []
        self.precursors: List[Interval] = []

        self._first_year: Union[None, int] = None
        self._last_year: Union[None, int] = None

        self.n_targets = 0
        self._max_lag: int = 0
        self._allow_overlap: bool = False

    @property
    def anchor(self):
        "Makes anchor a property so it easier to access."
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        self._anchor, self._anchor_fmt = self._parse_anchor(value)

    @property
    def allow_overlap(self):
        return self._allow_overlap

    @allow_overlap.setter
    def allow_overlap(self, value: bool):
        if isinstance(value, bool):
            self._allow_overlap = value
        else:
            raise ValueError(
                f"allow_overlap should be either True or False, not {value}"
                f"of type {type(value)}"
            )

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
        # non string check
        if not isinstance(anchor_str, str):
            raise ValueError("Anchor input must be a string with expected format.")
        # format match
        if re.fullmatch("\\d{1,2}-\\d{1,2}", anchor_str):
            utils.check_month_day(*[int(x) for x in anchor_str.split("-")])
            fmt = "%m-%d"
        elif re.fullmatch("\\d{1,2}", anchor_str):
            utils.check_month_day(int(anchor_str))
            fmt = "%m"
        elif re.fullmatch("W\\d{1,2}-\\d", anchor_str):
            utils.check_week_day(*[int(x) for x in anchor_str[1:].split("-")])
            fmt = "W%W-%w"
        elif re.fullmatch("W\\d{1,2}", anchor_str):
            utils.check_week_day(int(anchor_str[1:]))
            fmt = "W%W-%w"
            anchor_str += "-1"
        elif anchor_str.lower() in utils.get_month_names():
            anchor_str = str(utils.get_month_names()[anchor_str.lower()])
            fmt = "%m"
        else:
            raise ValueError(
                f"Anchor input '{anchor_str}' does not match expected format."
            )
        return anchor_str, fmt

    def _append(self, interval):
        """Append target/precursor periods to the calendar."""
        if interval.is_target:
            self.targets.append(interval)
        else:
            self.precursors.append(interval)

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
        intervals_target = self._concatenate_periods(year, self.targets, True)
        intervals_precursor = self._concatenate_periods(year, self.precursors, False)

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
                left_date += block.gap_dateoffset
                right_date = left_date + block.length_dateoffset
                intervals.append(pd.Interval(left_date, right_date, closed="left"))
                # update left date
                left_date = right_date
        else:
            # build from right to left
            right_date = self._get_anchor(year)
            # loop through all the building blocks to
            for block in list_periods:
                right_date -= block.gap_dateoffset
                left_date = right_date - block.length_dateoffset
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
        if self._allow_overlap:
            return 0

        proto_year = 2000
        skip_years = 0

        start_calendar = self._get_anchor(proto_year)
        for prec in self.precursors:
            start_calendar -= prec.gap_dateoffset
            start_calendar -= prec.length_dateoffset

        while True:
            prev_end_calendar = self._get_anchor(proto_year - 1 - skip_years)
            for target in self.targets:
                prev_end_calendar += target.gap_dateoffset
                prev_end_calendar += target.length_dateoffset
            if prev_end_calendar > start_calendar:
                skip_years += 1
            else:
                break

        return skip_years

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

        self._first_timestamp = None
        self._last_timestamp = None

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
        self._first_year = None
        self._last_year = None

        return self

    def _set_year_range_from_timestamps(self):
        min_year = self._first_timestamp.year  # type: ignore
        max_year = self._last_timestamp.year  # type: ignore

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

    def _set_mapping(self, mapping):
        if mapping is None:
            pass  # NOSONAR
        elif mapping[0] == "years":
            self.map_years(mapping[1], mapping[2])
        elif mapping[0] == "data":
            self._mapping = "data"
            self._first_timestamp = mapping[1]
            self._last_timestamp = mapping[2]
        else:
            raise ValueError(
                "Unknown mapping passed to calendar. Valid options are"
                "either 'years' or 'data'."
            )

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
            self._last_year, self._first_year - 1, -(self._get_skip_nyears() + 1)  # type: ignore
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

    #  pylint: disable=too-many-arguments
    def visualize(
        self,
        n_years: int = 3,
        interactive: bool = False,
        relative_dates: bool = False,
        show_length: bool = False,
        add_legend: bool = True,
        ax=None,
        **bokeh_kwargs,
    ) -> None:
        """Plots a visualization of the current calendar setup, to aid in user setup.

        Note: The interactive visualization requires the `bokeh` package to be installed
        in the active Python environment.

        Args:
            n_years: Sets the maximum number of anchor years that should be shown. By
                default only the most recent 3 are visualized, to ensure that they
                fit within the plot.
            interactive: If False, matplotlib will be used for the visualization. If
                True, bokeh will be used.
            relative_dates: Toggles if the intervals should be displayed relative to the
                anchor date, or as absolute dates.
            show_length: Toggles if the frequency of the intervals should be displayed.
                Defaults to False (Matplotlib plotter only).
            add_legend: Toggles if a legend should be added to the plot (Matplotlib only)
            ax: Matplotlib axis object to plot the visualization into.
            **bokeh_kwargs: Keyword arguments to pass to Bokeh's plotting.Figure. See
                https://docs.bokeh.org/en/2.4.2/docs/reference/plotting/figure.html
                for a list of possible keyword arguments.
        """
        calendar = copy.deepcopy(self)
        if calendar._mapping is None:  # pylint: disable=protected-access
            calendar.map_years(2000, 2000)
            if not relative_dates:
                print(
                    "Setting relative_dates=True. To see absolute dates, first call "
                    "calendar.map_years or calendar.map_data"
                )
                relative_dates = True
            add_yticklabels = False
        else:
            add_yticklabels = True

        n_years = max(n_years, 1)
        n_years = min(n_years, len(calendar.get_intervals().index))

        if interactive:
            utils.assert_bokeh_available()
            # pylint: disable=import-outside-toplevel
            from ._bokeh_plots import bokeh_visualization

            if ax is not None:
                warnings.warn(
                    "ax is only a valid keyword argument for the non-interactive "
                    "matplotlib backend. Bokeh's figure can be controlled by passing "
                    "Bokeh Figure keyword arguments (e.g. width=800).",
                    UserWarning,
                )
            bokeh_visualization(
                calendar, n_years, relative_dates, add_yticklabels, **bokeh_kwargs
            )
        else:
            if bokeh_kwargs:
                warnings.warn(
                    "kwargs for bokeh have been passed to visualize(), but the "
                    "matplotlib backend does not support these. Use the 'ax' kwarg "
                    "instead to control the generated figure.",
                    UserWarning,
                )

            _plot.matplotlib_visualization(
                calendar,
                n_years,
                relative_dates,
                show_length,
                add_legend,
                add_yticklabels,
                ax=ax,
            )

    @property
    def flat(self) -> pd.DataFrame:
        """Returns the flattened intervals."""
        return self.get_intervals().stack()  # type: ignore


class Interval:
    """Basic construction element of calendar for defining precursors and targets."""

    def __init__(
        self,
        role: Literal["target", "precursor"],
        length: Union[str, dict],
        gap: Union[str, dict] = "0d",
    ) -> None:
        """This is the basic construction element of the calendar.

        The Interval is characterised by its type (either target or precursor), its
        length and the gap between it and the previous interval of its type (or the
        anchor date, if the interval is the first target/first precursor).

        Args:
            role: The type of interval. Either "target" or "precursor".
            length: The length of the interval. This can either be a pandas-like
                frequency string (e.g. "10d", "2W", or "3M"), or a pandas.DateOffset
                compatible dictionary such as {days=10}, {weeks=2}, or
                {months=1, weeks=2}.
            gap: The gap between the previous interval and this interval. Valid inputs
                are the same as the length keyword argument. Defaults to "0d".
        """
        self.length = length
        self.gap = gap
        self._role = role
        self._target = role == "target"

        self._gap_dateoffset: pd.DateOffset
        self._length_dateoffset: pd.DateOffset

        # TO DO: support lead_time
        # self.lead_time = lead_time

    @property
    def is_target(self):
        return self._target

    @property
    def role(self):
        "Returns the type of interval."
        return self._role

    @property
    def length(self):
        "Returns the length of the interval, as a pandas.DateOffset."
        return self._length

    @length.setter
    def length(self, value: Union[str, dict]):
        self._length = value
        if isinstance(value, str):
            self._length_dateoffset = DateOffset(
                **utils.parse_freqstr_to_dateoffset(value)
            )
        else:
            self._length_dateoffset = DateOffset(**value)

    @property
    def length_dateoffset(self):
        return self._length_dateoffset

    @property
    def gap(self):
        """Returns the gap of the interval, as a pandas.DateOffset."""
        return self._gap

    @gap.setter
    def gap(self, value: Union[str, dict]):
        self._gap = value
        if isinstance(value, str):
            self._gap_dateoffset = DateOffset(
                **utils.parse_freqstr_to_dateoffset(value)
            )
        else:
            self._gap_dateoffset = DateOffset(**value)

    @property
    def gap_dateoffset(self):
        return self._gap_dateoffset

    def __repr__(self):
        """String representation of the Interval class."""
        props = [
            ("role", self.role),
            ("length", self.length),
            ("gap", self.gap),
        ]

        propstr = ", ".join([f"{k}={repr(v)}" for k, v in props])
        return f"{self.__class__.__name__}({propstr})"
