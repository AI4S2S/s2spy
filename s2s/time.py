"""AI4S2S time utils.

Utilities designed to aid in seasonal-to-subseasonal prediction experiments in
which we search for skillful predictors preceding a certain event of interest.

Time indices are anchored to the target period of interest. By keeping keeping
observations from the same cycle (typically 1 year) together and paying close
attention to the treatment of adjacent cycles, we avoid information leakage
between train and test sets.

Example:

    >>> import s2s.time
    >>>
    >>> # Countdown the weeks until New Year's Eve
    >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq="7d")
    >>> calendar
    AdventCalendar(month=12, day=31, freq=7d, n=52)
    >>> print(calendar)
    52 periods of 7d leading up to 12/31.

    >>> # Get the 60-day periods leading up to New Year's eve for the year 2020
    >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq='60d')
    >>> calendar.map_year(2020)
    t-0    (2020-11-01, 2020-12-31]
    t-1    (2020-09-02, 2020-11-01]
    t-2    (2020-07-04, 2020-09-02]
    t-3    (2020-05-05, 2020-07-04]
    t-4    (2020-03-06, 2020-05-05]
    t-5    (2020-01-06, 2020-03-06]
    Name: 2020, dtype: interval

    >>> # Get the 180-day periods leading up to New Year's eve for 2020 - 2022 inclusive.
    >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq='180d')
    >>> # note the leap year:
    >>> calendar.map_years(2020, 2022)
                               t-0                       t-1
    2022  (2022-07-04, 2022-12-31]  (2022-01-05, 2022-07-04]
    2021  (2021-07-04, 2021-12-31]  (2021-01-05, 2021-07-04]
    2020  (2020-07-04, 2020-12-31]  (2020-01-06, 2020-07-04]

    >>> # To get a stacked representation:
    >>> calendar.map_years(2020, 2022, flat=True)
    0    (2022-07-04, 2022-12-31]
    1    (2022-01-05, 2022-07-04]
    2    (2021-07-04, 2021-12-31]
    3    (2021-01-05, 2021-07-04]
    4    (2020-07-04, 2020-12-31]
    5    (2020-01-06, 2020-07-04]
    dtype: interval

"""
from typing import Optional, Tuple
import pandas as pd

from s2s import traintest

class AdventCalendar:
    """Countdown time to anticipated anchor date or period of interest."""

    def __init__(
        self, anchor_date: Tuple[int, int] = (11, 30), freq: str = "7d"
    ) -> None:
        """Instantiate a basic calendar with minimal configuration.

        Set up the calendar with given freq ending exactly on the anchor date.
        The index will extend back in time as many periods as fit within the
        cycle time of one year.

        Args:
            anchor_date: Tuple of the form (month, day). Effectively the origin
                of the calendar. It will countdown until this date. freq:
                Frequency of the calendar.

        Example:
            Instantiate a calendar counting down the weeks until new-year's
            eve.

            >>> import s2s.time
            >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq="7d")
            >>> calendar
            AdventCalendar(month=12, day=31, freq=7d, n=52)
            >>> print(calendar)
            "52 periods of 7d leading up to 12/31."

        """
        self.month = anchor_date[0]
        self.day = anchor_date[1]
        self.freq = freq
        self.n = pd.Timedelta("365days") // pd.to_timedelta(freq)

    def map_year(self, year: int) -> pd.Series:
        """Return a concrete IntervalIndex for the given year.

        Since the AdventCalendar represents a periodic event, it is first
        instantiated without a specific year. This method adds a specific year
        to the calendar and returns an intervalindex, applying the
        AdvenctCalendar to this specific year.

        Args:
            year: The year for which the AdventCalendar will be realized

        Returns:
            Pandas Series filled with Intervals of the calendar's frequency, counting
            backwards from the calendar's anchor_date.

        Example:

            >>> import s2s.time
            >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq='60d')
            >>> calendar.map_year(2020)
            t-0    (2020-11-01, 2020-12-31]
            t-1    (2020-09-02, 2020-11-01]
            t-2    (2020-07-04, 2020-09-02]
            t-3    (2020-05-05, 2020-07-04]
            t-4    (2020-03-06, 2020-05-05]
            t-5    (2020-01-06, 2020-03-06]
            Name: 2020, dtype: interval
        """
        anchor = pd.Timestamp(year, self.month, self.day)
        intervals = pd.interval_range(end=anchor, periods=self.n, freq=self.freq)
        intervals = pd.Series(intervals[::-1], name=str(year))
        intervals.index = intervals.index.map(lambda i: f"t-{i}")
        return intervals

    def map_years(
        self, start: int = 1979, end: int = 2020, flat: bool = False
    ) -> pd.DataFrame:
        """Return a periodic IntervalIndex for the given years.

        Like ``map_year``, but for multiple years.

        Args:
            start: The first year for which the calendar will be realized
            end: The last year for which the calendar will be realized
            flat: If False, years are rows and lag times are columns in the
                dataframe. If True, years and lags are stacked to form a
                continous index.

        Returns:
            Pandas DataFrame filled with Intervals of the calendar's frequency,
            counting backwards from the calendar's anchor_date.

        Example:

            >>> import s2s.time
            >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq='180d')
            >>> # note the leap year:
            >>> calendar.map_years(2020, 2022)
                                       t-0                       t-1
            2022  (2022-07-04, 2022-12-31]  (2022-01-05, 2022-07-04]
            2021  (2021-07-04, 2021-12-31]  (2021-01-05, 2021-07-04]
            2020  (2020-07-04, 2020-12-31]  (2020-01-06, 2020-07-04]

            >>> # To get a stacked representation:
            >>> calendar.map_years(2020, 2022, flat=True)
            0    (2022-07-04, 2022-12-31]
            1    (2022-01-05, 2022-07-04]
            2    (2021-07-04, 2021-12-31]
            3    (2021-01-05, 2021-07-04]
            4    (2020-07-04, 2020-12-31]
            5    (2020-01-06, 2020-07-04]
            dtype: interval
        """
        index = pd.concat(
            [self.map_year(year) for year in range(start, end + 1)], axis=1
        ).T[::-1]

        if flat:
            return index.stack().reset_index(drop=True)

        return index

    def __str__(self):
        return f"{self.n} periods of {self.freq} leading up to {self.month}/{self.day}."

    def __repr__(self):
        props = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"AdventCalendar({props})"

    def discard(self, max_lag):   # or "set_max_lag"
        """Only keep indices up to the given max lag."""
        # or think of a nicer way to discard unneeded info
        raise NotImplementedError

    def mark_target_period(self, start=None, periods=None):
        """Mark indices that fall within the target period."""
        # eg in pd.period_range you have to specify 2 of 3 (start/end/periods)
        if start is not None:
            pass
        elif periods:
            pass
        else:
            raise ValueError("Of start/end/periods, specify exactly 2")

        raise NotImplementedError

    def resample(self, input_data):
        """Resample input data onto this Index' axis.

        Pass in pandas dataframe or xarray object with a datetime axis.
        It will return the same object with the datetimes resampled onto
        this DateTimeIndex.
        """
        raise NotImplementedError

    def get_lagged_indices(self, lag=1):  # noqa
        """Return indices shifted backward by given lag."""
        raise NotImplementedError

    def get_traintest(self, method: str, method_kwargs: Optional[dict]) -> pd.DataFrame:
        """Shorthand for getting both train and test indices.

        Args:
            method: one of the methods available in s2s.traintest
            method_kwargs: keyword arguments that will be passed to `method`

        Returns:
            Pandas DataFrame with an column specifying whether the interval is
                part of the train or test datasets.

        Example:

            >>> import s2s.time
            >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq='180d')


        """
        # TODO: overwrite if method is different?
        # TODO: need to store self._df internally after calling map_years/map_data
        # TODO: implement tests
        if self._traintest is not None:
            self._traintest = traintest.ALL_METHODS[method](self._df, **method_kwargs)
        return self._traintest

    def get_train(self, method: str, method_kwargs: Optional[dict]) -> pd.DataFrame:
        """Return indices for training data indices using given strategy."""
        return self.get_traintest.query(label='train')

    def get_test(self, method: str, method_kwargs: Optional[dict]) -> pd.DataFrame:
        """Return indices for test data indices using given strategy."""
        return self.get_traintest.query(label='test')
