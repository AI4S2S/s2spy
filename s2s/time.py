"""AI4S2S time utils.

Utilities designed to aid in seasonal-to-subseasonal prediction experiments in
which we search for skillful predictors preceding a certain event of interest.

Time indices are anchored to the target period of interest. By keeping keeping
observations from the same cycle (typically 1 year) together and paying close
attention to the treatment of adjacent cycles, we avoid information leakage
between train and test sets.
"""
import pandas as pd


class AdventCalendar:
    """Countdown time to anticipated anchor date or period of interest."""

    def __init__(self, anchor_date=(11, 30), freq="7d"):
        """Instantiate a basic calendar with minimal configuration.

        Set up the calendar with given freq ending exactly on the anchor date. The
        index will extend back in time as many periods as fit within the cycle
        time.
        """
        self.month = anchor_date[0]
        self.day = anchor_date[1]
        self.freq = freq
        self.n = pd.Timedelta("365days") // pd.to_timedelta(freq)

    def map_to_data(input_data):
        """Map the calendar to input data."""

        raise NotImplementedError

    def map_year(self, year):
        """Return a concrete IntervalIndex for the given year."""
        anchor = pd.Timestamp(year, self.month, self.day)
        intervals = pd.interval_range(end=anchor, periods=self.n, freq=self.freq)
        intervals = pd.Series(intervals[::-1], name=str(year))
        intervals.index = intervals.index.map(lambda i: f"t-{i}")
        return intervals

    def map_years(self, start=1979, end=2020, flat=False):
        """Return a periodic IntervalIndex for the given years."""
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

    def discard(self, max_lag):
        """Only keep indices up to the given max lag."""
        # or think of a nicer way to discard unneeded info
        raise NotImplementedError

    def mark_target_period(self, start=None, end=None, periods=None): # we can drop end since we have anchor date
        """Mark indices that fall within the target period."""
        # eg in pd.period_range you have to specify 2 of 3 (start/end/periods)
        if start and end:
            pass
        elif start and periods:
            pass
        elif end and periods:
            pass
        else:
            raise ValueError("Of start/end/periods, specify exactly 2")
        raise NotImplementedError

    def _get_resample_bins(self, input_data):
        """Label bins for resampling."""


    def resample(self, input_data):
        """Resample input data onto this Index' axis.

        Pass in pandas dataframe or xarray object with a datetime axis.
        It will return the same object with the datetimes resampled onto
        this DateTimeIndex.
        """
        bins = self._get_resample_bins(input_data)
        resample_data = input_data.groupby_bins(bins) # check pandas/xarray

        return resample_data.mean() #check agg() in panadas and xarray

    def get_lagged_indices(self, lag=1):  # noqa
        """Return indices shifted backward by given lag."""
        raise NotImplementedError

    def get_train_indices(self, strategy, params):  # noqa
        """Return indices for training data indices using given strategy."""
        raise NotImplementedError

    def get_test_indices(self, strategy, params):  # noqa
        """Return indices for test data indices using given strategy."""
        raise NotImplementedError

    def get_train_test_indices(self, strategy, params):  # noqa
        """Shorthand for getting both train and test indices."""
        train = self.get_train_sets()
        test = self.get_test_sets()
        return train, test
