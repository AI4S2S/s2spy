"""AI4S2S time utils.

Utilities designed to aid in seasonal-to-subseasonal prediction experiments in
which we search for skillful predictors preceding a certain event of interest.

Time indices are anchored to the target period of interest. By keeping keeping
observations from the same cycle (typically 1 year) together and paying close
attention to the treatment of adjacent cycles, we avoid information leakage
between train and test sets.
"""


class TimeIndex:
    """TimeIndex anchored to a date or period of interest."""

    def __init__(self, anchor_date='20201130', freq='7d', cycle_time='1yr'):
        """Instantiate a basic index with minimal configuration.

        Set up the index with given freq ending exactly on the anchor date. The
        index will extend back in time as many periods as fit within the cycle
        time.
        """
        # set self.index or self.dataframe or something like that
        raise NotImplementedError

    def __str__(self):
        """Return nicely formatted representation of self."""
        return f"I'm {self}"

    def discard(self, max_lag):
        """Only keep indices up to the given max lag."""
        # or think of a nicer way to discard unneeded info
        raise NotImplementedError

    def mark_target_period(self, start, end, periods):
        """Mark indices that fall within the target period."""
        # eg in pd.period_range you have to specify 2 of 3 (start/end/periods)
        raise NotImplementedError

    def resample(self, input_data):
        """Resample input data onto this Index' axis.

        Pass in pandas dataframe or xarray object with a datetime axis.
        It will return the same object with the datetimes resampled onto
        this DateTimeIndex.
        """
        raise NotImplementedError

    def get_lagged_indices(self, lag=1):
        """Return indices shifted backward by given lag."""

    def get_train_indices(self, strategy, params):
        """Return indices for training data indices using given strategy."""
        raise NotImplementedError

    def get_test_indices(self, strategy, params):
        """Return indices for test data indices using given strategy."""
        raise NotImplementedError

    def get_train_test_indices(self, strategy, params):
        """Shorthand for getting both train and test indices."""
        train = self.get_train_sets()
        test = self.get_test_sets()
        return train, test
