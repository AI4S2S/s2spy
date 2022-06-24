"""AI4S2S time utils.

Utilities designed to aid in seasonal-to-subseasonal prediction experiments in
which we search for skillful predictors preceding a certain event of interest.

Time indices are anchored to the target period of interest. By keeping
observations from the same cycle (typically 1 year) together and paying close
attention to the treatment of adjacent cycles, we avoid information leakage
between train and test sets.

Example:

    >>> import s2s.time
    >>>
    >>> # Countdown the weeks until New Year's Eve
    >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq="7d")
    >>> calendar
    AdventCalendar(month=12, day=31, freq=7d)
    >>> print(calendar)
    52 periods of 7d leading up to 12/31.

    >>> # Get the 180-day periods leading up to New Year's eve for the year 2020
    >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq='180d')
    >>> calendar.map_years(2020, 2020)
    i_interval                          0                         1
    anchor_year
    2020         (2020-07-04, 2020-12-31]  (2020-01-06, 2020-07-04]

    >>> # Get the 180-day periods leading up to New Year's eve for 2020 - 2022 inclusive.
    >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq='180d')
    >>> # note the leap year:
    >>> calendar.map_years(2020, 2022)
    i_interval                          0                         1
    anchor_year
    2022         (2022-07-04, 2022-12-31]  (2022-01-05, 2022-07-04]
    2021         (2021-07-04, 2021-12-31]  (2021-01-05, 2021-07-04]
    2020         (2020-07-04, 2020-12-31]  (2020-01-06, 2020-07-04]

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
import warnings
from typing import Optional
from typing import Tuple
from typing import Union
import pandas as pd
import xarray as xr


PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)

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
                of the calendar. It will countdown until this date.
            freq: Frequency of the calendar.

        Example:
            Instantiate a calendar counting down the weeks until new-year's
            eve.

            >>> import s2s.time
            >>> calendar = s2s.time.AdventCalendar(anchor_date=(12, 31), freq="7d")
            >>> calendar
            AdventCalendar(month=12, day=31, freq=7d)
            >>> print(calendar)
            "52 periods of 7d leading up to 12/31."

        """
        self.month = anchor_date[0]
        self.day = anchor_date[1]
        self.freq = freq
        self._n_intervals = pd.Timedelta("365days") // pd.to_timedelta(freq)
        self._n_target = 1

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
            end=anchor, periods=self._n_intervals, freq=self.freq)
        intervals = pd.Series(intervals[::-1], name=str(year))
        intervals.index.name = 'i_interval'
        return intervals

    def map_years(
        self, start: int = 1979, end: int = 2020, flat: bool = False
    ) -> pd.DataFrame:
        """Return a periodic IntervalIndex for the given years.
        If the start and end years are the same, the Intervals for only that single
        year are returned.

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
            i_interval                          0                         1
            anchor_year
            2022         (2022-07-04, 2022-12-31]  (2022-01-05, 2022-07-04]
            2021         (2021-07-04, 2021-12-31]  (2021-01-05, 2021-07-04]
            2020         (2020-07-04, 2020-12-31]  (2020-01-06, 2020-07-04]

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
        self.intervals = pd.concat(
            [self._map_year(year) for year in range(start, end + 1)], axis=1
        ).T[::-1]

        self.intervals.index.name = 'anchor_year'

        if flat:
            self.intervals = self.intervals.stack().reset_index(drop=True)
            return self.intervals

        return self.intervals

    def map_to_data(
        self,
        input_data: Union[pd.Series, pd.DataFrame, xr.Dataset, xr.DataArray],
        flat: bool = False
        ) -> Union[pd.DataFrame, xr.Dataset]:
        """Map the calendar to input data period.

        Get datetime range from input data and generate corresponding interval index.
        This method guarantees that the generated interval (calendar) indices would be
        covered by the input
        data.
        Args:
            input_data: Input data for datetime mapping. Its index must be either
                pandas.DatetimeIndex, or an xarray `time` coordinate with datetime
                data.
            flat: Same as the argument in ``map_years``.
        Returns:
            Pandas DataFrame or xarray Dataset filled with Intervals of the calendar's
            frequency. (see also ``map_years``)
        """
        # check the datetime order of input data
        if isinstance(input_data, PandasData):
            first_timestamp = input_data.index.min()
            last_timestamp = input_data.index.max()
            map_last_year = last_timestamp.year
            map_first_year = first_timestamp.year
        elif isinstance(input_data, XArrayData):
            first_timestamp = input_data.time.min()
            last_timestamp = input_data.time.max()
            map_last_year = last_timestamp.dt.year.values
            map_first_year = first_timestamp.dt.year.values
        else:
            raise ValueError(
                "incompatible input data format, please pass a pandas or xarray object"
            )

        # ensure that the input data could always cover the advent calendar
        # last date check
        if self._map_year(map_last_year).iloc[0].right > last_timestamp:
            map_last_year -= 1
        # first date check
        if self._map_year(map_first_year).iloc[-1].left < first_timestamp:
            map_first_year += 1

        # map year(s) and generate year realized advent calendar
        if map_last_year >= map_first_year:
            self.intervals = self.map_years(map_first_year, map_last_year, flat)
        else:
            raise ValueError(
                "The input data could not cover the target advent calendar.")

        return self.intervals

    def _resample_bins_constructor(
        self, intervals: Union[pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Restructures the interval object into a tidy DataFrame.

        Args:
            intervals: the output interval `pd.Series` or `pd.DataFrame` from the
                `map_to_data` function.

        Returns:
            Pandas DataFrame with 'anchor_year', 'i_interval', and 'interval' as
                columns.
        """
        # Make a tidy dataframe where the intervals are linked to the anchor year
        if isinstance(intervals, pd.DataFrame):
            bins = intervals.copy()
            bins.index.rename("anchor_year", inplace=True)
            bins = bins.melt(
                var_name="i_interval", value_name="interval", ignore_index=False
            )
            bins = bins.sort_values(by=["anchor_year", "i_interval"])
        else:
            # Massage the dataframe into the same tidy format for a single year
            bins = pd.DataFrame(intervals)
            bins = bins.melt(
                var_name="anchor_year", value_name="interval", ignore_index=False
            )
            bins.index.rename("i_interval", inplace=True)
        bins = bins.reset_index()

        return bins

    def _resample_pandas(
        self, input_data: Union[pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """Internal function to handle resampling of Pandas data.

        Args:
            input_data (pd.Series or pd.DataFrame): Data provided by the user to the
                `resample` function

        Returns:
            pd.DataFrame: DataFrame containing the intervals and data resampled to
                these intervals."""

        intervals = self.map_to_data(input_data, flat=False)
        bins = self._resample_bins_constructor(intervals)

        interval_index = pd.IntervalIndex(bins["interval"])
        interval_groups = interval_index.get_indexer(input_data.index)
        interval_means = input_data.groupby(interval_groups).mean()

        # drop the -1 index, as it represents data outside of all intervals
        interval_means = interval_means.loc[0:]

        if isinstance(input_data, pd.DataFrame):
            for name in input_data.keys():
                bins[name] = interval_means[name].values
        else:
            name = "mean_data" if input_data.name is None else input_data.name
            bins[name] = interval_means.values

        return bins

    def _resample_xarray(
        self, input_data: Union[xr.DataArray, xr.Dataset]
    ) -> xr.Dataset:
        """Internal function to handle resampling of xarray data.

        Args:
            input_data (xr.DataArray or xr.Dataset): Data provided by the user to the
                `resample` function

        Returns:
            xr.Dataset: Dataset containing the intervals and data resampled to
                these intervals."""

        intervals = self.map_to_data(input_data)
        bins = self._resample_bins_constructor(intervals)

        # Create the indexer to connect the input data with the intervals
        interval_index = pd.IntervalIndex(bins["interval"])
        interval_groups = interval_index.get_indexer(input_data["time"])
        interval_means = input_data.groupby(
            xr.IndexVariable("time", interval_groups)
        ).mean()
        interval_means = interval_means.rename({"time": "index"})

        # drop the indices below 0, as it represents data outside of all intervals
        interval_means = interval_means.sel(index=slice(0, None))

        # Turn the bins dataframe into an xarray object and merge the data means into it
        bins = bins.to_xarray()
        if isinstance(input_data, xr.Dataset):
            bins = xr.merge([bins, interval_means])
        else:
            if interval_means.name is None:
                interval_means = interval_means.rename("mean_values")
            bins = xr.merge([bins, interval_means])
        bins["anchor_year"] = bins["anchor_year"].astype(int)

        # Turn the anchor year and interval count into coordinates
        bins = bins.assign_coords(
            {"anchor_year": bins["anchor_year"], "i_interval": bins["i_interval"]}
        )

        return bins

    def resample(
        self, input_data: Union[pd.Series, pd.DataFrame, xr.DataArray, xr.Dataset]
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """Resample input data to the calendar frequency.

        Pass a pandas Series/DataFrame with a datetime axis, or an
        xarray DataArray/Dataset with a datetime coordinate called 'time'.
        It will return the same object with the datetimes resampled onto
        the Calendar's Index by binning the data into the Calendar's intervals
        and calculating the mean of each bin.

        Note: this function is intended for upscaling operations, which means
        the calendar frequency is larger than the original frequency of input data (e.g.
        `freq` is "7days" and the input is daily data). It supports downscaling
        operations but the user need to be careful since the returned values may contain
        "NaN".

        Args:
            input_data: Input data for resampling. For a Pandas object its index must be
                either a pandas.DatetimeIndex. An xarray object requires a dimension
                named 'time' containing datetime values.

        Raises:
            UserWarning: If the calendar frequency is smaller than the frequency of
                input data

        Returns:
            Input data resampled based on the calendar frequency, similar data format as
                given inputs.

        Example:
            Assuming the input data is pd.DataFrame containing random values with index
            from 2021-11-11 to 2021-11-01 at daily frequency.

            >>> import s2s.time
            >>> import pandas as pd
            >>> import numpy as np
            >>> cal = s2s.time.AdventCalendar(freq='180d')
            >>> time_index = pd.date_range('20191201', '20211231', freq='1d')
            >>> var = np.arange(len(time_index))
            >>> input_data = pd.Series(var, index=time_index)
            >>> bins = cal.resample(input_data)
            >>> bins
              anchor_year  i_interval                  interval  mean_data
            0        2020           0  (2020-06-03, 2020-11-30]      275.5
            1        2020           1  (2019-12-06, 2020-06-03]       95.5
            2        2021           0  (2021-06-03, 2021-11-30]      640.5
            3        2021           1  (2020-12-05, 2021-06-03]      460.5

        """
        if not isinstance(input_data, PandasData + XArrayData):
            raise ValueError("The input data is neither a pandas or xarray object")

        if isinstance(input_data, PandasData):
            if not isinstance(input_data.index, pd.DatetimeIndex):
                raise ValueError("The input data does not have a datetime index.")

            # raise a warning for upscaling
            # target frequency must be larger than the (absolute) input frequency
            if input_data.index.freq:
                input_freq = input_data.index.freq
                input_freq = input_freq if input_freq.n > 0 else -input_freq
                if pd.Timedelta(self.freq) < input_freq:
                    warnings.warn(
                        """Target frequency is smaller than the original frequency.
                        The resampled data will contain NaN values, as there is no data
                        available within all intervals."""
                    )

            return self._resample_pandas(input_data)

        # Data must be xarray
        if "time" not in input_data.dims:
            raise ValueError(
                "The input DataArray/Dataset does not contain a `time` dimension"
            )
        if not xr.core.common.is_np_datetime_like(input_data["time"].dtype):
            raise ValueError("The `time` dimension is not of a datetime format")

        return self._resample_xarray(input_data)

    def __str__(self):
        return f"{self._n_intervals} periods of {self.freq} leading up to {self.month}/{self.day}."

    def __repr__(self):
        props = ", ".join([f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith('_')])
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

    def get_lagged_indices(self, lag=1):  # noqa
        """Return indices shifted backward by given lag."""
        raise NotImplementedError

    def get_cv_groups(self):
        """Group intervals into bins.

        Group intervals into bins for each anchor year by labelling.
        args:
            intervals: output of ``map_years``.
        """
        raise NotImplementedError

    def get_traintest(self) -> pd.DataFrame:
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
        # TODO: implement tests

        # checker if generated intervals are flat.
        if self.intervals.ndim != 1:
            raise ValueError("Please set `flat = True` when calling `map_years` or `map_data`")

        # checker if the method is configured.
        if self._traintest_method is not None:
            self.traintest = traintest.ALL_METHODS[self._traintest_method](self.intervals,
                **self._method_kwargs)
        return self.traintest

    def set_traintest_method(self, method: str, **method_kwargs: Optional[dict]):
        """
        The user must choose a method here. And the method will be used by
        all traintest splitting methods.
        """
        self._traintest_method = method
        self._method_kwargs = method_kwargs

    def get_train(self) -> pd.DataFrame:
        """Return indices for training data indices using given strategy.
        """
        return self.get_traintest.query(label='train')

    def get_test(self) -> pd.DataFrame:
        """Return indices for test data indices using given strategy.
        """
        return self.get_traintest.query(label='test')
    
    def get_train_test_indices(self, strategy, params):  # noqa
        """Shorthand for getting both train and test indices."""
        # train = self.get_train_indices()
        # test = self.get_test_indices()
        raise NotImplementedError
