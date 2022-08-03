"""BaseCalendar is a template for specific implementations of different calendars.
It includes most methods required for all calendar operations, except a specific __init__
and map_year method. These two have to be customized for each specific calendar.
"""
from typing import Optional
from typing import Union
import pandas as pd
import xarray as xr
from s2spy import traintest


PandasData = (pd.Series, pd.DataFrame)
XArrayData = (xr.DataArray, xr.Dataset)


class BaseCalendar:
    def __init__(self):
        self._nintervals = None
        self.n_targets = None
        self._traintest = None
        self.intervals = None
        self._skip_year = None

    def map_years(self, start: int = 1979, end: int = 2020) -> pd.DataFrame:
        """Return a periodic IntervalIndex for the given years.
        If the start and end years are the same, the Intervals for only that single
        year are returned.

        Args:
            start: The first year for which the calendar will be realized
            end: The last year for which the calendar will be realized

        Returns:
            Pandas DataFrame filled with Intervals of the calendar's frequency,
            counting backwards from the calendar's anchor_date.

        Example:

            >>> import s2spy.time
            >>> calendar = s2spy.time.AdventCalendar(anchor_date=(12, 31), freq='180d')
            >>> # note the leap year:
            >>> calendar.map_years(2020, 2022) # doctest: +NORMALIZE_WHITESPACE
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
        year_range = range(end, start - 1, -(self._skip_years + 1))

        self.intervals = pd.concat(
            [self._map_year(year) for year in year_range], axis=1
        ).T

        self.intervals.index.name = "anchor_year"

        return self

    def map_to_data(
        self,
        input_data: Union[pd.Series, pd.DataFrame, xr.Dataset, xr.DataArray],
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
            self.map_years(map_first_year, map_last_year)
        else:
            raise ValueError(
                "The input data could not cover the target advent calendar."
            )

        return self

    def __str__(self):
        return f"{self._nintervals} periods of {self.freq} leading up to {self.month}/{self.day}."

    def __repr__(self):
        if self.intervals is not None:
            return repr(self._label_targets(self.intervals))

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"AdventCalendar({props})"

    def _repr_html_(self):
        """For jupyter notebook to load html compatiable version of __repr__."""
        if self.intervals is not None:
            # pylint: disable=protected-access
            return self._label_targets(self.intervals)._repr_html_()

        props = ", ".join(
            [f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        )
        return f"AdventCalendar({props})"

    @property
    def flat(self):
        if self.intervals is not None:
            return self.intervals.stack()
        raise ValueError(
            "The calendar is not initialized with intervals yet."
            "use `map_years` or `map_to_data` methods to set it up."
        )

    def discard(self, max_lag):  # or "set_max_lag"
        """Only keep indices up to the given max lag."""
        # or think of a nicer way to discard unneeded info
        raise NotImplementedError

    def get_lagged_indices(self, lag=1):  # noqa
        """Return indices shifted backward by given lag."""
        raise NotImplementedError

    @property
    def traintest(self) -> pd.DataFrame:
        """Shorthand for getting both train and test indices.

        The user must first set a method using `set_traintest_method`. Then after calling
        this, the user will get a intervals list similar to the output of `map_years`
        or `map_data`, but with extra columns containing `train` and `test` labels.

        Returns:
            Pandas DataFrame with columns specifying whether the interval is
                part of the train or test datasets.

        Example:

            >>> import s2spy.time
            >>> calendar = s2spy.time.AdventCalendar(anchor_date=(10, 15), freq='180d')
            >>> calendar.map_years(2020, 2021).flat
            anchor_year  i_interval
            2021         0             (2021-04-18, 2021-10-15]
                         1             (2020-10-20, 2021-04-18]
            2020         0             (2020-04-18, 2020-10-15]
                         1             (2019-10-21, 2020-04-18]
            dtype: interval

            >>> calendar.set_traintest_method("kfold", n_splits = 2)
            >>> traintest_group = calendar.traintest
            >>> traintest_group # doctest: +NORMALIZE_WHITESPACE
                                                              0                         1
            anchor_year fold_0 fold_1
            2021        test   train   (2021-04-18, 2021-10-15]  (2020-10-20, 2021-04-18]
            2020        train  test    (2020-04-18, 2020-10-15]  (2019-10-21, 2020-04-18]
        """
        if self._traintest is None:
            raise RuntimeError("The train/test splitting method has not been set yet.")
        df_combined = self.intervals.join(self._traintest)
        new_index_cols = [self.intervals.index.name] + list(self._traintest.columns)
        return df_combined.reset_index().set_index(new_index_cols)  # .stack()

    def set_traintest_method(
        self, method: str, overwrite: bool = False, **method_kwargs: Optional[dict]
    ):
        """
        Configure the train/test splitting strategy for this calendar instance.
        The list of train/test splitting methods supported by this function can be found:
        https://ai4s2s.readthedocs.io/en/latest/autoapi/s2spy/traintest/index.html

        Args:
            method: one of the methods available in `s2spy.traintest`
            method_kwargs: keyword arguments that will be passed to `method`
        """
        if self._traintest is not None:
            if overwrite:
                pass
            else:
                raise ValueError(
                    "The traintest method has already been set. Set `overwrite = True` if you want to overwrite it."
                )

        func = traintest.ALL_METHODS.get(method)
        if not func:
            raise ValueError("The given method is not supported by `s2spy.traintest`.")

        self._traintest = func(self.intervals, **method_kwargs)
