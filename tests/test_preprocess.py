"""Tests for the s2spy.preprocess module."""

import numpy as np
import pytest
import scipy.signal
import xarray as xr
from s2spy import preprocess
from s2spy.preprocess import Preprocessor


TEST_FILE_PATH = "./tests/test_rgdr/test_data"

# pylint: disable=protected-access
# ruff: noqa: B018
# functions are called without variable assignment for testing the error message
# therefore need to suppress flake-burger B018


class TestPreprocessMethods:
    """Test preprocess methods."""

    # Define inputs as fixtures
    @pytest.fixture
    def raw_field(self):
        return xr.open_dataset(
            f"{TEST_FILE_PATH}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc"
        ).sel(
            time=slice("2010-01-01", "2011-12-31"),
            latitude=slice(40, 30),
            longitude=slice(180, 190),
        )

    def test_check_input_data_incorrect_type(self):
        dummy_data = np.ones((3, 3))
        with pytest.raises(ValueError):
            preprocess._check_input_data(dummy_data)

    def test_check_input_data_incorrect_dims(self, raw_field):
        dummy_dataarray = raw_field["sst"][0, ...]  # drop time dimension
        with pytest.raises(ValueError):
            preprocess._check_input_data(dummy_dataarray)

    def test_unknown_method_get_trend(self, raw_field):
        with pytest.raises(ValueError):
            preprocess._get_trend(raw_field, "unknown")

    def test_get_and_subtract_linear_trend(self, raw_field):
        expected = xr.apply_ufunc(
            scipy.signal.detrend,
            raw_field,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
        ).transpose("time", ...)
        trend = preprocess._get_trend(raw_field, "linear")
        result = preprocess._subtract_trend(raw_field, "linear", trend)
        np.testing.assert_array_almost_equal(result["sst"], expected["sst"])

    def test_check_temporal_resolution(self):
        with pytest.raises(ValueError):
            preprocess._check_temporal_resolution("hourly")  # type: ignore

    def test_get_climatology_daily(self, raw_field):
        result = preprocess._get_climatology(raw_field, timescale="daily")
        expected = (
            raw_field["sst"].sel(time=slice("2010-01-01", "2010-12-31")).data
            + raw_field["sst"].sel(time=slice("2011-01-01", "2011-12-31")).data
        ) / 2
        np.testing.assert_array_almost_equal(result["sst"], expected)

    def test_get_climatology_weekly(self, raw_field):
        raw_field_weekly = raw_field.resample(time="W").mean()
        result = preprocess._get_climatology(raw_field_weekly, timescale="weekly")
        # need to consider the actual calendar week number for the expected climatology
        raw_field_weekly["time"] = raw_field_weekly["time"].dt.isocalendar().week
        expected = raw_field_weekly.groupby("time").mean()
        np.testing.assert_array_almost_equal(result["sst"], expected["sst"])

    def test_get_climatology_monthly(self, raw_field):
        raw_field_monthly = raw_field.resample(time="M").mean()
        result = preprocess._get_climatology(raw_field_monthly, timescale="monthly")
        expected = (
            raw_field_monthly["sst"].sel(time=slice("2010-01-01", "2010-12-31")).data
            + raw_field_monthly["sst"].sel(time=slice("2011-01-01", "2011-12-31")).data
        ) / 2
        np.testing.assert_array_almost_equal(result["sst"], expected)

    def test_get_climatology_wrong_timescale(self, raw_field):
        with pytest.raises(ValueError):
            preprocess._get_climatology(raw_field, timescale="hourly")  # type: ignore

    @pytest.mark.parametrize("timescale", ("weekly", "monthly"))
    def test_check_data_resolution_mismatch(self, raw_field, timescale):
        with pytest.warns(UserWarning):
            preprocess._check_data_resolution_match(raw_field, timescale)

    def test_check_data_resolution_match(self, raw_field):
        preprocess._check_data_resolution_match(raw_field, "daily")


class TestPreprocessor:
    """Test preprocessor."""

    @pytest.fixture
    def preprocessor(self, request):
        method = request.param[0]
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            timescale="daily",
            detrend=method,
            subtract_climatology=True,
        )
        return prep

    @pytest.fixture(params=[None, 1])
    def preprocessor_no_rolling(self, request):
        prep = preprocess.Preprocessor(
            rolling_window_size=request.param,
            timescale="monthly",
            detrend=None,
            subtract_climatology=True,
        )
        return prep

    @pytest.fixture
    def preprocessor_no_climatology(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            timescale="daily",
            detrend="linear",
            subtract_climatology=False,
        )
        return prep

    @pytest.fixture
    def preprocessor_no_detrend(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            timescale="daily",
            detrend=None,
            subtract_climatology=True,
        )
        return prep

    @pytest.fixture
    def raw_field(self):
        return xr.open_dataset(
            f"{TEST_FILE_PATH}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc"
        ).sel(
            time=slice("2010-01-01", "2014-12-31"),  # including leap year
            latitude=slice(40, 30),
            longitude=slice(180, 190),
        )

    def test_init(self):
        prep = preprocess.Preprocessor(
            rolling_window_size=25,
            timescale="weekly",
            detrend="linear",
            subtract_climatology=True,
        )
        assert isinstance(prep, Preprocessor)

    # pytest.mark.parametrize("preprocessor", ["linear", "polynomial"])

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_fit(self, preprocessor, raw_field):
        preprocessor.fit(raw_field)
        assert (
            preprocessor.climatology is not None
        )  # climatology with leap years are counted

    def test_fit_no_rolling(self, preprocessor_no_rolling, raw_field):
        preprocessor_no_rolling.fit(raw_field)
        assert preprocessor_no_rolling.climatology == preprocess._get_climatology(
            raw_field, timescale="daily"
        )

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_transform(self, preprocessor, raw_field):
        preprocessor.fit(raw_field)
        preprocessed_data = preprocessor.transform(raw_field)
        assert preprocessed_data is not None

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_transform_without_fit(self, preprocessor, raw_field):
        with pytest.raises(ValueError):
            preprocessor.transform(raw_field)

    def test_fit_and_transform_no_climatology(
        self, preprocessor_no_climatology, raw_field
    ):
        preprocessor_no_climatology.fit(raw_field)
        results = preprocessor_no_climatology.transform(raw_field)
        expected = preprocess._subtract_trend(
            raw_field, "linear", preprocessor_no_climatology.trend
        )
        assert results == expected

    def test_fit_and_transform_no_detrend(self, preprocessor_no_detrend, raw_field):
        preprocessor_no_detrend.fit(raw_field)
        results = preprocessor_no_detrend.transform(raw_field)
        expected = (
            raw_field.groupby("time.dayofyear") - preprocessor_no_detrend.climatology
        )
        assert results == expected

    def test_fit_and_transform_no_climatology_and_detrend(self, raw_field):
        prep = preprocess.Preprocessor(
            rolling_window_size=10,
            timescale="daily",
            detrend=None,
            subtract_climatology=False,
        )
        prep.fit(raw_field)
        results = prep.transform(raw_field)

        assert results == raw_field

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_fit_transform_ds(self, preprocessor, raw_field):
        preprocessed_data = preprocessor.fit_transform(raw_field)
        assert preprocessed_data is not None

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_fit_transform_da(self, preprocessor, raw_field):

        raw_field = raw_field.to_array().squeeze("variable").drop_vars("variable")
        raw_field.name = "da_name"
        years = np.unique(raw_field.time.dt.year.values)
        train = raw_field.sel(time=raw_field.time.dt.year.isin([years[:-1]]))
        tranform_to = raw_field.sel(time=raw_field.time.dt.year.isin([years[-2:]]))
        fit_transformed = preprocessor.fit_transform(train)
        transformed = preprocessor.transform(tranform_to)
        assert fit_transformed is not None
        assert bool(
            (
                fit_transformed.sel(time="2013-01-01")
                == transformed.sel(time="2013-01-01")
            ).all()
        )

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_trend_property_not_fit(self, preprocessor):
        with pytest.raises(ValueError, match="The preprocessor has to be fit"):
            preprocessor.trend

    def test_trend_property_no_detrend(self, preprocessor_no_detrend, raw_field):
        preprocessor_no_detrend.fit(raw_field)
        with pytest.raises(ValueError, match="Detrending is set to `None`"):
            preprocessor_no_detrend.trend

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_climatology_property_not_fit(self, preprocessor):
        with pytest.raises(ValueError, match="The preprocessor has to be fit"):
            preprocessor.climatology

    def test_trend_property_no_climatology(self, preprocessor_no_climatology):
        with pytest.raises(ValueError, match="subtract_climatology is set to `False`"):
            preprocessor_no_climatology.climatology

    def test_trend_with_nan(self, raw_field):
        prep = preprocess.Preprocessor(
            rolling_window_size=1,
            timescale="daily",
            detrend="linear",
            subtract_climatology=True,
            nan_mask="complete",
        )
        single_doy = raw_field["sst"].sel(time=raw_field["sst"].time.dt.dayofyear == 1)
        single_doy[:2, 0, 0] = np.nan  # [0,0] lat/lon NaN at timestep 0, 1
        single_doy[2:, 1, 1] = np.nan  # [1:,1,1] lat/lon NaN at timestep 2:end

        pp_field = prep.fit_transform(single_doy)
        nans_in_pp_field = np.isnan(pp_field).sum("time")[0, 0]
        assert int(nans_in_pp_field) == np.unique(pp_field.time.dt.year).size, (
            "If any NaNs are present in the data, "
            "the entire timeseries should have become completely NaN in the output."
        )

        prep = preprocess.Preprocessor(
            rolling_window_size=1,
            timescale="daily",
            detrend="linear",
            subtract_climatology=True,
            nan_mask="individual",
        )

        pp_field = prep.fit_transform(single_doy)
        assert (
            np.isnan(pp_field).sum("time") == np.isnan(single_doy).sum("time")
        ).all(), (
            "If any NaNs are present in the data, "
            "the pp will only ignore those NaNs, but still fit a trendline."
            "Hence, the NaNs should remain the same in this case."
        )

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_get_trendtimeseries_dataset(self, preprocessor, raw_field):
        preprocessor.fit(raw_field)
        trend = preprocessor.get_trend_timeseries(raw_field)
        assert trend is not None
        assert trend.dims == raw_field.dims
        assert trend.sst.shape == raw_field.sst.shape

        # get timeseries if single lat-lon point is seleted:
        subset_latlon = raw_field.isel(latitude=[0], longitude=[0])
        trend = preprocessor.get_trend_timeseries(subset_latlon, align_coords=True)
        assert trend is not None
        assert trend.dims == subset_latlon.dims
        assert trend.sst.shape == subset_latlon.sst.shape

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_get_trendtimeseries_dataarray(self, preprocessor, raw_field):
        raw_field = raw_field.to_array().squeeze("variable").drop_vars("variable")
        preprocessor.fit(raw_field)
        trend = preprocessor.get_trend_timeseries(raw_field)
        assert trend is not None
        assert (
            trend.dims == raw_field.dims
        ), f"dims do not match \n {trend.dims} \n {raw_field.dims}"
        assert (
            trend.shape == raw_field.shape
        ), f"shape does not match \n {trend.shape} \n {raw_field.shape}"

    @pytest.mark.parametrize(
        "preprocessor",
        [
            ("linear",),
            ("polynomial",),
        ],
        indirect=True,
    )
    def test_get_climatology_timeseries(self, preprocessor, raw_field):
        preprocessor.fit(raw_field)
        climatology = preprocessor.get_climatology_timeseries(raw_field)
        assert climatology is not None
        assert climatology.dims == raw_field.dims
        assert climatology.sst.shape == raw_field.sst.shape

        # get timeseries if single lat-lon point is seleted:
        subset_latlon = raw_field.isel(latitude=[0], longitude=[0])
        climatology = preprocessor.get_climatology_timeseries(
            subset_latlon, align_coords=True
        )
        assert climatology is not None
        assert climatology.dims == subset_latlon.dims
        assert climatology.sst.shape == subset_latlon.sst.shape
