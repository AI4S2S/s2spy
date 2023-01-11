"""Tests for the s2s.nwp_calendar module.
"""
import copy
import numpy as np
import pytest
import xarray as xr
import pandas as pd
from s2spy.time import Calendar
from s2spy import calendar_shifter


class TestCalendarShifter:
    """Test the calendar shifter methods."""

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = Calendar(anchor="12-31")
        cal.add_intervals("target", "7d")
        cal.add_intervals("precursor", "7d", "7d")
        cal.add_intervals("precursor", "7d")
        return cal

    @pytest.fixture(autouse=True)
    def dummy_data(self):
        time_index = pd.date_range("20211001", "20220501", freq="1d")
        np.random.seed(0)
        temperature = 15 + 8 * np.random.randn(2, 2, len(time_index))
        lon = [[-99.83, -99.32], [-99.79, -99.23]]
        lat = [[42.25, 42.21], [42.63, 42.59]]
        ds = xr.Dataset(
            data_vars=dict(
                temperature=(["x", "y", "time"], temperature),
            ),
            coords=dict(
                lon=(["x", "y"], lon),
                lat=(["x", "y"], lat),
                time=time_index,
            ),
            attrs=dict(description="Weather related data."),
        )
        return ds

    def test_calendar_shifter(self, dummy_calendar):
        shift = {"days": 7}
        cal_shifted = calendar_shifter.calendar_shifter(dummy_calendar, shift)
        cal_expected = copy.deepcopy(dummy_calendar)
        cal_expected.targets[0].gap = {"days": 7}
        cal_expected.precursors[0].gap = {"days": 0}
        assert len(cal_shifted.targets + cal_shifted.precursors) == len(
            cal_expected.targets + cal_expected.precursors
        )
        assert str(cal_shifted.targets) == str(cal_expected.targets)
        assert str(cal_expected.precursors) == str(cal_shifted.precursors)

    def test_staggered_calendar(self, dummy_calendar):
        cal_list = calendar_shifter.staggered_calendar(dummy_calendar, "7d", n_shifts=3)
        assert len(cal_list) == 4

    def test_calendar_list_resampler(self, dummy_calendar, dummy_data):
        cal_list = calendar_shifter.staggered_calendar(
            dummy_calendar, "14d", n_shifts=5
        )
        for cal in cal_list:
            cal.map_to_data(dummy_data)
        ds_resampled = calendar_shifter.calendar_list_resampler(cal_list, dummy_data)
        expected_coords = np.array([0, 1, 2, 3, 4, 5])
        assert np.array_equal(ds_resampled.step.values, expected_coords)
