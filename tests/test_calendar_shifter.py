"""Tests for the s2s.nwp_calendar module.
"""
import copy
import numpy as np
import pytest
import xarray as xr
import pandas as pd
from s2spy.time import Calendar
from s2spy.time import resample
from s2spy import calendar_shifter


class TestCalendarShifter:
    """Test the calendar shifter methods."""

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = Calendar(anchor="12-31")
        # append building blocks
        cal.add_interval("target", "7d")
        cal.add_interval("precursor", "7d", "7d")
        cal.add_interval("precursor", "7d")
        return cal

    def test_nwp_calendar_builder(self, dummy_calendar):
        cal = calendar_shifter.cal_builder(
            anchor_date="12-31",
            target_length="7d",
            precur_length="7d",
            n_precurperiods=2,
            precur_leadtime="7d",
        )
        expected = dummy_calendar
        assert np.array_equal(cal, expected)
    
    def test_shifter(self, dummy_calendar):
        shift = {'days':7}
        cal_shifted = calendar_shifter.cal_shifter(dummy_calendar, shift)
        cal_expected = copy.deepcopy(dummy_calendar)
        cal_expected.targets[0].gap = shift
        cal_expected.precursors[0].gap = {(key, value * -1) for key, value in shift.items()}
        assert np.array_equal(cal_shifted, cal_expected)

    @pytest.fixture(autouse=True)
    def dummy_data():

        time_index = pd.date_range('20211201', '20220131', freq='1d')

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

    def test_cal_list_resampler(self, dummy_calendar, dummy_data):
        #generate cal_list
        n_runs = 3
        shift = '7d'
        cal_list = [dummy_calendar]
        for _ in range(n_runs - 1):
            cal_shifted = calendar_shifter.cal_shifter(cal_list[-1], shift)
            cal_list.append(cal_shifted)
        #use list resampler
        ds_resampled = calendar_shifter.cal_list_resampler(cal_list, dummy_data)
        #generate expected
        ds_expected = 
        assert np.array_equal(ds_resampled, ds_expected)