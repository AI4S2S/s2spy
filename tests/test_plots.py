"""Tests for the s2spy._bokeh_plots module.
"""
import matplotlib.pyplot as plt
import pytest
from bokeh import io as bokeh_io
from s2spy import time


class TestPlots:
    """Test the visualizations (rough check for errors only)."""
    @pytest.fixture(autouse=True)
    def dummy_bokeh_file(self, tmp_path):
        bokeh_io.output_file(tmp_path / "test.html")

    custom_cal_pre = time.CustomCalendar(anchor="12-31")
    custom_cal_pre.append(time.PrecursorPeriod(length="10d"))
    custom_cal_tar = time.CustomCalendar(anchor="12-31")
    custom_cal_tar.append(time.TargetPeriod(length="10d"))

    calendars = [
        time.AdventCalendar(anchor="12-31", freq="60d"),
        time.MonthlyCalendar(anchor="December", freq="1M"),
        time.WeeklyCalendar(anchor="W40", freq="2W"),
        custom_cal_pre,
        custom_cal_tar,
    ]

    @pytest.fixture(params=calendars, autouse=True)
    def dummy_calendars(self, request):
        "Dummy that tests all available calendars."
        cal = request.param
        return cal.map_years(2018, 2021)

    @pytest.fixture(params=[True, False], autouse=True)
    def isinteractive(self, request):
        return request.param

    def test_visualize_relative(self, dummy_calendars, isinteractive):
        dummy_calendars.visualize(interactive=isinteractive, relative_dates=True)
        plt.close('all')

    def test_visualize_absolute(self, dummy_calendars, isinteractive):
        dummy_calendars.visualize(interactive=isinteractive, relative_dates=False)
        plt.close('all')

    def test_visualize_without_text(self, dummy_calendars, isinteractive):
        dummy_calendars.visualize(interactive=isinteractive, add_length=False)
        plt.close('all')

    def test_visualize_with_text(self, dummy_calendars, isinteractive):
        dummy_calendars.visualize(interactive=isinteractive, add_length=True)
        plt.close('all')

    def test_visualize_unmapped(self, isinteractive):
        time.AdventCalendar(anchor="12-31").visualize(interactive=isinteractive)
        plt.close('all')


class TestPlotsSingle:
    """Test the visualization routines, where a single calendar suffices"""
    @pytest.fixture(autouse=True)
    def dummy_bokeh_file(self, tmp_path):
        bokeh_io.output_file(tmp_path / "test.html")

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        "Dummy that will only test for AdventCalendar (to avoid excess testing)"
        cal = time.AdventCalendar(anchor="12-31", freq="60d")
        return cal.map_years(2018, 2021)

    def test_bokeh_kwargs(self, dummy_calendar):
        """Testing kwargs that overwrite default kwargs."""
        dummy_calendar.visualize(
            interactive=True,
            width=800,
            height=200,
            tooltips=[("Interval", "@desc")],
        )

    def test_bokeh_kwarg_mpl(self, dummy_calendar):
        with pytest.warns():
            dummy_calendar.visualize(interactive=False, width=800)
            plt.close('all')


    def test_mpl_no_legend(self, dummy_calendar):
        dummy_calendar.visualize(interactive=False, add_legend=False)
        plt.close('all')

    def test_mpl_ax_kwarg(self, dummy_calendar):
        _, ax = plt.subplots()
        dummy_calendar.visualize(interactive=False, ax=ax)
        plt.close('all')

    def test_bokeh_ax_warning(self, dummy_calendar):
        _, ax = plt.subplots()
        with pytest.warns():
            dummy_calendar.visualize(interactive=True, ax=ax)
