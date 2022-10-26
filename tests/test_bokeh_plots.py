"""Tests for the s2spy._bokeh_plots module.
"""
import pytest
import s2spy._bokeh_plots
from s2spy.time import AdventCalendar


# pylint: disable=protected-access
class TestBokehPlots:
    """Test the bokeh visualizations (rough check)."""

    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = AdventCalendar(anchor="12-31", freq="60d")
        cal.map_years(2018, 2021)
        return cal

    def test_visualize_bokeh_relative(self, dummy_calendar):
        s2spy._bokeh_plots._bokeh_visualization(
            dummy_calendar, n_years=2, relative_dates=True
        )

    def test_visualize_bokeh_absolute(self, dummy_calendar):
        s2spy._bokeh_plots._bokeh_visualization(
            dummy_calendar, n_years=2, relative_dates=False
        )
