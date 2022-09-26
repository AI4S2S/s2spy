"""Tests for the s2spy.plot module.
"""
import pytest
from s2spy.time import AdventCalendar
import s2spy.plot
from bokeh import plotting

# pylint: disable=protected-access
class TestBokehPlots:
    """Test the bokeh visualizations (rough check)."""
    @pytest.fixture(autouse=True)
    def dummy_calendar(self):
        cal = AdventCalendar(anchor=(12, 31), freq="60d")
        cal.map_years(2018, 2021)
        return cal

    def test_visualize_bokeh_single(self, dummy_calendar):
        _ = s2spy.plot._bokeh_visualization_single(dummy_calendar, plotting=plotting)

    def test_visualize_bokeh_multiple(self, dummy_calendar):
        _ = s2spy.plot._bokeh_visualization_multiple(dummy_calendar, plotting=plotting)
