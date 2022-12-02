import sys
import numpy as np
from bokeh import plotting
from ._plot import generate_plot_data


def _generate_rectangle(figure: plotting.Figure, source: plotting.ColumnDataSource):
    """Adds intervals to the figure as rectangles.

    Args:
        figure: Bokeh figure in which the rectangles should be added.
        source: Bokeh source data containing the required parameters.

    Returns:
        plotting.figure
    """
    return figure.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        line_color="#000000",
        fill_color="color",
        fill_alpha=0.7,
        line_width=1.5,
        source=source,
    )


def _bokeh_visualization(
    calendar,
    n_years: int,
    relative_dates: bool,
    add_yticklabels: bool = True,
    **kwargs,
) -> plotting.Figure:
    """Visualization routine for generating a calendar visualization with Bokeh.

    Args:
        calendar: Mapped calendar which should be visualized.
        n_years: Number of years which should be displayed (most recent years only).
        relative_dates: If False, absolute dates will be used. If True, each anchor year
                        is aligned by the anchor date, so that all anchor years line up
                        vertically.
        add_yticklabels: If the years should be displayed on the y-axis ticks.

    Returns:
        plotting.Figure
    """

    if add_yticklabels:
        tooltips = [
            ("Interval", "@desc"),
            ("Size", "@width_days days"),
            ("Type", "@type"),
        ]
    else:
        # Do not show the actual intervals, as the calendar is not mapped.
        tooltips = [("Size", "@width_days days"), ("Type", "@type")]

    if "width" not in kwargs:
        kwargs["width"] = 500
    if "height" not in kwargs:
        kwargs["height"] = 300
    if "tooltips" not in kwargs:
        kwargs["tooltips"] = tooltips
    if "x_axis_type" not in kwargs:
        kwargs["x_axis_type"] = "linear" if relative_dates else "datetime"

    figure = plotting.figure(
        **kwargs,
        )

    intervals = calendar.get_intervals()[:n_years]

    for _, year_intervals in intervals.iterrows():
        data = generate_plot_data(
            calendar=calendar,
            relative_dates=relative_dates,
            year_intervals=year_intervals,
        )
        _generate_rectangle(figure, plotting.ColumnDataSource(data))

    figure.xaxis.axis_label = (
        "Days relative to anchor date" if relative_dates else "Date"
    )
    figure.yaxis.axis_label = "Anchor year"

    if relative_dates:
        figure.x_range.start = (
            np.min(data["x"]) - data["width"][np.argmin(data["x"])] / 2 - 14  # type: ignore
        )
        figure.x_range.end = (
            np.max(data["x"]) + data["width"][np.argmax(data["x"])] / 2 + 14  # type: ignore
        )

    if add_yticklabels:
        figure.yaxis.ticker = [int(x) for x in intervals.index.to_list()]
    else:
        figure.yaxis.ticker = []

    return figure


def bokeh_visualization(
    calendar,
    n_years: int,
    relative_dates: bool,
    add_yticklabels: bool = True,
    **kwargs,
) -> None:
    bokeh_fig = _bokeh_visualization(
        calendar, n_years, relative_dates, add_yticklabels, **kwargs
    )
    if "pytest" in sys.modules:  # Do not open browser if we are testing.
        plotting.save(bokeh_fig)
    else:
        plotting.show(bokeh_fig)
