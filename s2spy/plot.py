import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from calendar import month_abbr
from matplotlib.patches import Patch

XLABEL = "Anchor year"

def plot_mpl_interval(
    anchor_date: pd.Timestamp,
    interval: pd.Interval,
    ax: plt.Axes,
    color: str,
    add_freq: bool = False,
):
    """Utility for the calendar visualization.

    Plots a rectangle representing a single interval.

    Args:
        anchor_date: Pandas timestamp representing the anchor date.
        interval: Interval that should be added to the plot.
        ax: Axis to plot the interval in.
        color: (Matplotlib compatible) color that the rectangle should have.
        add_freq: Toggles if the frequency will be displayed on the rectangle
    """
    right = (anchor_date - interval.right).days
    hwidth = (interval.right - interval.left).days

    ax.add_patch(
        Rectangle(
            (right, anchor_date.year - 0.4),
            hwidth,
            0.8,
            facecolor=color,
            alpha=0.7,
            edgecolor="k",
            linewidth=1.5,
        )
    )

    if add_freq:
        ax.text(
            x=right + hwidth / 2,
            y=anchor_date.year,
            s=hwidth,
            c="k",
            size=8,
            ha="center",
            va="center",
        )


def matplotlib_visualization(calendar, n_years, add_freq):
    intervals = calendar.get_intervals()[:n_years]

    _, ax = plt.subplots()

    for year_intervals in intervals.values:
        anchor_date = year_intervals[0].right

        # Plot the anchor intervals
        for interval in year_intervals[0 : calendar.n_targets : 2]:
            plot_mpl_interval(
                anchor_date, interval, ax=ax, color="#ff7700", add_freq=add_freq
            )
        for interval in year_intervals[1 : calendar.n_targets : 2]:
            plot_mpl_interval(
                anchor_date, interval, ax=ax, color="#ffa100", add_freq=add_freq
            )

        # Plot the precursor intervals
        for interval in year_intervals[calendar.n_targets :: 2]:
            plot_mpl_interval(
                anchor_date, interval, ax=ax, color="#1f9ce9", add_freq=add_freq
            )
        for interval in year_intervals[calendar.n_targets + 1 :: 2]:
            plot_mpl_interval(
                anchor_date, interval, ax=ax, color="#137fc1", add_freq=add_freq
            )

    left_bound = (anchor_date - intervals.values[-1][-1].left).days
    ax.set_xlim([left_bound + 5, -5])
    ax.set_xlabel(
        f"Days before anchor date ({anchor_date.day}"
        f" {month_abbr[anchor_date.month]})"
    )

    anchor_years = intervals.index.astype(int).values
    ax.set_ylim([anchor_years.min() - 0.5, anchor_years.max() + 0.5])
    ax.set_yticks(anchor_years)
    ax.set_ylabel(XLABEL)

    # Add a custom legend to explain to users what the colors mean
    legend_elements = [
        Patch(
            facecolor="#ff8c00",
            label="Target interval",
            linewidth=1.5,
        ),
        Patch(
            facecolor="#137fc1",
            label="Precursor interval",
            linewidth=1.5,
        ),
    ]
    ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))


def bokeh_available():
    """Util that attempts to load the optional module bokeh"""
    try:
        import bokeh as _  # pylint: disable=import-outside-toplevel
        return True
    except ImportError as e:
        raise ImportError(
            "Could not import the `bokeh` module.\nPlease install this"
            " before continuing, with either `pip` or `conda`."
        ) from e


def make_color_array(n_targets, n_intervals):
    """Util that generates the colormap for the intervals."""
    colors = np.array(["#ffffff"] * n_intervals)
    colors[0:n_targets:2] = "#ff7700"
    colors[1:n_targets:2] = "#ffa100"
    colors[n_targets::2] = "#1f9ce9"
    colors[n_targets + 1 :: 2] = "#137fc1"
    return colors


def _bokeh_visualization_single(calendar, plotting):
    """Visualization routine for a single anchor year. Has a datetime x-axis."""

    intervals = calendar.get_intervals()

    year_intervals = intervals.values[0]
    anchor_date = year_intervals[0].right

    y_data = np.ones(len(intervals.values[0])) * anchor_date.year
    x_data = np.array([i.right for i in year_intervals])
    widths = np.array([(i.right - i.left) for i in year_intervals])
    interval_str = np.array(
        [f"{str(i.left)[:10]} -> {str(i.right)[:10]}" for i in year_intervals]
    )
    heights = np.ones(len(intervals.values[0])) * 0.8

    colors = make_color_array(
        n_targets=calendar.n_targets, n_intervals=len(intervals.values[0])
    )

    types = np.array(["Precursor"] * len(intervals.values[0]))
    types[: calendar.n_targets] = "Target"

    source = plotting.ColumnDataSource(
        data={
            "x": x_data - 0.5 * widths,
            "y": y_data,
            "height": heights,
            "width": widths,
            "width_days": np.array([x.days for x in widths]),
            "color": colors,
            "desc": interval_str,
            "type": types,
        }
    )

    tooltips = [("Interval", "@desc"), ("Size", "@width_days days"), ("Type", "@type")]

    plot = plotting.figure(
        width=500, height=300, tooltips=tooltips, x_axis_type="datetime"
    )

    plot.rect(
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

    plot.xaxis.axis_label = "Date"
    plot.yaxis.axis_label = XLABEL

    plot.yaxis.ticker = [anchor_date.year]

    return plot


def _bokeh_visualization_multiple(calendar, plotting):
    """Visualization routine for multiple anchor years. Has a datetime x-axis."""

    tooltips = [("Interval", "@desc"), ("Size", "@width days"), ("Type", "@type")]

    plot = plotting.figure(
        width=500,
        height=300,
        tooltips=tooltips,
    )

    intervals = calendar.get_intervals()

    for year_intervals in intervals.values:
        anchor_date = year_intervals[0].right

        y_data = np.ones(len(year_intervals)) * anchor_date.year
        x_data = np.array([(anchor_date - i.right).days for i in year_intervals])
        widths = np.array([(i.right - i.left).days for i in year_intervals])
        interval_str = np.array(
            [f"{str(i.left)[:10]} -> {str(i.right)[:10]}" for i in year_intervals]
        )
        heights = np.ones(len(year_intervals)) * 0.8

        colors = make_color_array(
            n_targets=calendar.n_targets, n_intervals=len(year_intervals)
        )

        types = np.array(["Precursor"] * len(year_intervals))
        types[: calendar.n_targets] = "Target"

        source = plotting.ColumnDataSource(
            data={
                "x": x_data + 0.5 * widths,
                "y": y_data,
                "height": heights,
                "width": widths,
                "color": colors,
                "desc": interval_str,
                "type": types,
            }
        )

        plot.rect(
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

    plot.xaxis.axis_label = "Days before anchor date"
    plot.yaxis.axis_label = XLABEL

    plot.x_range.start = np.max(x_data) + np.max(widths) + 10
    plot.x_range.end = -10

    plot.yaxis.ticker = [int(x) for x in intervals.index.to_list()]

    return plot


def bokeh_visualization(calendar, n_years):
    if not bokeh_available():
        return None
    from bokeh import plotting  # pylint: disable=import-outside-toplevel

    if n_years == 1:
        figure = _bokeh_visualization_single(calendar, plotting)
    else:
        figure = _bokeh_visualization_multiple(calendar, plotting)

    plotting.show(figure)

    return None
