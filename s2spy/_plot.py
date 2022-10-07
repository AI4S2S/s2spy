from calendar import month_abbr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle


def plot_interval(
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
    left = (interval.left - anchor_date).days
    hwidth = (interval.right - interval.left).days

    ax.add_patch(
        Rectangle(
            (left, anchor_date.year - 0.4),
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
            x=left + hwidth / 2,
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
        anchor_date = year_intervals[-calendar.n_targets].left

        # Plot the target intervals
        for interval in year_intervals[-calendar.n_targets :: 2]:
            plot_interval(
                anchor_date, interval, ax=ax, color="#ff7700", add_freq=add_freq
            )
        for interval in year_intervals[-calendar.n_targets + 1 :: 2]:
            plot_interval(
                anchor_date, interval, ax=ax, color="#ffa100", add_freq=add_freq
            )

        # Plot the precursor intervals
        for interval in year_intervals[: -calendar.n_targets : 2]:
            plot_interval(
                anchor_date, interval, ax=ax, color="#1f9ce9", add_freq=add_freq
            )
        for interval in year_intervals[1: -calendar.n_targets : 2]:
            plot_interval(
                anchor_date, interval, ax=ax, color="#137fc1", add_freq=add_freq
            )

    left_bound = (anchor_date - intervals.values[-1][0].left).days
    right_bound = (intervals.values[-1][-1].right - anchor_date).days
    ax.set_xlim([-left_bound - 5, right_bound + 5])
    ax.set_xlabel(
        f"Days with respect to anchor date ({anchor_date.day}"
        f" {month_abbr[anchor_date.month]})"
    )

    anchor_years = intervals.index.astype(int).values
    ax.set_ylim([anchor_years.min() - 0.5, anchor_years.max() + 0.5])
    ax.set_yticks(anchor_years)
    ax.set_ylabel('Anchor year')

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
