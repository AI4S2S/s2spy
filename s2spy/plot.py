import numpy as np


def import_bokeh():
    """Util that attempts to load the optional module bokeh"""
    try:
        import bokeh as _  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError("Could not import the `bokeh` module.\nPlease install this"
                          " before continuing, with either `pip` or `conda`.") from e


def make_color_array(n_targets, n_intervals):
    colors = np.array(['#ffffff'] * n_intervals)
    colors[0 : n_targets : 2] = "#ff7700"
    colors[1 : n_targets : 2] = "#ffa100"
    colors[n_targets :: 2] = "#1f9ce9"
    colors[n_targets + 1 :: 2] = "#137fc1"
    return colors


def bohek_visualization_single(calendar):
    import_bokeh()
    from bokeh import plotting  # pylint: disable=import-outside-toplevel

    intervals = calendar.get_intervals()

    year_intervals = intervals.values[0]
    anchor_date = year_intervals[0].right

    y_data = np.ones(len(intervals.values[0])) * anchor_date.year
    x_data = np.array([i.right for i in year_intervals])
    widths = np.array([(i.right - i.left) for i in year_intervals])
    interval_str = np.array([f"{str(i.left)[:10]} -> {str(i.right)[:10]}"
                             for i in year_intervals])
    heights = np.ones(len(intervals.values[0])) * 0.8

    colors = make_color_array(
        n_targets = calendar.n_targets,
        n_intervals = len(intervals.values[0])
    )

    types = np.array(['Precursor'] * len(intervals.values[0]))
    types[:calendar.n_targets] = 'Target'

    source = plotting.ColumnDataSource(data={
        'x': x_data - 0.5 * widths,
        'y': y_data,
        'height': heights,
        'width': widths,
        'width_days': np.array([x.days for x in widths]),
        'color': colors,
        'desc': interval_str,
        'type': types,
        })

    tooltips = [
        ("Interval", "@desc"),
        ("Size", "@width_days days"),
        ("Type", "@type")
    ]

    plot = plotting.figure(
        width=500,
        height=300,
        tooltips=tooltips,
        x_axis_type='datetime'
    )

    plot.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        line_color='#000000',
        fill_color="color",
        fill_alpha=0.7,
        line_width=1.5,
        source=source
        )

    plot.xaxis.axis_label = "Date"
    plot.yaxis.axis_label = "Anchor year"

    plot.yaxis.ticker = [anchor_date.year]

    plotting.show(plot)


def bohek_visualization_multiple(calendar):
    import_bokeh()
    from bokeh import plotting  # pylint: disable=import-outside-toplevel

    tooltips = [
        ("Interval", "@desc"),
        ("Size", "@width days"),
        ("Type", "@type")
    ]

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
        interval_str = np.array([f"{str(i.left)[:10]} -> {str(i.right)[:10]}"
                                for i in year_intervals])
        heights = np.ones(len(year_intervals)) * 0.8

        colors = make_color_array(
            n_targets = calendar.n_targets,
            n_intervals = len(year_intervals)
        )

        types = np.array(['Precursor'] * len(year_intervals))
        types[:calendar.n_targets] = 'Target'

        source = plotting.ColumnDataSource(data={
            'x': x_data + 0.5 * widths,
            'y': y_data,
            'height': heights,
            'width': widths,
            'color': colors,
            'desc': interval_str,
            'type': types,
            })

        plot.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            line_color='#000000',
            fill_color="color",
            fill_alpha=0.7,
            line_width=1.5,
            source=source
            )

    plot.xaxis.axis_label = "Days before anchor date"
    plot.yaxis.axis_label = "Anchor year"

    plot.x_range.start = np.max(x_data) + np.max(widths) + 10
    plot.x_range.end = -10

    plot.yaxis.ticker = [int(x) for x in intervals.index.to_list()]

    plotting.show(plot)