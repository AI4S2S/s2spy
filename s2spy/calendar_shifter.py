import copy
from typing import Union
import xarray as xr
import s2spy.time
from s2spy.time import resample  # pylint: disable=unused-import

# pylint: disable=too-many-arguments
def cal_builder(
    anchor_date,
    target_length,
    precur_length,
    n_targetperiods=1,
    n_precurperiods=1,
    target_overlap="0d",
    precur_overlap="0d",
    precur_leadtime="0d",
):
    """
    Build a calendar that resembles a typical NWP-type S2S forecast. Typically contains:
        - One or more targetperiods, for example the 1st, 2nd, 3rd and 4th week of June
        - One or more precursorperiods, for example the weeks preceeding the target
        periods
        - A leadtime for the precursors. A forecast run is typically done >2 weeks up to
        7 months ahead
    This timeframe is usually repeated on a regular basis as new data comes in. For
    example, every first of the month the forecast is reinitiated.
    - Args:
        - anchor_date : the referral date of your calendar. This is the first day of the
            first target period.
        - target_length : length of a target period
        - precur_length : length of a precursor period
        - n_targetperiods: number of targetperiods
        - target_overlap: overlap between target periods
        - n_precurperiods: number of precursorperiods
        - precur_overlap: overlap between precursor periods
        - precur_leadtime: gap (leadtime) between first target (i_interval = 0) and
            first precursor period (i_interval = -1)
    - Returns:
        - A calendar
    - Example:
        >>> import s2spy.time
        >>> cal = nwp_calendar_builder(start_date, '7d', '7d', precur_leadtime='14d', n_precurperiods=4)
        >>> cal
        Calendar(
        anchor='07-01',
        allow_overlap=False,
        mapping=None,
        intervals=[
            Interval(role='target', length='7d', gap='0d'),
            Interval(role='precursor', length='7d', gap='14d'),
            Interval(role='precursor', length='7d', gap='0d'),
            Interval(role='precursor', length='7d', gap='0d'),
            Interval(role='precursor', length='7d', gap='0d'),
            ]
        )
    """
    # create calendar instance
    cal = s2spy.time.Calendar(anchor=anchor_date)
    # add target periods to calendar
    for i in range(n_targetperiods):
        if i == 0:  # don't add overlap for first targetperiod = shifting
            cal.add_interval("target", length=target_length)
        else:  # overlap
            cal.add_interval("target", length=target_length, gap=target_overlap)
    # add precursor periods to calendar
    for j in range(n_precurperiods):
        if j == 0:  # add leadtime for first precurperiod
            cal.add_interval("precursor", length=precur_length, gap=precur_leadtime)
        else:  # overlap
            cal.add_interval("precursor", length=precur_length, gap=precur_overlap)

    return cal


def _gap_shift(interval: s2spy.time.Interval, shift: str) -> dict:
    """
    Shift a gap from a calendar interval by an input of a pandas-like
    frequency string (e.g. "10d", "2W", or "3M"), or a pandas.DateOffset
    compatible dictionary such as {days=10}, {weeks=2}, or {months=1, weeks=2}..
    Args:
        - gap = a pandas DateOffset from a calendar instance.
        - shift = the shift for the gap
    Returns:
        - dict with gap offset by shift
    Example:
        >>> import s2spy.time
        >>> gap_shifted = gap_shift()
        >>> cal = nwp_calendar_builder(start_date, '7d', '7d', precur_leadtime='14d', n_precurperiods=4)
        >>> interval = cal.targets[0]
        >>> gap_shifted = gap_shift(interval, '7d')
        >>> gap_shifted
        {'days': 7}
    """
    # parse timestrings to timedicts
    # pylint: disable=protected-access
    if isinstance(interval.gap, str):
        gap_time_dict = interval._parse_timestring(interval.gap)
    else:
        gap_time_dict = interval.gap
    # pylint: disable=protected-access
    shift_time_dict = interval._parse_timestring(shift)
    # make the shift negative for the precursor to shift forward in time
    if interval.role == "precursor":
        shift_time_dict.update(
            (key, value * -1) for key, value in shift_time_dict.items()
        )
    # combine gap and shift time dict
    gap_shifted = {
        k: gap_time_dict.get(k, 0) + shift_time_dict.get(k, 0)
        for k in set(gap_time_dict) | set(shift_time_dict)
    }

    return gap_shifted


def cal_shifter(
    calendar: s2spy.time.Calendar, shift: Union[str, dict]
) -> s2spy.time.Calendar:
    """
    Shift a Calendar instance by a given time offset. Instead of shifting the
    anchor date, this function shifts two things in reference to the anchor date:
    - the target period(s): as a gap between the anchor date and the start of the first
        target period
    - the precursor period(s): as a gap between the anchor date and the start of the
        first precursor period
    This way, the anchor year from the input calendar is maintained on the returned
    calendar. This is important for train-test splitting at later stages.
    - Args:
        - calendar : a s2spy.time.Calendar instance
        - shift: a pandas-like
        frequency string (e.g. "10d", "2W", or "3M"), or a pandas.DateOffset
        compatible dictionary such as {days=10}, {weeks=2}, or {months=1, weeks=2}
    - Example:
        Shift a calendar by a given dateoffset.
        >>> import s2spy.time
        >>> gap_shifted = gap_shift()
        >>> cal = nwp_calendar_builder(start_date, '7d', '7d', precur_leadtime='14d', n_precurperiods=4)
        >>> cal_shifted = shifter(cal, '7d')
        Calendar(
            anchor='07-01',
            allow_overlap=False,
            mapping=None,
            intervals=[
                Interval(role='target', length='7d', gap={'days': 7}),
                Interval(role='precursor', length='7d', gap={'days': 7}),
                Interval(role='precursor', length='7d', gap='0d'),
                Interval(role='precursor', length='7d', gap='0d'),
                Interval(role='precursor', length='7d', gap='0d'),
            ]
        )
    """
    # get 1st target interval and 1st precursor interval from calendar
    target_interval = calendar.targets[0]
    precursor_interval = calendar.precursors[0]

    # shift gaps
    target_gap_shifted = _gap_shift(target_interval, shift)
    precursor_gap_shifted = _gap_shift(precursor_interval, shift)

    # set the new gaps on the calendar
    calendar_shifted = copy.deepcopy(calendar)
    calendar_shifted.targets[0].gap = target_gap_shifted
    calendar_shifted.precursors[0].gap = precursor_gap_shifted

    return calendar_shifted


def cal_list_resampler(cal_list: list, ds: xr.Dataset) -> xr.DataArray:
    """
    Resample a dataset to every calendar in a list of calendars and concatenate them
    to a new dataset with dimension 'run'.
    - Args:
        - cal_list : list of shifted custom calendars
        - ds: dataset to resample
    """
    # loop over calendar in the nwp_calendar_list
    for x, cal_x in enumerate(cal_list):
        cal_x_mapped = cal_x.map_to_data(ds)
        # resample and concatenate on dimension 'run'
        if x == 0:
            ds_r = s2spy.time.resample(cal_x_mapped, ds)
        else:
            ds_toconcat = s2spy.time.resample(cal_x_mapped, ds)
            ds_r = xr.concat([ds_r, ds_toconcat], dim="run")
    # add coordinates for dimension 'run'
    ds_r = ds_r.assign_coords({"run": ds_r.run.values})

    return ds_r
