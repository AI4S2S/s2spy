"""s2spy dimensionality reduction module.

A module handling dimensionality reduction tasks.
This module provides a collection of dimensionality reduction
"""


def RGDR(series, field, lag_shift: int = 0):
    """Instantiate a basic calendar with minimal configuration.

    Set up the calendar with given freq ending exactly on the anchor date.
    The index will extend back in time as many periods as fit within the
    cycle time of one year.

    Args:
        series: Target timeseies.
        field: Target fields.
    """
    raise NotImplementedError

def PCA(field):
    raise NotImplementedError

def MCA(field):
    raise NotImplementedError

def _traintest_binder():
    raise NotImplementedError