"""s2spy dimensionality reduction module.

A module handling dimensionality reduction tasks, which provides a
collection of dimensionality reduction approaches.
"""

def RGDR(series, lag_shift: int = 0):
    """Wrapper for Response Guided Dimensionality Reduction function.

    Configure RGDR operations using this function. It manages input training,
    data and loop through data with lag shift upto the given `lag_shift` value.
    And it also invokes the RGDR operator and execute the relevant correlation
    and clustering processes via the RGDR module.

    Args:
        series: Target timeseies.
        field: Target fields.
        lag_shift: Number of lag shifts that will be tested.
    """
    # To do: loop through lags
    # To do: invoke RGDR functions without execution.
    # e.g. import functools
    # RGDR = functools.partial(s2s.RGDR.operator, series)
    # return RGDR
    raise NotImplementedError

def PCA():
    """Wrapper for Principle Component Analysis."""
    raise NotImplementedError

def MCA():
    """Wrapper for Maximum Covariance Analysis."""
    raise NotImplementedError
