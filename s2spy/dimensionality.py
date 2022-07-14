"""s2spy dimensionality reduction module.

A module handling dimensionality reduction tasks, which provides a
collection of dimensionality reduction approaches.
"""

def RGDR(series, lag_shift: int = 0):
    """Wrapper for Response Guided Dimensionality Reduction function.

    Configure RGDR operations using this function. It will invoke the
    RGDR operator and execute the relevant correlation and clustering
    processes via the RGDR module.

    Args:
        series: Target timeseies.
        field: Target fields.
        lag_shift: Number of lag shifts that will be tested.
    """
    # To do: invoke RGDR functions without execution.
    raise NotImplementedError

def PCA():
    """Wrapper for Principle Component Analysis."""
    raise NotImplementedError

def MCA():
    """Wrapper for Maximum Covariance Analysis."""
    raise NotImplementedError
