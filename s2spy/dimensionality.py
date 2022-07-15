"""s2spy dimensionality reduction module.

A module handling dimensionality reduction tasks, which provides a
collection of dimensionality reduction approaches.
"""


def rgdr(series, lag_shift: int = 0):
    """Wrapper for Response Guided Dimensionality Reduction function.

    Configure RGDR operations using this function. It manages input training
    data and creats the RGDR object for the relevant correlation
    and clustering processes via the RGDR module.

    Args:
        series: Target timeseies.
        lag_shift: Number of lag shifts that will be tested.
    """
    # To do: invoke RGDR functions without execution.
    # return RGDR object
    raise NotImplementedError

def pca():
    """Wrapper for Principle Component Analysis."""
    raise NotImplementedError

def mca():
    """Wrapper for Maximum Covariance Analysis."""
    raise NotImplementedError
