"""Response Guided Dimensionality Reduction."""
from . import map_analysis
from . import map_regions


def operator(series, fields, lag_shift):
    """Operator for lag shifting and execute RGDR functions.
    
    Invoke functions to compute correlations and to perform clustering.
    This function will loop through data with lag shift upto the given
    `lag_shift` value.
    """
    # To do: loop through lags
    # To do: call `map_analysis` and `map_regions` and manage
    #  inputs and outputs.
    raise NotImplementedError
