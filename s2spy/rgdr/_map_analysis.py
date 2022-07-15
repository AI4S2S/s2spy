"""Spatial-temporal data analysis utils.

A toolbox for spatial-temporal data analysis, including regression,
correlation, auto-correlation and relevant utilities functions.
"""


def correlation(field, target):
    '''Calculate correlation maps.'''
    raise NotImplementedError

def partial_correlation(field, target, z):
    '''Calculate partial correlation maps.'''
    raise NotImplementedError

def regression(field, target):
    '''Regression analysis on entire maps.

    Methods include Linear, Ridge, Lasso.
    '''
    raise NotImplementedError

def save_map():
    """Save calculated coefficients.
    
    Store calculated coefficients and significance values, and
    save them as netcdf files.
    """
    raise NotImplementedError

def load_map():
    """Load coefficients from saved netcdf maps."""
    raise NotImplementedError
