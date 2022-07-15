"""Regions clustering utils.

A module for clustering regions based on the given correlation
between spatial fields and target timeseries.
"""

def spatial_mean():
    """Calculate 1-d timeseries for each precursor region. Precursor regions 
    are integer label masks within the np.ndarray labels.
    """
    raise NotImplementedError

def cluster_dbscan():
    """Perform DBSCAN clustering from vector array or distance matrix.

    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Clusters gridcells together which are of the same sign and in proximity to
    each other using DBSCAN.

    The input data will be processed in this function to ensure that the distance
    is free of the impact from spherical curvature. The actual Euclidean distance
    will be obtained and passed to the DBSCAN clustering function.
    """
    raise NotImplementedError
