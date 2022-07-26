"""Regions clustering utils.

A module for clustering regions based on the given correlation
between spatial fields and target timeseries.
"""
import numpy as np
from sklearn.cluster import DBSCAN


def cluster_dbscan(
        map_analysis, threshold, eps, min_samples,
        n_jobs: int=-1, **method_kwargs
    ):
    """Perform DBSCAN clustering from vector array or distance matrix.

    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Clusters gridcells together which are of the same sign and in proximity to
    each other using DBSCAN.

    The input data will be processed in this function to ensure that the distance
    is free of the impact from spherical curvature. The actual Euclidean distance
    will be obtained and passed to the DBSCAN clustering function.


    """
    # To do: check if the longitude is between -180 : 180
    # To do: input with lags dimension to vectorize dbscan and keep labels
    # To do: handle clustering for +/- correlation separately
    # To do: mask data using significance (p_value)
    # To do: check if it is necessary to have weight by area
    #  (the impact of area is already reflected by the haversine distance)
    # To do: add relevant checks for the results and raise warnings
    #  (e.g. check whether the results are noise or not)
    # To do: calculate and verify DBSCAN
    mask = map_analysis.p_values.values < threshold
    grid_lon, grid_lat = np.meshgrid((map_analysis.longitude.values + 180) % 360 - 180,
        map_analysis.latitude.values)
    # prepare [lat, lon] pairs as input for DBSCAN
    grid_pair_up = [[lat, lon] for lat, lon in zip(np.radians(grid_lat.reshape(-1)),
        np.radians(grid_lon.reshape(-1)))]
    cluster = DBSCAN(eps = eps, min_samples = min_samples, metric = 'haversine',
        n_jobs = n_jobs, **method_kwargs).fit(grid_pair_up)

    return cluster.labels_

def spatial_mean():
    """Calculate 1-d timeseries for each precursor region. Precursor regions 
    are integer label masks within the np.ndarray labels.
    """
    raise NotImplementedError
