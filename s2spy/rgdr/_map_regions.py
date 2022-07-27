"""Regions clustering utils.

A module for clustering regions based on the given correlation
between spatial fields and target timeseries.
"""
import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN
from . import _map_utils


radius_earth_km = 6371


def weighted_groupby_mean(ds, groupby: str, weight: str):
    """Apply a weighted mean after a groupby call. xarray does not currently support
    combining `weighted` and `groupby`. An open PR adds supports for this functionality
    (https://github.com/pydata/xarray/pull/5480/files), but this branch was never merged.

    Args:
        ds (xr.Dataset): Dataset containing the coordinates or variables specified in
        the `groupby` and `weight` kwargs.
        groupby (str): Coordinate which should be used to make the groups.
        weight (str): Variable in the Dataset containing the weights that should be used.

    Returns:
        xr.Dataset: Dataset reduced using the `groupby` coordinate, using weighted average
            based on `ds[weight]`.
    """
    groups = ds.groupby(groupby)
    means = [
        g.weighted(g[weight]).mean(dim=["stacked_latitude_longitude"])
        for _, g in groups
    ]
    return xr.concat(means, dim=groupby)


def dbscan(
    ds: xr.Dataset, alpha: float = 0.05, eps_km: float = 600, min_area_km2: float = None
) -> xr.Dataset:
    """Determines the clusters based on sklearn's DBSCAN implementation. Alpha determines
    the mask based on the minimum p_value. Grouping can be adjusted using the `eps_km`
    kwarg.

    Args:
        ds (xr.Dataset): Dataset containing 'latitude' and 'longitude' dimensions in
            degrees. Must also contain 'p_val' and 'corr' to base the groups on.
        alpha (float): Value below which the correlation is significant enough to be
            considered
        eps_km (float): The maximum distance (in km) between two samples for one to be
            considered as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. This is the most important
            DBSCAN parameter to choose appropriately.

    Returns:
        xr.Dataset: Dataset grouped by the DBSCAN clusters. Cluster labels are negative
            for areas with a negative correlation coefficient and positive for areas
            with a positive correlation coefficient.
    """
    ds = ds.stack(coord=["latitude", "longitude"])
    coords = np.asarray(list(ds["coord"].values))  # turn array of tuples to 2d-array

    # Prepare labels, default value is 0 (not in cluster)
    labels = np.zeros(len(coords))

    for sign, sign_mask in zip([1, -1], [ds["corr"] >= 0, ds["corr"] < 0]):
        mask = np.logical_and(ds["p_val"] < alpha, sign_mask)

        if np.sum(mask) > 0:  # Check if the mask contains any points to cluster
            masked_coords = np.radians(coords[mask])
            db = DBSCAN(
                eps=eps_km / radius_earth_km,
                min_samples=1,
                algorithm="auto",
                metric="haversine",
            ).fit(masked_coords)

            labels[mask] = sign * (db.labels_ + 1)

    ds["cluster_labels"] = ("coord", labels)

    ds = ds.unstack(("coord"))

    resolution = np.abs(ds.longitude.values[1] - ds.longitude.values[0])
    ds["area"] = _map_utils.spherical_area(ds.latitude, resolution=resolution)

    if min_area_km2:
        ds = _map_utils.remove_small_area_clusters(ds, min_area_km2)

    return ds


def cluster(
    ds: xr.Dataset, alpha: float = 0.05, eps_km: float = 600, min_area_km2: float = None
) -> xr.Dataset:
    """Perform DBSCAN clustering on a prepared Dataset, and then group the data by their
    determined clusters, taking the weighted mean. The weight is based on the area of
    each grid cell.

    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    Clusters gridcells together which are of the same sign and in proximity to
    each other using DBSCAN.

    The input data will be processed in this function to ensure that the distance
    is free of the impact from spherical curvature. The actual geodesic distance
    will be obtained and passed to the DBSCAN clustering function.

    Args:
        ds: Dataset prepared to have p_val and corr.
        alpha (float): Value below which the correlation is significant enough to be
            considered
        eps_km (float): The maximum distance (in km) between two samples for one to be
            considered as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. This is the most important
            DBSCAN parameter to choose appropriately.

    """
    ds = dbscan(ds, alpha=alpha, eps_km=eps_km, min_area_km2=min_area_km2)

    ds = weighted_groupby_mean(ds, groupby="cluster_labels", weight="area")

    return ds
