"""Commonly used utility functions for s2spy."""
from typing import TypeVar
import numpy as np
import xarray as xr


XrType = TypeVar("XrType", xr.DataArray, xr.Dataset)


def weighted_groupby(
    ds: XrType, groupby: str, weight: str, method: str = "mean"
) -> XrType:
    """Apply a weighted reduction after a groupby call.

    xarray does not currently support combining `weighted` and `groupby`. An open PR
    adds supports for this functionality (https://github.com/pydata/xarray/pull/5480),
    but this branch was never merged.

    Args:
        ds: Data containing the coordinates or variables
            specified in the `groupby` and `weight` kwargs.
        groupby: Coordinate which should be used to make the groups.
        weight: Variable in the Dataset containing the weights that should be used.
        method: Method that should be used to reduce the dataset, by default
            'mean'. Supports any of xarray's builtin methods, e.g. 'median', 'min',
            'max'.

    Returns:
        Same as input: Dataset reduced using the `groupby` coordinate, using weights =
            based on `ds[weight]`.
    """
    groups = ds.groupby(groupby)

    # find stacked dim name
    group0 = list(groups)[0][1]
    dims = list(group0.dims)
    stacked_dims = [dim for dim in dims if "stacked_" in str(dim)]

    reduced_groups = [
        getattr(g.weighted(g[weight]), method)(dim=stacked_dims) for _, g in groups
    ]

    reduced_data: XrType = xr.concat(reduced_groups, dim=groupby)

    if isinstance(reduced_data, xr.DataArray):  # Add back the labels of the groupby dim
        reduced_data[groupby] = np.unique(ds[groupby])
    return reduced_data


def geographical_cluster_center(
    masked_data: xr.DataArray, reduced_data: xr.DataArray
) -> xr.DataArray:
    """Add the geographical centers to the clusters.

    Args:
        masked_data (xr.DataArray): Precursor data before being reduced to clusters,
            with the dimensions latitude and longitude, and cluster labels added.
        reduced_data (xr.DataArray): Data reduced to the clusters, to which the
            geographical centers will be added

    Returns:
        xr.DataArray: Reduced data with the latitude and longitude of the geographical
            centers added as coordinates of the cluster labels.
    """
    clusters = np.unique(masked_data["cluster_labels"])
    stacked_data = masked_data.stack(coords=("latitude", "longitude"))

    cluster_lats = np.zeros(clusters.shape)
    cluster_lons = np.zeros(clusters.shape)

    for i, cluster in enumerate(clusters):
        # Select only the grid cells within the cluster
        cluster_area = stacked_data["area"].where(
            stacked_data["cluster_labels"] == cluster
        )

        if "i_interval" in cluster_area.dims:
            cluster_area = cluster_area.dropna("i_interval", how="all")
        cluster_area = cluster_area.dropna("coords")

        # Area weighted mean to get the geographical center of the cluster
        cluster_lats[i] = cluster_area["latitude"].weighted(cluster_area).mean().item()
        cluster_lons[i] = cluster_area["longitude"].weighted(cluster_area).mean().item()

    reduced_data["latitude"] = ("cluster_labels", cluster_lats)
    reduced_data["longitude"] = ("cluster_labels", cluster_lons)

    return reduced_data


def intervals_subtract(intervals: list[int], n: int) -> list[int]:
    """Subtracts n from the interval indices, skipping 0."""
    if n < 0:
        raise ValueError("Lag values below 0 are not supported")

    lag_intervals = [i - n for i in intervals]
    # pylint: disable=chained-comparison
    return [
        i - 1 if (i <= 0 and j > 0) else i for i, j in zip(lag_intervals, intervals)
    ]
