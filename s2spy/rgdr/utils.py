from typing import TypeVar
import numpy as np
import xarray as xr


XrType = TypeVar("XrType", xr.DataArray, xr.Dataset)


def weighted_groupby(
    ds: XrType, groupby: str, weight: str, method: str = "mean"
) -> XrType:
    """Apply a weighted reduction after a groupby call. xarray does not currently support
    combining `weighted` and `groupby`. An open PR adds supports for this functionality
    (https://github.com/pydata/xarray/pull/5480), but this branch was never merged.

    Args:
        ds (xr.DataArray or xr.Dataset): Data containing the coordinates or variables
            specified in the `groupby` and `weight` kwargs.
        groupby (str): Coordinate which should be used to make the groups.
        weight (str): Variable in the Dataset containing the weights that should be used.
        method (str): Method that should be used to reduce the dataset, by default
            'mean'. Supports any of xarray's builtin methods, e.g. median, min, max.

    Returns:
        Same as input: Dataset reduced using the `groupby` coordinate, using weights =
            based on `ds[weight]`.
    """
    groups = ds.groupby(groupby)

    # find stacked dim name
    group0 = list(groups)[0][1]
    dims = list(group0.dims)
    stacked_dims = [
        dim for dim in dims if "stacked_" in str(dim)
    ]  # str() is just for mypy

    reduced_groups = [
        getattr(g.weighted(g[weight]), method)(dim=stacked_dims) for _, g in groups
    ]
    reduced_data: XrType
    reduced_data = xr.concat(reduced_groups, dim=groupby)

    if isinstance(reduced_data, xr.DataArray):  # Add back the labels of the groupby dim
        reduced_data[groupby] = np.unique(ds[groupby])
    return reduced_data


def geographical_cluster_center(
    masked_data: xr.DataArray, reduced_data: xr.DataArray
) -> xr.DataArray:
    """Function that adds the geographical centers to the clusters.

    Args:
        masked_data (xr.DataArray): Precursor data before being reduced to clusters,
            with the dimensions latitude and longitude, and cluster labels added.
        reduced_data (xr.DataArray): Data reduced to the clusters, to which the
            geographical centers will be added

    Returns:
        xr.DataArray: Reduced data with the latitude and longitude of the  geographical
            centers added as coordinates of the cluster labels.
    """
    clusters = np.unique(masked_data["cluster_labels"])
    stacked_data = masked_data.stack(coords=("latitude", "longitude"))

    cluster_lats = np.zeros(clusters.shape)
    cluster_lons = np.zeros(clusters.shape)

    for i, cluster in enumerate(clusters):
        # Select only the grid cells within the cluster
        cluster_data = stacked_data.where(
            stacked_data["cluster_labels"] == cluster
        ).dropna(dim="coords")

        # Area weighted mean to get the geographical center of the cluster
        # for the 0 clusters (leftovers), set to nan as this will avoid them in e.g.
        # plots
        if cluster == 0:
            cluster_lats[i] = np.nan
            cluster_lons[i] = np.nan
        else:
            cluster_lats[i] = (
                cluster_data["latitude"].weighted(cluster_data["area"]).mean().item()
            )
            cluster_lons[i] = (
                cluster_data["longitude"].weighted(cluster_data["area"]).mean().item()
            )

    reduced_data["latitude"] = ("cluster_labels", cluster_lats)
    reduced_data["longitude"] = ("cluster_labels", cluster_lons)

    return reduced_data
