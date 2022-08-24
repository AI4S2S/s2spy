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
            cluster_area = cluster_area.dropna('i_interval', how='all')
        cluster_area = cluster_area.dropna("coords")

        # Area weighted mean to get the geographical center of the cluster
        cluster_lats[i] = (
            cluster_area["latitude"].weighted(cluster_area).mean().item()
        )
        cluster_lons[i] = (
            cluster_area["longitude"].weighted(cluster_area).mean().item()
        )

    reduced_data["latitude"] = ("cluster_labels", cluster_lats)
    reduced_data["longitude"] = ("cluster_labels", cluster_lons)

    return reduced_data


def cluster_labels_to_ints(clustered_data: xr.DataArray) -> xr.DataArray:
    """Converts the labels of already clustered data to integers.

    Args:
        clustered_data: Data already clustered and grouped by cluster.

    Returns:
        Same as input, but with the labels converted to integers
    """
    un_labels = np.unique(clustered_data.cluster_labels)
    label_vals = [int(lb[-2:].replace(":","")) for lb in un_labels]
    label_lookup = dict(zip(un_labels, label_vals))

    clustered_data['cluster_labels'] = xr.apply_ufunc(
        lambda val: label_lookup[val],
        clustered_data['cluster_labels'],
        vectorize=True
    )

    return clustered_data
