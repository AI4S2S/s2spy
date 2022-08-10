"""Response Guided Dimensionality Reduction."""
from typing import Tuple
import numpy as np
import xarray as xr
from scipy.stats import pearsonr as _pearsonr
from sklearn.cluster import DBSCAN


class RGDR:
    """Response Guided Dimensionality Reduction."""

    def __init__(self, timeseries, lag_shift: int = 0):
        """Instantiate an RGDR operator."""
        self.timeseries = timeseries
        self.lag_shift = lag_shift

    def _map_analysis(self):
        """Perform map analysis.

        Use chosen method from `map_analysis` and perform map analysis.
        """
        raise NotImplementedError

    def _clustering_regions(self):
        """Perform regions clustering.

        Use chosen method from `map_regions` and perform clustering of regions
        based on the results from `map_analysis`.
        """
        raise NotImplementedError

    def fit(self, data):
        """Perform RGDR calculations with given data."""
        raise NotImplementedError

    def transform(self, data):
        """Apply RGDR on data, based on the fit model"""
        raise NotImplementedError


radius_earth_km = 6371
surface_area_earth_km2 = 5.1e8


def spherical_area(latitude: float, dlat: float, dlon: float = None) -> float:
    """Approximate the area of a square grid cell on a spherical (!) earth.
    Returns the area in square kilometers of earth surface.

    Args:
        latitude (float): Latitude at the center of the grid cell (deg)
        dlat (float): Latitude grid resolution (deg)
        dlon (float): Longitude grid resolution (deg), optional in case of a square grid.

    Returns:
        float: Area of the grid cell (km^2)
    """
    if dlon is None:
        dlon = dlat
    dlon = np.radians(dlon)
    dlat = np.radians(dlat)

    lat = np.radians(latitude)
    h = np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2)
    spherical_area = h * dlon / np.pi * 4

    return spherical_area * surface_area_earth_km2


def cluster_area(ds: xr.Dataset, cluster_label: float) -> np.ndarray:
    """Determines the total area of a cluster. Requires the input dataset to have the
    variables `area` and `cluster_labels`.

    Args:
        ds (xr.Dataset): Dataset containing the variables `area` and `cluster_labels`.
        cluster_label (float): The label for which the area should be calculated.

    Returns:
        float: Area of the cluster `cluster_label`.
    """
    return (
        ds["area"].where(ds["cluster_labels"] == cluster_label).sum(skipna=True).values
    )


def remove_small_area_clusters(ds: xr.Dataset, min_area_km2: float) -> xr.Dataset:
    """Removes the clusters where the area is under the input threshold.

    Args:
        ds (xr.Dataset): Dataset containing `cluster_labels` and `area`.
        min_area_km2 (float): The minimum allowed area of each cluster

    Returns:
        xr.Dataset: The input dataset with the labels of the clusters set to 0 when the
            area of the cluster is under the `min_area_km2` threshold.
    """
    clusters = np.unique(ds["cluster_labels"])
    areas = [cluster_area(ds, c) for c in clusters]
    valid_clusters = np.array([c for c, a in zip(clusters, areas) if a > min_area_km2])

    ds["cluster_labels"] = ds["cluster_labels"].where(
        np.isin(ds["cluster_labels"], valid_clusters), 0
    )

    return ds


def weighted_groupby(ds: xr.Dataset, groupby: str, weight: str, method: str = "mean") -> xr.Dataset:
    """Apply a weighted reduction after a groupby call. xarray does not currently support
    combining `weighted` and `groupby`. An open PR adds supports for this functionality
    (https://github.com/pydata/xarray/pull/5480), but this branch was never merged.

    Args:
        ds (xr.Dataset): Dataset containing the coordinates or variables specified in
        the `groupby` and `weight` kwargs.
        groupby (str): Coordinate which should be used to make the groups.
        weight (str): Variable in the Dataset containing the weights that should be used.
        method (str): Method that should be used to reduce the dataset, by default
            'mean'. Supports any of xarray's builtin methods, e.g. median, min, max.

    Returns:
        xr.Dataset: Dataset reduced using the `groupby` coordinate, using weights =
            based on `ds[weight]`.
    """
    groups = ds.groupby(groupby)

    # find stacked dim name:
    group_dims = list(groups)[0][1].dims  # Get ds of first group
    stacked_dims = [dim for dim in group_dims.keys() if "stacked_" in dim] # type: ignore

    reduced_data = [
        getattr(g.weighted(g[weight]), method)(dim=stacked_dims) for _, g in groups
    ]
    return xr.concat(reduced_data, dim=groupby)


def masked_spherical_dbscan(
    ds: xr.Dataset, alpha: float = 0.05, eps_km: float = 600, min_area_km2: float = None
) -> xr.Dataset:
    """Determines the clusters based on sklearn's DBSCAN implementation. Alpha determines
    the mask based on the minimum p_value. Grouping can be adjusted using the `eps_km`
    kwarg. Cluster labels are negative for areas with a negative correlation coefficient
    and positive for areas with a positive correlation coefficient. Areas without any
    significant correlation are put in the cluster labelled '0'.

    Args:
        ds (xr.Dataset): Dataset containing 'latitude' and 'longitude' dimensions in
            degrees. Must also contain 'p_val' and 'corr' to base the groups on.
        alpha (float): Value below which the correlation is significant enough to be
            considered
        eps_km (float): The maximum distance (in km) between two samples for one to be
            considered as in the neighborhood of the other. This is not a maximum bound
            on the distances of points within a cluster. This is the most important
            DBSCAN parameter to choose appropriately.
        min_area_km2 (float): The minimum area of a cluster. Clusters smaller than this
            minimum area will be discarded.

    Returns:
        xr.Dataset: Dataset grouped by the DBSCAN clusters.
    """
    ds = ds.stack(coord=["latitude", "longitude"])
    coords = np.asarray(list(ds["coord"].values))  # turn array of tuples to 2d-array
    coords = np.radians(coords)

    # Prepare labels, default value is 0 (not in cluster)
    labels = np.zeros(len(coords))

    for sign, sign_mask in zip([1, -1], [ds["corr"] >= 0, ds["corr"] < 0]):
        mask = np.logical_and(ds["p_val"] < alpha, sign_mask)

        if np.sum(mask) > 0:  # Check if the mask contains any points to cluster
            db = DBSCAN(
                eps=eps_km / radius_earth_km,
                min_samples=1,
                algorithm="auto",
                metric="haversine",
            ).fit(coords[mask])

            labels[mask] = sign * (db.labels_ + 1)

    ds["cluster_labels"] = ("coord", labels)

    ds = ds.unstack(("coord"))

    dlat = np.abs(ds.latitude.values[1] - ds.latitude.values[0])
    dlon = np.abs(ds.longitude.values[1] - ds.longitude.values[0])
    ds["area"] = spherical_area(ds.latitude, dlat, dlon)

    if min_area_km2:
        ds = remove_small_area_clusters(ds, min_area_km2)
    return ds


def reduce_to_clusters(
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
    ds = masked_spherical_dbscan(
        ds, alpha=alpha, eps_km=eps_km, min_area_km2=min_area_km2
    )

    ds = weighted_groupby(ds, groupby="cluster_labels", weight="area")

    return ds


def _pearsonr_nan(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """NaN friendly implementation of scipy.stats.pearsonr. Calculates the correlation
    coefficient between two arrays, as well as the p-value of this correlation. However,
    instead of raising an error when encountering NaN values, this function will return
    both the correlation coefficient and the p-value as NaN.

    Args:
        x: 1-D array
        y: 1-D array
    Returns:
        r_coefficient
        p_value

    """
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return np.nan, np.nan
    return _pearsonr(x, y)


def correlation(
    field: xr.DataArray, target: xr.DataArray, corr_dim: str = "time"
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Calculate correlation maps.

    Args:
        field: Spatial data with a dimension named `corr_dim`, over which each
            location should have the Pearson correlation coefficient calculated with the
            target data.
        target: Data which has to be correlated with the spatial data. Requires a
            dimension named `corr_dim`.
        corr_dim: Dimension over which the correlation coefficient should be calculated.

    Returns:
        r_coefficient: DataArray filled with the correlation coefficient for each
            non-`corr_dim` coordinate.
        p_value: DataArray filled with the two-tailed p-values for each computed
            correlation coefficient.
    """
    assert (
        corr_dim in target.dims
    ), f"input target does not have contain the '{corr_dim}' dimension"
    assert (
        corr_dim in field.dims
    ), f"input field does not have contain the '{corr_dim}' dimension"
    assert np.all(
        [dim in field.dims for dim in target.dims]
    ), "Field and target dims do not match"

    return xr.apply_ufunc(
        _pearsonr_nan,
        field,
        target,
        input_core_dims=[[corr_dim], [corr_dim]],
        vectorize=True,
        output_core_dims=[[], []],
    )


def partial_correlation(field, target, z):
    """Calculate partial correlation maps."""
    raise NotImplementedError


def regression(field, target):
    """Regression analysis on entire maps.

    Methods include Linear, Ridge, Lasso.
    """
    raise NotImplementedError
