import numpy as np
import xarray as xr


surface_area_earth_km2 = 5.1e8


def spherical_area(latitude: float, resolution: float) -> float:
    """Approximate the area of a square grid cell on a spherical (!) earth.
    Returns the area in square kilometers of earth surface.

    Args:
        latitude (float): Latitude at the center of the grid cell (deg)
        resolution (float): Grid resolution (deg)

    Returns:
        float: Area of the grid cell (km^2)
    """
    lat = np.radians(latitude)
    resolution = np.radians(resolution)
    h = np.sin(lat + resolution / 2) - np.sin(lat - resolution / 2)
    spherical_area = h * resolution / np.pi * 4
    return spherical_area * surface_area_earth_km2


def cluster_area(ds: xr.Dataset, cluster_label: float) -> float:
    """Determines the total area of a cluster. Requires the input dataset to have the
    variables `area` and `cluster_labels`.

    Args:
        ds (xr.Dataset): Dataset containing the variables `area` and `cluster_labels`.
        cluster_label (float): The label for which the area should be calculated.

    Returns:
        float: Area of the cluster `cluster_label`.
    """
    # Use where statement to
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
