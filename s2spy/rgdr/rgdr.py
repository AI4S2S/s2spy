"""Response Guided Dimensionality Reduction."""
import warnings
from os import linesep
from typing import Optional
from typing import TypeVar
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import QuadMesh
from scipy.stats import pearsonr as _pearsonr
from sklearn.cluster import DBSCAN
from . import utils


RADIUS_EARTH_KM = 6371
SURFACE_AREA_EARTH_KM2 = 5.10072e8
XrType = TypeVar("XrType", xr.DataArray, xr.Dataset)


def spherical_area(latitude: float, dlat: float, dlon: Optional[float] = None) -> float:
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
    spherical_area = h * dlon / (np.pi * 4)

    return spherical_area * SURFACE_AREA_EARTH_KM2


def cluster_area(ds: XrType, cluster_label: float) -> float:
    """Determine the total area of a cluster.

    Requires the input dataset to have the variables `area` and `cluster_labels`.

    Args:
        ds (xr.Dataset or xr.DataArray): Dataset/DataArray containing the variables
            `area` and `cluster_labels`.
        cluster_label (float): The label (as float) for which the area should be
            calculated.

    Returns:
        float: Area of the cluster `cluster_label`.
    """
    return (
        ds["area"]
        .where(ds["cluster_labels"] == cluster_label)
        .sum(skipna=True)
        .values.item()
    )


def remove_small_area_clusters(ds: XrType, min_area_km2: float) -> XrType:
    """Remove the clusters where the area is under the input threshold.

    Args:
        ds (xr.DataArray, xr.Dataset): Dataset containing `cluster_labels` and `area`.
        min_area_km2 (float): The minimum allowed area of each cluster

    Returns:
        xr.DataArray, xr.Dataset: The input dataset with the labels of the clusters set
            to 0 when the area of the cluster is under the `min_area_km2` threshold.
    """
    clusters = np.unique(ds["cluster_labels"])
    areas = [cluster_area(ds, c) for c in clusters]
    valid_clusters = np.array([c for c, a in zip(clusters, areas) if a > min_area_km2])

    ds["cluster_labels"] = ds["cluster_labels"].where(
        np.isin(ds["cluster_labels"], valid_clusters), 0
    )

    return ds


def add_gridcell_area(data: xr.DataArray):
    """Add the area of each gridcell (latitude) in km2.

    Note: Assumes an even grid (in degrees)

    Args:
        data: Data containing lat, lon coordinates in degrees.

    Returns:
        Input data with an added coordinate "area".
    """
    dlat = np.abs(data.latitude.values[1] - data.latitude.values[0])
    dlon = np.abs(data.longitude.values[1] - data.longitude.values[0])
    data["area"] = spherical_area(data.latitude, dlat, dlon)
    return data


def assert_clusters_present(data: xr.DataArray) -> None:
    """Assert that any (non-'0') clusters are present in the data."""
    if np.unique(data.cluster_labels).size == 1:
        warnings.warn(
            "No significant clusters found in the input DataArray", stacklevel=2
        )


def _get_dbscan_clusters(
    data: xr.Dataset, coords: np.ndarray, dbscan_params: dict
) -> np.ndarray:
    """Generate the DBSCAN cluster labels based on the correlation and p-value.

    Args:
        data: DataArray of the precursor field, of only a single
             i_interval. Requires the 'latitude' and 'longitude' dimensions to be stacked
             into a "coords" dimension.
        coords: 2-D array containing the coordinates of each (lat, lon) grid
            point, in radians.
        dbscan_params: Dictionary containing the elements 'alpha', 'eps',
            'min_area_km2'. See the documentation of RGDR for more information.

    Returns:
        np.ndarray: 1-D array of the same length as `coords`, containing cluster labels
            for every coordinate.
    """
    labels = np.zeros(len(coords), dtype=int)

    for sign, sign_mask in zip([1, -1], [data["corr"] >= 0, data["corr"] < 0]):
        mask = np.logical_and(data["p_val"] < dbscan_params["alpha"], sign_mask)

        if np.sum(mask) > 0:  # Check if the mask contains any points to cluster
            db = DBSCAN(
                eps=dbscan_params["eps"] / RADIUS_EARTH_KM,
                min_samples=1,
                algorithm="auto",
                metric="haversine",
            ).fit(coords[mask])

            labels[mask] = sign * (db.labels_ + 1)

    return labels


def _find_clusters(
    precursor: xr.DataArray,
    corr: xr.DataArray,
    p_val: xr.DataArray,
    dbscan_params: dict,
) -> xr.DataArray:
    """Compute clusters and adds their labels to the precursor dataset.

    For clustering the DBSCAN algorithm is used, with a Haversine distance metric.

    Args:
        precursor (xr.DataArray): DataArray of the precursor field, containing
            'latitude' and 'longitude' dimensions in degrees.
        corr (xr.DataArray): DataArray with the correlation values, generated by
            correlation_map()
        p_val (xr.DataArray): DataArray with the p-values, generated by
            correlation_map()
        dbscan_params (dict): Dictionary containing the elements 'alpha', 'eps',
            'min_area_km2'. See the documentation of RGDR for more information.

    Returns:
        xr.DataArray: The input precursor data, with as extra coordinate labelled
            clusters.
    """
    data = precursor.to_dataset()
    data["corr"], data["p_val"] = corr, p_val  # Will require less tracking of indices

    data = data.stack(coord=["latitude", "longitude"])
    coords = np.asarray(data["coord"].values.tolist())
    coords = np.radians(coords)

    labels = _get_dbscan_clusters(data, coords, dbscan_params)

    precursor = precursor.stack(coord=["latitude", "longitude"])
    precursor["cluster_labels"] = ("coord", labels)
    precursor["cluster_labels"] = precursor["cluster_labels"].astype("int16")
    precursor = precursor.unstack("coord")

    return precursor


def masked_spherical_dbscan(
    precursor: xr.DataArray,
    corr: xr.DataArray,
    p_val: xr.DataArray,
    dbscan_params: dict,
) -> xr.DataArray:
    """Determine the clusters based on sklearn's DBSCAN implementation.

    Alpha determines the mask based on the minimum p_value. Grouping can be
    adjusted using the `eps_km` parameter. Cluster labels are negative for
    areas with a negative correlation coefficient and positive for areas with
    a positive correlation coefficient. Areas without any significant correlation
    are put in the cluster labelled '0'.

    Args:
        precursor (xr.DataArray): DataArray of the precursor field, containing
            'latitude' and 'longitude' dimensions in degrees.
        corr (xr.DataArray): DataArray with the correlation values, generated by
            correlation_map()
        p_val (xr.DataArray): DataArray with the p-values, generated by
            correlation_map()
        dbscan_params (dict): Dictionary containing the elements 'alpha', 'eps',
            'min_area_km2'. See the documentation of RGDR for more information.

    Returns:
        xr.DataArray: Precursor data grouped by the DBSCAN clusters.
    """
    precursor = add_gridcell_area(precursor)
    precursor = _find_clusters(precursor, corr, p_val, dbscan_params)

    if dbscan_params["min_area"] is not None:
        precursor = remove_small_area_clusters(precursor, dbscan_params["min_area"])

    assert_clusters_present(precursor)

    return precursor


def _pearsonr_nan(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """NaN friendly implementation of scipy.stats.pearsonr.

    Calculates the correlation coefficient between two arrays, as well as the p-value
    of this correlation. However, instead of raising an error when encountering NaN
    values, this function will return both the correlation coefficient and the p-value as NaN.

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
) -> tuple[xr.DataArray, xr.DataArray]:
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


def stack_input_data(precursor, target, precursor_intervals, target_intervals):
    """Stack input data."""
    target = target.sel(i_interval=target_intervals).stack(
        anch_int=["anchor_year", "i_interval"]
    )
    precursor = precursor.sel(i_interval=precursor_intervals).stack(
        anch_int=["anchor_year", "i_interval"]
    )

    precursor = precursor.drop_vars({"anchor_year", "anch_int", "i_interval"})
    target = target.drop_vars({"anchor_year", "anch_int", "i_interval"})

    precursor["anch_int"] = range(precursor["anch_int"].size)
    target["anch_int"] = range(target["anch_int"].size)

    return precursor, target


class RGDR:
    """Response Guided Dimensionality Reduction."""

    def __init__(  # noqa: PLR0913 (too-many-arguments)
        self,
        target_intervals: Union[int, list[int]],
        lag: int,
        eps_km: float,
        alpha: float,
        min_area_km2: Optional[float] = None,
    ) -> None:
        """Response Guided Dimensionality Reduction (RGDR).

        Dimensionality reduction based on the correlation between precursor field and
        target timeseries.

        Args:
            target_intervals: The target interval indices which should be correlated
                with the precursors. Input in the form of "[1, 2, 3]" or "1"
                The precursor intervals will be determined based on the `lag` kwarg.
            lag: The lag between the precursor and target intervals to compute the
                correlation, akin to lag in cross-correlation. E.g. if the target
                intervals are [1, 2], and lag is 2, the precursor intervals will be
                [-2, -1]
            alpha (float): p-value below which the correlation is considered significant
                enough for a location to be included in a cluster.
            eps_km (float): The maximum distance (in km) between two grid cells for them
                to be considered to be in the same cluster. This is not a maximum bound
                on the distances between grid cells within a cluster.

                The minimum appropriate value is equal to the maximum width/height of a
                grid cell (i.e. the width of the grid cell closest to the equator).
                The upper bound depends on when cells or clusters would still be
                considered to be part of the same climate signal (i.e., precursor region).

                Higher values can lead to fewer clusters, but also clusters in which the
                cells of the same cluster are separated by large geographical distances.
            min_area_km2 (float): The minimum area of a cluster (in square km). Clusters
                smaller than this minimum area will be discarded.

        Attributes:
            corr_map: correlation coefficient map of given precursor field and
                target series.
            pval_map: p-values map of correlation
            cluster_map: cluster labels for precursor field masked by p-values
        """
        self._lag = lag
        self._dbscan_params = {"eps": eps_km, "alpha": alpha, "min_area": min_area_km2}

        self._target_intervals = (
            [target_intervals]
            if isinstance(target_intervals, int)
            else target_intervals
        )
        self._precursor_intervals = utils.intervals_subtract(
            self._target_intervals, lag
        )

        self._corr_map: Union[None, xr.DataArray] = None
        self._pval_map: Union[None, xr.DataArray] = None
        self._cluster_map: Union[None, xr.DataArray] = None

        self._area = None

    @property
    def target_intervals(self) -> list[int]:
        """Return target intervals."""
        return self._target_intervals

    @property
    def precursor_intervals(self) -> list[int]:
        """Return precursor intervals."""
        return self._precursor_intervals

    @property
    def cluster_map(self) -> xr.DataArray:
        """Return cluster map."""
        if self._cluster_map is None:
            raise ValueError(
                "No cluster map exists yet, .fit() has to be called first."
            )
        return self._cluster_map

    @property
    def pval_map(self) -> xr.DataArray:
        """Return p-value map."""
        if self._pval_map is None:
            raise ValueError(
                "No p-value map exists yet, .fit() has to be called first."
            )
        return self._pval_map

    @property
    def corr_map(self) -> xr.DataArray:
        """Return correlation map."""
        if self._corr_map is None:
            raise ValueError(
                "No correlation map exists yet, .fit() has to be called first."
            )
        return self._corr_map

    def get_correlation(
        self,
        precursor: xr.DataArray,
        target: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Calculate the correlation and p-value between input precursor and target.

        Args:
            precursor: Precursor field data with the dimensions
                'latitude', 'longitude', and 'anchor_year'
            target: Timeseries data with only the dimension 'anchor_year'

        Returns:
            (correlation, p_value): DataArrays containing the correlation and p-value.
        """
        if not isinstance(precursor, xr.DataArray):
            raise ValueError("Please provide an xr.DataArray, not a dataset")

        p, t = stack_input_data(
            precursor, target, self._precursor_intervals, self._target_intervals
        )

        return correlation(p, t, corr_dim="anch_int")

    def get_clusters(
        self,
        precursor: xr.DataArray,
        target: xr.DataArray,
    ) -> xr.DataArray:
        """Generate clusters for the precursor data.

        Args:
            precursor: Precursor field data with the dimensions
                'latitude', 'longitude', 'anchor_year', and 'i_interval'
            target: Target timeseries data with only the dimensions 'anchor_year' and
                'i_interval'

        Returns:
            DataArray containing the clusters as masks.
        """
        corr, p_val = self.get_correlation(precursor, target)
        return masked_spherical_dbscan(precursor, corr, p_val, self._dbscan_params)

    def preview_correlation(  # noqa: PLR0913 (too-many-arguments)
        self,
        precursor: xr.DataArray,
        target: xr.DataArray,
        add_alpha_hatch: bool = True,
        ax1: Optional[plt.Axes] = None,
        ax2: Optional[plt.Axes] = None,
    ) -> list[QuadMesh]:
        """Preview correlation and p-value results with given inputs.

        Generate a figure showing the correlation and p-value results with the
        initiated RGDR class and input precursor field.

        Args:
            precursor: Precursor field data with the dimensions
                'latitude', 'longitude', 'anchor_year', and 'i_interval'
            target: Target timeseries data with only the dimensions 'anchor_year' and
                'i_interval'
            add_alpha_hatch: Adds a red hatching when the p-value is lower than the
                RGDR's 'alpha' value.
            ax1: a matplotlib axis handle to plot the correlation values into.
                If None, an axis handle will be created instead.
            ax2: a matplotlib axis handle to plot the p-values into. If None, an axis
                handle will be created instead.

        Returns:
            List of matplotlib QuadMesh artists.
        """
        corr, p_val = self.get_correlation(precursor, target)

        if (ax1 is None) and (ax2 is None):
            _, (ax1, ax2) = plt.subplots(ncols=2)
        elif (ax1 is None) or (ax2 is None):
            raise ValueError(
                "Either pass axis handles for both ax1 and ax2, or pass neither."
            )

        plot1 = corr.plot.pcolormesh(ax=ax1, cmap="coolwarm")  # type: ignore
        plot2 = p_val.plot.pcolormesh(ax=ax2, cmap="viridis_r", vmin=0, vmax=1)  # type: ignore

        if add_alpha_hatch:
            coords = plot2.get_coordinates()
            plt.rcParams["hatch.color"] = "r"
            plt.pcolor(
                coords[:, :, 0],
                coords[:, :, 1],
                p_val.where(p_val < self._dbscan_params["alpha"]).values,
                hatch="x",
                alpha=0.0,
            )

        ax1.set_title("correlation")
        ax2.set_title("p-value")

        return [plot1, plot2]

    def preview_clusters(
        self,
        precursor: xr.DataArray,
        target: xr.DataArray,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> QuadMesh:
        """Preview clusters.

        Generates a figure showing the clusters resulting from the initiated RGDR
        class and input precursor field.

        Args:
            precursor: Precursor field data with the dimensions
                'latitude', 'longitude', 'anchor_year', and 'i_interval'
            target: Target timeseries data with only the dimensions 'anchor_year' and
                'i_interval'
            ax (plt.Axes, optional): a matplotlib axis handle to plot the clusters
                into. If None, an axis handle will be created instead.
            **kwargs: Keyword arguments that should be passed to QuadMesh.

        Returns:
            Matplotlib QuadMesh artist.
        """
        if ax is None:
            _, ax = plt.subplots()

        clusters = self.get_clusters(precursor, target)

        return clusters["cluster_labels"].plot(cmap="viridis", ax=ax, **kwargs)  # type: ignore

    def fit(self, precursor: xr.DataArray, target: xr.DataArray):
        """Fit RGDR clusters to precursor data.

        Performs DBSCAN clustering on a prepared DataArray, and then groups the data by
        their determined clusters, using an weighted mean. The weight is based on the
        area of each grid cell.

        Density-Based Spatial Clustering of Applications with Noise (DBSCAN) clusters
        gridcells together which are of the same sign and in proximity to
        each other using DBSCAN.

        Clusters labelled with a positive value represent a positive correlation with
        the target timeseries, the clusters labelled with a negative value represent a
        negative correlation. All locations not in a cluster are grouped together under
        the label '0'.

        Args:
            precursor: Precursor field data with the dimensions
                'latitude', 'longitude', 'anchor_year', and 'i_interval'
            target: Target timeseries data with only the dimensions 'anchor_year' and
                'i_interval', which will be correlated with the precursor field.

        Returns:
            xr.DataArray: The precursor data, with the latitute and longitude dimensions
                reduced to clusters.
        """
        corr, p_val = self.get_correlation(precursor, target)

        masked_data = masked_spherical_dbscan(
            precursor, corr, p_val, self._dbscan_params
        )
        self._corr_map = corr
        self._pval_map = p_val
        self._cluster_map = masked_data.cluster_labels
        self._area = masked_data.area

    def transform(self, data: xr.DataArray) -> xr.DataArray:
        """Apply RGDR on the input data, based on the previous fit.

        Transform will use the clusters previously generated when RGDR was fit, and use
        these clusters to reduce the latitude and longitude dimensions of the input
        data.
        """
        if self.cluster_map is None:
            raise ValueError(
                "Transform requires the model to be fit on other data first"
            )
        data = data.sel(i_interval=self._precursor_intervals)
        data["cluster_labels"] = self.cluster_map
        data["area"] = self._area

        reduced_data = utils.weighted_groupby(
            data, groupby="cluster_labels", weight="area"
        )

        # Add the geographical centers for later alignment between, e.g., splits
        reduced_data = utils.geographical_cluster_center(data, reduced_data)
        # Include explanations about geographical centers as attributes
        reduced_data.attrs[
            "data"
        ] = "Clustered data with Response Guided Dimensionality Reduction."
        reduced_data.attrs[
            "coordinates"
        ] = "Latitudes and longitudes are geographical centers associated with clusters."

        # Remove the '0' cluster
        reduced_data = reduced_data.where(reduced_data["cluster_labels"] != 0).dropna(
            dim="cluster_labels"
        )

        return reduced_data.transpose(..., "cluster_labels")

    def fit_transform(self, precursor: xr.DataArray, timeseries: xr.DataArray):
        """Fit RGDR clusters to precursor data, and applies RGDR on the input data.

        Args:
            precursor: Precursor field data with the dimensions 'latitude', 'longitude',
                and 'anchor_year'
            timeseries: Timeseries data with only the dimension 'anchor_year', which
                will be correlated with the precursor field.

        Returns:
            xr.DataArray: The precursor data, with the latitute and longitude dimensions
                reduced to clusters.
        """
        self.fit(precursor, timeseries)
        return self.transform(precursor)

    def __repr__(self) -> str:
        """Represent the RGDR transformer with strings."""
        props = [
            ("target_intervals", repr(self.target_intervals)),
            ("lag", repr(self._lag)),
            ("eps_km", repr(self._dbscan_params["eps"])),
            ("alpha", repr(self._dbscan_params["alpha"])),
            ("min_area_km2", repr(self._dbscan_params["min_area"])),
        ]

        propstr = f"{linesep}\t" + f",{linesep}\t".join(
            [f"{k}={v}" for k, v in props]
        )  # sourcery skip: use-fstring-for-concatenation
        return f"{self.__class__.__name__}({propstr}{linesep})".replace("\t", "    ")
