"""Label alignment tools for RGDR clusters."""
import itertools
import string
from copy import copy
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import xarray as xr


if TYPE_CHECKING:
    from s2spy.rgdr.rgdr import RGDR


def _get_split_cluster_dict(cluster_labels: xr.DataArray) -> dict:
    """Generate a dictionary of all cluster labels in each split.

    Args:
        cluster_labels: DataArray containing all the cluster maps, with the dimension
            "split" for the different clusters over splits.

    Returns:
        Dictionary in the form {0: [cluster_a, cluster_b], 1: [cluster_a], ...}
    """
    return {
        i_split: list(np.unique(split_data.values)[np.unique(split_data.values) != 0])
        for i_split, split_data in enumerate(cluster_labels)
    }


def _flatten_cluster_dict(cluster_dict: dict) -> list[tuple[int, int]]:
    """Flattens a cluster dictionary to a list with (split, cluster) as values.

    For example, if the input is {0: [-1, -2, 1], 1: [-1, 1]}, this function will return
    the following list: [(0, -1), (0, -2), (0, 1), (1, -1), (1, 1)]

    Args:
        cluster_dict: The cluster dictionary which should be flattened

    Returns:
        A list of the clusters and their splits
    """
    flat_clusters: list[tuple[int, int]] = []
    for split in cluster_dict:
        flat_clusters.extend((split, cluster) for cluster in cluster_dict[split])
    return flat_clusters


def _init_overlap_df(cluster_labels: xr.DataArray):
    """Build an empty dataframe with multi-indexes for clusters and labels.

    The structure will be something like the following table:

    split       |        0         1
    label       |       -1        -1
    ------------|-------------------
    split label |
    0     -1    |      NaN  0.583333
    1     -1    | 0.333333       NaN

    The same multi-index is used for both rows and columns, such that the dataframe can
    be populated with the overlap between labels from different splits.

    Args:
        cluster_labels: DataArray containing all the cluster maps, with the dimension
            "split" for the different clusters over splits.

    Returns:
        A pandas dataframe containing a table
    """
    split_label_dict = _get_split_cluster_dict(cluster_labels)
    flat_clusters = _flatten_cluster_dict(split_label_dict)
    multi_index = pd.MultiIndex.from_tuples(flat_clusters, names=("split", "label"))

    return pd.DataFrame(np.nan, index=multi_index, columns=multi_index)


def _calculate_overlap(
    cluster_labels: xr.DataArray,
    split_a: int,
    cluster_a: int,
    split_b: int,
    cluster_b: int,
) -> float:
    """Calculate the overlapping fraction between two clusters, over different splits.

    The overlap is defines as:
        overlap = n_overlapping_cells / total_cells_cluster_a

    Args:
        cluster_labels: DataArray containing all the cluster maps, with the dimension
            "split" for the different clusters over splits.
        split_a: The index of the split of the first cluster
        cluster_a: The value of the first cluster in the clusters_da DataArray.
        split_b: The index of the split of the second cluster
        cluster_b: The value of the second cluster in the clusters_da DataArray.

    Returns:
        Overlap of the first cluster with the second cluster, as a fraction (0.0 - 1.0)
    """
    mask_a = xr.where(cluster_labels.sel(split=split_a) == cluster_a, 1, 0).values
    mask_b = xr.where(cluster_labels.sel(split=split_b) == cluster_b, 1, 0).values

    return np.sum(np.logical_and(mask_a, mask_b)) / np.sum(mask_a)


def calculate_overlap_table(cluster_labels: xr.DataArray) -> pd.DataFrame:
    """Fill the overlap table with the overlap between clusters over different splits.

    Args:
        cluster_labels: DataArray containing all the cluster maps, with the dimension
            "split" for the different clusters over splits.

    Returns:
        The overlap table with all valid combinations filled in. Non valid combinations
            of clusters (the cluster itself, or within the same split) will have NaN
            values.
    """
    overlap_df = _init_overlap_df(cluster_labels)

    all_clusters = _get_split_cluster_dict(cluster_labels)

    pos_clusters = {
        split: [el for el in cluster_list if np.sign(el) == 1]
        for (split, cluster_list) in all_clusters.items()
    }
    neg_clusters = {
        split: [el for el in cluster_list if np.sign(el) == -1]
        for (split, cluster_list) in all_clusters.items()
    }

    for clusters in (pos_clusters, neg_clusters):
        flat_clusters = _flatten_cluster_dict(clusters)

        for split, cluster in flat_clusters:
            other_clusters = _flatten_cluster_dict(
                {k: v for (k, v) in clusters.items() if k != split}
            )
            for other_split, other_cluster in other_clusters:
                overlap = _calculate_overlap(
                    cluster_labels, split, cluster, other_split, other_cluster
                )
                overlap_df.at[(split, cluster), (other_split, other_cluster)] = overlap

    return overlap_df


def get_overlapping_clusters(
    cluster_labels: xr.DataArray, min_overlap: float = 0.1
) -> set:
    """Create sets of overlapping clusters.

    Clusters will be considered to have sufficient overlap if they overlap at least by
    the minimum threshold. Note that this is a one way criterion.

    For example, if the overlap table is like the following:
    split       |    0       1
    label       |   -1      -1
    ------------|-------------
    split label |
    0     -1    |  NaN    0.05
    1     -1    | 0.20     NaN

    Then cluster (split: 0, label: -1) will overlap with cluster (1, -1) by 0.05. This
    is insufficient to be considered the same cluster. However, cluster (1, -1) does
    overlap by 0.20 with cluster (0, -1), so they *will* be considered the same cluster.
    This situation can arise when one cluster is much bigger than another one.

    In this example, the overlapping set will be {frozenset("0_-1", "1_-1")}.
    Note that if we would use a threshold of 0.05, the output would not change, as the
    two nexted sets {"0_-1", "1_-1"} and {"1_-1", "0_-1"} are the same.

    Args:
        cluster_labels: DataArray containing all the cluster maps, with the dimension
            "split" for the different clusters over splits.
        min_overlap: Minimum overlap (0.0 - 1.0) when clusters are considered to be
            sufficiently overlapping to belong to the same signal. Defaults to 0.1.

    Returns:
        A set of (frozen) sets, each set corresponding to a possible combination of
            clusters that overlap.
    """
    overlap_df = calculate_overlap_table(cluster_labels)

    overlap_df.columns = ["_".join([str(el) for el in col]) for col in overlap_df.columns.values]  # type: ignore
    overlap_df.index = ["_".join([str(el) for el in idx]) for idx in overlap_df.index.values]  # type: ignore

    clusters = set()
    for row, _ in enumerate(overlap_df.index):
        overlapping = (
            overlap_df.iloc[row].where(overlap_df.iloc[row] > min_overlap).dropna()
        )
        cluster = set(overlapping.index)
        cluster.add(overlap_df.iloc[row].name)
        clusters.add(frozenset(cluster))

    return clusters


def remove_subsets(clusters: set) -> set:
    """Remove subsets from the clusters.

    For example: {{"A"}, {"A", "B"}} will become {{"A", "B"}}, as "A" is a subset of the
    bigger cluster.
    """
    no_subsets = set(clusters)
    for cluster in clusters:
        no_subsets.remove(cluster)
        if not any(cluster.issubset(cl) for cl in no_subsets):
            no_subsets.add(cluster)
    return no_subsets


def remove_overlapping_clusters(clusters: set) -> set:
    """Remove clusters shared between two different groups of clusters.

    Largest cluster gets priority.

    For example: {{"A", "D"}, {"A", "B", "C"}} will become {{"D"}, {"A", "B", "C"}}
    """
    clusterlist = [set(cl) for cl in clusters]
    clusterlist.sort(key=len)
    sanitised_clusters = []
    for _ in range(len(clusterlist)):
        cluster = clusterlist.pop(0)
        for cl in clusterlist:
            cluster.difference_update(cl)
        sanitised_clusters.append(cluster)
    return {frozenset(cluster) for cluster in sanitised_clusters}


def name_clusters(clusters: set) -> dict:
    """Give each cluster a unique name.

    Note: the first 26 names will be from A - Z. If more than 26 clusters are present,
    these will get names with two uppercase letters (AA - ZZ).

    Args:
        clusters: A set of different clusters. Each element is a list of clusters and
            their splits.

    Returns:
        A dictionary in the form {clustername0: clusters0, cluster_name1: cluster1}
    """
    cluster_names = list(string.ascii_uppercase)
    # Ensure a sufficiently long list of possible names, by extending (A-Z) with (AA-ZZ)
    cluster_names += [a + b for a, b in itertools.product(cluster_names, repeat=2)]
    clusters_list = list(clusters)

    # Ensure reproducable behavior, as sets are unordered.
    clusters_list.sort(
        key=lambda letters: sum(ord(c) for c in str(letters)), reverse=True
    )

    named_clusters = {}
    for cluster in clusters_list:
        name = cluster_names.pop(0)
        named_clusters[name] = set(cluster)

    return named_clusters


def create_renaming_dict(aligned_clusters: dict) -> dict[int, list[tuple[int, str]]]:
    """Create a dictionary that can be used to rename the clusters to the aligned names.

    Args:
        aligned_clusters: A dictionary containing the different splits, and the mapping
            of RGDR clusters to new names.

    Returns:
        A dictionary with the structure {split: [(old_name0, new_name0),
                                                 (old_name1, new_name1)]}.
    """
    inversed_names = {}
    for key in aligned_clusters:
        for el in aligned_clusters[key]:
            inversed_names[el] = key

    renaming_dict: dict[int, list[tuple[int, str]]] = {}
    for key in inversed_names:  # pylint: disable=consider-using-dict-items
        ky = key.split("_")
        split = int(ky[0])
        cluster = int(ky[1])
        if split in renaming_dict:
            apd = renaming_dict[split] + [(cluster, inversed_names[key])]
            renaming_dict[split] = apd
        else:
            renaming_dict[split] = [(cluster, inversed_names[key])]
    return renaming_dict


def ensure_unique_names(renaming_dict: dict[int, list[tuple[int, str]]]) -> dict:
    """Ensure that in every split, every cluster has a unique name.

    The function finds the non-unqiue names within each split, and will rename these by
    adding a number. For example, there are three clusters in the first split with the
    name "C". The new names will be "C", "C1" and "C2".

    If renaming_dict is the following:
        {0: [(-1, "A"), (1, "B")], 1: [(-1, "A"), (-2, "A")]}
    The renamed dictionary will be:
        {0: [(-1, "A1"), (1, "B")], 1: [(-1, "A1"), (-2, "A2")]}

    Args:
        renaming_dict: Renaming dictionary with non unique names.

    Returns:
        Renaming dictionary with only unique names
    """
    renamed_dict = deepcopy(renaming_dict)

    double_names_any: set[str] = set()
    for split in renamed_dict:
        cluster_old_names = [cl for cl, _ in renamed_dict[split]]
        cluster_new_names = [cl for _, cl in renamed_dict[split]]

        double_names = [
            x for x in set(cluster_new_names) if cluster_new_names.count(x) > 1
        ]

        for double_name in double_names:
            double_names_any.add(double_name)
            ndouble = cluster_new_names.count(double_name)
            for i in range(ndouble):
                el = cluster_new_names.index(double_name)
                cluster_new_names[el] = double_name + str(i + 1)
        renamed_dict[split] = list(zip(cluster_old_names, cluster_new_names))

    # If any new label was renamed in a different cluster, e.g. A -> A1, make sure that
    # any other "A" label is also renamed.
    for split, clusters in renamed_dict.items():
        for i, (old_name, new_name) in enumerate(clusters):
            if new_name in double_names_any:
                renamed_dict[split][i] = (old_name, f"{new_name}1")

    return renamed_dict


def _rename_datasets(
    rgdr_list: list["RGDR"], clustered_data: list[xr.DataArray], renaming_dict: dict
) -> list[xr.DataArray]:
    """Apply the renaming dictionary to the labels of the clustered data.

    Args:
        rgdr_list: List of RGDR objects that were used to fit and transform the data.
        clustered_data: List of the RGDR-transformed data. This can either be the
            training data or the test data.
        renaming_dict: Dictionary containing the mapping {old_label: new_label}

    Returns:
        A list of the input clustered data, with the labels renamed.
    """
    renamed = []
    for split, _ in enumerate(rgdr_list):
        # A copy is required to not modify labels of the input data
        data = copy(clustered_data[split])
        labels = data["cluster_labels"].values.astype("U2")
        for cl in renaming_dict[split]:
            if f"{cl[0]}" not in labels:
                break
            labels[labels == f"{cl[0]}"] = cl[1]
        data["cluster_labels"] = labels
        renamed.append(data)
    return renamed


def rename_labels(
    rgdr_list: list["RGDR"], clustered_data: list[xr.DataArray]
) -> list[xr.DataArray]:
    """Return a new object with renamed cluster labels aligned over different splits.

    To aid in users comparing the clustering over different splits, this function tries
    to match the clusters over different splits, and give clusters that are in the same
    region the same name (e.g. "A"). The clusters themselves are not changed, only the
    labels renamed.

    Args:
        rgdr_list: List of RGDR objects that were used to fit and transform the data.
        clustered_data: List of the RGDR-transformed datasets. This can either be the
            training data or the test data.

    Returns:
        A list of the input clustered data, with the labels renamed.
    """
    n_splits = len(rgdr_list)
    cluster_maps = [rgdr_list[split].cluster_map.reset_coords() for split in range(n_splits)]  # type: ignore

    cluster_map_ds = xr.concat(cluster_maps, dim="split")

    clusters = get_overlapping_clusters(cluster_map_ds["cluster_labels"])
    clusters = remove_subsets(clusters)
    clusters = remove_overlapping_clusters(clusters)
    cluster_names = name_clusters(clusters)

    renaming_dict = create_renaming_dict(cluster_names)
    renaming_dict = ensure_unique_names(renaming_dict)

    return _rename_datasets(rgdr_list, clustered_data, renaming_dict)
