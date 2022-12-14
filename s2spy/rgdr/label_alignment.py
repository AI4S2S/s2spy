import string
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
import numpy as np
import pandas as pd
import xarray as xr
from s2spy.rgdr import utils


if TYPE_CHECKING:
    from s2spy.rgdr.rgdr import RGDR


def _extract_split_labels(split_array: xr.DataArray):
    """Generate a dictionary of all cluster labels in each split."""

    split_labels_dict = {}
    # loop over splits
    for split_number, split in enumerate(split_array):
        unique_labels = np.unique(split)
        unique_labels = unique_labels[unique_labels != 0]  # 0 are no precursor fields
        split_labels_dict[split_number] = list(unique_labels)

    return split_labels_dict


def array_label_count(array: np.ndarray):
    """Count the total occurrence of each unique label."""
    unique, counts = np.unique(array, return_counts=True)
    return dict(zip(unique, counts))


def _init_overlap_df(split_array: xr.DataArray):
    """Build an empty dataframe with multi-indexes for clusters and labels.

    The structure will be something like {split1: [label1, label2], split2: [label1, label3], ...}.
    The same multi-index is used for both rows and columns, such that the dataframe can
    be populated with the overlap between labels from different splits.
    """

    split_label_dict = _extract_split_labels(split_array)

    # create a list of tuples:
    tuple_list = []  # type: ignore
    for split_number, label_list in split_label_dict.items():
        tuple_list.extend((split_number, label) for label in label_list)

    # create multi index with tuple list
    multi_index = pd.MultiIndex.from_tuples(tuple_list, names=("split", "label"))

    # create empty dataframe
    df = pd.DataFrame(np.nan, index=multi_index, columns=multi_index)

    # add label cell count column
    df["label_cell_count"] = 0

    # get label cell counts
    for split_number, label_list in split_label_dict.items():
        for label in label_list:
            split = split_array.sel({"split": split_number})
            split_labelmask = split.where(split == label, 0)
            label_count = array_label_count(split_labelmask.values)
            df.at[(split_number, label), "label_cell_count"] = label_count[label]

    return df


# pylint: disable=too-many-locals
def overlap_labels(da_splits: xr.DataArray):
    """
    Function to create a dataframe that shows how many grid cells
    of a found precursor region of a lag in a training split overlap with
    a region found in another split at the same lag.
    Args:
        - da_splits: an xr.DataArray containing the cluster labels with
        dimensions 'lag' and 'split'
        - lag: optional, if given one can give the df and lag separately,
        else the da_splits has no dimension i_interval
    """

    # initialize empty dataframe
    overlap_df = _init_overlap_df(da_splits)

    for split_number in da_splits.split.values:

        # select a split
        split = da_splits.sel({"split": split_number})
        # get labels
        labels = overlap_df.loc[split_number, :].index.get_level_values(0)

        for label in labels:

            # mask the array for label
            split_labelmask = split.where(split == label, 0)
            # convert where equal to label to 1
            split_labelmask = split_labelmask.where(split_labelmask != label, 1)

            # create other split list
            other_split_number_list = [
                other_split_number
                for other_split_number in da_splits.split.values
                if other_split_number != split_number
            ]

            # loop over other splits
            for other_split_number in other_split_number_list:

                # select other split
                other_split = da_splits.sel({"split": other_split_number})
                # get list of labels in this split
                other_split_labels = overlap_df.loc[
                    other_split_number, :
                ].index.get_level_values(0)
                # create list with labels in other split with same sign as label
                other_split_labels_samesign = [
                    other_label
                    for other_label in other_split_labels
                    if np.sign(other_label) == np.sign(label)
                ]

                # loop over labels of same sign
                for other_label in other_split_labels_samesign:

                    # mask for other_label
                    other_split_labelmask = other_split.where(
                        other_split == other_label, 0
                    )
                    # convert where equal to label to 1
                    other_split_labelmask = other_split.where(
                        other_split_labelmask != other_label, 1
                    )

                    # check where regions overlap
                    overlap_array = split_labelmask.where(
                        split_labelmask == other_split_labelmask
                    )

                    # count the number of cells in the overlap array
                    overlap_count_dict = array_label_count(overlap_array.values)
                    overlap_count = (
                        0 if 1 not in overlap_count_dict else overlap_count_dict[1]
                    )

                    # append to df
                    overlap_df.at[(split_number, label), (other_split_number, other_label)] = (
                        overlap_count
                        / overlap_df.loc[(split_number, label), "label_cell_count"]
                    )  # type: ignore

    return overlap_df


def get_overlapping_clusters(cluster_labels, min_overlap: float = 0.1) -> Set:
    """Creates sets of overlapping clusters.

    Clusters will be considered to have sufficient overlap if they overlap at least by
    the minimum threshold. Note that this is a one way criterion.

    Args:
        cluster_labels
        min_overlap: Minimum overlap when clusters are considered to be sufficiently
            overlapping to belong to the same signal. Defaults to 0.1.

    Returns:
        A set of (frozen) sets, each set corresponding to a possible combination of
            clusters that overlap.
    """
    overlap_df = overlap_labels(cluster_labels)
    overlap_df = overlap_df.drop(columns="label_cell_count")
    overlap_df.columns = [
        "_".join([str(el) for el in col]) for col in overlap_df.columns.values
    ]  # type: ignore
    overlap_df.index = [
        "_".join([str(el) for el in idx]) for idx in overlap_df.index.values
    ]  # type: ignore

    clusters = set()
    for row in range(overlap_df.index.size):  # type: ignore
        overlapping = (
            overlap_df.iloc[row].where(overlap_df.iloc[row] > min_overlap).dropna()
        )
        cluster = set(overlapping.index)
        cluster.add(overlap_df.iloc[row].name)
        clusters.add(frozenset(cluster))

    return clusters


def remove_subsets(clusters: Set) -> Set:
    """Removes subsets from the clusters.

    For example: {{"A"}, {"A", "B"}} will become {{"A", "B"}}, as "A" is a subset of the
    bigger cluster.
    """
    no_subsets = set(clusters)
    for cluster in clusters:
        no_subsets.remove(cluster)
        if not any(cluster.issubset(cl) for cl in no_subsets):
            no_subsets.add(cluster)
    return no_subsets


def remove_overlapping_clusters(clusters: Set) -> Set:
    """Removes clusters shared between two different groups of clusters.
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


def name_clusters(clusters: Set) -> Dict:
    """Gives each cluster a unique letter.

    Args:
        clusters: A set of different clusters. Each element is a list of clusters and
            their splits.

    Returns:
        A dictionary in the form {clustername0: clusters0, cluster_name1: cluster1}
    """
    # Ensure a sufficiently long list of possible names
    cluster_names = list(string.ascii_uppercase) + list(string.ascii_lowercase)

    clusters_list = list(clusters)

    # Ensure reproducable behavior, as sets are unordered.
    clusters_list.sort(key=lambda l: sum(ord(c) for c in str(l)), reverse=True)

    named_clusters = {}
    for cluster in clusters_list:
        name = cluster_names.pop(0)
        named_clusters[name] = set(cluster)

    return named_clusters


def create_renaming_dict(aligned_clusters: Dict) -> Dict:
    """Creates a dictionary that can be used to rename the clusters to the aligned names.

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

    renaming_dict: Dict[int, List[Tuple[int, str]]] = {}
    for key in inversed_names: # pylint: disable=consider-using-dict-items
        ky = key.split("_")
        split = int(ky[0])
        cluster = int(ky[1])
        if split in renaming_dict:
            apd = renaming_dict[split] + [(cluster, inversed_names[key])]
            renaming_dict[split] = apd
        else:
            renaming_dict[split] = [(cluster, inversed_names[key])]
    return renaming_dict


def ensure_unique_names(renaming_dict: Dict) -> Dict:
    """This function ensures that in every split, every cluster has a unique name.

    The function finds the non-unqiue names within each split, and will rename these by
    adding a number. For example, there are three clusters in the first split with the
    name "C". The new names will be "C", "C1" and "C2".

    Args:
        renaming_dict: Renaming dictionary with non unique names.

    Returns:
        Renaming dictionary with only unique names
    """
    renamed_dict = deepcopy(renaming_dict)
    for split in renamed_dict:
        cluster_old_names = [cl for cl, _ in renamed_dict[split]]
        cluster_new_names = [cl for _, cl in renamed_dict[split]]

        double_names = [
            x if cluster_new_names.count(x) > 1 else None
            for x in set(cluster_new_names)
        ]
        # pylint:disable=singleton-comparison
        double_names = list(np.array(double_names)[np.array(double_names) != None])  # noqa

        zero_names = []
        for double_name in double_names:
            ndouble = cluster_new_names.count(double_name)
            for i in range(ndouble):
                el = cluster_new_names.index(double_name)
                cluster_new_names[el] = double_name + str(i)  # type: ignore
                if i == 0:
                    zero_names.append(double_name + str(i))
        # Re-rename the "name0" cluster to "name"
        for name in zero_names:
            cluster_new_names[cluster_new_names.index(name)] = name.replace("0", "")
        renamed_dict[split] = list(zip(cluster_old_names, cluster_new_names))
    return renamed_dict


def _rename_datasets(
    rgdr_list: List["RGDR"], clustered_data: List[xr.DataArray], renaming_dict: Dict
) -> List[xr.DataArray]:
    """Applies the renaming dictionary to the labels of the clustered data.

    Args:
        rgdr_list: List of RGDR objects that were used to fit and transform the data.
        clustered_data: List of the RGDR-transformed data. This can either be the
            training data or the test data.
        renaming_dict: Dictionary containing the mapping {old_label: new_label}

    Returns:
        A list of the input clustered data, with the labels renamed.
    """
    renamed = []
    for split, _rgdr in enumerate(rgdr_list):
        data = clustered_data[split]
        labels = data["cluster_labels"].values
        i_interval = str(_rgdr.cluster_map["i_interval"].values)  # type: ignore
        for cl in renaming_dict[split]:
            if f"i_interval:{i_interval}_cluster:{cl[0]}" not in labels:
                break
            labels[labels == f"i_interval:{i_interval}_cluster:{cl[0]}"] = cl[1]
        data["cluster_labels"] = labels
        renamed.append(data)
    return renamed


def rename_labels(
    rgdr_list: List["RGDR"], clustered_data: List[xr.DataArray]
) -> List[xr.DataArray]:
    """Renames labels of clustered data over different splits to have similar names.

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
    cluster_maps = [
        utils.cluster_labels_to_ints(
            rgdr_list[split].cluster_map.reset_coords()  # type: ignore
        )
        for split in range(n_splits)
    ]
    cluster_map_ds = xr.concat(cluster_maps, dim="split")

    clusters = get_overlapping_clusters(cluster_map_ds["cluster_labels"])
    clusters = remove_subsets(clusters)
    clusters = remove_overlapping_clusters(clusters)
    cluster_names = name_clusters(clusters)

    renaming_dict = create_renaming_dict(cluster_names)
    renaming_dict = ensure_unique_names(renaming_dict)

    return _rename_datasets(rgdr_list, clustered_data, renaming_dict)
