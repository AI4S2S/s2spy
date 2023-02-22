import lilio
import numpy as np
import pytest
import xarray as xr
from sklearn.model_selection import ShuffleSplit
import s2spy
import s2spy.rgdr
from s2spy import RGDR
from s2spy.rgdr import label_alignment


TEST_FILE_PATH = "./tests/test_rgdr/test_data"


class TestAlignmentSubfunctions:
    labels = np.array(
        [
            [
                [0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -2, 0, 0],
                [-3, -3, -3, -3, 0, 0, 0, -2, -2, -2, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, -2, -2, -2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -2, -2, -2, 0, 0],
                [-3, 0, 0, 0, 0, 0, -4, 0, -2, -2, 0, -2, 0],
            ],
            [
                [0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0],
                [0, 0, 0, 0, 0, -2, -2, -2, -2, -2, -2, 0, 0],
            ],
        ]
    )

    cluster_labels = xr.DataArray(labels, dims=("split", "latitude", "longitude"))

    expected_set_of_sets = {
        frozenset({"0_-3", "2_-3"}),
        frozenset({"0_-1", "2_-1", "3_-1"}),
        frozenset({"0_-2", "1_-1", "2_-2", "3_-2"}),
        frozenset({"0_-2", "1_-1", "2_-2", "2_-4", "3_-2"}),
        frozenset({"1_1", "2_1", "3_1"}),
        frozenset({"1_-1", "2_-4", "3_-2"}),
    }

    expected_reduced_sets = {
        frozenset({"0_-3", "2_-3"}),
        frozenset({"0_-1", "2_-1", "3_-1"}),
        frozenset({"0_-2", "1_-1", "2_-2", "2_-4", "3_-2"}),
        frozenset({"1_1", "2_1", "3_1"}),
    }

    def test_df_to_full_sets(self):
        set_of_sets = label_alignment.get_overlapping_clusters(
            self.cluster_labels, min_overlap=0.10
        )
        assert self.expected_set_of_sets == set_of_sets

    def test_remove_subsets(self):
        reduced_set_of_sets = label_alignment.remove_subsets(self.expected_set_of_sets)
        assert self.expected_reduced_sets == reduced_set_of_sets

    def test_remove_overlapping_clusters(self):
        dummy_input = {frozenset({"0_-1", "1_-2"}), frozenset({"0_-1", "1_-1", "2_-1"})}
        expected = {frozenset({"1_-2"}), frozenset({"0_-1", "1_-1", "2_-1"})}
        assert expected == label_alignment.remove_overlapping_clusters(dummy_input)

    def test_name_clusters(self):
        dummy_input = {frozenset({"0_-1", "1_-2"}), frozenset({"0_-1", "1_-1", "2_-1"})}
        expected = {
            "A": {"0_-1", "1_-1", "2_-1"},
            "B": {"0_-1", "1_-2"},
        }
        assert expected == label_alignment.name_clusters(dummy_input)

    def test_create_renaming_dict(self):
        dummy_cluster_names = {
            "A": {"0_-1", "1_-1", "2_-1"},
            "B": {"0_-2", "1_-2"},
        }
        expected_dict = {
            0: [(-1, "A"), (-2, "B")],
            1: [(-1, "A"), (-2, "B")],
            2: [(-1, "A")],
        }
        assert expected_dict == label_alignment.create_renaming_dict(
            dummy_cluster_names
        )

    def test_ensure_unique_names(self):
        dummy_names = {0: [(-1, "B")], 1: [(-2, "B"), (-1, "B")], 2: [(-1, "B")]}
        expected = {0: [(-1, "B1")], 1: [(-2, "B1"), (-1, "B2")], 2: [(-1, "B1")]}
        assert expected == label_alignment.ensure_unique_names(dummy_names)


@pytest.fixture(scope="class")
def dummy_calendar():
    return lilio.daily_calendar(anchor="08-01", length="30d")


@pytest.fixture(scope="module")
def raw_target():
    return xr.open_dataset(f"{TEST_FILE_PATH}/tf5_nc5_dendo_80d77.nc").sel(cluster=3)


@pytest.fixture(scope="module")
def raw_field():
    return xr.open_dataset(
        f"{TEST_FILE_PATH}/sst_daily_1979-2018_5deg_Pacific_175_240E_25_50N.nc"
    )


@pytest.fixture(scope="class")
def example_field(raw_field, dummy_calendar):
    cal = dummy_calendar.map_to_data(raw_field)
    return lilio.resample(cal, raw_field).sst


@pytest.fixture(scope="class")
def example_target(raw_target, raw_field, dummy_calendar):
    cal = dummy_calendar.map_to_data(raw_field)
    return lilio.resample(cal, raw_target).ts


def test_alignment_example(example_field, example_target):
    seed = 1  # same 'randomness'
    n_splits = 4

    shufflesplit = ShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=seed)
    cv = lilio.traintest.TrainTestSplit(shufflesplit)

    rgdrs = [
        RGDR(target_intervals=[1], lag=5, eps_km=800, alpha=0.10, min_area_km2=0)
        for _ in range(n_splits)
    ]

    target_timeseries_splits = []
    precursor_field_splits = []
    for x_train, _, _, _ in cv.split(example_field, y=example_target):
        target_timeseries_splits.append(
            example_target.sel(anchor_year=x_train["anchor_year"].values)
        )
        precursor_field_splits.append(
            example_field.sel(anchor_year=x_train["anchor_year"].values)
        )

    clustered_precursors = [
        rgdrs[i].fit_transform(precursor_field_splits[i], target_timeseries_splits[i])
        for i in range(n_splits)
    ]

    aligned_precursors = s2spy.rgdr.label_alignment.rename_labels(
        rgdrs, clustered_precursors
    )

    expected = [["A1"], ["A1", "A2"], ["A1"], ["A1"]]
    clusters_list = [sorted(el.cluster_labels.values) for el in aligned_precursors]

    assert expected == clusters_list
