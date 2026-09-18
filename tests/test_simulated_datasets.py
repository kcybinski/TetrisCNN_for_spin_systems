"""The simulated datasets: ILGT, 1D TFIM, XXZ.

TetrisCNN was developed as a proof of concept on numerically simulated data before
it was applied to the experimental Rydberg snapshots, and both routes are meant to
keep working. These tests pin the parts of that compatibility that are easy to break
from the experimental side: that each simulated dataset is reachable through
`create_datasets()`, that the per-snapshot tuning-parameter ("times") array the
phase-indicator machinery reads is populated on the pre-split path, and that the
failure modes a user is most likely to hit say what to do about them.

Tests needing snapshots on disk are skipped when the dataset tree is absent.
"""
import numpy as np
import pytest
import torch

from tests.conftest import DATA_ROOT
from tetriscnn.dataprocessing import (
    DatasetNotAvailableError,
    partition_threshold,
    relabel_by_threshold,
)
from tetriscnn.datasets import create_datasets, get_no_partitions
from tetriscnn.utils import AttrDict, set_kernels, validate_kernels_fit


def _cf(**overrides):
    """A cf carrying just the keys create_datasets() reads."""
    cf = AttrDict()
    cf.update(dict(
        task="regression", label_param="beta", normalize_labels=True,
        label_subset_count=None, filter_t_values=None, data_fraction=1,
        param_cutoffs={"delta": 6, "omega": 0}, even_split=False,
        samples_per_pt_cap=None, partition_index=None, seed=42,
    ))
    cf.update(overrides)
    return cf


def _have(relative):
    return (DATA_ROOT / relative).is_dir()


ilgt_needed = pytest.mark.skipif(
    not _have("simulated/ILGT_regression"), reason="ILGT dataset not present")
tfim_needed = pytest.mark.skipif(
    not _have("simulated/1D_TFIM/FM_PM"), reason="1D TFIM dataset not present")


# ---------------------------------------------------------------- dispatch ---

def test_unknown_dataset_lists_the_valid_choices():
    with pytest.raises(ValueError) as excinfo:
        create_datasets(_cf(dataset="not_a_dataset"))
    message = str(excinfo.value)
    for name in ("Paris_Ising", "ILGT", "1D_TFIM_Z", "XXZ"):
        assert name in message


def test_absent_dataset_raises_a_dataset_error_naming_the_path():
    # XXZ is not distributed with the repository, so this is the message a user
    # meets first; it has to name the path it looked in.
    with pytest.raises(DatasetNotAvailableError) as excinfo:
        create_datasets(_cf(dataset="XXZ", label_param="Jz"))
    assert "XXZ" in str(excinfo.value)


# -------------------------------------------------------------------- ILGT ---

@ilgt_needed
@pytest.mark.parametrize("task,label_param,output_dim", [
    ("regression", "beta", 1),
    ("classification", "t", 2),
])
def test_ilgt_builds_datasets(task, label_param, output_dim):
    train, val = create_datasets(_cf(dataset="ILGT", task=task, label_param=label_param,
                                     normalize_labels=(task == "regression")))
    assert len(train) > 0 and len(val) > 0
    # Two channels, one per link orientation, on a 16x16 lattice.
    assert tuple(train[0][0].shape) == (2, 16, 16)
    assert train.output_dim == output_dim


@ilgt_needed
def test_ilgt_populates_the_tuning_parameter_axis():
    # Regression guard: the pre-split path once fell through to PhaseDataset without
    # defining `times`, which raised UnboundLocalError and made every simulated
    # dataset unreachable while the experimental path kept working.
    train, _ = create_datasets(_cf(dataset="ILGT", task="regression", label_param="beta"))
    assert len(train.times) == len(train)
    assert len(train.unique_labels_unnormalized) > 1


@ilgt_needed
def test_ilgt_learning_by_confusion_reads_the_regression_labels():
    # The classification file stores only the two phase classes, so a threshold in
    # the inverse temperature can only come from the regression file.
    cf = _cf(dataset="ILGT", task="partition", label_param="beta", partition_index=0)
    assert get_no_partitions(cf) > 1

    cf = _cf(dataset="ILGT", task="partition", label_param="beta", partition_index=10)
    train, _ = create_datasets(cf)
    assert train.output_dim == 2
    # Both classes have to be present, or the threshold landed outside the sweep.
    assert set(train.labels.argmax(dim=1).unique().tolist()) == {0, 1}


# -------------------------------------------------------------------- TFIM ---

@tfim_needed
@pytest.mark.parametrize("task,label_param,output_dim", [
    ("regression", "g", 1),
    ("classification", "t", 2),
])
def test_tfim_builds_datasets(task, label_param, output_dim):
    train, val = create_datasets(_cf(dataset="1D_TFIM_Z", task=task, label_param=label_param,
                                     phase_path="FM_PM",
                                     normalize_labels=(task == "regression")))
    assert len(train) > 0 and len(val) > 0
    # A 150-site chain stored as a one-column image, so the 2D branches apply unchanged.
    assert tuple(train[0][0].shape) == (1, 150, 1)
    assert train.output_dim == output_dim


@tfim_needed
def test_tfim_classification_keeps_the_sweep_as_its_time_axis():
    # The phase class is the label; the transverse field g stays the tuning parameter,
    # so the snapshot-average plots still resolve the sweep rather than two points.
    train, _ = create_datasets(_cf(dataset="1D_TFIM_Z", task="classification",
                                   label_param="t", phase_path="FM_PM"))
    assert len(train.unique_labels_unnormalized) > 2


@tfim_needed
def test_tfim_basis_suffix_selects_the_measurement_basis():
    for dataset in ("1D_TFIM_X", "1D_TFIM_Z"):
        train, _ = create_datasets(_cf(dataset=dataset, task="regression",
                                       label_param="g", phase_path="FM_PM"))
        assert len(train) > 0


# ----------------------------------------------------------------- kernels ---

def test_chainkernels_is_a_one_column_set():
    cf = AttrDict(); cf.kernel_set = "chainkernels"; cf.dataset = "1D_TFIM_Z"
    set_kernels(cf)
    assert all(width == 1 for (_, width), *_ in cf.kernels)
    # Areas 1..4 give the one- through four-point correlators of the chain.
    assert {h for (h, _), *_ in cf.kernels} == {1, 2, 3, 4}


def test_two_dimensional_kernels_on_a_chain_point_at_chainkernels():
    cf = AttrDict(); cf.kernel_set = "smallkernels"; cf.dataset = "1D_TFIM_Z"
    set_kernels(cf)
    with pytest.raises(ValueError) as excinfo:
        validate_kernels_fit(cf.kernels, torch.Size([150, 1]), kernel_set=cf.kernel_set)
    assert "chainkernels" in str(excinfo.value)


def test_chainkernels_fits_a_chain():
    cf = AttrDict(); cf.kernel_set = "chainkernels"; cf.dataset = "1D_TFIM_Z"
    set_kernels(cf)
    validate_kernels_fit(cf.kernels, torch.Size([150, 1]), kernel_set=cf.kernel_set)


def test_kernel_validation_accounts_for_dilation():
    # A dilated (2,1) spans 3 sites, so it does not fit a 2-site column.
    with pytest.raises(ValueError):
        validate_kernels_fit([[(2, 1), 1, 2, None, 1]], torch.Size([2, 1]))
    validate_kernels_fit([[(2, 1), 1, 2, None, 1]], torch.Size([3, 1]))


def test_missing_kernel_set_is_rejected():
    cf = AttrDict(); cf.kernel_set = "bigkernels"; cf.dataset = "1D_TFIM_Z"
    with pytest.raises(ValueError) as excinfo:
        set_kernels(cf)
    assert "chainkernels" in str(excinfo.value)


# -------------------------------------------------- partition helpers (pure) ---

def test_partition_threshold_sits_between_neighbouring_sweep_points():
    threshold, no_partitions = partition_threshold(np.array([0.0, 1.0, 2.0]), 0)
    assert threshold == pytest.approx(0.5)
    assert no_partitions == 2


def test_partition_threshold_rejects_an_out_of_range_index():
    with pytest.raises(ValueError):
        partition_threshold(np.array([0.0, 1.0, 2.0]), 2)


def test_partition_threshold_needs_two_sweep_points():
    with pytest.raises(ValueError):
        partition_threshold(np.array([0.0]), 0)


def test_relabel_by_threshold_keeps_the_tuning_values():
    samples = [(torch.zeros(1, 2, 1), torch.tensor(v)) for v in (0.0, 1.0, 2.0)]
    tuning = [0.0, 1.0, 2.0]
    relabelled, carried = relabel_by_threshold(samples, tuning, 1.5)
    assert [int(lbl) for _, lbl in relabelled] == [0, 0, 1]
    assert carried == tuning


def test_xxz_classification_is_refused_with_a_usable_alternative():
    from tetriscnn.dataprocessing import XXZDataProcessor

    # The dataset carries no phase labels, so plain classification cannot work. The
    # message has to name what does, since learning by confusion DOES reach it.
    with pytest.raises(NotImplementedError) as excinfo:
        XXZDataProcessor(discrete_labels=True, learning_by_confusion=False)
    message = str(excinfo.value)
    assert "regression" in message and "lbc" in message


def test_xxz_learning_by_confusion_is_reachable():
    from tetriscnn.dataprocessing import DatasetNotAvailableError, XXZDataProcessor

    # create_datasets() sets discrete_labels=True for lbc/partition, so this is the
    # combination the standard path actually produces. It must get past the label
    # check and fail only on the (undistributed) data itself.
    with pytest.raises(DatasetNotAvailableError):
        XXZDataProcessor(discrete_labels=True, learning_by_confusion=True, partition_index=0)
