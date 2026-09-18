"""register_dataset(): plugging a user-defined data source into create_datasets().

Uses a tiny in-memory processor, so nothing here needs the dataset tree. The worked,
user-facing version of the same route is notebooks/BYODataset_Tutorial.ipynb.
"""
import numpy as np
import pytest
import torch

from tetriscnn.dataprocessing import partition_threshold, relabel_by_threshold
from tetriscnn.datasets import _DATASET_REGISTRY, create_datasets, register_dataset
from tetriscnn.utils import AttrDict

SWEEP = (0.0, 0.1, 0.2, 0.3)
SHOTS = 10


class _PooledProcessor:
    """One pool of snapshots; create_datasets() does the split."""
    label_param = "p"
    label_dict = None

    def __init__(self, discrete_labels, learning_by_confusion, partition_index):
        rng = np.random.default_rng(0)
        self.samples, self.sample_times = [], []
        for p in SWEEP:
            for _ in range(SHOTS):
                x = torch.from_numpy(rng.choice([-1.0, 1.0], size=(1, 4, 4)).astype(np.float32))
                y = float(p > 0.15) if discrete_labels else p
                self.samples.append((x, torch.tensor(y, dtype=torch.float32)))
                self.sample_times.append(p)
        if learning_by_confusion:
            threshold, self.no_partitions = partition_threshold(SWEEP, partition_index)
            self.samples, self.sample_times = relabel_by_threshold(
                self.samples, self.sample_times, threshold)
        self.unique_labels = np.unique([float(y) for _, y in self.samples])


class _PresplitProcessor(_PooledProcessor):
    """The same data, handed over with its own train/val split."""

    def __init__(self, *args):
        super().__init__(*args)
        self.train_samples, self.val_samples = self.samples[::2], self.samples[1::2]
        self.train_times, self.val_times = self.sample_times[::2], self.sample_times[1::2]
        del self.samples, self.sample_times


@pytest.fixture
def registry():
    """Register test datasets for one test only, recording each factory call."""
    calls = []

    def factory_for(cls):
        def factory(cf, *, discrete_labels, label_param, learning_by_confusion, partition_index):
            calls.append(dict(discrete_labels=discrete_labels, label_param=label_param,
                              learning_by_confusion=learning_by_confusion,
                              partition_index=partition_index))
            return cls(discrete_labels, learning_by_confusion, partition_index)
        return factory

    saved = dict(_DATASET_REGISTRY)
    register_dataset("_Pooled", factory_for(_PooledProcessor))
    register_dataset("_Presplit", factory_for(_PresplitProcessor))
    yield calls
    _DATASET_REGISTRY.clear()
    _DATASET_REGISTRY.update(saved)


def _cf(**overrides):
    cf = AttrDict(task="classification", label_param="p", even_split=True,
                  samples_per_pt_cap=None, data_fraction=1, partition_index=None,
                  param_cutoffs={"delta": None, "omega": None})
    cf.update(overrides)
    return cf


def test_classification_on_a_pooled_processor(registry):
    train, val = create_datasets(_cf(dataset="_Pooled"))
    assert registry == [dict(discrete_labels=True, label_param="t",
                             learning_by_confusion=False, partition_index=None)]
    assert len(train) + len(val) == len(SWEEP) * SHOTS
    # even_split takes 70% of every sweep point, not 70% of the pool
    assert len(train) == len(SWEEP) * int(0.7 * SHOTS)
    assert train.output_dim == 2
    assert tuple(train[0][0].shape) == (1, 4, 4)
    assert sorted(torch.unique(train.times).tolist()) == pytest.approx(SWEEP)


def test_partition_task_passes_the_index_through(registry):
    train, _ = create_datasets(_cf(dataset="_Pooled", task="partition", partition_index=2))
    assert registry[0]["learning_by_confusion"] is True
    assert registry[0]["partition_index"] == 2
    # threshold between 0.2 and 0.3: only the last sweep point is class 1
    labels = train.labels.argmax(dim=1).cpu()
    times = train.times.cpu()
    assert torch.all(labels[times > 0.25] == 1) and torch.all(labels[times < 0.25] == 0)


def test_regression_on_a_presplit_processor(registry):
    cf = _cf(dataset="_Presplit", task="regression", normalize_labels=True,
             label_subset_count=None)
    train, val = create_datasets(cf)
    assert registry[0]["discrete_labels"] is False
    assert len(train) == len(val) == len(SWEEP) * SHOTS // 2
    assert train.output_dim == 1
    assert float(train.labels.min()) == 0.0 and float(train.labels.max()) == 1.0


def test_unknown_name_lists_registered_datasets(registry):
    with pytest.raises(ValueError) as excinfo:
        create_datasets(_cf(dataset="_Missing"))
    message = str(excinfo.value)
    assert "Paris_Ising" in message and "_Pooled" in message


def test_factory_must_be_callable():
    with pytest.raises(TypeError):
        register_dataset("_Broken", "not a function")
    assert "_Broken" not in _DATASET_REGISTRY


# ------------------------------------------------------- config defaults ---

def test_minimal_config_is_enough_to_build_datasets(registry):
    """Keys irrelevant to a dataset (data_fraction, param_cutoffs, ...) may be left out."""
    cf = AttrDict(dataset="_Pooled", task="classification")
    train, val = create_datasets(cf)
    assert len(train) + len(val) == len(SWEEP) * SHOTS
    assert cf.data_fraction == 1 and cf.param_cutoffs == {"delta": None, "omega": None}
    assert cf.even_split is True                 # the default, recorded on cf


def test_explicit_keys_are_not_overwritten(registry):
    cf = AttrDict(dataset="_Pooled", task="classification", even_split=False, data_fraction=0.5)
    train, val = create_datasets(cf)
    assert cf.even_split is False
    assert len(train) + len(val) == len(SWEEP) * SHOTS // 2


def test_missing_required_keys_are_named():
    from tetriscnn.utils import apply_config_defaults
    with pytest.raises(ValueError, match="task"):
        apply_config_defaults(AttrDict(dataset="_Pooled"))
    with pytest.raises(ValueError, match="logdir"):
        apply_config_defaults(AttrDict(dataset="_Pooled", task="classification"), for_training=True)


def test_training_defaults_warn_only_about_the_penalty(tmp_path):
    from tetriscnn.utils import DEFAULT_PENALTY_PARAMS, apply_config_defaults
    cf = AttrDict(dataset="_Pooled", task="classification", logdir=str(tmp_path / "run"))
    with pytest.warns(UserWarning, match="penalty_params") as record:
        apply_config_defaults(cf, for_training=True)
    assert len(record) == 1
    assert cf.penalty_params == DEFAULT_PENALTY_PARAMS
    assert len(cf.kernels) == 10 and cf.loss_str == "CEL" and cf.goodness_str == "acc"


@pytest.mark.slow
def test_train_runs_from_a_minimal_config(registry, tmp_path):
    from tetriscnn.train import train
    cf = AttrDict(dataset="_Pooled", task="classification", logdir=str(tmp_path / "run"),
                  penalty_params=[10, -3, 0, 1], epochs=2)
    cf.train_dataset, cf.val_dataset = create_datasets(cf)
    train(cf, "metrics.json")
    assert (tmp_path / "run" / "metrics.json").exists()


# ------------------------------------------------------ SnapshotProcessor ---

class _PoolFormat:
    """In-memory stand-in for a file format: one 0/1 array of (n, 3, 3) per t."""

    def _load_basis_files(self, file_list):
        rng = np.random.default_rng(len(file_list))
        return {t: rng.integers(0, 2, size=(SHOTS, 3, 3)).astype(np.float32) for t in SWEEP}


def _snapshot_processor(with_phases, **kwargs):
    from tetriscnn.dataprocessing import SnapshotProcessor

    class Proc(_PoolFormat, SnapshotProcessor):
        def __init__(self, bases=("Z",), **kw):
            super().__init__(grid_size=(3, 3), label_param="p",
                             file_dict={b: [b] * (i + 1) for i, b in enumerate(bases)}, **kw)

    if with_phases:
        Proc._phase_label = lambda self, t: int(t > 0.15)
    return Proc(**kwargs)


def test_snapshot_processor_maps_to_pm1_and_labels_regression_by_t():
    proc = _snapshot_processor(False)
    assert len(proc.samples) == len(SWEEP) * SHOTS
    x, y = proc.samples[0]
    assert tuple(x.shape) == (1, 3, 3)
    assert set(torch.unique(x).tolist()) <= {-1.0, 1.0}
    assert float(y) == proc.sample_times[0]


def test_snapshot_processor_stacks_bases_into_channels():
    proc = _snapshot_processor(False, bases=("X", "Z"))
    assert tuple(proc.samples[0][0].shape) == (2, 3, 3)


def test_snapshot_processor_classification_uses_phase_label():
    proc = _snapshot_processor(True, discrete_labels=True)
    for (_, y), t in zip(proc.samples, proc.sample_times):
        assert int(y) == int(t > 0.15)


def test_snapshot_processor_without_phase_labels_points_to_lbc():
    with pytest.raises(NotImplementedError, match="_phase_label"):
        _snapshot_processor(False, discrete_labels=True)


def test_snapshot_processor_learning_by_confusion():
    proc = _snapshot_processor(False, discrete_labels=True, learning_by_confusion=True,
                               partition_index=2)
    assert proc.no_partitions == len(SWEEP) - 1
    for (_, y), t in zip(proc.samples, proc.sample_times):
        assert int(y) == int(t > 0.25)
