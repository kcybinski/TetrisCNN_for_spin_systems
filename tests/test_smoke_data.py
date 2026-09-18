"""Group D: data-dependent smoke tests.

Every test here needs the (not checked into a fresh clone) data/ tree. The whole
module is skipped cleanly if data/ is absent, so `pytest` still passes end-to-end
right after `git clone`.
"""
import json

import pytest
import torch

from tests.conftest import data_available
from tetriscnn.utils import AttrDict, set_kernels, set_plotting_logging_strings
from tetriscnn.datasets import CLASSIFICATION_PARTITION_INDEX, create_datasets
from tetriscnn.plots import single_seed_plots
from tetriscnn.train import train

pytestmark = pytest.mark.data

if not data_available():
    pytest.skip("data/ directory not present; skipping data-dependent smoke tests",
                allow_module_level=True)


def _regression_cf(**overrides):
    """Minimal cf for create_datasets(Paris_Ising, task='regression')."""
    cf = AttrDict()
    cf.task = "regression"
    cf.label_param = "delta"
    cf.normalize_labels = True
    cf.label_subset_count = None
    cf.dataset = "Paris_Ising"
    cf.filter_t_values = None
    cf.data_fraction = 1
    cf.param_cutoffs = {"delta": 6, "omega": 0}
    cf.even_split = False
    cf.samples_per_pt_cap = None
    cf.seed = 42
    cf.update(overrides)
    return cf


def _classification_cf(**overrides):
    """Minimal cf for create_datasets(..., task='classification')."""
    cf = AttrDict()
    cf.task = "classification"
    cf.dataset = "Paris_Ising"
    cf.filter_t_values = None
    cf.data_fraction = 1
    cf.label_subset_count = None
    cf.param_cutoffs = {"delta": 6, "omega": 0}
    cf.even_split = False
    cf.samples_per_pt_cap = None
    cf.seed = 42
    cf.partition_index = None
    cf.update(overrides)
    return cf


def _trainable_cf(tmp_path, **overrides):
    """A full cf sufficient to call tetriscnn.train.train(), kept small/fast:
    even_split + a tight samples_per_pt_cap to shrink the dataset, small hidden_size,
    few epochs, early stopping effectively disabled (warmup > epochs) so the run
    always does exactly `epochs` epochs (deterministic wall-clock cost)."""
    cf = _regression_cf(even_split=True, samples_per_pt_cap=8, goodness_str="r2agg")
    cf.kernel_set = "smallkernels"
    cf.equivariant = False
    set_kernels(cf)
    cf.model = "tetriscnn"
    cf.hidden_size = 4
    cf.init = "kaiming"
    cf.epochs = 2
    cf.learning_rate = 1e-2
    cf.weight_decay = 1e-5
    cf.patience = 10
    cf.early_stop_warmup = 1000
    cf.early_stop_min_delta = 1e-5
    cf.batch_size = 16
    cf.VRAM_batch_size = 1024
    cf.num_workers = 0
    cf.pin_memory = False
    cf.experiment_name = "singlerun"
    cf.lam = -1
    cf.penalty_params = [10, -5, cf.lam, 1]
    cf.weight_penalty = None
    cf.save_final_values = False
    cf.save_models = True
    cf.use_lr_scheduler = False
    cf.save_histories = True
    set_plotting_logging_strings(cf)
    cf.update(overrides)
    cf.logdir = str(tmp_path / "run")
    return cf


class TestCreateDatasetsDeterminism:
    def test_determinism_independent_of_cf_seed(self):
        # the train/val split uses an isolated generator seeded with a fixed 42,
        # independent of cf.seed -- two different cf.seed values must still produce
        # identical splits/labels.
        cf_a = _regression_cf(seed=1)
        cf_b = _regression_cf(seed=999)
        train_a, val_a = create_datasets(cf_a)
        train_b, val_b = create_datasets(cf_b)

        assert torch.equal(train_a.labels, train_b.labels)
        assert torch.equal(val_a.labels, val_b.labels)
        assert torch.equal(train_a.times, train_b.times)

    def test_repeated_call_is_deterministic(self):
        cf1 = _regression_cf()
        cf2 = _regression_cf()
        train1, val1 = create_datasets(cf1)
        train2, val2 = create_datasets(cf2)
        assert torch.equal(train1.labels, train2.labels)
        assert torch.equal(val1.labels, val2.labels)
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)

    def test_70_30_split_proportion(self):
        cf = _regression_cf()  # even_split=False -> global random 70/30 split
        train_ds, val_ds = create_datasets(cf)
        total = len(train_ds) + len(val_ds)
        # split = int(0.7 * n_subset); allow the int()-truncation slack
        assert train_ds.__len__() == pytest.approx(0.7 * total, abs=1)

    def test_val_normalization_shares_train_y_min_max(self):
        cf = _regression_cf()
        train_ds, val_ds = create_datasets(cf)
        assert torch.equal(train_ds.y_min, val_ds.y_min)
        assert torch.equal(train_ds.y_max, val_ds.y_max)
        # sanity: these are genuinely derived from train labels, not defaults
        assert train_ds.y_min.numel() > 0


class TestEvenSplit:
    def test_every_time_point_represented_in_both_train_and_val(self):
        cf = _regression_cf(even_split=True)
        train_ds, val_ds = create_datasets(cf)
        train_times = set(train_ds.times.tolist())
        val_times = set(val_ds.times.tolist())
        all_times = set(train_ds.times.tolist()) | set(val_ds.times.tolist())
        assert train_times == all_times
        assert val_times == all_times

    def test_samples_per_pt_cap_is_respected(self):
        cap = 6
        cf = _regression_cf(even_split=True, samples_per_pt_cap=cap)
        train_ds, val_ds = create_datasets(cf)

        from collections import Counter
        train_counts = Counter(train_ds.times.tolist())
        val_counts = Counter(val_ds.times.tolist())
        all_tp = set(train_counts) | set(val_counts)
        for tp in all_tp:
            total_at_tp = train_counts.get(tp, 0) + val_counts.get(tp, 0)
            assert total_at_tp <= cap, f"time point {tp} has {total_at_tp} samples > cap {cap}"


@pytest.mark.slow
class TestTrainSmoke:
    def test_two_epoch_run_produces_finite_losses_and_expected_files(self, tmp_path):
        cf = _trainable_cf(tmp_path)
        train_dataset, val_dataset = create_datasets(cf)
        cf.train_dataset = train_dataset
        cf.val_dataset = val_dataset

        train(cf, "metrics.json")

        logdir = cf.logdir  # train() overwrites cf.logdir via create_path()
        for fname in ("config.json", "metrics.json", "net1.pt", "net2.pt"):
            assert (logdir / fname).exists(), f"missing {fname} in {logdir}"

        with open(logdir / "metrics.json") as f:
            metrics = json.load(f)

        assert len(metrics[f"train_{cf.loss_str}"]) == 2
        assert len(metrics[f"val_{cf.loss_str}"]) == 2
        for v in metrics[f"train_{cf.loss_str}"] + metrics[f"val_{cf.loss_str}"]:
            assert v == v and abs(v) != float("inf")  # not NaN, not +/-inf
        for v in metrics[f"train_{cf.goodness_str}"] + metrics[f"val_{cf.goodness_str}"]:
            assert v == v and abs(v) != float("inf")

    def test_save_histories_false_completes_and_writes_valid_metrics(self, tmp_path):
        """Regression test for a fixed bug: tetriscnn/train.py, end of train(), used to have

            if net2 is not None:
                if not cf.save_histories:
                    metrics["MVUL_out"], metrics["MVUL_std_out"] = \\
                        cf.val_dataset.snapshot_average(out).tolist()

        PhaseDataset.snapshot_average always returns a 2-tuple of numpy arrays
        `(means, stds)` (see tetriscnn/datasets.py), and a plain Python tuple has no
        `.tolist()` method, so any run with cf.save_histories = False crashed with
        AttributeError at the very end of training, after doing all the epochs' work.
        The fix unpacks the tuple first (matching the cf.save_histories=True idiom
        used earlier in the same file) before calling .tolist() on each array. This
        test asserts the run now succeeds end-to-end and produces valid MVUL metrics.
        """
        cf = _trainable_cf(tmp_path, save_histories=False)
        train_dataset, val_dataset = create_datasets(cf)
        cf.train_dataset = train_dataset
        cf.val_dataset = val_dataset

        train(cf, "metrics.json")

        logdir = cf.logdir  # train() overwrites cf.logdir via create_path()
        with open(logdir / "metrics.json") as f:
            metrics = json.load(f)

        assert isinstance(metrics["MVUL_out"], list)
        assert isinstance(metrics["MVUL_std_out"], list)
        assert len(metrics["MVUL_out"]) > 0
        for v in metrics["MVUL_out"] + metrics["MVUL_std_out"]:
            assert v == v and abs(v) != float("inf")  # not NaN, not +/-inf


class TestClassificationTask:
    """cf.task == 'classification' used to be unreachable for the Paris datasets:
    create_datasets() left learning_by_confusion=False and partition_index=None for
    it, and ParisDataProcessor raises NotImplementedError for discrete labels unless
    learning_by_confusion=True. classification now auto-resolves a hardcoded
    partition index for the Paris datasets (the manuscript's transition-flanking
    pair) and falls back to requiring cf.partition_index for any other dataset."""

    def test_paris_dataset_gets_hardcoded_partition_index(self):
        for dataset, expected_index in CLASSIFICATION_PARTITION_INDEX.items():
            if not dataset.startswith("Paris_Ising"):
                continue  # only Paris_Ising's data is guaranteed present in this module
            cf = _classification_cf(dataset=dataset)
            create_datasets(cf)
            assert cf.partition_index == expected_index

    def test_binary_labels_produced(self):
        cf = _classification_cf(dataset="Paris_Ising")
        train_ds, val_ds = create_datasets(cf)
        labels = set(train_ds.labels.flatten().tolist()) | set(val_ds.labels.flatten().tolist())
        assert labels <= {0.0, 1.0}
        assert len(labels) == 2  # both sides of the split are actually represented

    def test_xxz_requires_explicit_partition_index(self):
        # XXZDataProcessor needs learning-by-confusion for discrete labels, just like
        # the Paris processors, but has no canonical transition index of its own.
        # Resolution happens before any dataset file is touched, so this needs no XXZ
        # data on disk.
        cf = _classification_cf(dataset="XXZ", partition_index=None)
        with pytest.raises(AssertionError, match="No canonical transition-flanking index"):
            create_datasets(cf)

    def test_ilgt_and_tfim_classification_need_no_partition_index(self):
        # Unlike the Paris datasets and XXZ, ILGT/TFIM carry a genuine ground-truth
        # phase label (ILGTDataProcessor's classification file / TFIMDataProcessor's
        # "class=" column), so classification there is NOT the partition mechanism
        # and must keep working with cf.partition_index left at None.
        cf = _classification_cf(dataset="1D_TFIM_X", partition_index=None, phase_path="FM_PM")
        train_ds, val_ds = create_datasets(cf)
        assert cf.partition_index is None
        assert len(train_ds) > 0 and len(val_ds) > 0


@pytest.mark.slow
class TestFitBranchesIntegration:
    """cf.fit_branches (renamed from the stale, single-branch cf.fit_branch_number)
    wires plot_branch_fits into single_seed_plots(): it must run to completion
    whether or not any branch actually clears the activity floor."""

    def test_fit_branches_true_does_not_crash(self, tmp_path, capsys):
        cf = _trainable_cf(tmp_path, save_final_values=True, fit_branches=True)
        train_dataset, val_dataset = create_datasets(cf)
        cf.train_dataset = train_dataset
        cf.val_dataset = val_dataset

        train(cf, "metrics.json")
        logdir = cf.logdir
        with open(logdir / "metrics.json") as f:
            metrics = json.load(f)

        single_seed_plots(cf, metrics)  # must not raise either way

        branch_fits_path = logdir / "branch_fits.png"
        out = capsys.readouterr().out
        if branch_fits_path.exists():
            assert branch_fits_path.stat().st_size > 0
        else:
            assert "no branch cleared the activity floor" in out

    def test_fit_branches_false_writes_no_branch_fits_file(self, tmp_path):
        cf = _trainable_cf(tmp_path, save_final_values=True, fit_branches=False)
        train_dataset, val_dataset = create_datasets(cf)
        cf.train_dataset = train_dataset
        cf.val_dataset = val_dataset

        train(cf, "metrics.json")
        logdir = cf.logdir
        with open(logdir / "metrics.json") as f:
            metrics = json.load(f)

        single_seed_plots(cf, metrics)
        assert not (logdir / "branch_fits.png").exists()
