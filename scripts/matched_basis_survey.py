#!/usr/bin/env python
"""Retrain the XY X / Z / XZ basis-ablation arms on a snapshot-count-matched dataset.

Motivation (from an earlier, un-matched basis-ablation study): comparing the plain ``lambdamax_Paris_XY_{X,Z,XZ}_...``
runs is confounded by two things that have nothing to do with the basis itself --
Z's raw acquisition has 21 time points against X/XZ's 13, and even on the times they
DO share, each basis carries a different number of snapshots per time point (Z has up
to ~14x more at some t, e.g. t=2000: X=2574, Z=6539). This script removes both
confounds by construction:

  1. ``filter_t_values`` is fixed to the 13 time points common to the X and Z
     acquisition runs (exactly the set ``Paris_XY_XZ`` already restricts itself to,
     since its own pairing intersects the two file dictionaries).
  2. ``samples_per_pt_cap`` is a *per-timepoint* dict, `{t: min(n_X(t), n_Z(t))}` --
     the same count XZ's own snapshot-pairing (``pairing_mode="index"``) already uses
     per time point. Passed to all three datasets, this makes X's, Z's and XZ's
     per-t sample counts identical by construction, not just their totals.

  A side effect of (1): restricting Z to these 13 times collapses its unique-label
  set from 21 to 13, identical to X/XZ's. That forces ``partition_index=2`` for Z
  too (625ns), instead of Z's original ``partition_index=4`` (587.5ns) -- confirmed
  and intentional, see the matched-run analysis notes. All three arms now train on
  the literal same classification task (same threshold), removing the need for the
  old ad-hoc "13-shared-times" evaluation-only control.

Everything else (equivariant C4 smallkernels, lam=3 / penalty_params=[10,-3,3,1],
epochs=250, lr schedule, batch/VRAM sizes, early-stop warmup) is copied verbatim from
the un-matched runs' own recorded config.json, so the ONLY things that differ from
those runs are filter_t_values and samples_per_pt_cap.

Usage:
    python scripts/matched_basis_survey.py --dataset X
    python scripts/matched_basis_survey.py --dataset Z --seeds 42 43
    python scripts/matched_basis_survey.py --dataset XZ --dry-run   # just print the match
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TAG = "matched"  # distinguishes these logs from the un-matched basis-ablation runs
SEEDS_DEFAULT = (42, 43, 44, 45, 46)
LAM = 3
PENALTY_PARAMS = [10, -3, LAM, 1]
PARTITION_INDEX = 2  # unified across X/Z/XZ once filter_t_values matches their time grids

DATASET_NAMES = {"X": "Paris_XY_X", "Z": "Paris_XY_Z", "XZ": "Paris_XY_XZ"}


def compute_matching():
    """Return (sorted shared t-values, {t: min(n_X(t), n_Z(t))}) from the raw data,
    independent of any cf / filtering -- this is the ground truth both bases actually
    have on disk, not something read back off a previous run's config."""
    from tetriscnn.dataprocessing import XYDataProcessor
    import collections

    counts = {}
    for basis in ("X", "Z"):
        proc = XYDataProcessor(basis=basis, data_format="-11", label_param="t",
                                discrete_labels=True, filter_t_values=None,
                                learning_by_confusion=True, partition_index=0)
        counts[basis] = collections.Counter(proc.sample_times)

    shared_times = sorted(set(counts["X"]) & set(counts["Z"]))
    cap = {t: min(counts["X"][t], counts["Z"][t]) for t in shared_times}
    return shared_times, cap


def _base_config(dataset_key, shared_times, cap):
    from tetriscnn.utils import AttrDict, set_kernels, set_plotting_logging_strings

    cf = AttrDict()
    cf.dataset = DATASET_NAMES[dataset_key]
    cf.data_fraction = 1
    cf.label_subset_count = None
    cf.param_cutoffs = {"delta": 6, "omega": 0}

    cf.even_split = True
    cf.filter_t_values = shared_times
    cf.samples_per_pt_cap = dict(cap)  # per-timepoint dict -- see module docstring

    cf.experiment_name = "lambdamax"
    cf.kernel_set = "smallkernels"
    cf.equivariant = True
    cf.equivariant_group = "C4"
    set_kernels(cf)

    cf.lambdas = [LAM]
    cf.lam = LAM
    cf.penalty_params = PENALTY_PARAMS

    cf.task = "partition"
    cf.partition_index = PARTITION_INDEX
    cf.label_param = "t"

    cf.save_models = True
    cf.save_histories = True
    cf.save_final_values = False

    cf.epochs = 250
    cf.learning_rate = 1e-2
    cf.weight_decay = 1e-5
    cf.patience = 10
    cf.early_stop_warmup = 150
    cf.early_stop_min_delta = 1e-5
    cf.init = "kaiming"

    cf.model = "tetriscnn"
    cf.num_workers = 0
    cf.pin_memory = False
    cf.batch_size = 64
    cf.VRAM_batch_size = 1024
    cf.hidden_size = 32

    cf.use_lr_scheduler = True
    cf.lr_scheduler_type = "reduce_on_plateau"
    cf.lr_reduce_factor = 0.5
    cf.lr_reduce_patience = 5
    cf.min_lr = 1e-9
    cf.weight_penalty = None

    cf.use_weighted_loss = False

    set_plotting_logging_strings(cf)
    return cf


def logdir_for(dataset_key, seed):
    return (REPO_ROOT / "logs" / "logs_after_bugfix" /
            f"lambdamax_{DATASET_NAMES[dataset_key]}_partition_t_smallkernels_C4_{TAG}" /
            f"lambdamax_{LAM}" / f"seed_{seed}" / f"partition_{PARTITION_INDEX}")


def run_one(dataset_key, seed, shared_times, cap, epochs_override=None):
    from tetriscnn.datasets import create_datasets
    from tetriscnn.train import train
    from tetriscnn.plots import single_seed_plots
    from tetriscnn.utils import load_json

    cf = _base_config(dataset_key, shared_times, cap)
    if epochs_override is not None:
        cf.epochs = epochs_override
    cf.seed = seed
    cf.seeds = list(SEEDS_DEFAULT)  # recorded for reference; this script trains one seed at a time

    cf.train_dataset, cf.val_dataset = create_datasets(cf)
    cf.logdir = str(logdir_for(dataset_key, seed))

    train(cf, "metrics.json")
    metrics = load_json(cf.logdir, "metrics.json")
    try:
        single_seed_plots(cf, metrics)
    except Exception as e:
        print(f"[matched_basis_survey] plotting failed for {dataset_key} seed={seed} (non-fatal): {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=("X", "Z", "XZ"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument("--dry-run", action="store_true", help="print the computed match and exit")
    p.add_argument("--epochs", type=int, default=None, help="override cf.epochs (smoke testing)")
    args = p.parse_args()

    shared_times, cap = compute_matching()
    print(f"[matched_basis_survey] {len(shared_times)} shared time points: {shared_times}")
    print(f"[matched_basis_survey] per-timepoint cap (min(n_X(t), n_Z(t))): {cap}")
    print(f"[matched_basis_survey] total matched snapshots per basis: {sum(cap.values())}")

    if args.dry_run:
        return

    for seed in args.seeds:
        print(f"\n=== {args.dataset} seed={seed} -> {logdir_for(args.dataset, seed)} ===")
        run_one(args.dataset, seed, shared_times, cap, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
