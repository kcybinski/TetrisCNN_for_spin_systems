#!/usr/bin/env python
"""Retrain ONLY the XY XZ basis-ablation arm on a *half-snapshot* matched dataset.

Follow-up to ``matched_basis_survey.py``.
That study matched X, Z and XZ to the SAME 13 time points and the SAME per-timepoint
snapshot count ``min(n_X(t), n_Z(t))``. But an XZ training example is a *pair*: it
carries one X-basis snapshot AND one Z-basis snapshot. So at an equal *example*
count, the matched XZ arm actually consumes ~2x the RAW snapshots that the X-alone
and Z-alone arms do.

This script removes that last asymmetry. XZ is retrained with HALF as many paired
examples per time point -- ``half_cap({t: min(n_X(t), n_Z(t))})`` -- so each XZ arm
now consumes about ``min(n_X(t), n_Z(t))`` RAW snapshots per time point, the SAME
raw-snapshot budget as the (unchanged) X-alone and Z-alone matched arms. X and Z are
NOT retrained: the X-alone and Z-alone arms keep their existing ``..._C4_matched``
folders, and only the XZ arm is repointed at the new folders written here.

Everything else -- ``filter_t_values`` (the 13 shared times), ``partition_index=2``,
equivariant C4 smallkernels, lam=3 / ``penalty_params=[10,-3,3,1]``, epochs=250, lr
schedule, batch/VRAM sizes, early-stop warmup -- is inherited verbatim from
``matched_basis_survey._base_config``, so the ONLY thing that differs from the
matched XZ arm is the per-timepoint cap (halved by ``half_cap``).

Usage:
    python scripts/matched_basis_survey_xzhalf.py --dry-run          # print the match
    python scripts/matched_basis_survey_xzhalf.py --seeds 42         # one seed
    python scripts/matched_basis_survey_xzhalf.py                    # all 5, sequential
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import matched_basis_survey` works

from matched_basis_survey import (  # noqa: E402
    compute_matching, _base_config, DATASET_NAMES, LAM, PARTITION_INDEX,
)

TAG = "matched_xzhalf"  # distinguishes these logs from the equal-example matched runs
SEEDS_DEFAULT = (42, 43, 44, 45, 46)
DATASET_KEY = "XZ"  # this script only ever retrains XZ


def half_cap(cap: dict) -> dict:
    """Given the matched per-timepoint cap ``{t: min(n_X(t), n_Z(t))}``, return the
    halved cap that defines the XZ-half arm's dataset.

    This is the one modelling choice in the whole experiment -- everything else is
    copied from the matched run. Floor division is used rather than ``round(v / 2)``
    because it guarantees ``2 * (v // 2) <= v`` at every t, so the XZ-half arm never
    consumes MORE raw snapshots than the matched X-alone and Z-alone arms it is
    compared against. Rounding would exceed that equal-budget goal on odd counts.
    The cost is at most one snapshot per odd-valued t (5 of the 13 time points here,
    all of whose caps are in the hundreds, e.g. {0: 1777, 2000: 2574, ...}).

    Returns a new ``{t: int}`` dict; ``cap`` is not mutated.
    """
    # Floor division: guarantees 2 * (v // 2) <= v at every t, so the XZ-half arm
    # never consumes more raw snapshots than the matched X-alone / Z-alone arms.
    # max(1, ...) is a safety floor only; every cap here is in the hundreds.
    return {t: max(1, v // 2) for t, v in cap.items()}


def logdir_for(seed):
    return (REPO_ROOT / "logs" / "logs_after_bugfix" /
            f"lambdamax_{DATASET_NAMES[DATASET_KEY]}_partition_t_smallkernels_C4_{TAG}" /
            f"lambdamax_{LAM}" / f"seed_{seed}" / f"partition_{PARTITION_INDEX}")


def run_one(seed, shared_times, cap, epochs_override=None):
    from tetriscnn.datasets import create_datasets
    from tetriscnn.train import train
    from tetriscnn.plots import single_seed_plots
    from tetriscnn.utils import load_json

    cf = _base_config(DATASET_KEY, shared_times, cap)  # cap already halved by caller
    if epochs_override is not None:
        cf.epochs = epochs_override
    cf.seed = seed
    cf.seeds = list(SEEDS_DEFAULT)  # recorded for reference; trained one seed at a time

    cf.train_dataset, cf.val_dataset = create_datasets(cf)
    cf.logdir = str(logdir_for(seed))

    train(cf, "metrics.json")
    metrics = load_json(cf.logdir, "metrics.json")
    try:
        single_seed_plots(cf, metrics)
    except Exception as e:
        print(f"[matched_basis_survey_xzhalf] plotting failed for seed={seed} (non-fatal): {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument("--dry-run", action="store_true", help="print the computed match and exit")
    p.add_argument("--epochs", type=int, default=None, help="override cf.epochs (smoke testing)")
    args = p.parse_args()

    shared_times, matched_cap = compute_matching()
    cap = half_cap(matched_cap)
    print(f"[xzhalf] {len(shared_times)} shared time points: {shared_times}")
    print(f"[xzhalf] matched per-t cap  min(n_X,n_Z): {matched_cap}  (total {sum(matched_cap.values())})")
    print(f"[xzhalf] halved  per-t cap (this arm)   : {cap}  (total {sum(cap.values())} pairs "
          f"= {2 * sum(cap.values())} raw snapshots)")

    if args.dry_run:
        return

    for seed in args.seeds:
        print(f"\n=== XZ-half seed={seed} -> {logdir_for(seed)} ===")
        run_one(seed, shared_times, cap, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
