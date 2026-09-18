"""Train the Figure5+6 XY C4 run once per X/Z snapshot-pairing convention.

The Fig. 5/6 pipeline (`Figures5_6_combined_pipeline.ipynb`) is quoted on ONE run:

    logs/logs_after_bugfix/lambdamax_Paris_XY_XZ_partition_t_smallkernels_C4_[...]
        /lambdamax_3/seed_46/partition_2

whose two input channels are the X and Z snapshots stacked by the historical
`pairing_mode="index"` convention (first min(n_X, n_Z) rows of each basis, in
acquisition order). Because X and Z come from *independent* experimental
realisations, that convention is arbitrary -- see the `ablation/xz-pairing-robustness`
branch, which introduced the swappable `pairing_mode` this script drives.

This trains the SAME config (same model seed, same lambdamax, same partition, same
equivariance -- everything is read from the anchor's own config.json, so the recorded
penalty_params, including the lambda_min = -3 setting, are inherited verbatim rather
than re-derived) under the two meaningful re-pairing conventions:

    repair_fixed     the surviving snapshots are fixed by `pairing_subset_seed`
                     and held constant; only the X<->Z assignment is permuted.
                     Isolates pairing variance from truncation variance.
    repair_resample  subset AND assignment redrawn together. The end-to-end
                     ensemble; also resamples the ~24% of raw snapshots that
                     min-truncation discards.

("cross_time" is a deliberately destructive null control, not a pairing
convention, so it is not part of this comparison.)

Each (mode, pairing seed) lands in its own log subtree,
`..._pm={mode}/lambdamax_3/ps={pairing_seed}/seed_46/partition_2`, so nothing can
overwrite the published anchor run or another draw.

Run:  python scripts/pairing_fig56_runs.py                      # 2 modes x 5 seeds
      python scripts/pairing_fig56_runs.py --pairing-seeds 1 2  # a subset
      python scripts/pairing_fig56_runs.py --replot-only        # plots, no training
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tetriscnn.datasets import create_datasets            # noqa: E402
from tetriscnn.plots import single_seed_plots             # noqa: E402
from tetriscnn.train import train                         # noqa: E402
from tetriscnn.utils import AttrDict, load_json           # noqa: E402

LOGROOT = REPO / "logs/logs_after_bugfix"
LAMBDAS = "[-3, -2, -1, 0, 1, 2, 3, 4, 5]"
ANCHOR = (LOGROOT / f"lambdamax_Paris_XY_XZ_partition_t_smallkernels_C4_{LAMBDAS}"
          / "lambdamax_3/seed_46/partition_2")

MODES = ("repair_fixed", "repair_resample")
PAIRING_SEEDS = (1, 2, 3, 4, 5)  # which re-pairing draws
PAIRING_SUBSET_SEED = 0  # which snapshots survive truncation (repair_fixed only)


def arm_path(mode: str, pairing_seed: int) -> Path:
    """Where one (mode, draw) run folder goes.

    The anchor tree with `_pm={mode}` on the experiment component and a `ps={seed}`
    level inserted under the lambda level -- the same hierarchical shape, and the same
    `ps` abbreviation, that `build_logdir_path()` would give a swept parameter.

    Built by hand rather than through `build_logdir_path()` because that function
    reconstructs the whole tree from a live `cf` (lambda level, sweep levels, seed),
    and here the anchor path is already known -- the only things that change are the
    experiment-name component and the extra draw level. Doing it this way keeps the arm
    folder guaranteed parallel to the anchor instead of merely intended to be.
    """
    parts = list(ANCHOR.relative_to(LOGROOT).parts)
    parts[0] = f"{parts[0]}_pm={mode}"
    # parts is [experiment, lambdamax_3, seed_46, partition_2]; the draw level goes
    # between the lambda level and the model seed, matching the sweep convention.
    parts.insert(2, f"ps={pairing_seed}")
    return LOGROOT / Path(*parts)


def build_cf(mode: str, pairing_seed: int) -> AttrDict:
    """The anchor's own config.json, with only the pairing keys changed.

    Reading the recorded config rather than re-deriving one from `setup_experiment()`
    is what makes this a controlled comparison: every hyperparameter, the kernel
    list, the penalty vector and the seed are the anchor's by construction, so the
    pairing convention is provably the only difference.
    """
    cf = AttrDict()
    cf.update(load_json(str(ANCHOR), "config.json"))

    # The recorded config carries repr()'d dataset objects and a stale logdir.
    for key in ("train_dataset", "val_dataset", "unique_labels"):
        cf.pop(key, None)

    cf.pairing_mode = mode
    cf.pairing_seed = pairing_seed
    cf.pairing_subset_seed = PAIRING_SUBSET_SEED
    cf.logdir = str(arm_path(mode, pairing_seed))
    cf.seed_folder = str(arm_path(mode, pairing_seed))
    cf.return_all_data = False
    return cf


def plot_run(out: Path) -> None:
    """Training-history plots for a finished run, from its own config + metrics.

    Separate from the training call so a run trained before plotting was wired in can
    be given its plots without being retrained -- and so `--replot-only` can refresh
    every arm in one pass. `single_seed_plots` reads `cf.logdir`, so the config is
    re-pointed at where it actually sits rather than where it was first written.
    """
    cf = AttrDict()
    cf.update(load_json(str(out), "config.json"))
    for key in ("train_dataset", "val_dataset"):
        cf.pop(key, None)
    cf.logdir = str(out)
    single_seed_plots(cf, load_json(str(out), "metrics.json"))
    print(f"   history.png -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--pairing-seeds", nargs="+", type=int, default=list(PAIRING_SEEDS))
    ap.add_argument("--replot-only", action="store_true",
                    help="regenerate history plots for finished runs, train nothing")
    args = ap.parse_args()

    todo = [(m, ps) for m in args.modes for ps in args.pairing_seeds]
    for i, (mode, ps) in enumerate(todo, 1):
        out = arm_path(mode, ps)
        trained = (out / "net1.pt").exists()

        if trained:
            print(f"[{i}/{len(todo)}] {mode} ps={ps}: already trained, "
                  f"{'replotting' if args.replot_only else 'plotting only'}")
        elif args.replot_only:
            print(f"[{i}/{len(todo)}] {mode} ps={ps}: not trained, skipped")
            continue
        else:
            cf = build_cf(mode, ps)
            cf.train_dataset, cf.val_dataset = create_datasets(cf)
            print(f"\n[{i}/{len(todo)}] {mode} ps={ps}: {len(cf.train_dataset)} train / "
                  f"{len(cf.val_dataset)} val -> {out}")
            train(cf, "metrics.json")

        plot_run(out)

    print("\ndone. Analyse with: python scripts/pairing_fig56_analysis.py")


if __name__ == "__main__":
    main()
