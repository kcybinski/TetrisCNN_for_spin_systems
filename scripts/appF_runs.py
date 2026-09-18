#!/usr/bin/env python
"""Retrain the three Appendix F (task-dependence figure) runs on the fixed codebase.

The models checked into ``Plots_data/App_F_data/models/{Ising_CLF,Ising_PDM_Delta,Ising_PDM_Omega}``
predate two fixes (commit 7f18997): the snapshot reshape orientation and the
batch-size-dependent L1 penalty (the bottleneck L1 term was a batch SUM against a
mean-reduced data-fit loss, so the effective penalty was ~batch_size times stronger
than the nominal one). Both change Ising training, so the figure has to be redone
from re-trained models.

Configurations are copied verbatim from the checked-in ``config.json`` files, except
for the penalty exponents, which are exposed on the command line:

    --lambdamin  low end of the log-linear branch penalty ramp (default -3, the value
                 the manuscript quotes; the old configs used the hardcoded -5)
    --lambdamax  high end (default -1, as in the old configs). Since the fix removed a
                 factor of ~batch_size from the effective penalty, reproducing the old
                 degree of sparsity may need this raised by up to log10(64) ~ 1.8.

Usage:
    python scripts/appF_runs.py --outdir Plots_data/App_F_data_new/models --lambdamin -3 --lambdamax -1
    python scripts/appF_runs.py --outdir ... --only Ising_CLF
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _base_config():
    """Settings shared by all three runs (identical in the three saved configs)."""
    from tetriscnn.utils import AttrDict, set_kernels

    cf = AttrDict()
    cf.dataset = "Paris_Ising"
    cf.data_fraction = 1
    cf.label_subset_count = None
    cf.filter_t_values = None
    cf.even_split = True
    cf.samples_per_pt_cap = None

    cf.model = "tetriscnn"
    cf.kernel_set = "smallkernels"
    cf.equivariant = False
    cf.hidden_size = 32
    cf.init = "kaiming"
    set_kernels(cf)

    cf.epochs = 200
    cf.learning_rate = 1e-2
    cf.weight_decay = 1e-5
    cf.patience = 10
    cf.early_stop_warmup = 100
    cf.early_stop_min_delta = 1e-5
    cf.batch_size = 64
    cf.VRAM_batch_size = 1024
    cf.num_workers = 0
    cf.pin_memory = False

    cf.use_lr_scheduler = True
    cf.lr_scheduler_type = "reduce_on_plateau"
    cf.lr_reduce_factor = 0.5
    cf.lr_reduce_patience = 5
    cf.min_lr = 1e-9
    cf.weight_penalty = None

    cf.experiment_name = "singlerun"
    cf.save_models = True
    cf.save_histories = True
    cf.save_final_values = False
    return cf


def _configure_run(name, lambdamin, lambdamax):
    from tetriscnn.utils import set_plotting_logging_strings

    cf = _base_config()
    if name == "Ising_CLF":
        cf.task = "partition"
        cf.partition_index = 2
        cf.label_param = "t"
        cf.param_cutoffs = {"delta": 6, "omega": 4}
        cf.seed = 42
    elif name in ("Ising_PDM_Delta", "Ising_PDM_Omega"):
        cf.task = "regression"
        cf.partition_index = None
        cf.label_param = "delta" if name.endswith("Delta") else "omega"
        cf.normalize_labels = True
        cf.goodness_str = "r2agg"
        cf.param_cutoffs = {"delta": 6, "omega": 0}
        cf.seed = 2137 if name.endswith("Delta") else 42
    else:
        raise ValueError(f"unknown run {name}")

    cf.lam = lambdamax
    cf.penalty_params = [10, lambdamin, lambdamax, 1]
    set_plotting_logging_strings(cf)
    return cf


RUNS = ("Ising_CLF", "Ising_PDM_Delta", "Ising_PDM_Omega")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="Plots_data/App_F_data_new/models")
    p.add_argument("--lambdamin", type=float, default=-3)
    p.add_argument("--lambdamax", type=float, default=-1)
    p.add_argument("--only", action="append", choices=RUNS)
    p.add_argument("--overwrite", action="store_true",
                   help="delete an existing output folder instead of erroring")
    args = p.parse_args()

    from tetriscnn.datasets import create_datasets
    from tetriscnn.train import train
    from tetriscnn.utils import load_json

    outroot = (REPO_ROOT / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    summary = {}

    for name in (args.only or RUNS):
        cf = _configure_run(name, args.lambdamin, args.lambdamax)
        cf.train_dataset, cf.val_dataset = create_datasets(cf)

        outdir = outroot / name
        if outdir.exists():
            if not args.overwrite:
                raise SystemExit(f"{outdir} already exists; pass --overwrite to replace it")
            shutil.rmtree(outdir)
        outdir.parent.mkdir(parents=True, exist_ok=True)
        cf.logdir = str(outdir)

        train(cf, "metrics.json")

        metrics = load_json(cf.logdir, "metrics.json")
        n_branches = len(cf.kernels)
        summary[name] = {
            "logdir": str(cf.logdir),
            "penalty_params": cf.penalty_params,
            "epochs_run": len(metrics[f"train_{cf.loss_str}"]),
            f"val_{cf.goodness_str}": metrics[f"val_{cf.goodness_str}"][-1],
            f"val_{cf.loss_str}": metrics[f"val_{cf.loss_str}"][-1],
            "z": [metrics[f"z_{k}"][-1] for k in range(n_branches)],
        }
        print(f"\n=== {name} done: {json.dumps(summary[name]['z'])}\n")

    print(json.dumps(summary, indent=2))
    (outroot / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
