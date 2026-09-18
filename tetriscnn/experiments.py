"""Experiment drivers: everything main.py runs around setup_experiment().

main.py only describes an experiment (``setup_experiment(cf)``) and which runs to
re-plot. This module does the rest:

* ``run_or_remake`` is the entry point main.py calls. It either trains the
  experiment from a fresh config or, when a remake flag is set, re-plots existing
  runs under a chosen folder from their own saved ``config.json``.
* ``run_experiments`` expands the lambda and parameter sweeps and dispatches each
  point to ``run_seeds`` (regression, classification), ``run_lbc`` (learning by
  confusion over every partition) or ``run_partition`` (one partition).
* ``attach_sample_weights`` sets up the weighted-loss experiment.
* ``remake_plots_for_runs``, ``find_run_directories`` and
  ``find_seed_directories_with_models`` discover and re-plot runs already on disk.

See docs/CONFIGURATION.md for the configuration fields these read, and its
sections 8-10 for sweeps, remake flags and the log layout.
"""
import re
from collections import Counter
from itertools import product
from pathlib import Path

import torch

from tetriscnn.datasets import create_datasets
from tetriscnn.plots import (
    experiment_plot, generate_sweep_plot_from_logs, plot_lbc, plot_parameter_sweep,
    single_seed_plots,
)
from tetriscnn.train import train
from tetriscnn.utils import (
    PARAM_ABBREVIATIONS, get_experiment_level_path, get_sweepable_params, load_json,
    load_or_init_config, pick_directory, resolve_logdir, save_json, should_train,
    update_metrics_per_partition, update_metrics_per_seed,
)


def run_experiments(cf):
    """Train (or re-plot) every run of the experiment cf describes.

    Loops over cf.lambdas, then over every swept config field (a list-valued field;
    zipped for the weighted_loss experiment, a cartesian product otherwise), and
    dispatches each point to run_seeds(), run_lbc() or run_partition() by cf.task.
    The per-lambda aggregate is written to metrics_per_lambda.json at the experiment
    level, and a single-parameter sweep is plotted if cf.visualize_sweep is set.
    """

    # Extract sweepable parameters (excluding lambdas, seeds)
    sweepable = get_sweepable_params(cf, exclude=['lambdas', 'seeds'])

    if not sweepable:
        # Backward compatibility: no sweeps, use old logic
        metrics_per_lambda = {}
        for lam in cf.lambdas:
            cf.lam = lam

            if cf.experiment_name in ("lambdamax", "singlerun"):
                cf.penalty_params = [10, -3, cf.lam, 1]
            elif cf.experiment_name == "lambdatot":
                cf.penalty_params = None
            elif cf.experiment_name == "weighted_loss":
                cf.penalty_params = [10, -3, cf.lam, 1]

            if cf.task in ("regression", "classification"):
                metrics_per_lambda[lam] = run_seeds(cf, param_values=None)
            elif cf.task == "lbc":
                metrics_per_lambda[lam] = run_lbc(cf, param_values=None)
            elif cf.task == "partition":
                assert cf.partition_index is not None
                metrics_per_lambda[lam] = run_partition(cf, param_values=None)

        # Navigate to experiment level
        cf.logdir = get_experiment_level_path(Path(cf.logdir))
        if (not cf.remake_history_and_pt_plots) and (not cf.remake_lambda_plot):
            save_json(metrics_per_lambda, cf.logdir, "metrics_per_lambda.json")
        else:
            save_json(metrics_per_lambda, cf.logdir, f"metrics_per_lambda_remake.json")
        return

    # Parameter sweep mode
    param_names = list(sweepable.keys())
    param_value_lists = [sweepable[name] for name in param_names]

    # For weighted_loss experiment: use zip (parallel iteration) instead of product (cartesian)
    # This ensures scenarios run in parallel: [a, b, c, d] instead of all combinations
    use_parallel_sweep = (cf.experiment_name == "weighted_loss")

    if use_parallel_sweep:
        # Verify all parameter lists have the same length
        lengths = [len(lst) for lst in param_value_lists]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"weighted_loss experiment requires all swept parameters to have the same length.\n"
                f"Got: {dict(zip(param_names, lengths))}"
            )
        print(f"[Weighted Loss] Running {lengths[0]} parallel scenarios (using zip, not product)")

    metrics_per_lambda = {}
    for lam in cf.lambdas:
        cf.lam = lam

        if cf.experiment_name in ("lambdamax", "singlerun"):
            cf.penalty_params = [10, -3, cf.lam, 1]
        elif cf.experiment_name == "lambdatot":
            cf.penalty_params = None
        elif cf.experiment_name == "weighted_loss":
            cf.penalty_params = [10, -3, cf.lam, 1]

        metrics_per_param_combo = {}

        # Choose iteration method: zip (parallel) or product (cartesian)
        if use_parallel_sweep:
            param_iterator = zip(*param_value_lists)
        else:
            param_iterator = product(*param_value_lists)

        for param_values_tuple in param_iterator:
            # Build param_values dict
            param_values = dict(zip(param_names, param_values_tuple))

            # Update cf with current param values
            for param_name, param_value in param_values.items():
                setattr(cf, param_name, param_value)

            # Run experiment with current parameters
            if cf.task in ("regression", "classification"):
                metrics = run_seeds(cf, param_values)
            elif cf.task == "lbc":
                metrics = run_lbc(cf, param_values)
            elif cf.task == "partition":
                assert cf.partition_index is not None
                metrics = run_partition(cf, param_values)

            # Store metrics with param combo key
            combo_key = "-".join(
                f"{PARAM_ABBREVIATIONS.get(k, k)}={v}"
                for k, v in sorted(param_values.items())
            )
            metrics_per_param_combo[combo_key] = metrics

        metrics_per_lambda[lam] = metrics_per_param_combo

    # Navigate to experiment level and save
    cf.logdir = get_experiment_level_path(Path(cf.logdir))
    if (not cf.remake_history_and_pt_plots) and (not cf.remake_lambda_plot):
        save_json(metrics_per_lambda, cf.logdir, "metrics_per_lambda.json")
    else:
        save_json(metrics_per_lambda, cf.logdir, f"metrics_per_lambda_remake.json")

    # Visualize single-parameter sweep if enabled
    if 'visualize_sweep' in cf.keys() and cf.visualize_sweep:
        if len(param_names) == 1 and len(cf.lambdas) == 1:
            # Single parameter sweep with single lambda
            swept_param_name = param_names[0]
            swept_param_values = sweepable[swept_param_name]

            # Extract metrics per parameter value for the single lambda
            lam = cf.lambdas[0]
            metrics_per_value = {}

            for param_val in swept_param_values:
                # Reconstruct the combo key
                abbrev = PARAM_ABBREVIATIONS.get(swept_param_name, swept_param_name)
                combo_key = f"{abbrev}={param_val}"
                metrics_per_value[param_val] = metrics_per_lambda[lam][combo_key]

            # Create sweep visualization
            filename = f"sweep_{PARAM_ABBREVIATIONS.get(swept_param_name, swept_param_name)}.png"
            plot_parameter_sweep(swept_param_name, swept_param_values, metrics_per_value, cf, filename)
        else:
            print(f"\nNote: Parameter sweep visualization requires exactly 1 swept parameter and 1 lambda.")
            print(f"Found {len(param_names)} swept parameters and {len(cf.lambdas)} lambdas.")


def attach_sample_weights(cf, verbose=False):
    """For the weighted_loss experiment (scenario d), attach per-sample importance
    weights to cf.train_dataset, keyed by acquisition time.

    The weights are inverse-frequency: a sample at a time point with N snapshots gets
    weight min_count / N, so under-represented times are up-weighted toward 1 and each
    acquisition time contributes comparably to the loss. This matches the appendix's
    ``scales inversely with the representation''. No-op unless the current scenario
    requests weighted loss. Must run after create_datasets, since it is re-created per seed.
    """
    if not "use_weighted_loss" in cf.keys() or not cf.use_weighted_loss:
        return

    # Imbalance lives across acquisition times, not the (phase) labels: for a
    # classification/partition dataset cf.train_dataset.labels is the one-hot phase.
    times = cf.train_dataset.times
    times_list = times.tolist()
    tp_counts = Counter(times_list)
    min_count = min(tp_counts.values())

    weights = torch.tensor([min_count / tp_counts[t] for t in times_list], dtype=torch.float32)
    cf.train_dataset.sample_weights = weights

    if verbose:
        print(f"[Weighted Loss] {len(tp_counts)} time points; per-sample weight "
              f"min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")


def run_seeds(cf, param_values=None):
    """Train (or re-plot) one run per seed, for the regression/classification tasks."""
    metrics_per_seed = {"z":[], f"val_{cf.loss_str}":[], f"val_{cf.goodness_str}":[],
                        "pt":[], "pt2":[], "epochs":[], "net2_norm":[]}

    for seed in cf.seeds:
        cf.seed = seed
        cf.train_dataset, cf.val_dataset = create_datasets(cf)   # deterministic; train-val split uses a separate seed.
        attach_sample_weights(cf, verbose=(seed == cf.seeds[0]))

        # resolve_logdir() prefers the actual on-disk folder when remaking, so a run
        # that was moved (or whose logdir was set explicitly by a script) is re-plotted
        # in place instead of at the path build_logdir_path() reconstructs from cf.
        cf.logdir = resolve_logdir(cf, param_values, seed=seed)
        if should_train(cf):
            train(cf, "metrics.json")

        metrics = load_json(cf.logdir, "metrics.json")

        single_seed_plots(cf, metrics)
        update_metrics_per_seed(cf, metrics, metrics_per_seed)

    return metrics_per_seed


def run_lbc(cf, param_values=None):
    """Learning by confusion: one run per seed and per partition of the tuning parameter."""
    cf.partition_index = 0
    _, temp_dataset = create_datasets(cf)   # deterministic; train-val split uses a separate seed.
    no_partitions = temp_dataset.processor.no_partitions

    is_tetriscnn = getattr(cf, 'model', 'tetriscnn') == "tetriscnn"

    metrics_per_seed = {}
    for seed in cf.seeds:
        cf.seed = seed
        metrics_per_partition = {
            "z": [],
            f"val_{cf.loss_str}": [], f"train_{cf.loss_str}": [],
            f"val_{cf.goodness_str}": [], f"train_{cf.goodness_str}": []
        } if is_tetriscnn else {
            f"val_{cf.loss_str}": [], f"train_{cf.loss_str}": [],
            f"val_{cf.goodness_str}": [], f"train_{cf.goodness_str}": []
        }
        for partition_index in range(no_partitions):
            cf.partition_index = partition_index
            cf.train_dataset, cf.val_dataset = create_datasets(cf)   # deterministic; train-val split uses a separate seed.

            cf.logdir = resolve_logdir(cf, param_values, seed=seed)
            if should_train(cf):
                train(cf, "metrics.json")

            metrics = load_json(cf.logdir, "metrics.json")

            single_seed_plots(cf, metrics)
            update_metrics_per_partition(cf, metrics, metrics_per_partition)

        metrics_per_seed[seed] = metrics_per_partition

    cf.logdir = Path(cf.logdir).parent.parent # up two levels
    enabled_metrics = None if is_tetriscnn else ["acc", "loss"]
    metrics_per_seed = plot_lbc(metrics_per_seed, cf, "lbc.png", enabled_metrics=enabled_metrics)

    return metrics_per_seed

def run_partition(cf, param_values=None):
    """One run per seed at the single partition cf.partition_index."""
    # One entry per seed, so downstream statistics (mean/spread across seeds) are meaningful.
    # Previously this ran only cf.seeds[0], which left a single sample and NaN error bars.
    metrics_per_seed = {"z":[], f"val_{cf.loss_str}":[], f"train_{cf.loss_str}":[],
                        f"val_{cf.goodness_str}":[], f"train_{cf.goodness_str}":[]}

    for seed in cf.seeds:
        cf.seed = seed
        cf.train_dataset, cf.val_dataset = create_datasets(cf)   # deterministic; train-val split uses a separate seed.
        attach_sample_weights(cf, verbose=(seed == cf.seeds[0]))

        cf.logdir = resolve_logdir(cf, param_values, seed=seed)
        if should_train(cf):
            train(cf, "metrics.json")

        metrics = load_json(cf.logdir, "metrics.json")

        single_seed_plots(cf, metrics)
        update_metrics_per_partition(cf, metrics, metrics_per_seed)

    return metrics_per_seed



def remake_plots_for_runs(run_dirs, remake_flags, plot_config, basepath):
    """
    Re-plot every discovered run directly, each from its OWN config.json.

    A parent folder can hold many runs that differ in lambda, cap, rebate, ... and
    that all end in a folder called "seed_42". Keying them by seed number collapses
    them to one arbitrary winner, and reconstructing a sweep from the first config
    instead re-plots that one winner once per (bogus) parameter combination. Both
    are wrong for the same reason: on a remake the runs are whatever is on disk,
    not whatever the config says the sweep was. So iterate the directories.

    Returns the last run's config, with logdir pointed at the experiment level, so
    the caller can hand a coherent cf to experiment_plot().
    """
    metrics_per_lambda = {}
    last_cf = None
    seen_seeds = set()

    for run_dir in run_dirs:
        cf = load_or_init_config(remake_flags, plot_config, run_dir)
        cf.logdir = str(run_dir)
        cf._remake_seed_logdir = str(run_dir)

        # For an lbc/partition run the leaf is partition_{i} and the seed number
        # lives one level up, so read each from wherever it actually is.
        partition_match = re.match(r"partition_(\d+)$", run_dir.name)
        if partition_match:
            cf.partition_index = int(partition_match.group(1))
            seed_dir = run_dir.parent
        else:
            seed_dir = run_dir

        seed_match = re.match(r"seed_(\d+)", seed_dir.name)
        if seed_match:
            cf.seed = int(seed_match.group(1))
            cf.seeds = [cf.seed]
            seen_seeds.add(cf.seed)

        # Only the regression phase-transition plot reads the datasets; skip the
        # (expensive) rebuild for every other task.
        if cf.task == "regression":
            cf.train_dataset, cf.val_dataset = create_datasets(cf)

        print(f"[remake] {run_dir}")
        metrics = load_json(cf.logdir, "metrics.json")
        single_seed_plots(cf, metrics)

        # Re-accumulate the cross-lambda aggregate that run_experiments would have
        # written. experiment_plot() reads metrics_per_lambda_remake.json for a
        # lambdamax/lambdatot remake, and this path skips run_experiments, so
        # without this the per-run plots succeed and the aggregate plot then dies
        # on a missing file. Grouped by each run's OWN cf.lam, and aggregated with
        # the same helpers run_experiments uses, so the shape matches exactly.
        lam = cf.lam
        if lam not in metrics_per_lambda:
            if cf.task in ("regression", "classification"):
                metrics_per_lambda[lam] = {
                    "z": [], f"val_{cf.loss_str}": [], f"val_{cf.goodness_str}": [],
                    "pt": [], "pt2": [], "epochs": [], "net2_norm": [],
                }
            else:
                metrics_per_lambda[lam] = {
                    "z": [], f"val_{cf.loss_str}": [], f"train_{cf.loss_str}": [],
                    f"val_{cf.goodness_str}": [], f"train_{cf.goodness_str}": [],
                }
        if cf.task in ("regression", "classification"):
            update_metrics_per_seed(cf, metrics, metrics_per_lambda[lam])
        else:
            update_metrics_per_partition(cf, metrics, metrics_per_lambda[lam])

        last_cf = cf

    if last_cf is not None and metrics_per_lambda:
        # Write at the level the user actually selected: that is the experiment level
        # for a whole-tree remake, and it is where experiment_plot() will look once
        # get_experiment_level_path() finds this file there.
        save_json(metrics_per_lambda, Path(basepath), "metrics_per_lambda_remake.json")
        last_cf.logdir = str(basepath)
        # Restore the full seed set: the loop pins cf.seeds to one seed per run, but
        # the aggregate that experiment_plot() is about to draw spans all of them
        # (it prints the count in the figure title).
        if seen_seeds:
            last_cf.seeds = sorted(seen_seeds)
    return last_cf


def _is_seed_level_folder(path: Path) -> bool:
    """Returns True if path directly contains net1.pt and net2.pt."""
    return (path / "net1.pt").exists() and (path / "net2.pt").exists()


def find_run_directories(basepath):
    """
    Find every directory that holds one trained run's own artifacts.

    A "run" is a folder with net1.pt + net2.pt + metrics.json, i.e. the level a
    remake actually has to read and write. For an lbc/partition experiment that is
    the partition_{i} folder, NOT its seed_{n} parent -- the parent holds no
    metrics.json at all, which is why remaking such a folder used to fail outright.
    Deliberately does no collapsing: 12 partitions under one seed are 12 runs.
    """
    basepath = Path(basepath)
    run_dirs = set()
    for net1_file in basepath.glob("**/net1.pt"):
        run_dir = net1_file.parent
        if (run_dir / "net2.pt").exists() and (run_dir / "metrics.json").exists():
            run_dirs.add(run_dir)
    return sorted(run_dirs)


def find_seed_directories_with_models(basepath):
    """
    Find the seed-level directories under basepath.

    Collapses an lbc/partition run's partition_{i} folders onto their seed_{n}
    parent, because the callers of THIS function want seed identity (to scope
    cf.seeds, and to pick one representative run per seed for the aggregate
    lambda/sweep plots). Per-run replotting must use find_run_directories()
    instead, which keeps the partitions distinct.

    Deduplicated: without it a 12-partition seed was returned 12 times, so anything
    iterating the result re-visited every seed once per partition.
    """
    basepath = Path(basepath)
    seed_dirs = set()
    for net1_file in basepath.glob("**/net1.pt"):
        seed_dir = net1_file.parent
        if (seed_dir / "net2.pt").exists():
            if "partition" in seed_dir.name:
                seed_dirs.add(seed_dir.parent)  # For lbc task, the seed level is one level up
            else:
                seed_dirs.add(seed_dir)
    return sorted(seed_dirs)


def run_or_remake(setup_experiment, remake_flags, plot_config, basepath=None):
    """Train the experiment set up by ``setup_experiment``, or re-plot existing runs.

    Args:
        setup_experiment: function filling a fresh config in place (main.py's).
            Only called when no remake flag is set.
        remake_flags: dict of the three remake switches (remake_lambda_plot,
            remake_history_and_pt_plots, remake_sweep_plot); at most one may be set.
            All False means "train".
        plot_config: plotting options stored on cf (enabled_metrics, fit_branches).
        basepath: for a remake, the folder to re-plot: one run folder, or any
            parent holding several. None opens a directory picker. Ignored for
            training, where the log folder is derived from the config.

    Returns:
        The final config (for a multi-run remake, the aggregated one).
    """
    basepath = Path(basepath) if basepath is not None else None
    if (basepath is None or not basepath.exists()) and any(remake_flags.values()):
        basepath = Path(pick_directory(title="Select the directory containing the experiment logs."))
    elif basepath is not None and not basepath.exists():
        basepath = Path("logs/")

    print(f"Experiment basepath: {basepath}")

    cf = load_or_init_config(remake_flags, plot_config, basepath)

    remake_run_dirs = None  # set when a parent folder holds several runs

    # --- Remake seed scoping: reduce cf.seeds to match selected basepath ---
    if any(remake_flags.values()):
        if _is_seed_level_folder(basepath):
            # basepath is a seed-level folder (contains net1.pt + net2.pt)
            # Scope run to this single seed only
            seed_match = re.match(r"seed_(\d+)", basepath.name)
            if seed_match:
                cf.seeds = [int(seed_match.group(1))]
                cf.seed = int(seed_match.group(1))
            else:
                cf.seeds = [cf.seeds[0]]  # fallback: keep first seed only
            cf.logdir = str(basepath)
            cf._remake_seed_logdir = str(basepath)  # stash for override in run_seeds()
        else:
            # Parent folder: discover all seed dirs, filter cf.seeds to only those present.
            # Stash the actual on-disk path per seed (not just the seed number) so
            # run_seeds() can use it instead of reconstructing a path from cf fields,
            # which would resolve to the original training-time location if this
            # folder was copied/moved elsewhere.
            found_dirs = find_seed_directories_with_models(basepath)
            # Per-run plots operate on the leaf run folders (partition_{i} included);
            # the seed-level list below is only for cf.seeds and the aggregate plots.
            remake_run_dirs = find_run_directories(basepath)

            found_seeds = set()
            seed_logdirs = {}
            for d in found_dirs:
                m = re.match(r"seed_(\d+)", d.name)
                if m:
                    seed = int(m.group(1))
                    found_seeds.add(seed)
                    # NOTE: lossy on purpose -- only used by the lambda/sweep aggregate
                    # paths, which do want one representative run per seed. The
                    # per-run history plots go through remake_plots_for_runs() instead.
                    seed_logdirs.setdefault(seed, str(d))
            if found_seeds:
                cf.seeds = sorted(found_seeds)
            cf._remake_seed_logdirs = seed_logdirs

    if not any(remake_flags.values()):
        setup_experiment(cf)

    if remake_run_dirs is not None and remake_flags["remake_history_and_pt_plots"]:
        # Parent folder + per-run plots: re-plot each discovered run from its own
        # config instead of driving run_experiments over a reconstructed sweep.
        print(f"Re-plotting {len(remake_run_dirs)} run(s) found under {basepath}")
        aggregated_cf = remake_plots_for_runs(remake_run_dirs, remake_flags,
                                              plot_config, basepath)
        if aggregated_cf is not None:
            cf = aggregated_cf
    elif not cf.remake_sweep_plot:
        run_experiments(cf)

    experiment_plot(cf, basepath)

    # Generate sweep plot from existing logs if remake flag is set
    if cf.remake_sweep_plot:
        generate_sweep_plot_from_logs(cf, basepath)

    return cf
