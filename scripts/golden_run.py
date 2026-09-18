#!/usr/bin/env python
"""Golden-run regression harness for the TetrisCNN refactor.

This runs one small, fixed-seed, deterministic training run (Paris_Ising, task
regression, experiment "singlerun", a single lambda, a capped/reduced dataset) via
the library functions directly (create_datasets + train), and fingerprints its
outcome: final train/val loss, final val goodness, and the final per-branch
bottleneck (z_k) vector.

HOW THE REFACTOR SHOULD USE THIS
---------------------------------
1. On a clean, pre-refactor checkout, record the baseline ONCE:

       python scripts/golden_run.py --record

   This writes tests/golden/baseline.json, which SHOULD be committed to git (it is
   small, deterministic, and is exactly the kind of artifact a behavior-preservation
   refactor needs as a reference point).

2. After each refactor phase (a PR, a module rewrite, etc.), verify nothing observable
   changed:

       python scripts/golden_run.py --check

   Exits 0 if the new run matches the recorded baseline within tolerance, exits 1
   (with a diff-style report) if it doesn't. Wire this into CI once the refactor
   branch exists.

Everything the run produces (checkpoints, config.json, metrics.json, ...) is written
to a Python tempdir and discarded when the process exits -- only the small JSON
fingerprint in tests/golden/baseline.json is meant to persist.

TOLERANCE
---------
Empirically (see test-development notes / this file's git history), repeating this
exact config back-to-back on this machine's PyTorch MPS backend reproduced every
recorded value bit-for-bit (0.0 diff across two independent runs). MPS does not
carry the same strict determinism guarantees as CUDA's deterministic-algorithms mode
though, and CPU-vs-MPS or PyTorch-version drift can plausibly perturb results at the
level of float32 rounding accumulated over a handful of epochs. We therefore do NOT
use exact equality: LOSS/GOODNESS_RTOL=1e-3 (abs floor 1e-6) and the looser
Z_RTOL=1e-2 (abs floor 1e-4) for the bottleneck vector (individual z_k can be very
close to 0, where pure relative tolerance is meaningless -- hence the absolute
floor). This is loose enough to absorb cross-machine/PyTorch-version float noise,
but tight enough that a real behavior change -- in ad hoc experiments while building
this script, e.g. accidentally training a different number of epochs -- produced z_k
differences of order 0.1-0.7 and loss differences of order 10x-100x, both many
orders of magnitude past this tolerance.
"""
import argparse
import json
import math
import platform
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BASELINE_PATH = REPO_ROOT / "tests" / "golden" / "baseline.json"

LOSS_RTOL, LOSS_ATOL = 1e-3, 1e-6
Z_RTOL, Z_ATOL = 1e-2, 1e-4


def _build_config():
    """The fixed, small, deterministic experiment config for the golden run."""
    from tetriscnn.utils import AttrDict, set_kernels, set_plotting_logging_strings

    cf = AttrDict()
    # --- dataset ---
    cf.task = "regression"
    cf.label_param = "delta"
    cf.normalize_labels = True
    cf.label_subset_count = None
    cf.dataset = "Paris_Ising"
    cf.filter_t_values = None
    cf.data_fraction = 1
    cf.param_cutoffs = {"delta": 6, "omega": 0}
    cf.even_split = True
    cf.samples_per_pt_cap = 10          # keeps the dataset tiny (~196 train / 84 val)
    cf.seed = 2024

    # --- model ---
    cf.kernel_set = "smallkernels"
    cf.equivariant = False
    set_kernels(cf)
    cf.model = "tetriscnn"
    cf.hidden_size = 8
    cf.init = "kaiming"

    # --- training ---
    cf.epochs = 10
    cf.learning_rate = 1e-2
    cf.weight_decay = 1e-5
    cf.patience = 10
    cf.early_stop_warmup = 1000          # >> epochs: early stopping never fires,
                                          # so exactly `epochs` epochs always run.
    cf.early_stop_min_delta = 1e-5
    cf.batch_size = 16
    cf.VRAM_batch_size = 1024
    cf.num_workers = 0
    cf.pin_memory = False
    cf.use_lr_scheduler = False
    cf.weight_penalty = None

    # --- experiment/penalty bookkeeping ---
    cf.experiment_name = "singlerun"
    cf.lam = -1
    cf.penalty_params = [10, -5, cf.lam, 1]
    cf.goodness_str = "r2agg"
    cf.save_histories = True
    cf.save_final_values = False
    cf.save_models = False               # fingerprint only needs metrics, not weights
    set_plotting_logging_strings(cf)

    return cf


def _recorded_config_dict(cf):
    """A small, JSON-safe subset of cf worth recording for provenance (NOT used in
    the --check comparison itself, only shown to a human diagnosing a mismatch)."""
    keys = [
        "task", "label_param", "normalize_labels", "dataset", "even_split",
        "samples_per_pt_cap", "seed", "kernel_set", "equivariant", "model",
        "hidden_size", "init", "epochs", "learning_rate", "weight_decay",
        "patience", "early_stop_warmup", "early_stop_min_delta", "batch_size", "experiment_name", "lam",
        "penalty_params", "goodness_str",
    ]
    return {k: cf[k] for k in keys}


def run_golden_training():
    """Runs the golden config once and returns the fingerprint dict."""
    from tetriscnn.datasets import create_datasets
    from tetriscnn.train import train
    from tetriscnn.utils import load_json, DEVICE

    cf = _build_config()
    train_dataset, val_dataset = create_datasets(cf)
    cf.train_dataset = train_dataset
    cf.val_dataset = val_dataset

    recorded_config = _recorded_config_dict(cf)

    with tempfile.TemporaryDirectory(prefix="tetriscnn_golden_run_") as tmpdir:
        cf.logdir = str(Path(tmpdir) / "run")
        train(cf, "metrics.json")
        metrics = load_json(cf.logdir, "metrics.json")

    n_branches = len(cf.kernels)
    z_final = [metrics[f"z_{k}"][-1] for k in range(n_branches)]

    fingerprint = {
        "val_goodness": metrics[f"val_{cf.goodness_str}"][-1],
        "train_loss": metrics[f"train_{cf.loss_str}"][-1],
        "val_loss": metrics[f"val_{cf.loss_str}"][-1],
        "z": z_final,
    }

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "device": str(DEVICE),
    }
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except Exception:
        pass

    return {
        "fingerprint": fingerprint,
        "config": recorded_config,
        "versions": versions,
    }


def _isclose(a, b, rtol, atol):
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def compare(baseline, current):
    """Returns a list of human-readable mismatch strings (empty if all match)."""
    problems = []
    bf, cf_ = baseline["fingerprint"], current["fingerprint"]

    for key in ("val_goodness", "train_loss", "val_loss"):
        b, c = bf[key], cf_[key]
        if not _isclose(b, c, LOSS_RTOL, LOSS_ATOL):
            problems.append(f"{key}: baseline={b!r} current={c!r} (rtol={LOSS_RTOL}, atol={LOSS_ATOL})")

    b_z, c_z = bf["z"], cf_["z"]
    if len(b_z) != len(c_z):
        problems.append(f"z: length mismatch, baseline has {len(b_z)} branches, current has {len(c_z)}")
    else:
        for i, (b, c) in enumerate(zip(b_z, c_z)):
            if not _isclose(b, c, Z_RTOL, Z_ATOL):
                problems.append(f"z[{i}]: baseline={b!r} current={c!r} (rtol={Z_RTOL}, atol={Z_ATOL})")

    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", action="store_true", help="run once and (over)write the baseline JSON")
    group.add_argument("--check", action="store_true", help="run once and compare against the recorded baseline")
    args = parser.parse_args()

    print(f"[golden_run] running golden training config (device selection is automatic; "
          f"see 'device' in the recorded versions)...", flush=True)
    # Imported here, not at module scope, for the same reason every other tetriscnn
    # import in this file is deferred: tetriscnn/__init__.py must run before torch.
    from tetriscnn.dataprocessing import DatasetNotAvailableError

    try:
        result = run_golden_training()
    except DatasetNotAvailableError as exc:
        # This run trains on the experimental Ising snapshots, which ship as a release
        # asset rather than in the repository. A fresh clone therefore cannot run it.
        # Skip rather than fail, matching the `data` marker in the pytest suite, so
        # that the README's installation check passes before the data is fetched.
        print(f"[golden_run] SKIPPED: this check trains on the experimental snapshots, "
              f"which are not present.\n  {exc}", file=sys.stderr)
        return 0
    print(f"[golden_run] fingerprint: {json.dumps(result['fingerprint'], indent=2)}")

    if args.record:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"[golden_run] baseline recorded to {BASELINE_PATH}")
        return 0

    # --check
    if not BASELINE_PATH.exists():
        print(f"[golden_run] ERROR: no baseline found at {BASELINE_PATH}. "
              f"Run with --record first.", file=sys.stderr)
        return 2

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    if baseline.get("versions", {}).get("device") != result["versions"]["device"]:
        print(f"[golden_run] WARNING: baseline was recorded on device="
              f"{baseline.get('versions', {}).get('device')!r}, current run is on "
              f"device={result['versions']['device']!r}. Cross-device float "
              f"differences may exceed tolerance for reasons unrelated to the "
              f"refactor.", file=sys.stderr)

    problems = compare(baseline, result)
    if problems:
        print("[golden_run] MISMATCH vs baseline:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("[golden_run] OK: current run matches recorded baseline within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
