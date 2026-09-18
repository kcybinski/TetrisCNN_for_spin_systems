"""
Rebuild the ``results_LBC_Lambdamax_{DATASET}_summary.pkl`` files consumed by Figure4.ipynb.

A lambdamax-LBC run is laid out as::

    {root}/lambdamax_{lam}/seed_{seed}/partition_{p}/metrics.json

For every (lambda, seed) pair this script reads the metric value at each partition,
builds the learning-by-confusion curve over the partition grid, and locates the
transition as the extremum of that curve (argmax for accuracy, argmin for loss).

That yields ``n_lambdas x n_seeds`` peak positions per metric, summarised two ways:

* ``centers``    - one value per lambda (median over seeds), then stats over lambdas.
* ``all_values`` - the raw n_lambdas*n_seeds peaks, pooled with no prior aggregation.

The partition grid is ``(unique_labels[:-1] + unique_labels[1:]) / 2``, i.e. midpoints
between consecutive acquisition times.  The ``*_env`` fields are therefore not
statistical spreads but *grid brackets*: the acquisition times immediately below the
smallest and immediately above the largest peak.

Usage::

    python scripts/lbc_lambdamax_summary.py \
        --root "logs/.../lambdamax_Paris_XY_XZ_lbc_smallkernels_[-5, ..., 5]" \
        --out  Plots_data/results_LBC_Lambdamax_XY_XZ_summary.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
from tetriscnn.utils import parse_array_field

# Metric name -> which extremum of the LBC curve marks the transition.
# Accuracy peaks at the transition; cross-entropy loss dips there.
METRIC_EXTREMUM = {
    "val_acc": "max",
    "train_acc": "max",
    "val_CEL": "min",
    "train_CEL": "min",
}


# ──────────────────────────────────────────────────────────────────────────────
#  Loading
# ──────────────────────────────────────────────────────────────────────────────

def _natural_index(path: Path, prefix: str) -> int:
    """Extract the trailing integer from e.g. ``partition_12`` -> 12."""
    m = re.fullmatch(rf"{prefix}_(-?\d+)", path.name)
    if m is None:
        raise ValueError(f"{path.name!r} does not look like '{prefix}_<int>'")
    return int(m.group(1))


def _parse_unique_labels(raw) -> np.ndarray:
    """``unique_labels`` as recorded in a config.json; see tetriscnn.utils.parse_array_field."""
    return parse_array_field(raw)


def _metric_at_partition(metrics: dict, key: str, patience: int, epochs: int) -> float:
    """
    Pull a single scalar out of one partition's metrics.json.

    Mirrors ``tetriscnn.utils.update_metrics_per_partition``: training histories are
    read at ``-patience`` rather than ``-1``, because early stopping rolls back to the
    best epoch and the final entries are the already-degraded ones.
    """
    series = metrics[key]
    if not isinstance(series, list):
        return float(series)
    index = patience if epochs > patience else 1
    return float(series[-index])


def load_run(root: Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[int], list[int]]:
    """
    Walk a lambdamax-LBC run.

    Returns
    -------
    partitions : (n_partitions,) float array - the partition grid.
    curves     : metric -> (n_lambdas, n_seeds, n_partitions) array.
    lambdas    : the lambda values, sorted ascending.
    seeds      : the seed values, sorted ascending.
    """
    lam_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("lambdamax_")),
        key=lambda p: _natural_index(p, "lambdamax"),
    )
    if not lam_dirs:
        raise FileNotFoundError(f"No 'lambdamax_*' folders under {root}")

    lambdas = [_natural_index(p, "lambdamax") for p in lam_dirs]
    seeds: list[int] | None = None
    partitions: np.ndarray | None = None
    curves: dict[str, list] = {m: [] for m in METRIC_EXTREMUM}

    for lam_dir in lam_dirs:
        seed_dirs = sorted(
            (p for p in lam_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")),
            key=lambda p: _natural_index(p, "seed"),
        )
        lam_seeds = [_natural_index(p, "seed") for p in seed_dirs]
        if seeds is None:
            seeds = lam_seeds
        elif lam_seeds != seeds:
            raise ValueError(f"{lam_dir} has seeds {lam_seeds}, expected {seeds}")

        per_seed: dict[str, list] = {m: [] for m in METRIC_EXTREMUM}
        for seed_dir in seed_dirs:
            part_dirs = sorted(
                (p for p in seed_dir.iterdir() if p.is_dir() and p.name.startswith("partition_")),
                key=lambda p: _natural_index(p, "partition"),
            )
            if not part_dirs:
                raise FileNotFoundError(f"No 'partition_*' folders under {seed_dir}")

            cfg = json.loads((part_dirs[0] / "config.json").read_text())
            unique_labels = _parse_unique_labels(cfg["unique_labels"])
            grid = (unique_labels[:-1] + unique_labels[1:]) / 2.0
            if partitions is None:
                partitions = grid
                globals()["_UNIQUE_LABELS"] = unique_labels
            elif not np.allclose(partitions, grid):
                raise ValueError(f"{seed_dir} has a different partition grid than earlier runs")

            patience = int(cfg.get("patience", 10))
            epochs = int(cfg.get("epochs", 0))

            per_metric: dict[str, list] = {m: [] for m in METRIC_EXTREMUM}
            for part_dir in part_dirs:
                metrics = json.loads((part_dir / "metrics.json").read_text())
                for m in METRIC_EXTREMUM:
                    per_metric[m].append(_metric_at_partition(metrics, m, patience, epochs))
            for m in METRIC_EXTREMUM:
                per_seed[m].append(per_metric[m])

        for m in METRIC_EXTREMUM:
            curves[m].append(per_seed[m])

    assert partitions is not None and seeds is not None
    return (
        partitions,
        {m: np.asarray(v, dtype=float) for m, v in curves.items()},
        lambdas,
        seeds,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Peak finding
# ──────────────────────────────────────────────────────────────────────────────

def find_peak(curve: np.ndarray, partitions: np.ndarray,
              extremum: str, trim_left: int = 0, trim_right: int = 0) -> float:
    """
    Locate the transition on a single LBC curve.

    ``trim_left``/``trim_right`` drop that many partition points from each edge before
    the search.  The edges of an LBC sweep are the least reliable part of the curve --
    the extreme partitions put almost all snapshots in one class, so the classifier can
    score well for reasons unrelated to a phase transition -- and an unguarded
    argmax/argmin will happily latch onto such an artefact.
    """
    lo = trim_left
    hi = len(partitions) - trim_right
    if hi - lo < 1:
        raise ValueError(f"trim_left={trim_left}, trim_right={trim_right} leaves no partitions")
    window = curve[lo:hi]
    idx = int(np.argmax(window) if extremum == "max" else np.argmin(window)) + lo
    return float(partitions[idx])


# ──────────────────────────────────────────────────────────────────────────────
#  Outlier policy
# ──────────────────────────────────────────────────────────────────────────────

def keep_peak(peak: float, curve: np.ndarray, partitions: np.ndarray,
              lam: int, seed: int, trim_left: int, trim_right: int) -> bool:
    """
    Decide whether one (lambda, seed) peak is trustworthy enough to enter the statistics.

    The published summaries keep every peak, which is what this hook does: it is a
    deliberate no-op, kept as a named seam so that a filtering rule can be tried
    without disturbing the call site. Changing it changes the committed pkls and
    therefore the numbers in the manuscript, so it is left alone.

    One peak is worth knowing about if the rule is ever revisited: the XY_XZ
    ``val_CEL`` peak sits at t=5000, far from the other 54, and looks like an
    artefact of an almost-flat loss curve rather than a real transition. It is
    included in the published statistics along with the rest.
    """
    return True


# ──────────────────────────────────────────────────────────────────────────────
#  Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def _bracket(value: float, unique_labels: np.ndarray, side: str) -> np.float32:
    """
    Grid bracket for a partition midpoint: the acquisition time just below (``side='lo'``)
    or just above (``side='hi'``) it.  Clamped at the ends of the grid.
    """
    if side == "lo":
        below = unique_labels[unique_labels < value]
        return np.float32(below[-1] if below.size else unique_labels[0])
    above = unique_labels[unique_labels > value]
    return np.float32(above[0] if above.size else unique_labels[-1])


def _stats(values: np.ndarray, unique_labels: np.ndarray, suffix: str) -> dict:
    v = np.asarray(values, dtype=np.float32)
    return {
        f"mean_{suffix}": np.float32(np.mean(v)),
        # ddof=1 (sample std) -- verified against the committed pkls: the recorded
        # Ising val_acc std 80.90398 is reproduced only with ddof=1, not ddof=0.
        f"std_{suffix}": np.float32(np.std(v, ddof=1)),
        f"median_{suffix}": np.float32(np.median(v)),
        f"min_{suffix}": np.float32(np.min(v)),
        f"max_{suffix}": np.float32(np.max(v)),
        f"min_{suffix}_env": _bracket(float(np.min(v)), unique_labels, "lo"),
        f"max_{suffix}_env": _bracket(float(np.max(v)), unique_labels, "hi"),
    }


def summarize(root: Path, trim_left: int = 0, trim_right: int = 0,
              verbose: bool = True, dump_peaks: bool = False) -> dict:
    """Build the full summary dict for one lambdamax-LBC run folder."""
    partitions, curves, lambdas, seeds = load_run(root)
    unique_labels = globals()["_UNIQUE_LABELS"]

    summary: dict[str, dict] = {}
    for metric, extremum in METRIC_EXTREMUM.items():
        cube = curves[metric]                      # (n_lambdas, n_seeds, n_partitions)
        peaks = np.full(cube.shape[:2], np.nan)     # (n_lambdas, n_seeds)
        for i in range(cube.shape[0]):
            for j in range(cube.shape[1]):
                pk = find_peak(cube[i, j], partitions, extremum, trim_left, trim_right)
                if keep_peak(pk, cube[i, j], partitions, lambdas[i], seeds[j],
                             trim_left, trim_right):
                    peaks[i, j] = pk

        # A lambda whose seeds were all rejected contributes no center.
        with np.errstate(all="ignore"):
            centers = np.nanmedian(peaks, axis=1)   # one value per lambda
        centers = centers[~np.isnan(centers)]
        all_values = peaks.ravel()                  # every (lambda, seed) peak, unpooled
        all_values = all_values[~np.isnan(all_values)]
        if centers.size == 0 or all_values.size == 0:
            raise ValueError(f"every peak was rejected for metric {metric!r}")

        if dump_peaks:
            print(f"\n--- {metric} peak matrix (rows=lambda, cols=seed) ---")
            print("       " + "".join(f"{sd:>9d}" for sd in seeds))
            for i, lam in enumerate(lambdas):
                cells = "".join("      nan" if np.isnan(x) else f"{x:>9.1f}"
                                for x in peaks[i])
                print(f"{lam:>5d}: {cells}")

        summary[metric] = {
            **_stats(centers, unique_labels, "centers"),
            **_stats(all_values, unique_labels, "all_values"),
        }
        if verbose:
            print(f"{metric:>10s} ({extremum}): centers median="
                  f"{summary[metric]['median_centers']:.1f}  "
                  f"all median={summary[metric]['median_all_values']:.1f}  "
                  f"range=[{summary[metric]['min_all_values']:.1f}, "
                  f"{summary[metric]['max_all_values']:.1f}]")

    if verbose:
        print(f"\n{len(lambdas)} lambdas x {len(seeds)} seeds = "
              f"{len(lambdas) * len(seeds)} peaks per metric")
        print(f"lambdas: {lambdas}\nseeds:   {seeds}")
        print(f"partition grid ({len(partitions)}): {partitions[:5]} ... {partitions[-3:]}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="lambdamax-LBC run folder containing lambdamax_*/seed_*/partition_*")
    ap.add_argument("--out", type=Path, default=None,
                    help="destination .pkl (omit for a dry run that only prints)")
    ap.add_argument("--trim-left", type=int, default=0,
                    help="partitions dropped from the low-t edge before the peak search")
    ap.add_argument("--trim-right", type=int, default=0,
                    help="partitions dropped from the high-t edge before the peak search")
    ap.add_argument("--dump-peaks", action="store_true",
                    help="print the per-(lambda, seed) peak matrix; use this to pick the trims")
    args = ap.parse_args()

    summary = summarize(args.root, args.trim_left, args.trim_right,
                        dump_peaks=args.dump_peaks)

    if args.out is None:
        print("\n[dry run] no --out given, nothing written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(summary, f)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
