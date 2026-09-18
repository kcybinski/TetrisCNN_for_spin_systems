"""Numerical fingerprint for the correlator best-subset selector (`select_terms`).

Companion to scripts/golden_run.py, same --record / --check contract, same job:
catch a silent change in numerics. Here the thing being pinned is which correlator
terms `fit_activation_to_correlators(..., select_terms=k)` picks, and how well the
selected subset fits -- the numbers that end up quoted as a branch's equation.

Deliberately built on DETERMINISTIC SYNTHETIC spin snapshots (a seeded RNG), not on
a trained run, so it needs neither the gitignored data/ tree nor a checkpoint and
gives the same answer on any machine. The trained-model behaviour it stands in for
is what notebooks/Figure5.ipynb and
notebooks/Figures5_6_combined_pipeline.ipynb do on the recorded runs in Plots_data/.

What is compared:
  - `support`: the selected column indices (exact match required -- a changed
    support means a different equation, not a rounding difference);
  - `r2_selected` and `r2_full`: within rtol, absorbing BLAS/platform float noise;
  - `coefficients`: within a looser rtol, since they are the most noise-prone.

Run:
    python scripts/golden_bfa_selection.py --check
    python scripts/golden_bfa_selection.py --record   # only after a deliberate change
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tetriscnn.interpret import fit_activation_to_correlators  # noqa: E402
from tetriscnn.pattern_generation import pattern_correlator  # noqa: E402

BASELINE = REPO / "tests/golden/bfa_selection.json"

RTOL_R2 = 1e-6
RTOL_COEF = 1e-4

# Stopping rule pinned by the forward cases.
PLATEAU_TOL = 0.01
PLATEAU_FLOOR = 0.95

# (name, kernel spec, channels, select_terms, seed). Spans the shapes that behave
# differently: a non-square footprint, a masked diagonal, a full square, and
# two-channel (cross-basis) cases.
#
# Term count is 2^(active_sites * channels) - 1, so an UNMASKED 2x2 on two channels
# has 255 terms and C(255, 3) = 2.7M refits -- past the guard in best_subset_terms
# and genuinely infeasible. It is deliberately not a case here; that combination is
# covered by the guard test in tests/test_interpret.py instead. Every branch the
# current kernel sets actually produce stays at <= 15 terms.
CASES = [
    ("domino_2x1_k2", [(2, 1), 1, 1, None, 1], 1, 2, 11),
    ("square_2x2_k2", [(2, 2), 1, 1, None, 1], 1, 2, 12),
    ("square_2x2_k3", [(2, 2), 1, 1, None, 1], 1, 3, 12),
    ("diag_mask_k2", [(2, 2), 1, 1, [[1, 0], [0, 1]], 1], 1, 2, 13),
    ("xz_domino_k3", [(2, 1), 1, 1, None, 1], 2, 3, 14),
    ("xz_diag_k3", [(2, 2), 1, 1, [[1, 0], [0, 1]], 1], 2, 3, 15),
]

# Cases whose target is (almost) a single enumerated correlator, so the forward
# path plateaus immediately and the pruning rule has something to prune.
CLEAN_CASES = [
    ("clean_square_k4", [(2, 2), 1, 1, None, 1], 1, 4, 16),
    ("clean_domino_k3", [(2, 1), 1, 1, None, 1], 1, 3, 17),
]


def build_case(kernel, n_channels, seed, B=500, H=6, W=6, clean=False):
    """Deterministic snapshots and a target that leans on the 1-point correlator.

    `clean=False` pools the target over a (1,1) window, which is deliberately NOT
    one of the branch's enumerated features (those are pooled over the footprint's
    valid-position grid). That mismatch caps R^2 near 0.7, which is why those cases
    land below the plateau floor and keep their full budget.

    `clean=True` pools over the branch footprint itself and cuts the noise, so one
    correlator explains almost everything and the plateau rule actually truncates.
    Both regimes need pinning: the golden would otherwise never exercise pruning.
    """
    rng = np.random.default_rng(seed)
    snaps = rng.choice(np.array([-1.0, 1.0]), size=(B, n_channels, H, W))
    window = tuple(kernel[0]) if clean else (1, 1)
    c_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=window)
    noise = 0.05 if clean else 0.25
    z = 3.0 * c_site + noise * rng.standard_normal(B)
    return snaps, z


def run() -> dict:
    names = {1: ["Z"], 2: ["X", "Z"]}
    out = {}
    for name, kernel, n_channels, k, seed, clean in (
        [(*c, False) for c in CASES] + [(*c, True) for c in CLEAN_CASES]
    ):
        snaps, z = build_case(kernel, n_channels, seed, clean=clean)

        # --- forward path + plateau/floor pruning -------------------------------
        # Pinned separately from best-subset because it answers a different
        # question (nested importance hierarchy, self-chosen length) and can
        # legitimately return a different support and a different term count.
        fwd = fit_activation_to_correlators(
            snaps, z, kernel, channel_names=names[n_channels],
            select_terms=k, select_strategy="forward",
            plateau_tol=PLATEAU_TOL, plateau_floor=PLATEAU_FLOOR, verbose=False,
        )
        out[f"{name}__forward"] = {
            "n_terms_available": len(fwd["patterns"]),
            "budget": k,
            "plateau_tol": PLATEAU_TOL,
            "plateau_floor": PLATEAU_FLOOR,
            "support": [int(j) for j in fwd["selected_terms"]],
            "path_columns": [int(c) for c, _ in fwd["selection_path"]],
            "path_r2": [float(r) for _, r in fwd["selection_path"]],
            "r2_selected": float(fwd["selected_r_squared"]),
            "r2_full": float(fwd["r_squared"]),
            "coefficients": [float(c) for c in np.asarray(fwd["selected_coefficients"])],
        }

        if clean:
            continue

        res = fit_activation_to_correlators(
            snaps, z, kernel, channel_names=names[n_channels],
            select_terms=k, verbose=False,
        )
        out[name] = {
            "n_terms_available": len(res["patterns"]),
            "select_terms": k,
            "support": [int(j) for j in res["selected_terms"]],
            "patterns": list(res["selected_patterns"]),
            "r2_selected": float(res["selected_r_squared"]),
            "r2_full": float(res["r_squared"]),
            "coefficients": [float(c) for c in np.asarray(res["selected_coefficients"])],
        }
    return out


def check(current: dict, baseline: dict) -> list[str]:
    problems = []
    missing = set(baseline) - set(current)
    extra = set(current) - set(baseline)
    if missing:
        problems.append(f"cases missing from this run: {sorted(missing)}")
    if extra:
        problems.append(f"cases not in the baseline: {sorted(extra)}")

    for name in sorted(set(current) & set(baseline)):
        cur, base = current[name], baseline[name]
        if cur["support"] != base["support"]:
            problems.append(
                f"{name}: support changed {base['support']} -> {cur['support']} "
                "(a different equation, not float noise)"
            )
        if cur.get("path_columns") != base.get("path_columns"):
            problems.append(
                f"{name}: forward path order changed "
                f"{base.get('path_columns')} -> {cur.get('path_columns')}"
            )
        if "path_r2" in base and len(cur.get("path_r2", [])) == len(base["path_r2"]):
            if not np.allclose(cur["path_r2"], base["path_r2"], rtol=RTOL_R2, atol=0):
                problems.append(f"{name}: forward path R^2 moved past rtol {RTOL_R2}")
        if cur["n_terms_available"] != base["n_terms_available"]:
            problems.append(
                f"{name}: term enumeration changed "
                f"{base['n_terms_available']} -> {cur['n_terms_available']}"
            )
        for key, rtol in (("r2_selected", RTOL_R2), ("r2_full", RTOL_R2)):
            if not np.isclose(cur[key], base[key], rtol=rtol, atol=0):
                problems.append(
                    f"{name}: {key} {base[key]:.10f} -> {cur[key]:.10f} (rtol {rtol})"
                )
        if len(cur["coefficients"]) != len(base["coefficients"]):
            problems.append(f"{name}: coefficient count changed")
        elif not np.allclose(cur["coefficients"], base["coefficients"],
                             rtol=RTOL_COEF, atol=1e-8):
            problems.append(f"{name}: coefficients moved past rtol {RTOL_COEF}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="(re)record the baseline")
    ap.add_argument("--check", action="store_true", help="compare against the baseline")
    args = ap.parse_args()
    if not (args.record or args.check):
        ap.error("pass --record or --check")

    current = run()

    if args.record:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"[golden_bfa] recorded {len(current)} cases -> {BASELINE}")
        return 0

    if not BASELINE.exists():
        print(f"[golden_bfa] no baseline at {BASELINE}; run --record first")
        return 1

    baseline = json.loads(BASELINE.read_text())
    problems = check(current, baseline)
    if problems:
        print("[golden_bfa] MISMATCH:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"[golden_bfa] OK: {len(current)} cases match the recorded baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
