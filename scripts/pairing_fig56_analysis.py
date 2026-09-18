"""Run `Figures5_6_combined_pipeline.ipynb`'s pipeline over several X/Z-pairing arms.

The notebook analyses ONE run: BFA readout of each dominant branch -> a plane and a
square decision boundary in the latent space -> the same two boundaries rewritten in
correlator space -> the pairwise accuracy comparison. This script does exactly that
(the machinery below is the notebook's, transcribed unchanged, minus the plotting)
for the anchor run and for each re-pairing arm trained by
`scripts/pairing_fig56_runs.py`, and prints the three side by side.

The question it answers: are the quoted equations -- the affine BFA maps, the latent
plane/square, and their correlator rewrites -- properties of the physics, or of the
arbitrary convention by which independent X and Z snapshots were stacked into two
channels?

Run:  python scripts/pairing_fig56_analysis.py
      python scripts/pairing_fig56_analysis.py --split val --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sympy as sp
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tetriscnn.datasets import create_datasets                        # noqa: E402
from tetriscnn.interpret import (fit_activation_to_correlators,       # noqa: E402
                                 plateau_index)
from tetriscnn.models import ShapeAdaptiveConvNet, SmallModel         # noqa: E402
from tetriscnn.utils import (AttrDict, DEVICE, branch_label,          # noqa: E402
                             logistic_penalty_kwargs,
                             load_json)

# ── configuration: identical to the notebook's, so the anchor numbers reproduce ──
LOGROOT = REPO / "logs/logs_after_bugfix"
LAMBDAS = "[-3, -2, -1, 0, 1, 2, 3, 4, 5]"
ANCHOR_EXP = f"lambdamax_Paris_XY_XZ_partition_t_smallkernels_C4_{LAMBDAS}"
RUN_LEAF = "lambdamax_3/seed_46/partition_2"

PAIRING_SEEDS = (1, 2, 3, 4, 5)
MODES = ("repair_fixed", "repair_resample")

# The anchor first, then every (mode, draw). Labels are the dict keys and double as the
# row/column headers of the transfer matrix, so they are kept short.
ARMS = {"anchor": LOGROOT / ANCHOR_EXP / RUN_LEAF}
for _m in MODES:
    for _ps in PAIRING_SEEDS:
        lam_level, seed_level, part_level = RUN_LEAF.split("/")
        ARMS[f"{'fix' if _m == 'repair_fixed' else 'res'}{_ps}"] = (
            LOGROOT / f"{ANCHOR_EXP}_pm={_m}"
            / lam_level / f"ps={_ps}" / seed_level / part_level)

#: Which family each label belongs to, for the per-mode aggregation.
FAMILY = {"anchor": "anchor",
          **{f"fix{ps}": "repair_fixed" for ps in PAIRING_SEEDS},
          **{f"res{ps}": "repair_resample" for ps in PAIRING_SEEDS}}

N_DIMS = 3
CORR_MAX_TERMS, CORR_PLATEAU_TOL, CORR_PLATEAU_FLOOR = 5, 0.05, 0.95
CORR_SUFFICIENT = None
DEGREES = (1, 2)
CS = (0.01, 0.1, 1.0)
MAX_TERMS, PLATEAU_TOL, PLATEAU_FLOOR = 6, 0.002, 0.95
SUFFICIENT_ACC = 0.99

# Cross-term penalty. A candidate's CV accuracy is reduced by
#     penalty = CROSS_TERM_MARGIN * (n_distinct_vars - 1)
#             + DEGREE_MARGIN     * (total_degree    - 1)
# before the forward search takes its argmax, so a term must EARN its complexity. This is
# the notebook's current `term_penalty` mechanism, not the older binary "beat the best
# pure term by MARGIN" rule; with DEGREE_MARGIN = 0 and only bilinear cross terms present
# the two coincide.
#
# 0.05 rather than the notebook's 0.02 (direct fit) / 0.005 (latent fit): on the previous
# 11-arm sweep the cross terms that won did so by 0.005-0.02 of CV accuracy, which is
# inside the fold-to-fold scatter, and they cost the equation its readability. The whole
# curvature gain (plane -> square) is only ~0.017, so 0.05 is ~3x the largest effect any
# single term has demonstrated here -- strong, but still soft, so an interaction that
# genuinely carries the boundary can win and would be visible in the printed path.
#
# DEGREE_MARGIN stays 0: pure squares are wanted (the X^2 term IS the curvature result);
# only products of DISTINCT coordinates are being discouraged.
CROSS_TERM_MARGIN = 0.05
DEGREE_MARGIN = 0.0
FORBID_CROSS = False   # a hard ban; the margin above is preferred so the path stays visible
ACCEPTABLE_DROP = 0.01
SEED = 0
VAR_NAMES = ["X", "Y", "Z"][:N_DIMS]
SYMS = tuple(sp.symbols(" ".join(VAR_NAMES), real=True))

import re                                                              # noqa: E402
_TOKEN = re.compile(r"^([A-Z])(?:\^(\d+))?$")


# ── notebook machinery, transcribed ─────────────────────────────────────────────
def load_run(run_folder: Path, split: str = "train"):
    cf = AttrDict()
    cf.update(load_json(str(run_folder), "config.json"))
    for k in range(len(cf.kernels)):
        if len(cf.kernels[k]) < 5:
            cf.kernels[k].append(1)
    cf.logdir = cf.seed_folder = str(run_folder)
    cf.return_all_data = False

    train_ds, val_ds = create_datasets(cf)
    dataset = {"train": train_ds, "val": val_ds}[split]

    net1 = ShapeAdaptiveConvNet(
        in_channels=dataset[0][0].shape[0], kernels=cf.kernels,
        equivariant=cf.equivariant, hidden_size=cf.hidden_size, init=cf.init,
        device=DEVICE).to(DEVICE)
    net2 = SmallModel([len(cf.kernels), 32, 16, dataset.output_dim],
                      device=DEVICE).to(DEVICE)
    net1.load_state_dict(torch.load(run_folder / "net1.pt", map_location=DEVICE))
    net2.load_state_dict(torch.load(run_folder / "net2.pt", map_location=DEVICE))
    net1.eval(); net2.eval()

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            z = net1(x)
            out = net2(z)
    return {"cf": cf, "path": run_folder, "split": split, "x": x.cpu().numpy(),
            "z": z.cpu().numpy(), "out": out.cpu().numpy(),
            "times": dataset.times.cpu().numpy(),
            "group": cf.equivariant_group if getattr(cf, "equivariant", False) else None,
            "channels": ["X", "Z"] if x.shape[1] == 2 else ["Z"]}


def active_branches(z, abs_tol=1e-3, rel_tol=1e-3):
    mags = np.abs(z).mean(axis=0)
    cut = max(abs_tol, rel_tol * mags.max())
    return [k for k in range(z.shape[1]) if mags[k] > cut]


def boundary_expr(scaler, clf, feat_names, var_names) -> sp.Expr:
    syms = tuple(sp.symbols(" ".join(var_names), real=True))
    scaled = {name: (s - float(m)) / float(sd)
              for name, s, m, sd in zip(var_names, syms, scaler.mean_, scaler.scale_)}
    expr = sp.Float(float(clf.intercept_[0]))
    for fname, coef in zip(feat_names, clf.coef_[0]):
        if abs(coef) < 1e-12:
            continue
        term = sp.Float(float(coef))
        for tok in fname.split():
            var, power = _TOKEN.match(tok).groups()
            term *= scaled[var] ** (int(power) if power else 1)
        expr += term
    return expr


def fit_surface(X, y, degree=2, C_=0.04, penalty="l1", n_splits=5, seed=SEED,
                var_names=None):
    var_names = var_names or VAR_NAMES[:X.shape[1]]
    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LogisticRegression(**logistic_penalty_kwargs(penalty == "l1"), C=C_,
                           solver="liblinear" if penalty == "l1" else "lbfgs",
                           max_iter=20000, class_weight="balanced", random_state=seed))
    pipe.fit(X, y)
    feat_names = pipe.named_steps["polynomialfeatures"].get_feature_names_out(var_names)
    coefs = pipe.named_steps["logisticregression"].coef_[0]
    nonzero = [(n, float(c)) for n, c in zip(feat_names, coefs) if abs(c) > 1e-8]
    cv = cross_val_score(pipe, X, y, scoring="accuracy", n_jobs=-1,
                         cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    return {"pipe": pipe, "degree": degree, "C": C_, "penalty": penalty,
            "acc": float(pipe.score(X, y)), "cv_mean": float(cv.mean()),
            "cv_std": float(cv.std()), "n_terms": len(nonzero),
            "n_feats": len(feat_names), "terms": nonzero,
            "feat_names": list(feat_names), "var_names": var_names,
            "expr": boundary_expr(pipe.named_steps["standardscaler"],
                                  pipe.named_steps["logisticregression"],
                                  feat_names, var_names)}


def parsimonious(rows, key="cv_mean"):
    best = max(rows, key=lambda r: r[key])
    ok = [r for r in rows if r[key] >= best[key] - best["cv_std"]]
    return min(ok, key=lambda r: (r["n_terms"], r["degree"], -r[key]))


def round_expr(expr, num_digits=3):
    rounded = {}
    for atom in expr.atoms(sp.Number):
        if abs(atom) > 1 / 10 ** num_digits:
            rounded[atom] = round(atom, num_digits)
        else:
            rounded[atom] = float(f"{float(atom):+.{num_digits}e}")
    return expr.xreplace(rounded)


def eval_boundary(expr, X, var_names=None):
    var_names = var_names or VAR_NAMES
    syms = tuple(sp.symbols(" ".join(var_names), real=True))
    f = sp.lambdify(syms, sp.expand(expr), "numpy")
    vals = np.broadcast_to(np.asarray(f(*X.T), dtype=float), (len(X),))
    return (vals > 0).astype(int)


def accuracy_of(expr, X, y, var_names=None):
    return float((eval_boundary(expr, X, var_names) == y).mean())


def poly_features(X, degree, var_names):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X), list(poly.get_feature_names_out(var_names))


def _subset_pipe(Xp, y, cols, C_=1e4, seed=SEED):
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_, solver="lbfgs", max_iter=20000,
                                            class_weight="balanced",
                                            random_state=seed))
    pipe.fit(Xp[:, cols], y)
    return pipe


def cv_accuracy(Xp, y, cols, C_=1e4, n_splits=5, seed=SEED):
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_, solver="lbfgs", max_iter=20000,
                                            class_weight="balanced",
                                            random_state=seed))
    cv = cross_val_score(pipe, Xp[:, cols], y, scoring="accuracy", n_jobs=-1,
                         cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    return float(cv.mean())


def is_pure(name):
    return len(name.split()) == 1


def term_complexity(name):
    """(number of distinct variables, total degree) of a `PolynomialFeatures` term."""
    toks = name.split()
    deg = 0
    for tok in toks:
        _, pw = _TOKEN.match(tok).groups()
        deg += int(pw) if pw else 1
    return len(toks), deg


def term_penalty(name, cross_margin=0.0, degree_margin=0.0):
    """Accuracy a candidate forfeits for being complicated.

        penalty = cross_margin * (n_vars - 1) + degree_margin * (total_degree - 1)

    A pure linear term pays nothing. `X^2` pays only the degree margin, `X Y` pays one
    cross margin plus one degree margin, `X^2 Y` pays one cross plus two degree.
    """
    n_vars, deg = term_complexity(name)
    return cross_margin * (n_vars - 1) + degree_margin * (deg - 1)


def forward_boundary_path(Xp, y, max_terms, names, cross_margin=0.0,
                          degree_margin=0.0, forbid_cross=False):
    """Greedy forward selection over polynomial features, scored by CV accuracy.

    Candidates are ranked by CV accuracy MINUS `term_penalty`, so complexity is charged
    against the thing being maximised rather than handled by a special case. The raw
    score is what gets recorded and printed, since that is the honest accuracy of the
    resulting equation; the penalty only decides the order terms are taken in.
    """
    chosen, remaining, steps = [], list(range(Xp.shape[1])), []
    for _ in range(min(max_terms, Xp.shape[1])):
        cand = [j for j in remaining if is_pure(names[j])] if forbid_cross else remaining
        if not cand:
            break                     # ran out of pure terms; a shorter path is the answer
        scored = {j: cv_accuracy(Xp, y, chosen + [j]) for j in cand}
        adjusted = {j: scored[j] - term_penalty(names[j], cross_margin, degree_margin)
                    for j in cand}
        pick = max(adjusted, key=lambda j: adjusted[j])

        pure = [j for j in cand if is_pure(names[j])]
        cross = [j for j in cand if not is_pure(names[j])]
        chosen.append(pick)
        remaining.remove(pick)
        steps.append({
            "col": pick, "score": scored[pick], "adjusted": adjusted[pick],
            "cross": not is_pure(names[pick]),
            "penalty": term_penalty(names[pick], cross_margin, degree_margin),
            "best_pure": max((scored[j] for j in pure), default=None),
            "best_cross": max((scored[j] for j in cross), default=None)})
    return steps


def subset_expr(pipe, names, var_names):
    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    syms = {v: sp.Symbol(v, real=True) for v in var_names}
    expr = sp.Float(float(clf.intercept_[0]))
    for name, coef, mu, sd in zip(names, clf.coef_[0], scaler.mean_, scaler.scale_):
        mono = sp.Integer(1)
        for tok in name.split():
            v, p = _TOKEN.match(tok).groups()
            mono *= syms[v] ** (int(p) if p else 1)
        expr += float(coef) * (mono - float(mu)) / float(sd)
    return sp.expand(expr)


def substitute_affine(expr, axes, syms=SYMS):
    tmp = tuple(sp.symbols(" ".join(f"_t{i}" for i in range(len(syms))), real=True))
    staged = expr.subs({s: t for s, t in zip(syms, tmp)}, simultaneous=True)
    return sp.expand(staged.subs(
        {t: ax_["a"] * s + ax_["b"] for t, s, ax_ in zip(tmp, syms, axes)}))


def rewrite_verdict(drop, flip_frac, cv_gap, tol=ACCEPTABLE_DROP):
    if drop > tol:
        return f"REJECT: loses {drop:.4f} accuracy, more than the {tol} budget"
    if flip_frac > 5 * tol:
        return (f"CAUTION: only {drop:.4f} accuracy lost, but {100 * flip_frac:.2f}% of "
                f"points change class -- the errors trade places rather than cancel")
    if cv_gap > tol:
        return (f"ACCEPT, but the affine map costs {cv_gap:.4f} against a direct "
                f"correlator-space fit -- a refit would be tighter")
    return f"ACCEPT: {drop:.4f} accuracy lost, {100 * flip_frac:.2f}% of points flipped"


# ── one arm, end to end ─────────────────────────────────────────────────────────
def analyse(run_folder: Path, split: str, label: str, quiet: bool = False) -> dict:
    import builtins, contextlib, io
    # `quiet` suppresses the per-arm trace only; every number it would have printed is
    # still computed and returned, so the cross-arm report is unaffected by the flag.
    sink = io.StringIO() if quiet else None
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    if quiet:
        builtins.print(f"[{label}] analysing {run_folder.name} ...", flush=True)
    with ctx:
        return _analyse(run_folder, split, label)


def _analyse(run_folder: Path, split: str, label: str) -> dict:
    print(f"\n{'=' * 78}\n### {label}\n    {run_folder}\n{'=' * 78}")
    D = load_run(run_folder, split)
    CF, GROUP, CHANNELS = D["cf"], D["group"], D["channels"]

    mags = np.abs(D["z"]).mean(axis=0)
    ranked = sorted(active_branches(D["z"]), key=lambda k: -mags[k])
    LATENT = ranked[:N_DIMS]
    if len(LATENT) < N_DIMS:
        raise ValueError(f"only {len(LATENT)} surviving branches in {run_folder}")

    Y_NET = D["out"].argmax(axis=1)
    Z = np.column_stack([D["z"][:, b] for b in LATENT])
    print(f"{CF.dataset}/{CF.task}  pairing_mode={getattr(CF, 'pairing_mode', 'index')} "
          f"pairing_seed={getattr(CF, 'pairing_seed', None)}  group={GROUP}")
    print(f"{len(CF.kernels)} branches, {len(ranked)} active, {len(Z)} {split} snapshots,"
          f" class balance {np.bincount(Y_NET)}")
    for k in ranked:
        mark = f"  <- slot {LATENT.index(k)}" if k in LATENT else ""
        print(f"  branch {k:>2}  shape={str(CF.kernels[k][0]):>6} "
              f"mask={str(CF.kernels[k][3]):>20}  mean|z|={mags[k]:.4f}{mark}")

    # --- 1. BFA readout, one correlator per branch ---
    FITS = {}
    for k in LATENT:
        probe = fit_activation_to_correlators(
            D["x"], D["z"][:, k], CF.kernels[k], channel_names=CHANNELS,
            group=GROUP, verbose=False)
        FITS[k] = fit_activation_to_correlators(
            D["x"], D["z"][:, k], CF.kernels[k], channel_names=CHANNELS, group=GROUP,
            select_terms=min(CORR_MAX_TERMS, len(probe["patterns"])),
            select_strategy="forward", plateau_tol=CORR_PLATEAU_TOL,
            plateau_floor=CORR_PLATEAU_FLOOR, plateau_sufficient=CORR_SUFFICIENT,
            verbose=False)

    print(f"\n{'branch':>6} {'kernel':>8} {'kept':>5} {'R2_full':>8} {'R2_kept':>8} "
          f"{'R2_lead':>8}  forward path")
    for k in LATENT:
        r = FITS[k]
        trail = "  ".join(f"{v:.4f}" for _, v in r["selection_path"])
        print(f"{k:>6} {str(CF.kernels[k][0]):>8} {len(r['selected_terms']):>5} "
              f"{r['r_squared']:>8.4f} {r['selected_r_squared']:>8.4f} "
              f"{r['selection_path'][0][1]:>8.4f}  {trail}")

    CORR_SYM = "C" if GROUP is None else r"C_{\mathrm{rot}}"
    AXES, C = [], np.empty((len(Z), len(LATENT)))
    for slot, k in enumerate(LATENT):
        res = FITS[k]
        lead_col = res["selection_path"][0][0]
        c_vals = res["correlators_with_const"][:, lead_col]
        ols1 = sm.OLS(res["outputs"], sm.add_constant(c_vals)).fit()
        b, a = float(ols1.params[0]), float(ols1.params[1])
        C[:, slot] = c_vals
        AXES.append({"slot": slot, "branch": k, "col": int(lead_col),
                     "pattern": res["patterns"][lead_col - 1], "a": a, "b": b,
                     "r2": float(ols1.rsquared), "var": VAR_NAMES[slot],
                     "glyph": branch_label(CF, k, full_latex=True).strip("$"),
                     "kept_terms": list(res["selected_patterns"]),
                     "r2_kept": float(res["selected_r_squared"])})

    print("\naffine maps  z = a*C + b:")
    for ax_ in AXES:
        print(f"  slot {ax_['slot']} ({ax_['var']})  branch {ax_['branch']}:  "
              f"z = {ax_['a']:+.4f} * C {ax_['b']:+.4f}   R^2={ax_['r2']:.4f}   "
              f"C range [{C[:, ax_['slot']].min():+.3f}, "
              f"{C[:, ax_['slot']].max():+.3f}]")

    Z_HAT = np.column_stack([ax_["a"] * C[:, ax_["slot"]] + ax_["b"] for ax_ in AXES])
    recon = []
    for ax_ in AXES:
        s = ax_["slot"]
        rms = float(np.sqrt(((Z[:, s] - Z_HAT[:, s]) ** 2).mean()))
        recon.append({"slot": s, "rms": rms, "pct_sd": 100 * rms / float(Z[:, s].std())})
        print(f"  slot {s}: RMS residual {rms:.5f} ({recon[-1]['pct_sd']:.2f}% of sd z)")

    # --- 2. boundary in latent space ---
    SWEEP = [fit_surface(Z, Y_NET, degree=deg, C_=c) for deg in DEGREES for c in CS]
    SWEEP.sort(key=lambda r: (-r["cv_mean"], r["n_terms"]))
    BEST_BY_DEGREE = {deg: max([r for r in SWEEP if r["degree"] == deg],
                               key=lambda r: r["cv_mean"]) for deg in DEGREES}
    ONESE_BY_DEGREE = {deg: parsimonious([r for r in SWEEP if r["degree"] == deg])
                       for deg in DEGREES}
    plane_s, square_s = BEST_BY_DEGREE[1], BEST_BY_DEGREE[2]
    d_cv = square_s["cv_mean"] - plane_s["cv_mean"]
    curvature_note = ("curvature justified" if d_cv > plane_s["cv_std"]
                      else "within noise -- a plane describes it just as well")
    print(f"\nLASSO sweep: best plane cv={plane_s['cv_mean']:.4f} "
          f"(sd {plane_s['cv_std']:.4f}), best square cv={square_s['cv_mean']:.4f}; "
          f"gain {d_cv:+.4f} -> {curvature_note}")

    FORWARD = {}
    for degree in DEGREES:
        Xp, names = poly_features(Z, degree, VAR_NAMES)
        steps = forward_boundary_path(Xp, Y_NET, MAX_TERMS, names,
                                      cross_margin=CROSS_TERM_MARGIN,
                                      degree_margin=DEGREE_MARGIN,
                                      forbid_cross=FORBID_CROSS)
        path = [(st["col"], st["score"]) for st in steps]
        n_keep = plateau_index(path, PLATEAU_TOL, floor=PLATEAU_FLOOR,
                               sufficient=SUFFICIENT_ACC)
        cols = [c for c, _ in path[:n_keep]]
        pipe = _subset_pipe(Xp, Y_NET, cols)
        expr = subset_expr(pipe, [names[c] for c in cols], VAR_NAMES)
        FORWARD[degree] = {"degree": degree, "cols": cols, "path": path, "steps": steps,
                           "n_keep": n_keep, "names": [names[c] for c in cols],
                           "pipe": pipe, "expr": expr,
                           "acc": float(pipe.score(Xp[:, cols], Y_NET)),
                           "cv_mean": path[n_keep - 1][1]}
        label_d = {1: "PLANE", 2: "SQUARE"}[degree]
        print(f"\n=== {label_d} (degree {degree})")
        prev = 0.0
        for i, st in enumerate(steps, 1):
            mark = "<<" if i == n_keep else "  "
            print(f"{mark} {i:>4} {st['score']:>8.4f} {st['score'] - prev:>+9.4f} "
                  f"{'cross' if st['cross'] else 'pure':>6}  {names[st['col']]}")
            prev = st["score"]
        print(f"   kept {n_keep}: {FORWARD[degree]['names']}  "
              f"acc={FORWARD[degree]['acc']:.4f} cv={FORWARD[degree]['cv_mean']:.4f}")
        print(f"   latent equation: {sp.latex(round_expr(sp.expand(expr), 3))} = 0")

    # --- 3/4. correlator rewrite + comparison ---
    CORR, DIRECT, rows = {}, {}, []
    for degree in DEGREES:
        e_lat = sp.expand(FORWARD[degree]["expr"])
        e_cor = substitute_affine(e_lat, AXES)
        CORR[degree] = {"degree": degree, "expr": e_cor,
                        "acc_latent": accuracy_of(e_lat, Z, Y_NET),
                        "acc_corr": accuracy_of(e_cor, C, Y_NET),
                        "acc_corr_via_zhat": accuracy_of(e_lat, Z_HAT, Y_NET)}
        print(f"\ncorrelator equation (degree {degree}): "
              f"{sp.latex(round_expr(e_cor, 3))} = 0")

        p_lat, p_cor = eval_boundary(e_lat, Z), eval_boundary(e_cor, C)
        drop = CORR[degree]["acc_latent"] - CORR[degree]["acc_corr"]
        flip_frac = float((p_lat != p_cor).mean())
        Cp, cnames = poly_features(C, degree, VAR_NAMES)
        all_cols = list(range(Cp.shape[1]))
        ceiling_pipe = _subset_pipe(Cp, Y_NET, all_cols)
        ceiling_acc = float(ceiling_pipe.score(Cp[:, all_cols], Y_NET))
        ceiling_cv = cv_accuracy(Cp, Y_NET, all_cols)
        cv_gap = ceiling_acc - CORR[degree]["acc_corr"]

        # The DIRECT fit: a boundary fitted in correlator space from scratch, under the
        # same cross-term penalty as the latent one. Not the same object as the
        # substituted rewrite -- it is free to place terms where the correlators, rather
        # than the activations, separate the classes -- so both are carried.
        d_steps = forward_boundary_path(Cp, Y_NET, MAX_TERMS, cnames,
                                        cross_margin=CROSS_TERM_MARGIN,
                                        degree_margin=DEGREE_MARGIN,
                                        forbid_cross=FORBID_CROSS)
        d_path = [(st["col"], st["score"]) for st in d_steps]
        d_keep = plateau_index(d_path, PLATEAU_TOL, floor=PLATEAU_FLOOR,
                               sufficient=SUFFICIENT_ACC)
        d_cols = [c for c, _ in d_path[:d_keep]]
        d_pipe = _subset_pipe(Cp, Y_NET, d_cols)
        d_expr = subset_expr(d_pipe, [cnames[c] for c in d_cols], VAR_NAMES)
        DIRECT[degree] = {
            "degree": degree, "n_keep": d_keep, "names": [cnames[c] for c in d_cols],
            "expr": d_expr, "acc": float(d_pipe.score(Cp[:, d_cols], Y_NET)),
            "cv_mean": d_path[d_keep - 1][1],
            "path": [(int(c), float(v)) for c, v in d_path],
            "steps_kind": ["cross" if st["cross"] else "pure" for st in d_steps]}
        print(f"direct correlator {name if False else degree}: kept {d_keep} "
              f"{DIRECT[degree]['names']}  acc={DIRECT[degree]['acc']:.4f} "
              f"cv={DIRECT[degree]['cv_mean']:.4f}")
        print(f"   {sp.latex(round_expr(sp.expand(d_expr), 3))} = 0")
        rows.append({"degree": degree,
                     "name": {1: "plane", 2: "square"}[degree],
                     "terms": FORWARD[degree]["n_keep"],
                     "acc_latent": CORR[degree]["acc_latent"],
                     "cv_latent": FORWARD[degree]["cv_mean"],
                     "acc_corr": CORR[degree]["acc_corr"], "drop": drop,
                     "flipped": int((p_lat != p_cor).sum()), "flip_frac": flip_frac,
                     "corr_refit_acc": ceiling_acc, "corr_refit_cv": ceiling_cv,
                     "cv_gap": cv_gap,
                     "direct_acc": DIRECT[degree]["acc"],
                     "direct_cv": DIRECT[degree]["cv_mean"],
                     "direct_terms": DIRECT[degree]["n_keep"],
                     "verdict": rewrite_verdict(drop, flip_frac, cv_gap)})

    COMPARE = pd.DataFrame(rows).set_index("name")
    print("\n" + COMPARE[["degree", "terms", "acc_latent", "cv_latent", "acc_corr",
                          "drop", "flipped", "corr_refit_acc",
                          "corr_refit_cv"]].to_string(
        float_format=lambda v: f"{v:.4f}"))
    for r in rows:
        print(f"{r['name']:>7}: {r['verdict']}")

    return {
        "label": label, "path": str(run_folder), "split": split,
        "pairing_mode": getattr(CF, "pairing_mode", "index"),
        "pairing_seed": getattr(CF, "pairing_seed", None),
        "n_samples": int(len(Z)), "class_balance": np.bincount(Y_NET).tolist(),
        "branches_active": ranked, "latent": LATENT,
        "mean_abs_z": {int(k): float(mags[k]) for k in range(len(mags))},
        "axes": [{kk: vv for kk, vv in ax_.items()} for ax_ in AXES],
        "bfa_r2_full": {int(k): float(FITS[k]["r_squared"]) for k in LATENT},
        "bfa_r2_kept": {int(k): float(FITS[k]["selected_r_squared"]) for k in LATENT},
        "bfa_terms_kept": {int(k): list(FITS[k]["selected_patterns"]) for k in LATENT},
        "recon": recon,
        "lasso": {int(d): {"best_cv": BEST_BY_DEGREE[d]["cv_mean"],
                           "best_cv_std": BEST_BY_DEGREE[d]["cv_std"],
                           "onese_terms": ONESE_BY_DEGREE[d]["n_terms"],
                           "onese_cv": ONESE_BY_DEGREE[d]["cv_mean"]}
                  for d in DEGREES},
        "curvature_note": curvature_note, "curvature_gain": float(d_cv),
        "forward": {int(d): {"n_keep": FORWARD[d]["n_keep"],
                             "names": FORWARD[d]["names"],
                             "path": [(int(c), float(s)) for c, s in FORWARD[d]["path"]],
                             "acc": FORWARD[d]["acc"],
                             "cv_mean": FORWARD[d]["cv_mean"],
                             "expr_latex":
                                 sp.latex(round_expr(sp.expand(FORWARD[d]["expr"]), 3)),
                             "expr_latex_2":
                                 sp.latex(round_expr(sp.expand(FORWARD[d]["expr"]), 2)),
                             "corr_latex": sp.latex(round_expr(CORR[d]["expr"], 3)),
                             "corr_latex_2": sp.latex(round_expr(CORR[d]["expr"], 2))}
                    for d in DEGREES},
        "compare": rows,
        # Not JSON-serialised (stripped in main): the coordinate matrices and the
        # network's class, kept so one arm's equation can be scored on another's data.
        "direct": {int(d): {"n_keep": DIRECT[d]["n_keep"], "names": DIRECT[d]["names"],
                            "acc": DIRECT[d]["acc"], "cv_mean": DIRECT[d]["cv_mean"],
                            "expr_latex":
                                sp.latex(round_expr(sp.expand(DIRECT[d]["expr"]), 3))}
                   for d in DEGREES},
        # Sign gauge of each latent axis: nothing pins the sign of a branch activation,
        # so z_k and -z_k are the same coordinate. `a` (the BFA slope onto a PHYSICAL
        # correlator) does carry a meaning, so its sign is what fixes the gauge -- see
        # `_gauge_signs`.
        "gauge": [1.0 if ax_["a"] > 0 else -1.0 for ax_ in AXES],
        "_arrays": {"Z": Z, "C": C, "Y": Y_NET,
                    "expr_lat": {int(d): sp.expand(FORWARD[d]["expr"]) for d in DEGREES},
                    "expr_cor": {int(d): CORR[d]["expr"] for d in DEGREES},
                    "expr_dir": {int(d): sp.expand(DIRECT[d]["expr"]) for d in DEGREES}},
    }


# ── cross-arm comparison ────────────────────────────────────────────────────────
def _normalised_coeffs(expr, signs=None):
    """Monomial -> coefficient, rescaled to unit L2 norm of the coefficient vector.

    A decision boundary is defined only up to POSITIVE rescaling (F=0 and aF=0 have the
    same zero set, and F>0 the same sign set), so raw coefficients are not comparable
    between fits. Dividing by the coefficient norm fixes that freedom and leaves the
    shape of the surface, which is the thing two arms either agree on or do not.

    Why the norm and not one chosen coefficient: dividing by, say, the Y coefficient is
    undefined whenever an arm's equation does not contain Y -- which happens here, and
    silently returned RAW coefficients that then polluted the ensemble mean by two orders
    of magnitude. The norm is always defined and never singular.

    The overall SIGN needs no convention: every expression here is a logistic decision
    function with F>0 -> class 1, so it is already fixed by construction.

    `signs`, when given, first applies a per-coordinate flip x_k -> s_k x_k. That is
    needed for LATENT equations only: nothing pins the sign of a branch activation, so
    two runs can describe the same surface with opposite-signed coordinates. Correlator
    coordinates are physical and must be passed `signs=None`.
    """
    expr = sp.expand(expr)
    if signs is not None:
        expr = sp.expand(expr.subs({sym: sgn * sym for sym, sgn in zip(SYMS, signs)},
                                   simultaneous=True))
    poly = sp.Poly(expr, *SYMS)
    terms = {}
    for mono, coef in zip(poly.monoms(), poly.coeffs()):
        name = "1" if not any(mono) else " ".join(
            f"{v}^{e}" if e > 1 else v for v, e in zip(VAR_NAMES, mono) if e)
        terms[name] = float(coef)
    scale = float(np.sqrt(sum(v ** 2 for v in terms.values())))
    if scale < 1e-12:
        return terms
    return {k: v / scale for k, v in terms.items()}


def _rescale_max(terms):
    """Rescale so the largest-magnitude coefficient is exactly +-1.

    The quotable form. Unit-L2 is the right gauge for AVERAGING (always defined, never
    singular), but it leaves every coefficient a fraction and the equation hard to read.
    Dividing by the largest |coefficient| instead pins the dominant term at 1, so the
    remaining numbers read directly as "fraction of the leading term". The divisor is
    positive, so predictions are untouched.
    """
    scale = max((abs(v) for v in terms.values()), default=0.0)
    if scale < 1e-12:
        return terms
    return {k: v / scale for k, v in terms.items()}


def _eq_string(terms, drop_zero=True, digits=3):
    """A normalised coefficient dict rendered as a readable `... = 0` equation."""
    order = sorted((k for k in terms if not drop_zero or abs(terms[k]) > 1e-9),
                   key=lambda m: (m == "1", len(m.split()), m))
    out = ""
    for k in order:
        v = terms[k]
        out += f" {'-' if v < 0 else '+'} {abs(v):.{digits}f}" + ("" if k == "1" else f"*{k}")
    return (out.strip().lstrip("+ ") or "0") + " = 0"


def _agg_table(results, tabs, title, anchor_label="anchor"):
    """Print per-arm normalised coefficients, then family means and the anchor's z-score."""
    monos = sorted({m for t in tabs.values() for m in t}, key=lambda m: (len(m), m))
    print(f"\n-- {title}: coefficients at unit L2 norm")
    print(f"{'arm':>8} " + " ".join(f"{m:>9}" for m in monos))
    for label in tabs:
        print(f"{label:>8} " + " ".join(f"{tabs[label].get(m, 0.0):>9.4f}" for m in monos))

    rep = [l for l in tabs if l != anchor_label]
    if len(rep) < 2:
        return
    print(f"{'':>8} " + " ".join("-" * 9 for _ in monos))
    mean = {m: float(np.mean([tabs[l].get(m, 0.0) for l in rep])) for m in monos}
    sd = {m: float(np.std([tabs[l].get(m, 0.0) for l in rep], ddof=1)) for m in monos}
    print(f"{'mean':>8} " + " ".join(f"{mean[m]:>+9.4f}" for m in monos)
          + f"   ({len(rep)} re-paired arms)")
    print(f"{'sd':>8} " + " ".join(f"{sd[m]:>9.4f}" for m in monos))
    if anchor_label in tabs:
        z = {m: (tabs[anchor_label].get(m, 0.0) - mean[m]) / sd[m] if sd[m] > 1e-12
                else float("nan") for m in monos}
        print(f"{'anchor z':>8} " + " ".join(
            ("       --" if np.isnan(z[m]) else f"{z[m]:>+9.2f}") for m in monos))
        worst = max((abs(v) for v in z.values() if not np.isnan(v)), default=float("nan"))
        print(f"   anchor's largest |z| across terms: {worst:.2f}"
              + ("   <- inside the ensemble" if worst < 2 else "   <- OUTLIER"))
    print(f"   mean equation (unit L2):  {_eq_string(mean)}")

    # Statistics in the QUOTABLE scaling. These must be recomputed from per-arm
    # max-normalised coefficients, not obtained by dividing the unit-L2 sd by a single
    # number: every arm has its own largest coefficient, so the two scalings are related
    # by a DIFFERENT factor per arm and the spread does not simply rescale.
    mx = {l: _rescale_max(tabs[l]) for l in tabs}
    mean_x = {m: float(np.mean([mx[l].get(m, 0.0) for l in rep])) for m in monos}
    sd_x = {m: float(np.std([mx[l].get(m, 0.0) for l in rep], ddof=1)) for m in monos}
    print(f"   mean equation (max = 1):  {_eq_string(mean_x)}")
    print(f"   sd (max = 1 scaling):     " + "  ".join(
        f"{m}:{sd_x[m]:.3f}" for m in monos if abs(mean_x[m]) > 1e-9 or sd_x[m] > 1e-9))
    if anchor_label in mx:
        zx = {m: (mx[anchor_label].get(m, 0.0) - mean_x[m]) / sd_x[m]
              if sd_x[m] > 1e-12 else float("nan") for m in monos}
        worst_x = max((abs(v) for v in zx.values() if not np.isnan(v)),
                      default=float("nan"))
        print(f"   anchor z (max = 1):       " + "  ".join(
            f"{m}:{zx[m]:+.2f}" for m in monos if not np.isnan(zx[m]))
            + f"   | largest |z| = {worst_x:.2f}")


def _support_census(results, key, degree, signs_key=None):
    counts = {}
    for a in results:
        names = tuple(sorted(a[key][str(degree) if str(degree) in a[key] else degree]["names"]))
        counts[names] = counts.get(names, 0) + 1
    return counts


def cross_arm_report(results):
    """Are the quoted equations the same object, or one per pairing convention?

    Reported in BOTH spaces, since they answer different questions and fail differently:

      * latent -- the boundary the network actually draws, in its own coordinates. Only
        comparable across arms after a per-axis sign gauge fix (see `_normalised_coeffs`).
      * correlator (substituted) -- the latent equation pushed through z_k -> a_k C_k + b_k.
        Same surface, physical coordinates, pays the BFA residual.
      * correlator (direct) -- a boundary FITTED in correlator space. A different surface
        the affine map cannot reach; the honest ceiling for a physically-worded equation.

    Then transfer: each arm's equation scored on every arm's data. The latent equation is
    deliberately NOT transferred -- z is network-internal, so two runs' activations share
    no common scale and the comparison would be meaningless rather than merely negative.
    """
    print(f"\n\n{'=' * 104}\n### CROSS-ARM COMPARISON ({len(results)} arms)   "
          f"cross margin = {CROSS_TERM_MARGIN}, degree margin = {DEGREE_MARGIN}, "
          f"forbid_cross = {FORBID_CROSS}\n{'=' * 104}")

    # -- 1. per-arm summary ------------------------------------------------------
    print("\n-- per-arm summary  (R2_lead = one-correlator BFA fit; kept = terms the "
          "plateau rule keeps per branch)")
    hdr = (f"{'arm':>8}{'act':>4} | {'R2_lead 0/1/2':>18}{'kept':>8} | "
           f"{'lat_pl':>8}{'sub_pl':>8}{'dir_pl':>8} | "
           f"{'lat_sq':>8}{'sub_sq':>8}{'dir_sq':>8} | {'gauge':>8}")
    print(hdr); print("-" * len(hdr))
    for a in results:
        lead = "/".join(f"{ax['r2']:.3f}" for ax in a["axes"])
        kept = "/".join(str(len(ax["kept_terms"])) for ax in a["axes"])
        pl = next(r for r in a["compare"] if r["degree"] == 1)
        sq = next(r for r in a["compare"] if r["degree"] == 2)
        g = "".join("+" if x > 0 else "-" for x in a["gauge"])
        print(f"{a['label']:>8}{len(a['branches_active']):>4} | {lead:>18}{kept:>8} | "
              f"{pl['acc_latent']:>8.4f}{pl['acc_corr']:>8.4f}{pl['direct_acc']:>8.4f} | "
              f"{sq['acc_latent']:>8.4f}{sq['acc_corr']:>8.4f}{sq['direct_acc']:>8.4f} | "
              f"{g:>8}")

    for fam in ("repair_fixed", "repair_resample"):
        rows = [a for a in results if FAMILY[a["label"]] == fam]
        for deg, tag in ((1, "plane"), (2, "square")):
            g = lambda k: np.array([next(r for r in a["compare"] if r["degree"] == deg)[k]
                                    for a in rows])
            print(f"   {fam:>16} {tag:>6}  latent {g('acc_latent').mean():.4f}"
                  f"+-{g('acc_latent').std(ddof=1):.4f}   "
                  f"substituted {g('acc_corr').mean():.4f}"
                  f"+-{g('acc_corr').std(ddof=1):.4f}   "
                  f"direct {g('direct_acc').mean():.4f}"
                  f"+-{g('direct_acc').std(ddof=1):.4f}   "
                  f"drop {g('drop').mean():.4f}+-{g('drop').std(ddof=1):.4f}")

    print("\n-- leading correlator per axis (the figure's three coordinates)")
    for slot in range(N_DIMS):
        pats = {a["axes"][slot]["pattern"] for a in results}
        brs = {a["axes"][slot]["branch"] for a in results}
        print(f"   slot {slot}: {len(pats)} distinct pattern(s) across {len(results)} "
              f"arms, branch(es) {sorted(brs)}"
              + ("   <- IDENTICAL" if len(pats) == 1 and len(brs) == 1 else "   <- DIFFERS"))

    # -- 2. term support -------------------------------------------------------
    print("\n-- term support (how often each equation shape is selected)")
    for space, key in (("latent", "forward"), ("correlator direct", "direct")):
        for degree in DEGREES:
            counts = {}
            for a in results:
                d = a[key]
                names = tuple(sorted(d[degree]["names"] if degree in d
                                     else d[str(degree)]["names"]))
                counts[names] = counts.get(names, 0) + 1
            tag = {1: "plane", 2: "square"}[degree]
            top = sorted(counts.items(), key=lambda kv: -kv[1])
            print(f"   {space:>18} {tag:>6}: " + ";  ".join(
                f"{n}/{len(results)} {list(k)}" for k, n in top))

    # -- 3. gauge-fixed coefficients, all three equation families ---------------
    for degree in DEGREES:
        tag = {1: "plane", 2: "square"}[degree]
        _agg_table(results,
                   {a["label"]: _normalised_coeffs(a["_arrays"]["expr_lat"][degree],
                                                   signs=a["gauge"]) for a in results},
                   f"LATENT {tag} (sign-gauge-fixed: z_k -> sign(a_k) z_k)")
        _agg_table(results,
                   {a["label"]: _normalised_coeffs(a["_arrays"]["expr_cor"][degree])
                    for a in results},
                   f"CORRELATOR {tag}, substituted")
        _agg_table(results,
                   {a["label"]: _normalised_coeffs(a["_arrays"]["expr_dir"][degree])
                    for a in results},
                   f"CORRELATOR {tag}, fitted directly")

    # -- 4. per-arm quotable equations, written out ----------------------------
    print("\n-- quotable equations, arm by arm (largest coefficient scaled to 1)")
    for degree in DEGREES:
        tag = {1: "plane", 2: "square"}[degree]
        for space, key, sgn in (("latent", "expr_lat", True),
                                ("substituted", "expr_cor", False),
                                ("direct", "expr_dir", False)):
            print(f"\n   [{space} {tag}]")
            for a in results:
                t = _normalised_coeffs(a["_arrays"][key][degree],
                                       signs=a["gauge"] if sgn else None)
                print(f"   {a['label']:>8}:  {_eq_string(_rescale_max(t))}")

    # -- 5. transfer -----------------------------------------------------------
    labels = [a["label"] for a in results]
    for key, what in (("expr_cor", "substituted"), ("expr_dir", "fitted directly")):
        for degree in DEGREES:
            tag = {1: "plane", 2: "square"}[degree]
            M = np.array([[accuracy_of(a["_arrays"][key][degree],
                                       b["_arrays"]["C"], b["_arrays"]["Y"])
                           for b in results] for a in results])
            print(f"\n-- transfer of the correlator {tag} ({what}): rows = equation's "
                  f"arm, cols = data's arm")
            print("eq / data".rjust(10) + " " + " ".join(f"{l:>7}" for l in labels))
            for i, l in enumerate(labels):
                print(f"{l:>10} " + " ".join(f"{M[i, j]:>7.4f}" for j in range(len(labels))))
            diag = np.diag(M)
            off = M[~np.eye(len(M), dtype=bool)]
            print(f"   diagonal (own equation on own data): {diag.mean():.4f} "
                  f"+- {diag.std(ddof=1):.4f}, min {diag.min():.4f}")
            print(f"   off-diagonal (transferred):          {off.mean():.4f} "
                  f"+- {off.std(ddof=1):.4f}, min {off.min():.4f}")
            cost = np.array([M[i, j] - M[j, j]
                             for j in range(len(M)) for i in range(len(M)) if i != j])
            print(f"   per-dataset transfer cost (foreign - native, same data): "
                  f"mean {cost.mean():+.4f}, worst {cost.min():+.4f}")
            ar = M[labels.index("anchor")]
            print(f"   ANCHOR's equation on the re-paired runs: {ar[1:].mean():.4f} "
                  f"+- {ar[1:].std(ddof=1):.4f}, vs their own {diag[1:].mean():.4f}; "
                  f"cost {(ar - diag)[1:].mean():+.4f} "
                  f"(worst {(ar - diag)[1:].min():+.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--json", default=None, help="write the full result dict here")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--quiet", action="store_true",
                    help="print only the cross-arm report, not each arm's full trace")
    args = ap.parse_args()

    out = []
    for label in args.arms:
        folder = ARMS[label]
        if not (folder / "net1.pt").exists():
            print(f"[missing] {label}: {folder} -- train it first")
            continue
        out.append(analyse(folder, args.split, label, quiet=args.quiet))

    if len(out) > 1:
        cross_arm_report(out)

    if args.json:
        slim = [{k: v for k, v in a.items() if k != "_arrays"} for a in out]
        Path(args.json).write_text(json.dumps(slim, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
