"""Correlator regression: the primary interpretability route for TetrisCNN.

Each surviving bottleneck branch activation ``z_k`` is, by the Boolean-Fourier
argument (see the manuscript's main text and Appendix B, "Boolean function theory"),
a linear combination of the spin correlators ``C[P']`` over the sub-patterns ``P' ``
of that branch's filter footprint. This module makes that statement *operational*: it
enumerates those sub-patterns, evaluates their correlators on real snapshots, and fits
an ordinary-least-squares regression of ``z_k`` onto them, so a branch activation is
read back as a closed-form physical formula, e.g. for the 2D Ising model a
single-site branch gives ``z[■] ≈ 6.41·C^Z[■] − 2.96`` (the Z-magnetization).

This is the *primary* interpretability route and is deliberately independent of
symbolic regression: importing this module does **not** pull in PySR/Julia (unlike
``tetriscnn.symbolic_regression``), so a branch can be interpreted with nothing but
numpy and statsmodels. PySR (``sr_toolbox.py``) is a secondary, fragile route that
tries to also express ``z_k`` in a richer operator basis; see
``docs/SYMBOLIC_REGRESSION.md``.

**Multibasis fits.** For a multi-channel dataset such as ``Paris_XY_XZ`` (channel 0
= X basis, channel 1 = Z basis), a branch activation can depend not only on the
per-basis correlators but on *cross-basis* products such as
``C^X[■]·C^Z[■]``. :func:`fit_activation_to_correlators` builds both, deduplicating
cross-basis terms up to the requested spatial symmetries, so the recovered formula
uses the full experimentally-measurable correlator dictionary.

The design that produced the manuscript's Fig. 5 equations lives here; it was
developed in the original (pre-release) Figure 5 notebook and ported into the library so it is reusable and
tested.
"""

from itertools import combinations, permutations
from math import comb
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import statsmodels.api as sm

from tetriscnn.pattern_generation import (
    generate_patterns,
    pattern_correlator,
    pattern_to_mask,
)
from tetriscnn.utils import mask_to_latex_pattern


# ---------------------------------------------------------------------------
# Cross-basis term construction
# ---------------------------------------------------------------------------

def _cross_channel_subsets_for_pattern(
    patt: List[Tuple[int, int]],
    n_channels: int,
    require_all_spatial_sites: bool = True,
    min_channels_in_term: int = 2,
    min_order: int = 2,
    max_order: Optional[int] = None,
) -> Iterable[List[Tuple[int, int, int]]]:
    """Yield cross-basis correlator terms drawn from one spatial pattern.

    Given a spatial pattern ``patt`` (a list of ``(row, col)`` sites) and a number
    of measurement channels, enumerate subsets of the ``site x channel`` occupancy
    grid that mix at least ``min_channels_in_term`` channels. Each yielded term is a
    list of ``(row, col, channel)`` triples defining a cross-basis product
    correlator, e.g. ``C^X[site0]·C^Z[site1]``.

    require_all_spatial_sites=True keeps only terms that touch every spatial site of
    the pattern (so the term has the same footprint as the branch), which is the
    setting used for the manuscript figures.
    """
    m = len(patt)
    universe = [(site_idx, ch) for site_idx in range(m) for ch in range(n_channels)]
    if max_order is None:
        max_order = len(universe)

    for order in range(max(min_order, 2), max_order + 1):
        for subset in combinations(universe, order):
            used_channels = {ch for _, ch in subset}
            if len(used_channels) < min_channels_in_term:
                continue

            used_sites = {site_idx for site_idx, _ in subset}
            if require_all_spatial_sites and len(used_sites) < m:
                continue

            yield [
                (int(patt[site_idx][0]), int(patt[site_idx][1]), int(ch))
                for site_idx, ch in subset
            ]


def _cross_pattern_label_two_row(
    pattern3d: Tuple[Tuple[int, int, int], ...],
    label_shape: Tuple[int, int],
    n_channels: int,
    channel_names: List[str],
    equation_env: bool = True,
) -> str:
    """Render a cross-basis term as a stacked, per-channel LaTeX pattern label."""
    rows = []
    for ch in range(n_channels):
        patt_ch = [(r, c) for (r, c, b) in pattern3d if b == ch]
        if len(patt_ch) == 0:
            mask = np.zeros((label_shape[0], label_shape[1]), dtype=int)
        else:
            mask = pattern_to_mask(patt_ch, width=label_shape[1], height=label_shape[0])
        patt_label = mask_to_latex_pattern(mask, equation_env=False, full_latex=True)
        ch_name = channel_names[ch] if ch < len(channel_names) else f"ch{ch}"
        rows.append(f"{ch_name}:{patt_label}")

    if equation_env:
        return "$\\substack{" + r" \\ ".join(rows) + "}$"
    return "\n".join(rows)


def _canonicalize_cross_pattern_3d(
    pattern3d: Iterable[Tuple[int, int, int]],
    bounds: Tuple[int, int, int],
    use_rotations: bool = True,
    use_reflections: bool = True,
    use_translations: bool = True,
    include_basis_axis: bool = False,
    include_only_valid: bool = True,
) -> frozenset:
    """Return a canonical, hashable key for a 3D (row, col, channel) cross-pattern.

    Used to deduplicate cross-basis terms that are equivalent under the requested
    symmetries:

    - spatial rotations/reflections are optional (disable them to keep mirrored
      cross-terms distinct, as the manuscript workflow does),
    - translations are normalized by shifting occupied coordinates to start at 0,
    - basis-axis symmetry (treating channel labels as interchangeable) is optional,
    - transformed variants leaving ``bounds`` are discarded when
      ``include_only_valid=True``.

    The returned frozenset is the minimal variant, so equivalent patterns collapse
    to the same dictionary key.
    """
    H, W, D = bounds
    coords = tuple((int(r), int(c), int(b)) for (r, c, b) in pattern3d)

    if len(coords) == 0:
        return frozenset()

    def _in_bounds(pts: Tuple[Tuple[int, int, int], ...]) -> bool:
        return all(0 <= r < H and 0 <= c < W and 0 <= b < D for (r, c, b) in pts)

    if include_only_valid and not _in_bounds(coords):
        raise ValueError("pattern3d contains entries outside bounds")

    if include_basis_axis and D > 1:
        basis_maps = list(permutations(range(D)))
    else:
        basis_maps = [tuple(range(D))]

    rot_angles = [0, 90, 180, 270] if use_rotations else [0]
    reflect_options = [False, True] if use_reflections else [False]
    variants = []

    for basis_map in basis_maps:
        remapped = tuple((r, c, basis_map[b]) for (r, c, b) in coords)

        for angle in rot_angles:
            if angle == 0:
                rotate = lambda rr, cc: (rr, cc)
                rotated_w = W
            elif angle == 90:
                rotate = lambda rr, cc: (cc, (H - 1) - rr)
                rotated_w = H
            elif angle == 180:
                rotate = lambda rr, cc: ((H - 1) - rr, (W - 1) - cc)
                rotated_w = W
            elif angle == 270:
                rotate = lambda rr, cc: ((W - 1) - cc, rr)
                rotated_w = H
            else:
                raise ValueError(f"Unsupported rotation angle: {angle}")

            for do_reflect in reflect_options:
                transformed = []
                for (rr, cc, bb) in remapped:
                    r2, c2 = rotate(rr, cc)
                    if do_reflect:
                        c2 = (rotated_w - 1) - c2
                    transformed.append((int(r2), int(c2), int(bb)))

                if use_translations:
                    min_r = min(r for (r, _, _) in transformed)
                    min_c = min(c for (_, c, _) in transformed)
                    if include_basis_axis:
                        min_b = min(b for (_, _, b) in transformed)
                        normalized = tuple(
                            sorted((r - min_r, c - min_c, b - min_b) for (r, c, b) in transformed)
                        )
                    else:
                        normalized = tuple(
                            sorted((r - min_r, c - min_c, b) for (r, c, b) in transformed)
                        )
                else:
                    normalized = tuple(sorted(transformed))

                if include_only_valid and not _in_bounds(normalized):
                    continue

                variants.append(normalized)

    if not variants:
        if use_translations:
            min_r = min(r for (r, _, _) in coords)
            min_c = min(c for (_, c, _) in coords)
            if include_basis_axis:
                min_b = min(b for (_, _, b) in coords)
                fallback = tuple(sorted((r - min_r, c - min_c, b - min_b) for (r, c, b) in coords))
            else:
                fallback = tuple(sorted((r - min_r, c - min_c, b) for (r, c, b) in coords))
        else:
            fallback = tuple(sorted(coords))
        return frozenset(fallback)

    return frozenset(min(variants))



# ---------------------------------------------------------------------------
# Group symmetrization (equivariant branches)
# ---------------------------------------------------------------------------

def orbit_site_maps(
    kernel_shape: Tuple[int, int],
    group: str = "C4",
) -> List[Tuple[Dict[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]]:
    """Site relabelling and valid-window shape for each element of a symmetry group.

    Returns one ``(site_map, window_shape)`` per group element, where
    ``site_map[(r, c)]`` is where footprint site ``(r, c)`` lands under that element
    and ``window_shape`` is the shape of the transformed footprint -- which fixes the
    valid-position grid the correlator is averaged over. For a non-square footprint
    such as ``(2, 1)`` the C4 orbit alternates between ``(2, 1)`` and ``(1, 2)``, i.e.
    between a ``(H-1) x W`` and an ``H x (W-1)`` grid of valid positions; getting that
    right is exactly what an equivariant branch's exact correlator reading needs.

    The transforms mirror ``models.ConvBranch_Equivariant._group_transformed_weights``
    (``np.rot90`` for C4, the Klein-four flips for D2/K4) and are derived by
    transporting an index grid rather than by hand-written coordinate algebra, so they
    cannot drift out of sync with the model.
    """
    kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
    ids = np.arange(kh * kw).reshape(kh, kw)

    if group == "C4":
        transformed = [np.rot90(ids, k=r) for r in range(4)]
    elif group in ("D2", "K4"):
        transformed = [ids, ids[::-1, :], ids[:, ::-1], ids[::-1, ::-1]]
    else:
        raise ValueError(
            f"Unknown group {group!r}: expected 'C4', 'D2' or 'K4' "
            "('D2' and 'K4' are aliases for the rectangle/Klein-four group)."
        )

    maps = []
    for t in transformed:
        pos = {int(v): (int(i), int(j)) for (i, j), v in np.ndenumerate(t)}
        site_map = {(r, c): pos[r * kw + c] for r in range(kh) for c in range(kw)}
        maps.append((site_map, (int(t.shape[0]), int(t.shape[1]))))
    return maps


def orbit_term_labels(
    term3d: Iterable[Tuple[int, int, int]],
    kernel_shape: Tuple[int, int],
    group: str,
    n_channels: int,
    channel_names: List[str],
) -> List[str]:
    """LaTeX glyphs for every DISTINCT member of a term's group orbit.

    An equivariant branch is linear in the orbit average, not in the canonical
    correlator, and for a non-square footprint the orbit members do not share a
    shape -- the C4 orbit of a 2x1 domino term also contains 1x2 terms. Showing only
    the canonical glyph therefore under-describes what was fitted: a reader cannot
    tell that the horizontal and vertical dominoes were collapsed together. This
    returns the whole orbit so a label can show it.

    Each member is rendered at ITS OWN window shape (that is what makes the shape
    change visible). Duplicates are removed while preserving orbit order, so a term
    invariant under part of the group -- e.g. a fully-occupied domino, unchanged by
    the 180 degree rotation -- collapses to the two distinct shapes rather than
    repeating each twice.

    Returns:
        Distinct glyph strings (no ``$``), canonical member first.
    """
    labels: List[str] = []
    for site_map, wshape in orbit_site_maps(kernel_shape, group):
        moved = [(*site_map[(int(r), int(c))], int(b)) for (r, c, b) in term3d]
        label = _cross_pattern_label_two_row(
            pattern3d=tuple(moved),
            label_shape=wshape,
            n_channels=n_channels,
            channel_names=channel_names,
            equation_env=False,
        )
        # _cross_pattern_label_two_row returns newline-joined rows when
        # equation_env=False; re-join them the way the $...$ form does.
        label = "\\substack{" + r" \\ ".join(label.split("\n")) + "}"
        if label not in labels:
            labels.append(label)
    return labels


def _correlator_from_term(
    in_snapshots: np.ndarray,
    term3d: Iterable[Tuple[int, int, int]],
    window_shape: Tuple[int, int],
) -> np.ndarray:
    """Correlator of one ``(row, col, channel)`` monomial over its valid-position grid.

    Equivalent to ``pattern_correlator`` for a single-channel term, but takes the
    channel per site so single-basis and cross-basis terms go through one code path.
    """
    nrows, ncols = in_snapshots.shape[-2], in_snapshots.shape[-1]
    h_prime = nrows - window_shape[0] + 1
    w_prime = ncols - window_shape[1] + 1
    if h_prime <= 0 or w_prime <= 0:
        raise ValueError(f"window {window_shape} does not fit in a {nrows}x{ncols} lattice")
    slices = [
        in_snapshots[:, b, r:r + h_prime, c:c + w_prime]
        for (r, c, b) in term3d
    ]
    return np.prod(np.stack(slices, axis=0), axis=0).mean(axis=(-2, -1))


def _symmetrized_correlator(
    in_snapshots: np.ndarray,
    term3d: Iterable[Tuple[int, int, int]],
    orbit: List[Tuple[Dict[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]],
) -> np.ndarray:
    """Group-symmetrized correlator ``(1/|G|) sum_g C_{g.F}[g.term]``.

    This is the feature an equivariant branch is actually linear in: the branch averages
    its per-group-element pooled scalars, and each element pools over its own transformed
    footprint and valid-position grid.
    """
    term3d = list(term3d)
    vals = [
        _correlator_from_term(
            in_snapshots,
            [(*site_map[(r, c)], b) for (r, c, b) in term3d],
            window_shape,
        )
        for site_map, window_shape in orbit
    ]
    return np.mean(vals, axis=0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Guard against an unusable exhaustive search. C(n, k) refits at n=25, k=4 is ~13k
# (fine); the blow-up only starts to bite for footprints far larger than anything
# set_kernels() produces today, so this is a tripwire, not a real limit.
_BEST_SUBSET_MAX_COMBOS = 500_000


def best_subset_terms(
    X_with_const: np.ndarray,
    y: np.ndarray,
    n_terms: int,
) -> Tuple[List[int], float, np.ndarray]:
    """Exhaustive best-subset selection over the correlator terms.

    Finds the ``n_terms`` columns of ``X_with_const`` (excluding the intercept in
    column 0) whose unpenalized OLS refit maximises R^2, i.e. the honest answer to
    "which n correlators alone explain this activation best".

    Why exhaustive rather than an L1 path: at these footprint sizes it is both
    exact AND cheaper. Measured on a 15-term XY branch, best-subset takes 0.15 s
    for k=3 (455 refits) against 12.4 s for a 400-point lasso path.

    Args:
        X_with_const: Design matrix with the intercept in column 0 (the
            ``correlators_with_const`` returned by
            :func:`fit_activation_to_correlators`).
        y: Target activation, shape ``(B,)``.
        n_terms: How many non-intercept terms to keep.

    Returns:
        ``(support, r_squared, coefficients)`` where ``support`` lists the chosen
        column indices into ``X_with_const`` (all >= 1, ascending), ``r_squared`` is
        the refit R^2, and ``coefficients`` is the refit parameter vector with the
        intercept first, aligned with ``[0] + support``.
    """
    n_avail = X_with_const.shape[1] - 1
    if n_terms < 1:
        raise ValueError(f"select_terms must be >= 1, got {n_terms}.")
    if n_terms > n_avail:
        raise ValueError(
            f"select_terms={n_terms} exceeds the {n_avail} correlator terms this "
            "branch footprint provides."
        )

    n_combos = comb(n_avail, n_terms)
    if n_combos > _BEST_SUBSET_MAX_COMBOS:
        raise ValueError(
            f"Exhaustive selection of {n_terms} from {n_avail} terms needs "
            f"{n_combos} OLS refits, over the {_BEST_SUBSET_MAX_COMBOS} guard. "
            "Reduce select_terms or narrow the term enumeration."
        )

    best_r2, best_support, best_params = -np.inf, None, None
    for support in combinations(range(1, n_avail + 1), n_terms):
        cols = (0,) + support
        fit = sm.OLS(y, X_with_const[:, cols]).fit()
        if fit.rsquared > best_r2:
            best_r2, best_support, best_params = fit.rsquared, list(support), fit.params
    return best_support, float(best_r2), best_params


def forward_selection_path(
    X_with_const: np.ndarray,
    y: np.ndarray,
    max_terms: int,
) -> List[Tuple[int, float]]:
    """Greedy forward selection: build a nested importance hierarchy of terms.

    Repeatedly adds whichever remaining correlator raises R^2 the most, giving an
    ORDERED path ``[(col, r2_with_that_many_terms), ...]``. Unlike
    :func:`best_subset_terms`, the supports here are nested, which is what makes an
    "importance ranking" and a stopping rule meaningful at all -- best-subset's
    answers at k and k+1 need not share a single term.

    The trade-off is that greedy is not the argmax at every size: a term can look
    weak alone yet be strong in company. That is not hypothetical here. On the plain
    XY (2,1) branch the path climbs 0.24 -> 0.36 -> 0.45 and then JUMPS to 0.99 on
    the fourth term, because those correlators are only jointly informative. Any
    stopping rule of the form "stop at the first small gain" truncates that at one
    term and reports R^2=0.24; :func:`plateau_index` therefore looks at the whole
    path instead. Compare against ``best_subset_terms`` at the chosen size when the
    distinction matters.

    Cost is ``max_terms * n_avail`` refits (75 for a 15-term branch at max_terms=5),
    i.e. cheaper than the exhaustive search, so no combinatorial guard is needed.

    Args:
        X_with_const: Design matrix with the intercept in column 0.
        y: Target activation, shape ``(B,)``.
        max_terms: How far to extend the path.

    Returns:
        ``[(column_index, r_squared), ...]`` in the order terms were added, where
        ``r_squared`` is for the model holding that term and all earlier ones.
    """
    n_avail = X_with_const.shape[1] - 1
    if max_terms < 1:
        raise ValueError(f"max_terms must be >= 1, got {max_terms}.")
    if max_terms > n_avail:
        raise ValueError(
            f"max_terms={max_terms} exceeds the {n_avail} correlator terms this "
            "branch footprint provides."
        )

    chosen: List[int] = []
    path: List[Tuple[int, float]] = []
    for _ in range(max_terms):
        best_r2, best_col = -np.inf, None
        for j in range(1, n_avail + 1):
            if j in chosen:
                continue
            r2 = sm.OLS(y, X_with_const[:, [0] + chosen + [j]]).fit().rsquared
            if r2 > best_r2:
                best_r2, best_col = r2, j
        chosen.append(best_col)
        path.append((int(best_col), float(best_r2)))
    return path


def plateau_index(
    path: List[Tuple[int, float]],
    tol: float = 0.01,
    floor: float = 0.95,
    sufficient: Optional[float] = None,
) -> int:
    """How many terms of a forward path to keep: the shortest *adequate* prefix.

    Keeps the smallest ``k`` satisfying BOTH conditions:

    * **plateau** -- its R^2 is within ``tol`` of the best R^2 anywhere on the path,
      so nothing worthwhile is being given up by stopping there;
    * **floor** -- its R^2 is at least ``floor``, so a branch is never summarised by
      an equation that is merely *stably* bad.

    i.e. ``target = max(best_on_path - tol, floor)`` and the first ``k`` reaching it
    wins. Both halves are load-bearing. Without the floor, a branch whose path tops
    out at 0.45 would be "summarised" by its first term at 0.42 -- plateaued, and
    useless. Without the plateau term, a branch that hits the floor early would keep
    collecting terms that buy nothing.

    Deliberately NOT "stop at the first gain below tol": that local rule breaks when
    terms are only jointly informative (see :func:`forward_selection_path` for the
    branch where it would report 0.24 in place of 0.99). Scanning the whole path
    costs nothing, since it is already computed.

    If the target is unreachable within the path, the full path length is returned;
    callers should compare the resulting R^2 against ``floor`` and say so rather
    than quoting the equation as if it were adequate.

    ``sufficient`` -- an OPT-IN absolute "good enough" bar (default None = off, so
    existing callers are unaffected). When set, the shortest prefix reaching it wins
    outright and the plateau comparison is skipped. This is a different question from
    the one ``tol``/``floor`` answer, and the difference matters once a path is
    strong throughout: the plateau rule is *relative to the best on the path*, so on
    a path running 0.941, 0.967, 0.986, 0.993, 0.996, 0.998 it demands 6 terms in
    order to come within ``tol`` of 0.998, even though 4 terms already reach 0.993.
    If any prefix clearing 0.99 is acceptable, say so with ``sufficient=0.99`` and
    get the 4-term equation. ``floor`` cannot express this: it is a lower bound that
    the ``max`` makes *stricter*, never a stopping condition.

    The two combine as: take the shortest prefix that is good enough if one exists,
    otherwise fall back to the shortest prefix that is as good as the path gets.
    """
    if not path:
        raise ValueError("empty selection path")
    r2s = [r for _, r in path]

    if sufficient is not None:
        for i, r in enumerate(r2s):
            if r >= sufficient:
                return i + 1

    target = max(max(r2s) - tol, floor)
    for i, r in enumerate(r2s):
        if r >= target:
            return i + 1
    return len(r2s)


def fit_activation_to_correlators(
    in_snapshots: np.ndarray,
    z_selected: np.ndarray,
    branch_to_fit: List,
    channel_names: Optional[List[str]] = None,
    include_cross_terms: bool = True,
    include_basis_axis_symmetry: bool = False,
    include_rotations: bool = False,
    include_reflections: bool = False,
    include_translations: bool = False,
    require_all_spatial_sites: bool = True,
    cross_max_order: Optional[int] = None,
    snapshot_averaged: Optional[Any] = None,
    group: Optional[str] = None,
    select_terms: Optional[int] = None,
    select_strategy: str = "best_subset",
    plateau_tol: Optional[float] = None,
    plateau_floor: float = 0.95,
    plateau_sufficient: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Regress a branch activation onto the correlators of its own filter footprint.

    Fits ``z_selected ≈ intercept + Σ_j β_j · C_j`` where the ``C_j`` are the
    single- and (optionally) cross-basis spin correlators of every sub-pattern of
    ``branch_to_fit``'s filter footprint. The R² of this fit measures how completely
    the branch has learned a closed-form function of measurable correlators; the
    coefficients ``β_j`` are the physical readout.

    This is the primary, non-symbolic interpretability route. It imports no
    torch/PySR machinery: pass it snapshots and activations you have already
    extracted from a trained network (push the data through ``net1`` once to get
    ``z``; select one branch's column as ``z_selected``).

    Args:
        in_snapshots: Spin snapshots, shape ``(B, C, H, W)`` (or ``(B, H, W)`` for a
            single channel). Values are ``±1``. ``C`` is the number of measurement
            bases (e.g. 2 for ``Paris_XY_XZ``: X then Z).
        z_selected: The chosen branch's activation, shape ``(B,)``.
        branch_to_fit: The branch's kernel spec ``[shape, n_filters, dilation, mask,
            stride]`` (an entry of ``cf.kernels``). ``shape`` is the footprint and
            ``mask`` (if not None) marks inactive sites; non-square footprints are
            handled by padding to a masked square.
        channel_names: Display names per channel for the LaTeX labels (e.g.
            ``["X", "Z"]``). Defaults to ``["ch0", "ch1", ...]``.
        include_cross_terms: Include cross-basis product correlators (only has an
            effect when ``C > 1``).
        include_basis_axis_symmetry: Treat channel labels as interchangeable when
            deduplicating cross terms.
        include_rotations, include_reflections, include_translations: Spatial
            symmetries used to deduplicate terms. All default to False, which keeps
            mirrored/translated cross-terms distinct (the manuscript setting: the
            window is fixed per branch, so translation dedup is unwanted).
        require_all_spatial_sites: Keep only cross terms spanning every spatial site
            of the pattern (same footprint as the branch).
        cross_max_order: Cap on the order (number of factors) of cross terms; None
            means no cap.
        group: Symmetry group to symmetrize the correlators over, for interpreting an
            **equivariant** branch (``"C4"``, or ``"D2"``/``"K4"``). None (default) is
            the plain reading, correct for ``ConvBranch_twoLayer``. An equivariant
            branch averages its pooled scalar over the group, so it is linear in the
            orbit-averaged correlators ``(1/|G|) sum_g C_{g.F}[g.P]`` rather than in
            ``C[P]``; passing the branch's group here builds those features and
            recovers R^2 = 1. Pass the group the model was trained with
            (``cf.equivariant_group``), and leave it None for a non-equivariant model.
        select_terms: If set, additionally search for a short list of correlators
            that alone explain the activation, returned under the ``selected_*``
            keys. Its meaning depends on ``select_strategy``: an EXACT size for
            ``"best_subset"``, a maximum BUDGET for ``"forward"``. The full fit is
            unaffected either way -- this is purely additive, so ``coefficients`` /
            ``r_squared`` mean exactly what they did before. None (default) skips
            the search entirely and the ``selected_*`` keys are None.
        select_strategy: ``"best_subset"`` (default) runs the exhaustive search for
            the single best set of exactly ``select_terms`` terms.  ``"forward"``
            runs greedy forward selection instead, producing a NESTED importance
            hierarchy (see :func:`forward_selection_path`) plus, if ``plateau_tol``
            is set, an automatically chosen length. Use ``"forward"`` when the
            question is "how few terms does this branch really need, and in what
            order of importance"; use ``"best_subset"`` when the size is fixed in
            advance and you want the provably best set of that size.
        plateau_tol: ``"forward"`` only. When set, truncate the path at
            :func:`plateau_index`, keeping the shortest prefix within ``plateau_tol``
            R² of the path's best AND at least ``plateau_floor``. None keeps the
            whole budget.
        plateau_sufficient: ``"forward"`` only, optional. An absolute R² that counts
            as good enough: the shortest prefix reaching it is kept and the plateau
            comparison is skipped. Use it when any adequate equation will do and the
            simplest one is preferred. See :func:`plateau_index`.
        plateau_floor: ``"forward"`` only, default 0.95. The minimum R² a truncated
            equation must reach; stops a branch being summarised by a short form
            that is stable but inadequate. If the budget is exhausted below it, the
            full path is kept and (under ``verbose``) a warning is printed --
            check ``selected_r_squared`` against this before quoting the equation.
        snapshot_averaged: Optional ``PhaseDataset``; when given, both correlators
            and the target are first averaged per unique label (via
            ``snapshot_average``) so the fit is on label-averaged quantities rather
            than per-snapshot. None fits per snapshot.
        verbose: Print an ``X`` shape / term count / R² one-liner.

    Returns:
        dict with:
            - ``correlators_with_const``: design matrix ``X`` with intercept column.
            - ``outputs``: the target ``y`` actually fitted (label-averaged if
              ``snapshot_averaged`` was given).
            - ``coefficients``: OLS coefficients (``params``); index 0 is the
              intercept, aligned with ``correlators_with_const`` columns.
            - ``r_squared``: OLS R².
            - ``patterns``: LaTeX label per non-intercept term, aligned with the
              coefficient tail ``coefficients[1:]``.
            - ``ols_object``: the fitted statsmodels OLS results object.
            - ``selected_terms``: column indices into ``correlators_with_const``
              of the term selection (None unless ``select_terms`` was set), plus
              ``selection_path`` / ``selection_path_patterns`` giving the greedy
              ``(column, R²)`` hierarchy (``"forward"`` only, else None).
            - ``selected_patterns``: LaTeX labels for those terms.
            - ``selected_coefficients``: refit coefficients, intercept first.
            - ``selected_r_squared``: refit R^2 of the selected subset alone.
            - ``ols_reg_object``: ``fit_regularized()`` with statsmodels' default
              ``alpha=0.0`` -- i.e. an UNPENALIZED coordinate-descent fit, not a
              sparse one. It selects nothing; do not read it as a sparse readout.
    """
    ker = branch_to_fit
    ker_shape = tuple(ker[0])
    ker_mask_array = ker[3]

    # Use the true branch shape for label rendering so 2x1/1x2 kernels are not
    # drawn as padded 2x2 squares.
    label_shape = ker_shape
    window_shape = label_shape

    # For an equivariant branch every correlator is replaced by its orbit average over
    # the branch's group; `orbit` is None for the ordinary (non-equivariant) reading.
    orbit = orbit_site_maps(window_shape, group) if group is not None else None

    # Sites switched off by the branch's own mask.
    if ker_mask_array is not None:
        ker_mask_nzlist = np.int8(np.invert(np.bool_(ker_mask_array))).nonzero()
        ker_mask_coords = np.array(list(zip(ker_mask_nzlist[0], ker_mask_nzlist[1])))
        ker_mask_set = {tuple(x) for x in ker_mask_coords}
    else:
        ker_mask_set = set()

    # A non-square footprint is embedded in the smallest square and the extra sites
    # are masked out, so pattern enumeration can work on a square grid.
    if ker_shape[0] != ker_shape[1]:
        max_dim = max(ker_shape)
        square_shape = (max_dim, max_dim)
        all_sites = {(i, j) for i in range(square_shape[0]) for j in range(square_shape[1])}
        all_sites_nonsquare = {(i, j) for i in range(ker_shape[0]) for j in range(ker_shape[1])}
        all_sites_nonsquare_mask = all_sites - all_sites_nonsquare
    else:
        square_shape = ker_shape
        all_sites_nonsquare_mask = set()

    combined_mask = ker_mask_set.union(all_sites_nonsquare_mask)
    n_valid_sites = square_shape[0] * square_shape[1] - len(combined_mask)
    possible_orders = list(range(1, n_valid_sites + 1))

    # All sub-patterns of the footprint, of every order, avoiding masked sites.
    all_patterns = []
    for order in possible_orders:
        patterns = [
            patt
            for patt in generate_patterns(
                shape=square_shape,
                order=order,
                rotations=False,
                reflections=False,
                translations=include_translations,
            )
            if combined_mask.isdisjoint({tuple(pt) for pt in patt})
        ]
        all_patterns += patterns

    n_channels = in_snapshots.shape[1] if in_snapshots.ndim == 4 else 1
    if channel_names is None:
        channel_names = [f"ch{c}" for c in range(n_channels)]

    npf_all: List[np.ndarray] = []
    latex_patterns: List[str] = []
    terms3d: List[Tuple[Tuple[int, int, int], ...]] = []

    # Single-basis terms: one correlator per (pattern, channel).
    for patt in all_patterns:
        base_mask = pattern_to_mask(patt, width=label_shape[1], height=label_shape[0])
        base_latex = mask_to_latex_pattern(base_mask, equation_env=False, full_latex=True)
        for c in range(n_channels):
            term3d = [(int(r), int(cc), c) for (r, cc) in patt]
            if orbit is None:
                npf = pattern_correlator(in_snapshots[:, c:c + 1, :, :], patt, window_shape=window_shape)
            else:
                npf = _symmetrized_correlator(in_snapshots, term3d, orbit)
            npf_all.append(npf)
            terms3d.append(tuple(term3d))
            if n_channels == 2:
                empty = mask_to_latex_pattern(
                    np.zeros((label_shape[0], label_shape[1]), dtype=int),
                    equation_env=False, full_latex=True,
                )
                rows = [
                    f"{channel_names[0]}:{base_latex if c == 0 else empty}",
                    f"{channel_names[1]}:{base_latex if c == 1 else empty}",
                ]
                latex_patterns.append("$\\substack{" + r" \\ ".join(rows) + "}$")
            else:
                suffix = f"_{channel_names[c]}" if n_channels > 1 else ""
                latex_patterns.append(r"$" + base_latex + suffix + r"$")

    # Cross-basis terms: products mixing channels, deduplicated by symmetry.
    if include_cross_terms and n_channels > 1:
        bounds = (square_shape[0], square_shape[1], n_channels)
        unique_cross: Dict[frozenset, Any] = {}

        for patt in all_patterns:
            for pattern3d in _cross_channel_subsets_for_pattern(
                patt=patt,
                n_channels=n_channels,
                require_all_spatial_sites=require_all_spatial_sites,
                min_channels_in_term=2,
                min_order=2,
                max_order=cross_max_order,
            ):
                canon = _canonicalize_cross_pattern_3d(
                    pattern3d=pattern3d,
                    bounds=bounds,
                    use_rotations=include_rotations,
                    use_reflections=include_reflections,
                    use_translations=include_translations,
                    include_basis_axis=include_basis_axis_symmetry,
                    include_only_valid=True,
                )
                # Store the canonical representative itself (as the ported Fig. 5
                # code does): with the default all-symmetries-off settings this is
                # just the sorted original coords, so the product is unchanged; with
                # symmetries on it is the dedup representative. Iterating a frozenset
                # is fine here since the correlator product is order-independent.
                if canon not in unique_cross:
                    unique_cross[canon] = canon

        for patt3d in unique_cross.values():
            if orbit is None:
                npf_all.append(_correlator_from_term(in_snapshots, patt3d, window_shape))
            else:
                npf_all.append(_symmetrized_correlator(in_snapshots, patt3d, orbit))
            terms3d.append(tuple(patt3d))
            latex_patterns.append(
                _cross_pattern_label_two_row(
                    pattern3d=patt3d,
                    label_shape=label_shape,
                    n_channels=n_channels,
                    channel_names=channel_names,
                    equation_env=True,
                )
            )

    if snapshot_averaged is None:
        X = np.array(npf_all).T
        y = z_selected
    else:
        corrs_avgd = []
        for npf in npf_all:
            avgd, _ = snapshot_averaged.snapshot_average(npf)
            corrs_avgd.append(avgd)
        X = np.array(corrs_avgd).T
        y, _ = snapshot_averaged.snapshot_average(z_selected)

    X_ = sm.add_constant(X)
    ols = sm.OLS(y, X_).fit()
    # NB: statsmodels' fit_regularized() defaults to alpha=0.0, so this is NOT a
    # sparse fit -- it is the same unpenalized solution reached by coordinate descent,
    # and its support is always full. Kept for backwards compatibility with callers
    # that already unpack it; best_subset/forward selection are the term selectors
    # this module recommends (see select_best_subset).
    ols_reg = sm.OLS(y, X_).fit_regularized()

    selected_terms = selected_patterns = selected_coefficients = None
    selected_r_squared = selection_path = selection_path_patterns = None
    if select_terms is not None:
        if select_strategy == "best_subset":
            selected_terms, selected_r_squared, selected_coefficients = (
                best_subset_terms(X_, y, select_terms)
            )
        elif select_strategy == "forward":
            # `select_terms` is a BUDGET here, not an exact size: build the greedy
            # path out to it, then keep the shortest prefix that is already as good
            # as the path gets (plateau_tol=None keeps the whole budget).
            selection_path = forward_selection_path(X_, y, select_terms)
            n_keep = (len(selection_path) if plateau_tol is None
                      else plateau_index(selection_path, plateau_tol,
                                         floor=plateau_floor,
                                         sufficient=plateau_sufficient))
            selected_terms = [col for col, _ in selection_path[:n_keep]]
            refit = sm.OLS(y, X_[:, [0] + selected_terms]).fit()
            selected_r_squared = float(refit.rsquared)
            selected_coefficients = refit.params
            selection_path_patterns = [latex_patterns[c - 1]
                                       for c, _ in selection_path]
        else:
            raise ValueError(
                f"Unknown select_strategy {select_strategy!r}: "
                "expected 'best_subset' or 'forward'."
            )
        # latex_patterns is aligned with the coefficient TAIL, so a design-matrix
        # column index j corresponds to latex_patterns[j - 1].
        selected_patterns = [latex_patterns[j - 1] for j in selected_terms]

    if verbose:
        print(
            f"[correlator OLS] X={X.shape}, y={np.asarray(y).shape}, "
            f"terms={len(latex_patterns)}, R^2={ols.rsquared:.4f}"
        )
        if select_terms is not None:
            print(
                f"[{select_strategy}] kept {len(selected_terms)} of "
                f"{len(latex_patterns)} terms, R^2={selected_r_squared:.4f}, "
                f"terms={selected_patterns}"
            )
            if selection_path is not None:
                trail = ", ".join(f"{i}:{r:.4f}"
                                  for i, (_, r) in enumerate(selection_path, 1))
                print(f"[{select_strategy}] path R^2 by size -> {trail}")
                if (plateau_tol is not None
                        and selected_r_squared < plateau_floor):
                    print(
                        f"[{select_strategy}] WARNING: budget exhausted at "
                        f"R^2={selected_r_squared:.4f}, below the floor "
                        f"{plateau_floor}; this branch has no adequate short form."
                    )

    return {
        "correlators_with_const": X_,
        "outputs": y,
        "coefficients": ols.params,
        "r_squared": ols.rsquared,
        "patterns": latex_patterns,
        "terms3d": terms3d,
        "label_shape": label_shape,
        "selected_terms": selected_terms,
        "selected_patterns": selected_patterns,
        "selected_coefficients": selected_coefficients,
        "selected_r_squared": selected_r_squared,
        "selection_path": selection_path,
        "selection_path_patterns": selection_path_patterns,
        "ols_object": ols,
        "ols_reg_object": ols_reg,
    }
