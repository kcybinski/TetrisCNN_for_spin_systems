"""Tests for tetriscnn.interpret.fit_activation_to_correlators.

The primary interpretability route regresses a branch activation onto the spin
correlators of its filter footprint. The core guarantee we characterize here is
recovery: when the activation IS a known linear combination of correlators, the
OLS fit must return those coefficients with R^2 = 1. We also lock in the term
enumeration (single- and cross-basis counts) and confirm the module has no
torch/PySR import cost.
"""

import numpy as np
import pytest

from tetriscnn.interpret import (
    best_subset_terms,
    fit_activation_to_correlators,
    forward_selection_path,
    orbit_site_maps,
    plateau_index,
)
from tetriscnn.pattern_generation import pattern_correlator


# Kernel specs follow cf.kernels entries: [shape, n_filters, dilation, mask, stride].
SINGLE_SITE = [(1, 1), 1, 1, None, 1]
DOMINO_V = [(2, 1), 1, 1, None, 1]
SQUARE = [(2, 2), 1, 1, None, 1]


def _random_spins(rng, B, C, H, W):
    """Batch of +/-1 spin snapshots, shape (B, C, H, W)."""
    return rng.choice(np.array([-1.0, 1.0]), size=(B, C, H, W))


class TestSingleBasisRecovery:
    def test_single_site_linear_recovery(self):
        # Build z exactly as a known affine function of the 1-point correlator,
        # then check the fit recovers slope and intercept and R^2 == 1.
        rng = np.random.default_rng(0)
        snaps = _random_spins(rng, B=400, C=1, H=6, W=6)

        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(1, 1))
        true_slope, true_intercept = 6.41, -2.96
        z = true_slope * C_site + true_intercept

        res = fit_activation_to_correlators(
            snaps, z, SINGLE_SITE, channel_names=["Z"], verbose=False,
        )

        assert res["r_squared"] == pytest.approx(1.0, abs=1e-9)
        # coefficients[0] is the intercept; the single-site branch has exactly one term.
        coefs = np.asarray(res["coefficients"])
        assert coefs[0] == pytest.approx(true_intercept, abs=1e-6)
        assert coefs[1] == pytest.approx(true_slope, abs=1e-6)
        assert len(res["patterns"]) == 1

    def test_domino_multilinear_recovery(self):
        # A 2x1 branch footprint has sub-patterns: two 1-point and one 2-point.
        rng = np.random.default_rng(1)
        snaps = _random_spins(rng, B=600, C=1, H=6, W=6)

        res_probe = fit_activation_to_correlators(
            snaps, np.arange(snaps.shape[0], dtype=float), DOMINO_V, channel_names=["Z"], verbose=False,
        )
        X = np.asarray(res_probe["correlators_with_const"])  # (B, 1 + n_terms)
        n_terms = X.shape[1] - 1
        assert n_terms == 3  # site (0,0), site (1,0), and the (0,0)-(1,0) pair

        beta = np.array([0.5, 1.3, -0.7, 0.2])  # intercept + 3 terms
        z = X @ beta

        res = fit_activation_to_correlators(
            snaps, z, DOMINO_V, channel_names=["Z"], verbose=False,
        )
        assert res["r_squared"] == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(np.asarray(res["coefficients"]), beta, atol=1e-6)

    def test_unexplained_activation_has_low_r2(self):
        # Pure noise uncorrelated with any correlator must not be "explained".
        rng = np.random.default_rng(2)
        snaps = _random_spins(rng, B=500, C=1, H=6, W=6)
        z = rng.normal(size=snaps.shape[0])
        res = fit_activation_to_correlators(
            snaps, z, SINGLE_SITE, channel_names=["Z"], verbose=False,
        )
        assert res["r_squared"] < 0.1


class TestCrossBasis:
    def test_cross_terms_appear_only_for_multichannel(self):
        rng = np.random.default_rng(3)
        # 2x2 square, single channel: no cross terms possible.
        snaps1 = _random_spins(rng, B=200, C=1, H=6, W=6)
        res1 = fit_activation_to_correlators(
            snaps1, np.arange(200, dtype=float), SQUARE, verbose=False,
        )
        n1 = np.asarray(res1["correlators_with_const"]).shape[1] - 1

        # 2x2 square, two channels with cross terms: strictly more terms.
        snaps2 = _random_spins(rng, B=200, C=2, H=6, W=6)
        res2 = fit_activation_to_correlators(
            snaps2, np.arange(200, dtype=float), SQUARE, channel_names=["X", "Z"],
            include_cross_terms=True, verbose=False,
        )
        n2 = np.asarray(res2["correlators_with_const"]).shape[1] - 1
        assert n2 > 2 * n1  # single-basis terms doubled, plus cross terms on top

    def test_cross_terms_toggle_off(self):
        rng = np.random.default_rng(4)
        snaps = _random_spins(rng, B=200, C=2, H=6, W=6)
        with_cross = fit_activation_to_correlators(
            snaps, np.arange(200, dtype=float), SQUARE, channel_names=["X", "Z"],
            include_cross_terms=True, verbose=False,
        )
        without_cross = fit_activation_to_correlators(
            snaps, np.arange(200, dtype=float), SQUARE, channel_names=["X", "Z"],
            include_cross_terms=False, verbose=False,
        )
        n_with = np.asarray(with_cross["correlators_with_const"]).shape[1]
        n_without = np.asarray(without_cross["correlators_with_const"]).shape[1]
        assert n_with > n_without

    def test_cross_basis_product_recovery(self):
        # z built from a genuine cross-basis product C^X[site]*C^Z[site]; the fit
        # must include such a term and reach R^2 = 1.
        rng = np.random.default_rng(5)
        snaps = _random_spins(rng, B=800, C=2, H=6, W=6)
        res_probe = fit_activation_to_correlators(
            snaps, np.arange(800, dtype=float), SQUARE, channel_names=["X", "Z"],
            include_cross_terms=True, verbose=False,
        )
        X = np.asarray(res_probe["correlators_with_const"])
        # Construct z from an arbitrary fixed direction in the correlator space.
        beta = np.zeros(X.shape[1])
        beta[0] = 0.1
        beta[3] = 0.8
        beta[-1] = -0.5
        z = X @ beta
        res = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["X", "Z"],
            include_cross_terms=True, verbose=False,
        )
        assert res["r_squared"] == pytest.approx(1.0, abs=1e-9)


class TestOutputContract:
    def test_return_keys_and_shapes(self):
        rng = np.random.default_rng(6)
        snaps = _random_spins(rng, B=150, C=1, H=5, W=5)
        z = rng.normal(size=150)
        res = fit_activation_to_correlators(snaps, z, SQUARE, verbose=False)
        for key in ("correlators_with_const", "outputs", "coefficients",
                    "r_squared", "patterns", "ols_object", "ols_reg_object"):
            assert key in res
        X = np.asarray(res["correlators_with_const"])
        # coefficients align with columns; patterns align with the non-intercept tail.
        assert len(np.asarray(res["coefficients"])) == X.shape[1]
        assert len(res["patterns"]) == X.shape[1] - 1

    def test_non_square_kernel_runs(self):
        # 2x1 footprint must be handled via square-padding + masking, not crash.
        rng = np.random.default_rng(7)
        snaps = _random_spins(rng, B=120, C=1, H=5, W=5)
        z = rng.normal(size=120)
        res = fit_activation_to_correlators(snaps, z, DOMINO_V, verbose=False)
        assert np.isfinite(res["r_squared"])


class TestGroupSymmetrization:
    """The `group=` path, for reading an equivariant branch exactly.

    An equivariant branch averages its pooled scalar over the group, so it is linear in
    the orbit-averaged correlators, not in the plain ones. These tests pin down that the
    orbit bookkeeping matches the model's own weight transforms, and that passing the
    branch's group is what turns an approximate reading into an exact one.
    """

    @pytest.mark.parametrize("shape", [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2)])
    @pytest.mark.parametrize("group", ["C4", "D2"])
    def test_orbit_maps_match_model_weight_transforms(self, shape, group):
        """orbit_site_maps must reproduce ConvBranch_Equivariant's weight transforms.

        The correlator features are only right if the group action assumed here is the
        one the branch actually applies, so compare against the model's own method
        rather than against a hand-written expectation.
        """
        import torch

        from tetriscnn.models import ConvBranch_Equivariant

        branch = ConvBranch_Equivariant(
            in_channels=1, kernel_shape=shape, number_of_filters=1,
            hidden_size=1, group=group,
        )
        W = torch.arange(float(shape[0] * shape[1])).reshape(1, 1, *shape)
        transformed = branch._group_transformed_weights(W)
        maps = orbit_site_maps(shape, group)
        assert len(maps) == len(transformed)

        for (site_map, window), Wg in zip(maps, transformed):
            assert window == tuple(Wg.shape[2:])
            for (r, c), (i, j) in site_map.items():
                assert Wg[0, 0, i, j].item() == W[0, 0, r, c].item()

    def test_group_arg_rejects_unknown_group(self):
        with pytest.raises(ValueError, match="Unknown group"):
            orbit_site_maps((2, 2), "C6")

    def test_equivariant_branch_needs_its_group_for_exact_recovery(self):
        """A trained-shaped equivariant (2,1) branch: plain fit is approximate, C4 is exact.

        (2,1) is the case that matters -- its C4 orbit alternates between a (2,1) and a
        (1,2) footprint, so orbit members pool over differently shaped valid-position
        grids. A group-blind fit cannot represent that.
        """
        import torch

        from tetriscnn.models import ConvBranch_Equivariant

        rng = np.random.default_rng(0)
        snaps = _random_spins(rng, 400, 1, 8, 8)

        torch.manual_seed(0)
        branch = ConvBranch_Equivariant(
            in_channels=1, kernel_shape=(2, 1), number_of_filters=1,
            hidden_size=4, group="C4",
        )
        with torch.no_grad():
            z = branch(torch.tensor(snaps, dtype=torch.float32)).squeeze(-1).numpy()

        plain = fit_activation_to_correlators(
            snaps, z, DOMINO_V, verbose=False)
        sym = fit_activation_to_correlators(
            snaps, z, DOMINO_V, group="C4", verbose=False)

        assert sym["r_squared"] == pytest.approx(1.0, abs=1e-6)
        assert plain["r_squared"] < sym["r_squared"]

    def test_symmetrizing_a_plain_branch_breaks_its_exact_reading(self):
        """Orbit-averaged correlators are not a universally better basis.

        A non-equivariant branch is exactly linear in the PLAIN correlators; symmetrizing
        must therefore make the reading worse, not better. Without this the `group=` gain
        could be explained as "more expressive features" rather than "matching the model".
        """
        rng = np.random.default_rng(1)
        snaps = _random_spins(rng, 400, 1, 8, 8)
        c_top = pattern_correlator(snaps, [(0, 0)], window_shape=(2, 1))
        c_bot = pattern_correlator(snaps, [(1, 0)], window_shape=(2, 1))
        z = 0.7 * c_top - 0.4 * c_bot + 0.1

        plain = fit_activation_to_correlators(snaps, z, DOMINO_V, verbose=False)
        sym = fit_activation_to_correlators(snaps, z, DOMINO_V, group="C4", verbose=False)

        assert plain["r_squared"] == pytest.approx(1.0, abs=1e-8)
        assert sym["r_squared"] < 0.999

    def test_default_path_is_unchanged_by_the_group_feature(self):
        """group=None must reproduce the plain reading bit-for-bit."""
        rng = np.random.default_rng(2)
        snaps = _random_spins(rng, 200, 2, 6, 7)
        z = rng.normal(size=200)
        a = fit_activation_to_correlators(snaps, z, DOMINO_V, verbose=False)
        b = fit_activation_to_correlators(snaps, z, DOMINO_V, group=None, verbose=False)
        assert np.array_equal(a["correlators_with_const"], b["correlators_with_const"])
        assert a["patterns"] == b["patterns"]


def test_module_does_not_require_pysr():
    """The primary interpretability route must not drag in PySR/Julia.

    Importing tetriscnn.interpret pulls in tetriscnn.utils (hence torch), which is
    expected: the whole library uses torch. The point of this module living apart
    from symbolic_regression is that it needs no PySR/Julia backend. We assert two
    things: pysr is not imported as a side effect, and the module declares no import
    of pysr or symbolic_regression (docstring mentions do not count).
    """
    import ast
    import importlib
    import sys

    importlib.import_module("tetriscnn.interpret")
    assert "pysr" not in sys.modules

    import tetriscnn.interpret as mod
    tree = ast.parse(open(mod.__file__).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("pysr" in m for m in imported)
    assert not any("symbolic_regression" in m for m in imported)


class TestBestSubsetSelection:
    """The `select_terms` opt-in: exhaustive best-subset over the correlator terms.

    This automates what was previously done by hand -- deciding which terms can be
    thrown out of an exact-but-unreadable full fit. The contract we pin down is
    that it is EXACT (it really is the argmax over all subsets, verified against a
    brute-force recomputation) and PURELY ADDITIVE (the full fit is untouched).
    """

    def _setup(self, seed=7, B=500):
        rng = np.random.default_rng(seed)
        snaps = _random_spins(rng, B=B, C=1, H=6, W=6)
        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(1, 1))
        # A target that leans on one correlator plus noise, so the selection has a
        # clear right answer rather than an arbitrary tie.
        z = 3.0 * C_site + 0.25 * rng.standard_normal(B)
        return snaps, z

    def test_selection_is_the_true_argmax(self):
        # Brute-force every subset independently and confirm the returned support
        # really is the maximiser -- the whole value of exhaustive over a lasso path.
        from itertools import combinations

        import statsmodels.api as sm

        snaps, z = self._setup()
        res = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=2, verbose=False,
        )
        X_ = np.asarray(res["correlators_with_const"])
        y = np.asarray(res["outputs"])

        best = max(
            (sm.OLS(y, X_[:, (0,) + s]).fit().rsquared, s)
            for s in combinations(range(1, X_.shape[1]), 2)
        )
        assert res["selected_r_squared"] == pytest.approx(best[0], abs=1e-12)
        assert tuple(res["selected_terms"]) == best[1]

    def test_selection_is_additive_and_off_by_default(self):
        # The full fit must be bit-identical with and without the opt-in.
        snaps, z = self._setup()
        plain = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], verbose=False,
        )
        picked = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=3, verbose=False,
        )
        assert plain["selected_terms"] is None
        assert plain["selected_r_squared"] is None
        assert picked["r_squared"] == pytest.approx(plain["r_squared"], abs=1e-15)
        np.testing.assert_allclose(
            np.asarray(picked["coefficients"]), np.asarray(plain["coefficients"]),
            rtol=0, atol=1e-15,
        )

    def test_more_terms_never_hurts(self):
        # Nested budgets: R^2 is monotone non-decreasing in select_terms, and a
        # full-width selection must recover the unrestricted fit exactly.
        snaps, z = self._setup()
        scores = []
        for k in (1, 2, 3):
            res = fit_activation_to_correlators(
                snaps, z, SQUARE, channel_names=["Z"], select_terms=k, verbose=False,
            )
            assert len(res["selected_terms"]) == k
            assert len(res["selected_patterns"]) == k
            # coefficients carry the intercept plus one per selected term.
            assert len(res["selected_coefficients"]) == k + 1
            scores.append(res["selected_r_squared"])
        assert scores == sorted(scores)

        n_avail = len(fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], verbose=False)["patterns"])
        full = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=n_avail, verbose=False,
        )
        assert full["selected_r_squared"] == pytest.approx(full["r_squared"], abs=1e-12)

    def test_patterns_align_with_selected_columns(self):
        # patterns[] is aligned with the coefficient TAIL, so column j maps to
        # patterns[j-1]; an off-by-one here would silently mislabel every equation.
        snaps, z = self._setup()
        res = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=3, verbose=False,
        )
        expected = [res["patterns"][j - 1] for j in res["selected_terms"]]
        assert res["selected_patterns"] == expected

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_nonpositive_budget(self, bad):
        snaps, z = self._setup(B=200)
        with pytest.raises(ValueError, match="must be >= 1"):
            fit_activation_to_correlators(
                snaps, z, SQUARE, channel_names=["Z"], select_terms=bad, verbose=False,
            )

    def test_rejects_budget_beyond_available_terms(self):
        snaps, z = self._setup(B=200)
        with pytest.raises(ValueError, match="exceeds"):
            fit_activation_to_correlators(
                snaps, z, SINGLE_SITE, channel_names=["Z"], select_terms=5, verbose=False,
            )

    def test_guard_rejects_infeasible_enumeration(self):
        # Term count is 2^(sites*channels) - 1, so an UNMASKED 2x2 on two channels
        # has 255 terms and C(255,3) = 2.7M refits. Exhaustive selection is genuinely
        # infeasible there and must say so loudly rather than hang. Every branch the
        # current kernel sets produce stays at <= 15 terms, so this is a tripwire.
        rng = np.random.default_rng(3)
        snaps = _random_spins(rng, B=200, C=2, H=6, W=6)
        z = rng.standard_normal(200)

        probe = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["X", "Z"], verbose=False,
        )
        assert len(probe["patterns"]) == 255

        with pytest.raises(ValueError, match="guard"):
            fit_activation_to_correlators(
                snaps, z, SQUARE, channel_names=["X", "Z"],
                select_terms=3, verbose=False,
            )


class TestForwardSelection:
    """Greedy forward selection + the plateau/floor stopping rule.

    The distinction that matters against best-subset: forward supports are NESTED,
    which is what makes an importance ordering and a stopping rule well defined.
    """

    def test_path_is_nested_and_monotone(self):
        rng = np.random.default_rng(31)
        snaps = _random_spins(rng, 400, 1, 6, 6)
        z = snaps[:, 0].mean(axis=(-2, -1)) * 3 + 0.2 * rng.standard_normal(400)
        res = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], verbose=False
        )
        path = forward_selection_path(res["correlators_with_const"], res["outputs"], 4)

        assert len(path) == 4
        cols = [c for c, _ in path]
        assert len(set(cols)) == 4, "a term was selected twice"
        r2s = [r for _, r in path]
        # Nested supports cannot lose explanatory power as they grow.
        assert all(b >= a - 1e-12 for a, b in zip(r2s, r2s[1:])), r2s

    def test_first_term_is_the_single_best(self):
        """Step 1 of the path must be the argmax over all single terms."""
        rng = np.random.default_rng(32)
        snaps = _random_spins(rng, 400, 1, 6, 6)
        z = snaps[:, 0].mean(axis=(-2, -1)) * 3 + 0.2 * rng.standard_normal(400)
        res = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], verbose=False
        )
        X_, y = res["correlators_with_const"], res["outputs"]
        path = forward_selection_path(X_, y, 1)
        brute = best_subset_terms(X_, y, 1)
        assert [c for c, _ in path] == brute[0]
        assert path[0][1] == pytest.approx(brute[1])

    def test_plateau_prefers_the_shortest_adequate_prefix(self):
        # Flat after one term, and comfortably above the floor -> keep one.
        assert plateau_index([(1, 0.9985), (2, 1.0), (3, 1.0)], tol=0.01, floor=0.95) == 1

    def test_plateau_survives_a_late_jump(self):
        """A local 'first small gain' rule is wrong; this is the case that proves it.

        Gains here are +0.24, +0.11, +0.10, then +0.54 -- terms that are only
        jointly informative. Stopping at the first small increment would quote
        R^2=0.24. Taken from the real plain-XY (2,1) branch.
        """
        path = [(1, 0.2434), (2, 0.3561), (3, 0.4536), (4, 0.9899), (5, 0.9954)]
        assert plateau_index(path, tol=0.01, floor=0.95) == 4

    def test_floor_overrides_an_early_plateau(self):
        """Plateaued but inadequate: the floor must force the longer model.

        The plateau target is relative to the path's own best, so it alone would
        accept 0.945 here (within tol of the 0.9510 ceiling). The floor is what
        rejects it.
        """
        path = [(1, 0.945), (2, 0.9505), (3, 0.9510)]
        assert plateau_index(path, tol=0.01, floor=0.95) == 2
        # Same path, no floor -> the plateau alone stops one term earlier.
        assert plateau_index(path, tol=0.01, floor=0.0) == 1

    def test_unreachable_floor_returns_the_full_path(self):
        path = [(1, 0.42), (2, 0.44), (3, 0.45)]
        assert plateau_index(path, tol=0.01, floor=0.95) == 3

    # ---- `sufficient`: the absolute "good enough" bar -----------------------
    # Answers a different question from tol/floor. tol is RELATIVE to the best on
    # the path, so a uniformly strong path still demands nearly every term; floor is
    # a lower bound that `max` only makes stricter. `sufficient` is the only one of
    # the three that can stop early on a strong path.

    # The real case that motivated it: the C4 XY boundary path. Six terms are needed
    # to come within tol of the path best, but four already clear 0.99.
    XY_BOUNDARY_PATH = [(0, 0.9411), (1, 0.9669), (2, 0.9857),
                        (3, 0.9929), (4, 0.9959), (5, 0.9984)]

    def test_sufficient_takes_the_shortest_adequate_prefix(self):
        assert plateau_index(self.XY_BOUNDARY_PATH, tol=0.002, floor=0.95,
                             sufficient=0.99) == 4

    def test_without_sufficient_the_same_path_keeps_everything(self):
        # Regression guard: the default must not drift, because Figure 5's published
        # equations are cut by this function with `sufficient` unset.
        assert plateau_index(self.XY_BOUNDARY_PATH, tol=0.002, floor=0.95) == 6

    def test_sufficient_none_is_exactly_the_old_behaviour(self):
        for path in ([(1, 0.2434), (2, 0.3561), (3, 0.4536), (4, 0.9899), (5, 0.9954)],
                     [(1, 0.945), (2, 0.9505), (3, 0.9510)],
                     [(1, 0.42), (2, 0.44), (3, 0.45)]):
            assert (plateau_index(path, tol=0.01, floor=0.95, sufficient=None)
                    == plateau_index(path, tol=0.01, floor=0.95))

    def test_unreachable_sufficient_falls_back_to_the_plateau_rule(self):
        path = [(1, 0.42), (2, 0.44), (3, 0.45)]
        assert plateau_index(path, tol=0.01, floor=0.95, sufficient=0.99) == 3

    def test_sufficient_takes_the_first_crossing_not_the_best(self):
        # Non-monotone paths are normal when the score is cross-validated accuracy.
        # The rule is "shortest prefix that is good enough", so a later, higher point
        # must not win.
        path = [(0, 0.980), (1, 0.994), (2, 0.991), (3, 0.997)]
        assert plateau_index(path, tol=0.001, floor=0.95, sufficient=0.99) == 2

    def test_sufficient_can_select_the_very_first_term(self):
        path = [(0, 0.995), (1, 0.996), (2, 0.999)]
        assert plateau_index(path, tol=0.001, floor=0.95, sufficient=0.99) == 1

    def test_fit_plumbs_plateau_sufficient_through(self):
        rng = np.random.default_rng(7)
        snaps = _random_spins(rng, 400, 1, 6, 6)
        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(2, 2))
        z = 3.0 * C_site + 0.05 * rng.standard_normal(400)
        kw = dict(in_snapshots=snaps, z_selected=z, branch_to_fit=[(2, 2), 1, 1, None, 1],
                  channel_names=["Z"], select_terms=4, select_strategy="forward",
                  plateau_tol=1e-9, plateau_floor=0.95, verbose=False)
        loose = fit_activation_to_correlators(**kw, plateau_sufficient=0.95)
        tight = fit_activation_to_correlators(**kw)
        assert len(loose["selected_terms"]) <= len(tight["selected_terms"])
        assert loose["selected_r_squared"] >= 0.95

    def test_forward_strategy_truncates_and_reports_the_path(self):
        rng = np.random.default_rng(33)
        snaps = _random_spins(rng, 500, 1, 6, 6)
        # Dominated by ONE correlator. window_shape must match the branch footprint:
        # the enumerated single-site term is pooled over the 2x2 window's valid
        # positions, so a (1,1)-pooled target is NOT in the span and caps R^2 at 0.84.
        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(2, 2))
        z = 3.0 * C_site + 0.05 * rng.standard_normal(500)
        res = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"],
            select_terms=4, select_strategy="forward",
            plateau_tol=0.01, plateau_floor=0.95, verbose=False,
        )
        assert res["selection_path"] is not None
        assert len(res["selection_path"]) == 4
        assert len(res["selected_terms"]) <= 4
        assert len(res["selected_patterns"]) == len(res["selected_terms"])
        # intercept + one coefficient per kept term
        assert len(res["selected_coefficients"]) == len(res["selected_terms"]) + 1
        assert res["selected_r_squared"] >= 0.95
        # plateaus immediately: one correlator is the whole story here
        assert len(res["selected_terms"]) == 1

    def test_forward_budget_is_a_maximum_not_an_exact_size(self):
        """best_subset returns exactly k; forward may return fewer."""
        rng = np.random.default_rng(34)
        snaps = _random_spins(rng, 500, 1, 6, 6)
        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(2, 2))
        z = 3.0 * C_site + 0.05 * rng.standard_normal(500)
        exact = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=4, verbose=False
        )
        pruned = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=4,
            select_strategy="forward", plateau_tol=0.01, verbose=False,
        )
        assert len(exact["selected_terms"]) == 4
        assert len(pruned["selected_terms"]) < 4

    def test_no_plateau_tol_keeps_the_whole_budget(self):
        rng = np.random.default_rng(35)
        snaps = _random_spins(rng, 400, 1, 6, 6)
        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(2, 2))
        z = 3.0 * C_site + 0.05 * rng.standard_normal(400)
        res = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=3,
            select_strategy="forward", plateau_tol=None, verbose=False,
        )
        assert len(res["selected_terms"]) == 3

    def test_best_subset_remains_the_default(self):
        """The existing contract must not shift under callers that never opt in."""
        rng = np.random.default_rng(36)
        snaps = _random_spins(rng, 400, 1, 6, 6)
        z = snaps[:, 0].mean(axis=(-2, -1)) * 3 + 0.2 * rng.standard_normal(400)
        default = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=2, verbose=False
        )
        explicit = fit_activation_to_correlators(
            snaps, z, SQUARE, channel_names=["Z"], select_terms=2,
            select_strategy="best_subset", verbose=False,
        )
        assert default["selected_terms"] == explicit["selected_terms"]
        assert default["selection_path"] is None

    def test_rejects_unknown_strategy(self):
        rng = np.random.default_rng(37)
        snaps = _random_spins(rng, 200, 1, 6, 6)
        z = rng.standard_normal(200)
        with pytest.raises(ValueError, match="select_strategy"):
            fit_activation_to_correlators(
                snaps, z, DOMINO_V, channel_names=["Z"],
                select_terms=2, select_strategy="lasso", verbose=False,
            )

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_nonpositive_budget(self, bad):
        rng = np.random.default_rng(38)
        snaps = _random_spins(rng, 200, 1, 6, 6)
        z = rng.standard_normal(200)
        res = fit_activation_to_correlators(
            snaps, z, DOMINO_V, channel_names=["Z"], verbose=False
        )
        with pytest.raises(ValueError, match="max_terms"):
            forward_selection_path(res["correlators_with_const"], res["outputs"], bad)

    def test_rejects_budget_beyond_available_terms(self):
        rng = np.random.default_rng(39)
        snaps = _random_spins(rng, 200, 1, 6, 6)
        z = rng.standard_normal(200)
        res = fit_activation_to_correlators(
            snaps, z, SINGLE_SITE, channel_names=["Z"], verbose=False
        )
        n_avail = len(res["patterns"])
        with pytest.raises(ValueError, match="exceeds"):
            forward_selection_path(
                res["correlators_with_const"], res["outputs"], n_avail + 1
            )

    def test_empty_path_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            plateau_index([], tol=0.01)
