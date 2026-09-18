"""Group A: pure-function characterization tests, no data files, no training.

These must stay fast (whole file well under 5s). They pin down the CURRENT numeric
behavior of small utility functions in tetriscnn/utils.py so a refactor can be
checked against them byte-for-byte (well, float-for-float).
"""
import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score as sk_r2_score

from tetriscnn.utils import (
    AttrDict,
    EarlyStopper,
    denormalize01,
    get_branch_penalties,
    l1_regularization,
    normalize01,
    r2_agg,
    set_kernels,
)


# ---------------------------------------------------------------------------
# 1. get_branch_penalties
# ---------------------------------------------------------------------------

class TestGetBranchPenalties:
    """penalty_params = [base, min_exp, lam, n_pen] = [10, -5, lam, 1].

    l = base ** linspace(min_exp, lam, max_area)[area - 1]
    Kernels with a mask use popcount(mask) as their area instead of shape[0]*shape[1].
    Dilated kernels (area > 1) get an extra linear `* dilation` factor.
    """

    def test_smallkernels_masks_use_popcount_area(self):
        # smallkernels (Paris_Ising, non-equivariant): 10 branches, several (2,2) kernels
        # share a mask with 2 or 3 nonzero entries, which changes their effective area.
        cf = AttrDict()
        cf.kernel_set = "smallkernels"
        cf.equivariant = False
        set_kernels(cf)
        assert len(cf.kernels) == 10

        penalties = get_branch_penalties([10, -5, -1, 1], cf.kernels)
        expected = [
            1e-05,
            0.00021544346900318823,  # (2,1), area 2
            0.00021544346900318823,  # (1,2), area 2
            0.00021544346900318823,  # (2,2) mask popcount=2 -> area 2
            0.00021544346900318823,  # (2,2) mask popcount=2 -> area 2
            0.004641588833612777,    # (2,2) mask popcount=3 -> area 3
            0.004641588833612777,
            0.004641588833612777,
            0.004641588833612777,
            0.1,                      # (2,2) full mask=None -> area 4
        ]
        assert penalties.tolist() == pytest.approx(expected, rel=1e-9)

        # A second lam value, same kernel set -> penalty_base rescales but the
        # per-branch *ordering* (which entries share a value) is unchanged.
        penalties2 = get_branch_penalties([10, -5, 4, 1], cf.kernels)
        expected2 = [1e-05, 0.01, 0.01, 0.01, 0.01, 10.0, 10.0, 10.0, 10.0, 10000.0]
        assert penalties2.tolist() == pytest.approx(expected2, rel=1e-9)

    def test_defaultkernels_dilation_doubles_penalty(self):
        # defaultkernels has two dilation=2 kernels of area 2 ((2,1) and (1,2) dilated);
        # their penalty should be exactly 2x the dilation=1, area-2 kernels' penalty
        # (factor = k_dilation when area > 1).
        cf = AttrDict()
        cf.kernel_set = "defaultkernels"
        set_kernels(cf)
        assert len(cf.kernels) == 9

        penalties = get_branch_penalties([10, -5, -1, 1], cf.kernels)
        expected = [
            1e-05,
            3.1622776601683795e-05,  # (2,1) dil=1, area 2
            3.1622776601683795e-05,  # (1,2) dil=1, area 2
            0.00031622776601683794,  # (2,2) area 4
            6.324555320336759e-05,   # (2,1) dil=2, area 2 -> 2x the dil=1 area-2 value
            6.324555320336759e-05,   # (1,2) dil=2, area 2
            0.0001,                  # (3,1) area 3
            0.0001,                  # (1,3) area 3
            0.1,                     # (3,3) area 9
        ]
        assert penalties.tolist() == pytest.approx(expected, rel=1e-9)
        # explicit dilation-doubling check, independent of the hardcoded values above
        assert penalties[4] == pytest.approx(2 * penalties[1], rel=1e-9)
        assert penalties[5] == pytest.approx(2 * penalties[2], rel=1e-9)


# ---------------------------------------------------------------------------
# 2. l1_regularization
# ---------------------------------------------------------------------------

class TestL1Regularization:
    def test_weighted_l1_of_activations(self):
        z = torch.tensor([[1.0, -2.0, 3.0], [0.0, 5.0, -1.0]])
        penalties = torch.tensor([0.1, 1.0, 10.0])
        result = l1_regularization(z, penalties)
        # row0: 0.1*1 + 1*2 + 10*3 = 32.1 ; row1: 0.1*0 + 1*5 + 10*1 = 15.0
        expected = torch.tensor([32.1, 15.0])
        assert torch.allclose(result, expected, rtol=1e-6)

    def test_shape_mismatch_asserts(self):
        z = torch.zeros(4, 3)
        penalties = torch.zeros(2)
        with pytest.raises(AssertionError):
            l1_regularization(z, penalties)


# ---------------------------------------------------------------------------
# 3. EarlyStopper
# ---------------------------------------------------------------------------

class TestEarlyStopper:
    def test_warmup_never_stops_even_with_worsening_loss(self):
        stopper = EarlyStopper(patience=1, min_delta=0, verbose=False, warmup_epochs=3)
        losses = [1.0, 2.0, 3.0]  # monotonically worsening
        for epoch, loss in enumerate(losses):
            assert stopper.early_stop(loss, epoch=epoch) is False

    def test_warmup_still_tracks_best_objective(self):
        stopper = EarlyStopper(patience=1, min_delta=0, verbose=False, warmup_epochs=3)
        for epoch, loss in enumerate([5.0, 1.0, 3.0]):
            stopper.early_stop(loss, epoch=epoch)
        # best seen during warmup (1.0 at epoch 1) must be tracked internally,
        # since post-warmup improvement is judged relative to it.
        assert stopper.min_validation_obj == pytest.approx(1.0)

    def test_fires_after_patience_non_improving_epochs_post_warmup(self):
        stopper = EarlyStopper(patience=3, min_delta=0, verbose=False, warmup_epochs=0)
        # epoch 0: establishes best=1.0
        assert stopper.early_stop(1.0, epoch=0) is False
        # epochs 1,2,3: non-improving (worse than best) -> counter 1,2,3
        assert stopper.early_stop(2.0, epoch=1) is False
        assert stopper.early_stop(2.0, epoch=2) is False
        assert stopper.early_stop(2.0, epoch=3) is True  # counter reaches patience=3

    def test_warmup_then_patience_end_to_end(self):
        # warmup=2 (epochs 0,1 disabled), patience=2 thereafter.
        stopper = EarlyStopper(patience=2, min_delta=0, verbose=False, warmup_epochs=2)
        assert stopper.early_stop(10.0, epoch=0) is False   # warmup, tracks best=10.0
        assert stopper.early_stop(20.0, epoch=1) is False   # warmup, worse, not tracked
        assert stopper.early_stop(15.0, epoch=2) is False   # post-warmup, worse than 10 -> counter=1
        assert stopper.early_stop(15.0, epoch=3) is True    # counter=2 >= patience

    def test_nan_stops_immediately_after_warmup(self):
        stopper = EarlyStopper(patience=5, min_delta=0, verbose=False, warmup_epochs=1)
        assert stopper.early_stop(1.0, epoch=0) is False  # warmup
        assert stopper.early_stop(float("nan"), epoch=1) is True


# ---------------------------------------------------------------------------
# 4. normalize01 / denormalize01
# ---------------------------------------------------------------------------

class TestNormalizeRoundTrip:
    def test_round_trip(self):
        y = torch.tensor([0.0, 2.5, 5.0, 10.0])
        y_min, y_max = 0.0, 10.0
        n = normalize01(y, y_min, y_max)
        assert torch.allclose(n, torch.tensor([0.0, 0.25, 0.5, 1.0]))
        back = denormalize01(n, y_min, y_max)
        assert torch.allclose(back, y)


# ---------------------------------------------------------------------------
# 5. r2_agg
# ---------------------------------------------------------------------------

class TestR2Agg:
    def _sample(self):
        # 2 outputs, each true value repeated (simulates repeated snapshots at the
        # same tuning-parameter value), predictions noisy around the true value.
        y_true = torch.tensor([
            [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
            [0.5, 1.0], [0.5, 1.0],
            [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0],
        ])
        y_pred = torch.tensor([
            [0.1, 0.9], [-0.1, 1.1], [0.05, 1.05],
            [0.6, 0.8], [0.4, 1.2],
            [0.9, 2.1], [1.1, 1.9], [1.05, 2.05], [0.95, 1.95],
        ])
        return y_true, y_pred

    def test_variance_weighted_matches_sklearn_on_aggregated_points(self):
        y_true, y_pred = self._sample()
        result = r2_agg(y_pred, y_true, multioutput="variance_weighted")

        # hand/sklearn cross-check: aggregate predictions per unique true value,
        # then r2_score(..., multioutput='variance_weighted') on the aggregated points.
        y_true_np, y_pred_np = y_true.numpy(), y_pred.numpy()
        agg_true, agg_pred = [], []
        for d in range(2):
            uniq = np.unique(y_true_np[:, d])
            agg_true.append(uniq)
            agg_pred.append(np.array([y_pred_np[y_true_np[:, d] == u, d].mean() for u in uniq]))
        # sklearn's variance_weighted needs equal-length columns; here both dims
        # happen to differ in unique-value count (3 vs 2), so weight by hand instead,
        # replicating r2_agg's own per-output variance-weighting formula.
        r2_0 = sk_r2_score(agg_true[0], agg_pred[0])
        r2_1 = sk_r2_score(agg_true[1], agg_pred[1])
        var0 = np.var(agg_true[0])
        var1 = np.var(agg_true[1])
        expected = (r2_0 * var0 + r2_1 * var1) / (var0 + var1)

        assert result.item() == pytest.approx(expected, rel=1e-5)
        assert result.item() == pytest.approx(0.9996577501296997, rel=1e-5)

    def test_uniform_average(self):
        y_true, y_pred = self._sample()
        result = r2_agg(y_pred, y_true, multioutput="uniform_average")
        assert result.item() == pytest.approx(0.9996222257614136, rel=1e-5)

    def test_raw_values(self):
        y_true, y_pred = self._sample()
        result = r2_agg(y_pred, y_true, multioutput="raw_values")
        assert result.tolist() == pytest.approx([0.9994444251060486, 0.9998000264167786], rel=1e-5)

    def test_1d_input_treated_as_single_output(self):
        y_true = torch.tensor([0.0, 0.0, 1.0, 1.0])
        y_pred = torch.tensor([0.1, -0.1, 0.9, 1.1])
        result = r2_agg(y_pred, y_true)
        # aggregated: true=[0,1], pred=[0.0, 1.0] -> perfect fit
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_unknown_multioutput_raises(self):
        y_true, y_pred = self._sample()
        with pytest.raises(ValueError):
            r2_agg(y_pred, y_true, multioutput="not_a_real_option")


# ---------------------------------------------------------------------------
# 6. set_kernels
# ---------------------------------------------------------------------------

class TestSetKernels:
    def test_smallkernels(self):
        cf = AttrDict()
        cf.kernel_set = "smallkernels"
        set_kernels(cf)
        assert len(cf.kernels) == 10
        for k in cf.kernels:
            assert len(k) == 5  # (shape, n_filters, dilation, mask, stride)
        # every entry got stride=1 appended by the compatibility loop
        assert all(k[4] == 1 for k in cf.kernels)
        # areas (by raw shape, ignoring masks) are non-decreasing: this is the
        # "nested pattern" ordering the physics narrative relies on for smallkernels.
        areas = [k[0][0] * k[0][1] for k in cf.kernels]
        assert areas == sorted(areas)

    def test_smallkernels_equivariant_flag_selects_canonical_5(self):
        """cf.equivariant branches smallkernels between the 10-branch non-equivariant
        list (one branch per orientation) and a 5-branch canonical-representative
        list (one branch per C4 orbit, realized via ConvBranch_Equivariant weight
        sharing). See models.ConvBranch_Equivariant."""
        cf = AttrDict()
        cf.kernel_set = "smallkernels"
        cf.equivariant = False
        set_kernels(cf)
        assert len(cf.kernels) == 10

        cf2 = AttrDict()
        cf2.kernel_set = "smallkernels"
        cf2.equivariant = True
        set_kernels(cf2)
        assert len(cf2.kernels) == 5
        for k in cf2.kernels:
            assert len(k) == 5  # (shape, n_filters, dilation, mask, stride) after set_kernels

    def test_defaultkernels(self):
        cf = AttrDict()
        cf.kernel_set = "defaultkernels"
        set_kernels(cf)
        assert len(cf.kernels) == 9
        for k in cf.kernels:
            assert len(k) == 5
        # NOTE (characterization, not a claim about intent): unlike smallkernels,
        # defaultkernels' area sequence is NOT monotonic: 1,2,2,4,2,2,3,3,9. The
        # (2,1)/(1,2) dilated-by-2 kernels (area 2) come *after* the (2,2) kernel
        # (area 4) in list order. The nested-pattern narrative (see docs/CONFIGURATION.md,
        # "Regularization and the L1 penalty") still holds by *penalty magnitude*
        # (get_branch_penalties indexes by area, not list position) but not by list
        # ordering, for this kernel set specifically.
        areas = [k[0][0] * k[0][1] for k in cf.kernels]
        assert areas == [1, 2, 2, 4, 2, 2, 3, 3, 9]
        assert areas != sorted(areas)

    def test_bigkernels_paris_ising(self):
        cf = AttrDict()
        cf.kernel_set = "bigkernels"
        cf.dataset = "Paris_Ising"
        set_kernels(cf)
        assert len(cf.kernels) == 4
        for k in cf.kernels:
            assert len(k) == 5
        areas = [k[0][0] * k[0][1] for k in cf.kernels]
        assert areas == sorted(areas)  # 1, 4, 16, 64 -> monotonic

    def test_bigkernels_paris_xy(self):
        cf = AttrDict()
        cf.kernel_set = "bigkernels"
        cf.dataset = "Paris_XY_Z"
        set_kernels(cf)
        assert len(cf.kernels) == 4

    def test_bigkernels_ilgt(self):
        cf = AttrDict()
        cf.kernel_set = "bigkernels"
        cf.dataset = "ILGT"
        set_kernels(cf)
        assert len(cf.kernels) == 5

    def test_kernels_already_5_long_are_not_double_appended(self):
        # compatibility-fix loop only appends stride if len(k) < 5; run set_kernels
        # twice on the same cf (as load_or_init_config's compat-fix loop does
        # elsewhere) and confirm entries stay 5-long, not 6.
        cf = AttrDict()
        cf.kernel_set = "defaultkernels"
        set_kernels(cf)
        first_pass = [list(k) for k in cf.kernels]
        for k in range(len(cf.kernels)):
            if len(cf.kernels[k]) < 5:
                cf.kernels[k].append(1)
        assert [list(k) for k in cf.kernels] == first_pass
        assert all(len(k) == 5 for k in cf.kernels)


# ---------------------------------------------------------------------------
# 7. AttrDict
# ---------------------------------------------------------------------------

class TestAttrDict:
    def test_attribute_set_and_get(self):
        d = AttrDict()
        d.foo = 42
        assert d["foo"] == 42
        assert d.foo == 42

    def test_dict_style_access_also_works(self):
        d = AttrDict()
        d["bar"] = "baz"
        assert d.bar == "baz"

    def test_is_a_real_dict(self):
        d = AttrDict()
        d.a, d.b = 1, 2
        assert isinstance(d, dict)
        assert set(d.keys()) == {"a", "b"}
        assert "a" in d

    def test_missing_attribute_raises_attributeerror_not_keyerror(self):
        # AttrDict.__getattr__ re-raises a missing-key KeyError as AttributeError,
        # so attribute access on a missing key behaves like normal Python attribute
        # access (and, in particular, is catchable by hasattr()/getattr()-with-default).
        d = AttrDict()
        with pytest.raises(AttributeError):
            d.nonexistent

    def test_hasattr_on_missing_key_returns_false(self):
        # hasattr() only suppresses AttributeError; now that AttrDict raises
        # AttributeError (not KeyError) for a missing key, hasattr(attrdict_instance,
        # "missing_key") correctly returns False, as callers throughout the codebase
        # (build_logdir_path, train.py, etc.) assume. See test_paths.py for a
        # concrete instance of this (cf.model absent).
        d = AttrDict()
        assert hasattr(d, "missing_key") is False

    def test_getattr_with_default_on_missing_key_returns_default(self):
        # Same root cause fixed: getattr(obj, name, default) falls back to `default`
        # on AttributeError, which AttrDict now raises for a missing key.
        d = AttrDict()
        assert getattr(d, "missing_key", "some_default") == "some_default"

    def test_missing_key_still_raises_keyerror_on_dict_style_access(self):
        # Dict-style access (d["missing_key"]) is untouched by the __getattr__ fix;
        # it still raises the normal dict KeyError.
        d = AttrDict()
        with pytest.raises(KeyError):
            d["missing_key"]

    def test_delattr_removes_key(self):
        d = AttrDict()
        d.foo = 42
        del d.foo
        assert "foo" not in d
        with pytest.raises(AttributeError):
            d.foo

    def test_delattr_on_missing_key_raises_attributeerror(self):
        d = AttrDict()
        with pytest.raises(AttributeError):
            del d.nonexistent
