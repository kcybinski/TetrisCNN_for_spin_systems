"""Mixed equivariant / non-equivariant branches (cf.kernel_set='smallkernels_mixed').

Characterization tests for the per-branch `branch_groups` mechanism and the
`equivariant_rebate` penalty multiplier. The overriding requirement is that the
DEFAULT path is untouched: every assertion about "no branch_groups" here pins the
historical behaviour so the mixed feature cannot leak into ordinary runs.
"""
import numpy as np
import pytest
import torch

from tetriscnn.models import (
    ShapeAdaptiveConvNet, ConvBranch_twoLayer, ConvBranch_Equivariant,
)
from tetriscnn.utils import AttrDict, get_branch_penalties, set_kernels, set_seeds


def mixed_cf(group="C4", dataset="Paris_Ising"):
    cf = AttrDict(dict(kernel_set="smallkernels_mixed", dataset=dataset,
                       equivariant=False, mixed_group=group,
                       hidden_size=8, init="kaiming"))
    set_kernels(cf)
    return cf


class TestMixedKernelSet:
    @pytest.mark.parametrize("group,n_eq", [("C4", 5), ("D2", 6), ("K4", 6)])
    def test_shape_and_groups(self, group, n_eq):
        cf = mixed_cf(group)
        assert len(cf.kernels) == 10 + n_eq
        assert cf.branch_groups[:10] == [None] * 10
        assert cf.branch_groups[10:] == [group] * n_eq

    @pytest.mark.parametrize("group", ["C4", "D2"])
    def test_orbits_cover_the_non_equivariant_patterns_exactly(self, group):
        # This is what makes the two halves a fair comparison: the equivariant
        # representatives stand in for precisely the ten ordinary patterns.
        cf = mixed_cf(group)
        assert sum(cf.branch_orbit_sizes[10:]) == 10

    def test_equivariant_flag_is_rejected(self):
        # Setting both would make EVERY branch equivariant and destroy the mixture.
        cf = AttrDict(dict(kernel_set="smallkernels_mixed", dataset="Paris_Ising",
                           equivariant=True))
        with pytest.raises(ValueError, match="must stay False"):
            set_kernels(cf)

    def test_unknown_group_raises(self):
        cf = AttrDict(dict(kernel_set="smallkernels_mixed", dataset="Paris_Ising",
                           equivariant=False, mixed_group="C3"))
        with pytest.raises(ValueError, match="Unknown cf.mixed_group"):
            set_kernels(cf)

    def test_other_kernel_sets_leave_branch_groups_none(self):
        # A cf reused across kernel sets must not carry a stale mixture forward.
        cf = AttrDict(dict(kernel_set="smallkernels_mixed", dataset="Paris_Ising",
                           equivariant=False))
        set_kernels(cf)
        assert cf.branch_groups is not None
        cf.kernel_set = "smallkernels"
        set_kernels(cf)
        assert cf.branch_groups is None
        assert cf.branch_orbit_sizes is None


class TestMixedModel:
    def test_branch_classes_follow_branch_groups(self):
        cf = mixed_cf()
        net = ShapeAdaptiveConvNet(in_channels=1, kernels=cf.kernels, device="cpu",
                                   branch_groups=cf.branch_groups, hidden_size=8)
        for br, g in zip(net.branches, cf.branch_groups):
            expected = ConvBranch_Equivariant if g else ConvBranch_twoLayer
            assert isinstance(br, expected)

    def test_equivariant_branches_are_invariant_and_others_are_not(self):
        cf = mixed_cf()
        set_seeds(0)
        net = ShapeAdaptiveConvNet(in_channels=1, kernels=cf.kernels, device="cpu",
                                   branch_groups=cf.branch_groups, hidden_size=8)
        x = torch.sign(torch.randn(32, 1, 8, 8))
        with torch.no_grad():
            d = np.abs(net(x).numpy() - net(torch.rot90(x, 1, dims=[2, 3])).numpy()).max(0)
        for k, g in enumerate(cf.branch_groups):
            if g is not None:
                assert d[k] < 1e-5, f"equivariant branch {k} is not C4 invariant"
        # The ordinary branches must genuinely differ -- except the 1x1, which is a
        # fixed point of the group action and so is trivially invariant either way.
        non_trivial = [k for k, g in enumerate(cf.branch_groups)
                       if g is None and tuple(cf.kernels[k][0]) != (1, 1)]
        assert all(d[k] > 1e-4 for k in non_trivial)

    def test_the_1x1_pair_is_exactly_degenerate(self):
        # rot90 of a 1x1 kernel is the identity, so the orbit average collapses back
        # to the plain branch: given identical weights the two columns agree EXACTLY.
        # This is why the survey excludes the 1x1 pair from its preference metric.
        cf = mixed_cf()
        set_seeds(0)
        net = ShapeAdaptiveConvNet(in_channels=1, kernels=cf.kernels, device="cpu",
                                   branch_groups=cf.branch_groups, hidden_size=8)
        ne, eq = net.branches[0], net.branches[10]
        with torch.no_grad():
            for a, b in ((eq.conv1, ne.conv1), (eq.conv2, ne.conv2)):
                a.weight.copy_(b.weight)
                a.bias.copy_(b.bias)
            z = net(torch.sign(torch.randn(32, 1, 8, 8))).numpy()
        assert np.abs(z[:, 0] - z[:, 10]).max() == 0.0

    def test_length_mismatch_raises(self):
        cf = mixed_cf()
        with pytest.raises(ValueError, match="one-to-one"):
            ShapeAdaptiveConvNet(in_channels=1, kernels=cf.kernels, device="cpu",
                                 branch_groups=cf.branch_groups[:-1], hidden_size=8)

    @pytest.mark.parametrize("equivariant", [False, True])
    def test_uniform_nets_are_unchanged_by_the_feature(self, equivariant):
        cf = AttrDict(dict(kernel_set="smallkernels", dataset="Paris_Ising",
                           equivariant=equivariant, equivariant_group="C4"))
        set_kernels(cf)
        set_seeds(0)
        net = ShapeAdaptiveConvNet(in_channels=1, kernels=cf.kernels, device="cpu",
                                   equivariant=equivariant, hidden_size=8)
        expected = ConvBranch_Equivariant if equivariant else ConvBranch_twoLayer
        assert all(isinstance(b, expected) for b in net.branches)
        assert net.branch_model is expected


class TestRebate:
    PP = [10, -5, 1, 1]

    def test_no_rebate_matches_the_historical_signature(self):
        cf = mixed_cf()
        base = get_branch_penalties(self.PP, cf.kernels)
        passthrough = get_branch_penalties(self.PP, cf.kernels,
                                           branch_groups=cf.branch_groups,
                                           equivariant_rebate=None)
        np.testing.assert_array_equal(base, passthrough)
        np.testing.assert_array_equal(
            base, get_branch_penalties(self.PP, cf.kernels,
                                       branch_groups=cf.branch_groups,
                                       equivariant_rebate=1.0))

    @pytest.mark.parametrize("r", [0.25, 0.5, 2.0, 8.0])
    def test_uniform_rebate_hits_only_the_equivariant_half(self, r):
        cf = mixed_cf()
        base = get_branch_penalties(self.PP, cf.kernels)
        got = get_branch_penalties(self.PP, cf.kernels,
                                   branch_groups=cf.branch_groups,
                                   equivariant_rebate=r)
        np.testing.assert_allclose(got[:10], base[:10])
        np.testing.assert_allclose(got[10:], base[10:] * r)

    def test_inverse_orbit_divides_by_orbit_size(self):
        cf = mixed_cf()
        base = get_branch_penalties(self.PP, cf.kernels)
        got = get_branch_penalties(self.PP, cf.kernels,
                                   branch_groups=cf.branch_groups,
                                   equivariant_rebate="inverse_orbit",
                                   branch_orbit_sizes=cf.branch_orbit_sizes)
        np.testing.assert_allclose(got[:10], base[:10])
        expected = base[10:] / np.array(cf.branch_orbit_sizes[10:], dtype=float)
        np.testing.assert_allclose(got[10:], expected)

    def test_area_keying_is_group_invariant(self):
        # At r=1 an equivariant branch and its ordinary orientation partners are
        # charged the SAME lambda -- the rebate is the only thing separating them.
        cf = mixed_cf()
        pen = get_branch_penalties(self.PP, cf.kernels,
                                   branch_groups=cf.branch_groups,
                                   equivariant_rebate=1.0)
        assert pen[11] == pytest.approx(pen[1])   # EQ (2,1) vs NE (2,1)
        assert pen[11] == pytest.approx(pen[2])   # ... and NE (1,2)
        assert pen[13] == pytest.approx(pen[5])   # EQ tromino vs an NE tromino

    def test_rebate_without_groups_raises(self):
        cf = mixed_cf()
        with pytest.raises(ValueError, match="without branch_groups"):
            get_branch_penalties(self.PP, cf.kernels, equivariant_rebate=0.5)

    def test_inverse_orbit_without_sizes_raises(self):
        cf = mixed_cf()
        with pytest.raises(ValueError, match="needs branch_orbit_sizes"):
            get_branch_penalties(self.PP, cf.kernels,
                                 branch_groups=cf.branch_groups,
                                 equivariant_rebate="inverse_orbit")
