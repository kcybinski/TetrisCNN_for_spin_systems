"""Group A-style: pure, data-free tests for the C4-equivariant branch
(tetriscnn.models.ConvBranch_Equivariant) and its wiring into ShapeAdaptiveConvNet
and set_kernels. The construction is Eq. (6) of the manuscript; the class docstring
gives the implementation details these tests check.

No data/ needed; must stay fast (a few seconds).
"""
import numpy as np
import pytest
import torch

from tetriscnn.models import ConvBranch_Equivariant, ConvBranch_twoLayer, ShapeAdaptiveConvNet
from tetriscnn.utils import AttrDict, set_kernels

torch.manual_seed(0)

# The 5 canonical C4 representatives (must match tetriscnn.utils.set_kernels'
# smallkernels/equivariant=True branch).
CANONICAL_SMALLKERNELS = [
    [(1, 1), 1, 1, None],
    [(2, 1), 1, 1, None],
    [(2, 2), 1, 1, [[1, 0], [0, 1]]],
    [(2, 2), 1, 1, [[1, 1], [1, 0]]],
    [(2, 2), 1, 1, None],
]


def _make_mask(mask_spec):
    if mask_spec is None:
        return None
    return torch.tensor(np.array(mask_spec), dtype=torch.float32)


def _make_branch(kernel_spec, in_channels=1, hidden_size=6):
    kernel_shape, n_filters, dilation, mask_spec = kernel_spec
    return ConvBranch_Equivariant(
        in_channels=in_channels,
        kernel_shape=kernel_shape,
        number_of_filters=n_filters,
        hidden_size=hidden_size,
        dilation=dilation,
        stride=1,
        mask=_make_mask(mask_spec),
        init="kaiming",
    )


# ---------------------------------------------------------------------------
# 1. C4 invariance -- the correctness gate.
# ---------------------------------------------------------------------------

class TestC4Invariance:
    """A 90-degree rotation of a rectangular image changes its shape, so invariance
    is only literally testable on a SQUARE input tensor; we use an 8x8 +/-1 random
    spin configuration on purpose."""

    @pytest.mark.parametrize("kernel_spec", CANONICAL_SMALLKERNELS)
    @pytest.mark.parametrize("channels", [1, 2])
    def test_invariant_to_rotation(self, kernel_spec, channels):
        branch = _make_branch(kernel_spec, in_channels=channels)
        branch.eval()

        x = torch.randint(0, 2, (8, channels, 8, 8), dtype=torch.float32) * 2 - 1  # +/-1 spins

        with torch.no_grad():
            z0 = branch(x)
            max_abs_diff = 0.0
            for k in (1, 2, 3):
                x_rot = torch.rot90(x, k=k, dims=(2, 3))
                z_rot = branch(x_rot)
                diff = (z0 - z_rot).abs().max().item()
                max_abs_diff = max(max_abs_diff, diff)
                assert torch.allclose(z0, z_rot, atol=1e-4), (
                    f"kernel={kernel_spec}, channels={channels}, k={k}: "
                    f"max abs diff {diff:.3e} exceeds tolerance 1e-4"
                )


# ---------------------------------------------------------------------------
# 2. Weight-sharing sanity: parameter count independent of |group|.
# ---------------------------------------------------------------------------

class TestWeightSharing:
    @pytest.mark.parametrize("kernel_spec", CANONICAL_SMALLKERNELS)
    def test_param_count_matches_two_layer_branch(self, kernel_spec):
        kernel_shape, n_filters, dilation, mask_spec = kernel_spec
        hidden_size = 6

        equiv = _make_branch(kernel_spec, in_channels=1, hidden_size=hidden_size)
        plain = ConvBranch_twoLayer(
            in_channels=1,
            kernel_shape=kernel_shape,
            number_of_filters=n_filters,
            hidden_size=hidden_size,
            dilation=dilation,
            stride=1,
            mask=_make_mask(mask_spec),
            init="kaiming",
        )

        n_equiv = sum(p.numel() for p in equiv.parameters())
        n_plain = sum(p.numel() for p in plain.parameters())
        assert n_equiv == n_plain, (
            f"kernel={kernel_spec}: equivariant branch has {n_equiv} params, "
            f"plain two-layer branch has {n_plain} (only one conv1 + one conv2 "
            f"should be held regardless of the group order)"
        )
        # exactly one conv1 + one conv2 module (i.e. no per-rotation weight copies):
        # weight+bias for each of the two convs, four tensors total.
        param_names = {name for name, _ in equiv.named_parameters()}
        assert param_names == {"conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"}


# ---------------------------------------------------------------------------
# 3. Net-level forward on the real XY image shape (6, 7) after the reshape fix.
# ---------------------------------------------------------------------------

class TestShapeAdaptiveConvNetEquivariant:
    def test_forward_shape(self):
        cf = AttrDict()
        cf.kernel_set = "smallkernels"
        cf.equivariant = True
        set_kernels(cf)
        assert len(cf.kernels) == 5

        net = ShapeAdaptiveConvNet(
            in_channels=2,
            kernels=cf.kernels,
            device="cpu",
            equivariant=True,
            hidden_size=8,
            init="kaiming",
        )

        x = torch.randn(4, 2, 6, 7)
        out = net(x)
        assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# 4. D2/K4 (rectangle symmetry) invariance -- the correctness gate for the
#    Klein four-group branch. Unlike C4, D2's flips are shape-preserving, so
#    this is directly testable on a RECTANGULAR input (the real XY (6,7) shape).
# ---------------------------------------------------------------------------

# The 6 canonical D2/K4 representatives (must match tetriscnn.utils.set_kernels'
# smallkernels/equivariant=True+equivariant_group="D2" branch).
CANONICAL_SMALLKERNELS_D2 = [
    [(1, 1), 1, 1, None],
    [(2, 1), 1, 1, None],
    [(1, 2), 1, 1, None],
    [(2, 2), 1, 1, [[1, 0], [0, 1]]],
    [(2, 2), 1, 1, [[1, 1], [1, 0]]],
    [(2, 2), 1, 1, None],
]


class TestD2Invariance:
    """D2 = K4 = the symmetry group of a non-square rectangle: identity, 180-deg
    rotation, horizontal mirror, vertical mirror. All four are shape-preserving, so
    invariance is testable directly on a rectangular (6,7) input (unlike C4's
    90-degree rotation, which changes shape)."""

    @pytest.mark.parametrize("kernel_spec", CANONICAL_SMALLKERNELS_D2)
    @pytest.mark.parametrize("channels", [1, 2])
    def test_invariant_to_flips(self, kernel_spec, channels):
        kernel_shape, n_filters, dilation, mask_spec = kernel_spec
        branch = ConvBranch_Equivariant(
            in_channels=channels,
            kernel_shape=kernel_shape,
            number_of_filters=n_filters,
            hidden_size=6,
            dilation=dilation,
            stride=1,
            mask=_make_mask(mask_spec),
            init="kaiming",
            group="D2",
        )
        branch.eval()

        x = torch.randint(0, 2, (6, channels, 6, 7), dtype=torch.float32) * 2 - 1  # +/-1 spins

        with torch.no_grad():
            z0 = branch(x)
            max_abs_diff = 0.0
            for dims in ([2], [3], [2, 3]):
                x_flipped = torch.flip(x, dims=dims)
                z_flipped = branch(x_flipped)
                diff = (z0 - z_flipped).abs().max().item()
                max_abs_diff = max(max_abs_diff, diff)
                print(
                    f"kernel={kernel_spec}, channels={channels}, flip dims={dims}: "
                    f"max abs diff {diff:.3e}"
                )
                assert torch.allclose(z0, z_flipped, atol=1e-4), (
                    f"kernel={kernel_spec}, channels={channels}, flip dims={dims}: "
                    f"max abs diff {diff:.3e} exceeds tolerance 1e-4"
                )


# ---------------------------------------------------------------------------
# 5. Unknown group string raises a clear ValueError.
# ---------------------------------------------------------------------------

class TestUnknownGroup:
    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            ConvBranch_Equivariant(
                in_channels=1,
                kernel_shape=(2, 2),
                number_of_filters=1,
                hidden_size=6,
                group="D4",
            )


# ---------------------------------------------------------------------------
# 6. set_kernels: group-dependent smallkernels branch counts.
# ---------------------------------------------------------------------------

class TestSetKernelsGroup:
    def test_smallkernels_branch_counts_by_group(self):
        cf_c4 = AttrDict()
        cf_c4.kernel_set = "smallkernels"
        cf_c4.equivariant = True
        cf_c4.equivariant_group = "C4"
        set_kernels(cf_c4)
        assert len(cf_c4.kernels) == 5

        cf_d2 = AttrDict()
        cf_d2.kernel_set = "smallkernels"
        cf_d2.equivariant = True
        cf_d2.equivariant_group = "D2"
        set_kernels(cf_d2)
        assert len(cf_d2.kernels) == 6

        cf_k4 = AttrDict()
        cf_k4.kernel_set = "smallkernels"
        cf_k4.equivariant = True
        cf_k4.equivariant_group = "K4"
        set_kernels(cf_k4)
        assert len(cf_k4.kernels) == 6

        cf_false = AttrDict()
        cf_false.kernel_set = "smallkernels"
        cf_false.equivariant = False
        set_kernels(cf_false)
        assert len(cf_false.kernels) == 10


# ---------------------------------------------------------------------------
# 7. Net-level forward with the D2 kernel set on the real XY image shape (6, 7).
# ---------------------------------------------------------------------------

class TestShapeAdaptiveConvNetD2:
    def test_forward_shape(self):
        cf = AttrDict()
        cf.kernel_set = "smallkernels"
        cf.equivariant = True
        cf.equivariant_group = "D2"
        set_kernels(cf)
        assert len(cf.kernels) == 6

        net = ShapeAdaptiveConvNet(
            in_channels=2,
            kernels=cf.kernels,
            device="cpu",
            equivariant=True,
            equivariant_group="D2",
            hidden_size=8,
            init="kaiming",
        )

        x = torch.randn(4, 2, 6, 7)
        out = net(x)
        assert out.shape == (4, 6)


# ---------------------------------------------------------------------------
# 8. Collapse check: the equivariant branch reproduces the group-pool of the
#    separate oriented branches it replaces.
# ---------------------------------------------------------------------------

class TestCollapseToOrbitPool:
    """The point of the feature: one weight-tied branch should compute exactly the
    mean over the group of what the corresponding separately-masked plain branches
    would compute from the same (transformed) weights. This is what licenses reading
    the 5 canonical C4 branches as a collapse of the 10 non-equivariant ones."""

    @pytest.mark.parametrize("kernel_spec", CANONICAL_SMALLKERNELS)
    def test_equals_mean_over_orbit_of_plain_branches(self, kernel_spec):
        kernel_shape, n_filters, dilation, mask_spec = kernel_spec
        equiv = _make_branch(kernel_spec, in_channels=1, hidden_size=6).double().eval()

        x = torch.randint(0, 2, (8, 1, 8, 8), dtype=torch.float64) * 2 - 1

        W = equiv.conv1.weight if equiv.mask is None else equiv.conv1.weight * equiv.mask
        outs = []
        with torch.no_grad():
            for r in range(4):
                W_rot = torch.rot90(W, r, (2, 3))
                plain = ConvBranch_twoLayer(
                    in_channels=1,
                    kernel_shape=tuple(W_rot.shape[2:]),
                    number_of_filters=n_filters,
                    hidden_size=6,
                    mask=None,
                ).double()
                plain.conv1.weight.data.copy_(W_rot)
                plain.conv1.bias.data.copy_(equiv.conv1.bias)
                plain.conv2.weight.data.copy_(equiv.conv2.weight)
                plain.conv2.bias.data.copy_(equiv.conv2.bias)
                outs.append(plain(x))
            pooled = torch.stack(outs, dim=0).mean(dim=0)
            diff = (equiv(x) - pooled).abs().max().item()

        assert diff < 1e-12, (
            f"kernel={kernel_spec}: equivariant activation differs from the C4 pool of "
            f"the equivalent plain branches by {diff:.3e}"
        )


# ---------------------------------------------------------------------------
# 9. Masked entries of conv1.weight are zeroed in the STORED parameter, not just
#    masked at forward time.
# ---------------------------------------------------------------------------

class TestMaskedWeightsAreZeroed:
    """forward() masks out of place, so masked entries get zero gradient and never
    move. They must therefore start at zero, or they keep their init values for the
    whole run: plots.plot_weights renders conv1.weight on a shared colour scale taken
    from the data, and cf.weight_penalty sums the L1 of every conv1 parameter."""

    def test_zero_at_init_and_after_training(self):
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        branch = ConvBranch_Equivariant(
            in_channels=1, kernel_shape=(2, 2), number_of_filters=1,
            hidden_size=4, mask=mask,
        )
        assert branch.conv1.weight[:, :, 1, 1].abs().max().item() == 0.0

        opt = torch.optim.AdamW(branch.parameters(), lr=1e-2, weight_decay=1e-5)
        x = torch.randn(8, 1, 6, 6)
        for _ in range(20):
            opt.zero_grad()
            branch(x).sum().backward()
            opt.step()

        assert branch.conv1.weight[:, :, 1, 1].abs().max().item() == 0.0, (
            "masked conv1 entries drifted away from zero during training"
        )
        assert branch.conv1.weight.grad[:, :, 1, 1].abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# 10. Settings that would silently break the invariance are rejected, not ignored.
# ---------------------------------------------------------------------------

class TestUnsupportedSettingsRaise:
    @pytest.mark.parametrize("kwargs", [
        pytest.param(dict(stride=2), id="stride"),
        pytest.param(dict(dilation=(2, 1)), id="anisotropic-dilation"),
        pytest.param(dict(padding_size=1), id="padding-size"),
        pytest.param(dict(padding_mode="circular"), id="padding-mode"),
    ])
    def test_raises(self, kwargs):
        with pytest.raises(NotImplementedError):
            ConvBranch_Equivariant(
                in_channels=1, kernel_shape=(2, 2), number_of_filters=1,
                hidden_size=4, **kwargs
            )

    def test_isotropic_dilation_is_allowed_and_stays_invariant(self):
        """Scalar dilation rotates with the kernel, so it is safe; only anisotropic
        dilation is rejected."""
        branch = ConvBranch_Equivariant(
            in_channels=1, kernel_shape=(2, 2), number_of_filters=1,
            hidden_size=6, dilation=2,
        ).double().eval()
        x = torch.randint(0, 2, (8, 1, 8, 8), dtype=torch.float64) * 2 - 1
        with torch.no_grad():
            z0 = branch(x)
            diff = max((z0 - branch(torch.rot90(x, k, (2, 3)))).abs().max().item()
                       for k in (1, 2, 3))
        assert diff < 1e-12


class TestEquivariantKernelSetGuard:
    @pytest.mark.parametrize("kernel_set", ["defaultkernels", "bigkernels", "bigkernels_stride2"])
    def test_non_smallkernels_raises(self, kernel_set):
        cf = AttrDict()
        cf.kernel_set = kernel_set
        cf.equivariant = True
        cf.dataset = "Paris_Ising"
        with pytest.raises(NotImplementedError):
            set_kernels(cf)

    def test_non_smallkernels_still_fine_when_not_equivariant(self):
        cf = AttrDict()
        cf.kernel_set = "defaultkernels"
        cf.equivariant = False
        cf.dataset = "Paris_Ising"
        set_kernels(cf)
        assert len(cf.kernels) == 9

    def test_unknown_group_raises_in_set_kernels(self):
        cf = AttrDict()
        cf.kernel_set = "smallkernels"
        cf.equivariant = True
        cf.equivariant_group = "D4"
        with pytest.raises(ValueError):
            set_kernels(cf)
