"""Group E: device placement (tetriscnn/models.py).

These tests exist because of a class of bug that is invisible on a CPU-only machine:
a module whose buffers live on a different device than its parameters. Such a module
builds and even trains fine as long as nothing touches the buffer before the whole
net is `.to(DEVICE)`'d, and then fails with

    RuntimeError: Expected all tensors to be on the same device,
    but found at least two devices, cuda:0 and cpu!

the moment something does. ConvBranch_Equivariant uses its mask at construction time
(to zero the masked-off weights once), which is exactly such a moment.

The accelerator-marked tests below skip on CPU-only machines and run for real on any
box with cuda or mps, which is the only place this class of defect is observable.
"""
import pytest
import torch

from tetriscnn.models import (
    ConvBranch_Equivariant,
    ConvBranch_twoLayer,
    ShapeAdaptiveConvNet,
)
from tetriscnn.utils import AttrDict, set_kernels


def _accelerators():
    """Every non-CPU device actually usable on this machine."""
    devs = []
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


ACCELERATORS = _accelerators()
requires_accelerator = pytest.mark.skipif(
    not ACCELERATORS, reason="no cuda/mps device available on this machine"
)

DIAGONAL_MASK = [[1, 0], [0, 1]]


def _kernels(equivariant, group="C4"):
    cf = AttrDict()
    cf.kernel_set = "smallkernels"
    cf.dataset = "Paris_Ising"
    cf.equivariant = equivariant
    if equivariant:
        cf.equivariant_group = group
    set_kernels(cf)
    return cf.kernels


def _net(device, equivariant, group="C4", hidden_size=8):
    kwargs = dict(
        in_channels=1,
        kernels=_kernels(equivariant, group),
        device=device,
        equivariant=equivariant,
        hidden_size=hidden_size,
        init="kaiming",
    )
    if equivariant:
        kwargs["equivariant_group"] = group
    return ShapeAdaptiveConvNet(**kwargs)


class TestMaskBufferDevicePlacement:
    """A branch's mask buffer must sit on the same device as that branch's own
    parameters, at construction time, whatever device the caller names.
    """

    @pytest.mark.parametrize("equivariant,group", [(False, None), (True, "C4"), (True, "D2")])
    def test_cpu_mask_and_params_agree(self, equivariant, group):
        net = _net("cpu", equivariant, group or "C4")
        for branch in net.branches:
            if branch.mask is not None:
                assert branch.mask.device == branch.conv1.weight.device

    @requires_accelerator
    @pytest.mark.accelerator
    @pytest.mark.parametrize("device", ACCELERATORS)
    @pytest.mark.parametrize("equivariant,group", [(False, None), (True, "C4"), (True, "D2")])
    def test_accelerator_mask_and_params_agree_before_moving_net(self, device, equivariant, group):
        """The regression test proper.

        Constructing with device="cuda" used to hand each masked branch a cuda mask
        while its conv layers were still on cpu. ConvBranch_Equivariant then multiplied
        the two together and raised. Note the net is deliberately NOT .to(device)'d
        here: the point is that construction alone must leave a coherent module.
        """
        net = _net(device, equivariant, group or "C4")
        for branch in net.branches:
            if branch.mask is not None:
                assert branch.mask.device == branch.conv1.weight.device

    @requires_accelerator
    @pytest.mark.accelerator
    @pytest.mark.parametrize("device", ACCELERATORS)
    def test_branch_accepts_a_mask_handed_in_on_another_device(self, device):
        """Unit-level version: a caller-supplied on-device mask must be adopted, not
        stored as-is, since the branch's own parameters are created on cpu."""
        mask = torch.tensor(DIAGONAL_MASK, dtype=torch.float32, device=device)
        branch = ConvBranch_Equivariant(
            in_channels=1, kernel_shape=(2, 2), number_of_filters=1, hidden_size=4,
            mask=mask, group="C4",
        )
        assert branch.mask.device == branch.conv1.weight.device
        # and the construction-time zeroing still actually happened
        assert torch.count_nonzero(branch.conv1.weight * (1 - branch.mask)) == 0

    @requires_accelerator
    @pytest.mark.accelerator
    @pytest.mark.parametrize("device", ACCELERATORS)
    def test_two_layer_branch_also_adopts_mask_device(self, device):
        mask = torch.tensor(DIAGONAL_MASK, dtype=torch.float32, device=device)
        branch = ConvBranch_twoLayer(
            in_channels=1, kernel_shape=(2, 2), number_of_filters=1, hidden_size=4,
            mask=mask,
        )
        assert branch.mask.device == branch.conv1.weight.device


class TestWholeNetIsDeviceCoherent:
    @pytest.mark.parametrize("equivariant,group", [(False, None), (True, "C4"), (True, "D2")])
    def test_all_params_and_buffers_share_one_device_after_to(self, equivariant, group):
        net = _net("cpu", equivariant, group or "C4").to("cpu")
        devices = {p.device for p in net.parameters()} | {b.device for b in net.buffers()}
        assert len(devices) == 1

    @requires_accelerator
    @pytest.mark.accelerator
    @pytest.mark.parametrize("device", ACCELERATORS)
    @pytest.mark.parametrize("equivariant,group", [(False, None), (True, "C4"), (True, "D2")])
    def test_all_params_and_buffers_share_one_device_on_accelerator(self, device, equivariant, group):
        net = _net(device, equivariant, group or "C4").to(device)
        devices = {p.device.type for p in net.parameters()} | {b.device.type for b in net.buffers()}
        assert devices == {torch.device(device).type}


@requires_accelerator
@pytest.mark.accelerator
class TestTrainingStepOnAccelerator:
    """End-to-end: build on the accelerator, forward, backward, optimizer step."""

    @pytest.mark.parametrize("device", ACCELERATORS)
    @pytest.mark.parametrize("equivariant,group,n_branches", [
        (False, None, 10), (True, "C4", 5), (True, "D2", 6),
    ])
    def test_forward_backward_step(self, device, equivariant, group, n_branches):
        net = _net(device, equivariant, group or "C4").to(device)
        x = torch.randn(4, 1, 8, 8, device=device)
        z = net(x)
        assert z.shape == (4, n_branches)
        assert z.device.type == torch.device(device).type

        opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-5)
        net(x).pow(2).mean().backward()
        opt.step()

    @pytest.mark.parametrize("device", ACCELERATORS)
    def test_c4_invariance_holds_on_accelerator(self, device):
        """The invariance is the whole point of the branch; assert it survives the
        move to a real accelerator (float32 there is not bit-identical to cpu)."""
        net = _net(device, True, "C4").to(device)
        net.eval()
        x = torch.randn(4, 1, 8, 8, device=device)
        with torch.no_grad():
            z = net(x)
            z_rot = net(torch.rot90(x, 1, dims=(2, 3)))
        assert torch.allclose(z, z_rot, atol=1e-5)

    @pytest.mark.parametrize("device", ACCELERATORS)
    def test_masked_weights_stay_zero_on_accelerator(self, device):
        net = _net(device, True, "C4").to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-5)
        x = torch.randn(8, 1, 8, 8, device=device)
        for _ in range(5):
            opt.zero_grad()
            net(x).pow(2).mean().backward()
            opt.step()
        for branch in net.branches:
            if branch.mask is not None:
                assert torch.count_nonzero(branch.conv1.weight * (1 - branch.mask)) == 0
