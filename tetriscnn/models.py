import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeAdaptiveConvNet(nn.Module):
    def __init__(
        self, in_channels, kernels, device, padding_size=0, padding_mode=None, equivariant=False,
        equivariant_group="C4", hidden_size=32, init="kaiming", branch_groups=None,
    ):
        super().__init__()

        # Equivariant convolutional branches (ConvBranch_Equivariant, hand-rolled C4
        # weight-tied group conv, Eq. (6) of the manuscript) replace the old
        # gcnn/escnn-backed lineage (ConvBranch_Equivariant2), which remains
        # discontinued. equivariant=False (the default) is fully unchanged.
        self.in_channels = in_channels
        self.kernels = kernels
        self.device = device
        self.branches = nn.ModuleList()

        # Per-branch group assignment. `branch_groups`, when given, is a list with one
        # entry per kernel: None for an ordinary ConvBranch_twoLayer, or a group name
        # ("C4", "D2"/"K4") for a ConvBranch_Equivariant. It lets a single net MIX both
        # branch types, which the uniform `equivariant` flag cannot express -- that is
        # what cf.kernel_set = "smallkernels_mixed" is built on. When branch_groups is
        # None (the default), this reduces exactly to the old uniform behaviour and the
        # `equivariant` / `equivariant_group` arguments alone decide the branch class.
        if branch_groups is not None:
            if len(branch_groups) != len(self.kernels):
                raise ValueError(
                    f"branch_groups has {len(branch_groups)} entries but there are "
                    f"{len(self.kernels)} kernels; they must correspond one-to-one."
                )
            self.branch_groups = [g if g else None for g in branch_groups]
            self.equivariant_group = equivariant_group
            # branch_model stays meaningful only for a uniform net; a mixed net has no
            # single branch class, so leave it None rather than lie about it.
            self.branch_model = None
            n_eq = sum(g is not None for g in self.branch_groups)
            groups_used = sorted({g for g in self.branch_groups if g is not None})
            print(
                f"\nUsing MIXED convolutional branches: "
                f"{len(self.branch_groups) - n_eq} standard, "
                f"{n_eq} equivariant ({'/'.join(groups_used) or 'none'})."
            )
        elif equivariant:
            self.branch_model = ConvBranch_Equivariant
            self.equivariant_group = equivariant_group
            self.branch_groups = [equivariant_group] * len(self.kernels)
            print(f"\nUsing {equivariant_group}-equivariant convolutional branches.")
        else:
            self.branch_model = ConvBranch_twoLayer
            self.branch_groups = [None] * len(self.kernels)
            print("\nUsing standard 2-layer convolutional branches.")
            print("Note: init is unused for standard convolutions.")

        for k in range(len(self.kernels)):
            # try:
            branch_kwargs = dict(
                in_channels=in_channels,
                kernel_shape=kernels[k][0],
                number_of_filters=kernels[k][1],
                hidden_size=hidden_size,
                # dilation=(kernels[k][2], 1), # TODO: dilation only in x direction? is this correct? K: maybe
                dilation=kernels[k][2] , # TODO: dilation only in x direction? is this correct? K: maybe
                stride = kernels[k][4],
                padding_size=padding_size,
                padding_mode=padding_mode,
                # Built on the default (CPU) device on purpose, to match the branch's
                # nn.Conv2d parameters, which are also created on CPU and only reach the
                # accelerator when the assembled net is .to(DEVICE)'d by the caller.
                # Pre-moving the mask to `device` here used to leave every masked branch
                # holding a cuda buffer alongside cpu weights. (The branches normalize
                # this themselves too, so a caller-supplied on-device mask is still fine.)
                mask=torch.tensor(np.array(kernels[k][3]),
                                  dtype=torch.float32)
                                  if kernels[k][3] is not None else None,
                init=init,
            )
            group_k = self.branch_groups[k]
            branch_cls = ConvBranch_Equivariant if group_k is not None else ConvBranch_twoLayer
            if group_k is not None:
                branch_kwargs["group"] = group_k
            self.branches.append(
                # ConvBranch_twoLayer(
                branch_cls(**branch_kwargs)
            )


    def forward(self, x):
        all_filters = []
        for k in range( len(self.kernels) ):  # can likely be sped up with block sparse approach
            h = self.branches[k](x) # shape: (batch_size, 1)
            all_filters.append(h)

        return torch.cat(all_filters, dim=1) # shape: (batch_size, number of branches)
        # need to adjust later for multiclass


class ConvBranch_twoLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_shape,
        number_of_filters,
        hidden_size,
        dilation=1,
        stride=1,
        padding_size=0,
        padding_mode=None,
        mask=None,
        init="kaiming",
    ):
        super().__init__()

        self.input_dim = in_channels
        self.kernel_shape = kernel_shape
        self.hidden_size = hidden_size
        # TODO/NOTE: this is an important hyperparameter not specified in the paper!
        # output channels in a CNN is a free hyperparameter, which is roughly equivalent to the width
        # of a hidden layer in a MLP.
        self.number_of_filters = number_of_filters  # this is only used in the second layer

        if padding_mode is None:
            padding_mode = "zeros"
            padding_size = 0

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_shape,
            stride=stride,
            padding=padding_size,
            padding_mode=padding_mode,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.number_of_filters, # TODO: is there any case where this is not 1? K: droppped because didn't help expressivity; messed up interpretation
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # print(self.conv1.weight.data.shape)
        # print(self.conv2.weight.data.shape)

        if mask is not None:
            # print("Using mask: ", mask)
            assert tuple(mask.squeeze().shape) == tuple(kernel_shape), "Mask does not match kernel shape."
            # Keep the mask on whatever device the branch's own parameters are on. The
            # caller may hand in a mask already moved to the training device, while the
            # nn.Conv2d layers above are always created on the default (CPU) device and
            # only reach the accelerator later, when the whole net is .to(DEVICE)'d. A
            # buffer registered on a different device than the parameters is a latent
            # cross-device error; register_buffer does not reconcile them.
            mask = mask.to(device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)
        # else:
            # print("No mask.")
        self.register_buffer("mask", mask)



    def forward(self, x):
        if self.mask is not None:
            self.conv1.weight.data *= self.mask  # Enforce mask constraint

        # print(f"conv 1 weight: {self.conv1.weight.shape}")
        # compute features (kxk convolution, then ReLU, then 1x1 convolution)
        f = self.conv1(x)
        f = F.relu(f)
        f = self.conv2(f)     # shape: N,1,H,W

        # average (global pool) over the entire image
        f = torch.mean(f, dim=(2, 3)) # shape: N,1

        return f


class ConvBranch_Equivariant(nn.Module):
    """Group-equivariant convolutional branch: one canonical conv1 kernel + one shared
    1x1 conv2, realized as a group conv by transforming the (masked) conv1 weight at
    forward time (hand-rolled weight tying, no escnn/gcnn). This is the rotation-averaged
    convolution of Eq. (6) in the manuscript. Pooling (averaging) the per-group-element
    GAP'd scalars makes the branch scalar invariant to the chosen group's action on the
    input, and by the same Boolean-Fourier argument as ConvBranch_twoLayer, a linear
    function of the group-symmetrized (orbit-averaged) correlators of the canonical
    footprint, the C_rot[P] of Eq. (4).

    The group elements are applied in a loop of F.conv2d calls rather than one stacked
    call, because a rotated non-square kernel changes shape ((2,1) becomes (1,2)). A
    single canonical mask generates the whole orbit, so one branch replaces the separate
    per-orientation branches of the unconstrained set (10 branches become 5 for C4).

    `group="C4"` (default): the 4 rotations {0,90,180,270} via `torch.rot90`, correct
    for the square Ising lattice. `group="D2"` (alias `"K4"`): the Klein four-group
    {identity, 180-deg rotation, horizontal mirror, vertical mirror}, i.e. the
    symmetry group of a non-square rectangle, correct for the rectangular XY lattice
    -- unlike C4 it is shape-preserving on non-square kernels/images and keeps the
    vertical (2,1) and horizontal (1,2) dominoes as separate orbits.

    Drop-in constructor: accepts the same kwargs as ConvBranch_twoLayer, plus an
    optional `group` kwarg ("C4", "D2", or "K4"; unknown strings raise ValueError).

    Only wired up and tested for the `smallkernels` set (see utils.set_kernels), which
    is the only kernel set with canonical group representatives. Settings that would
    silently break the invariance raise instead of being ignored; the exact boundaries
    were measured, not assumed:

    - `stride != 1` raises. Stride only preserves the invariance when the strided
      sampling grid happens to be group-symmetric: a 2x2 kernel with stride 2 or 3 on
      8x8 is exact, but a (2,1) kernel with stride 2 drifts by 1.3e-1 and a 3x3 kernel
      with stride 2 by 2.9e-1 (against a signal scale std(z) ~ 4e-2 and 9.5e-2).
    - Anisotropic `dilation` (a tuple with unequal entries) raises: rotating the kernel
      does not rotate the dilation, so the footprint stops being an orbit. A scalar
      (isotropic) dilation is safe and allowed, verified exact to 5.6e-17 for dilation
      2 and 3 on square and non-square kernels.
    - `padding_size`/`padding_mode` raise. Symmetric zero padding does preserve the
      invariance (verified exact for padding 1 and 2, square and non-square kernels),
      so this is a scope limit, not a mathematical one: the per-group-element F.conv2d
      loop does not reimplement nn.Conv2d's non-zero padding_mode handling. Note that
      padding is also unwanted for the correlator reading, since it injects fictitious
      boundary sites whose value is 0 rather than +/-1, and nothing in this repo
      constructs ShapeAdaptiveConvNet with padding arguments.

    Interpretability: because the branch scalar is a linear function of the
    group-symmetrized correlators, regressing it onto the plain correlators of its own
    canonical footprint does not reach R^2 = 1 (0.926 for the trained Ising (2,1) branch,
    0.873 for the XY diagonal). Pass `group=` to
    interpret.fit_activation_to_correlators to build the orbit-averaged features
    (1/|G|) sum_g C_{g.F}[g.P] instead -- evaluating each orbit member on its own
    valid-position grid, which differs between members for a non-square footprint -- and
    the exact reading is recovered (R^2 = 1 to float32 precision on every active branch
    of both C4 runs). interpret.orbit_site_maps mirrors _group_transformed_weights below.
    """

    def __init__(
        self,
        in_channels,
        kernel_shape,
        number_of_filters,
        hidden_size,
        dilation=1,
        stride=1,
        padding_size=0,
        padding_mode=None,
        mask=None,
        init="kaiming",
        group="C4",
    ):
        super().__init__()

        self.input_dim = in_channels
        self.kernel_shape = kernel_shape
        self.hidden_size = hidden_size
        self.number_of_filters = number_of_filters
        self.dilation = dilation
        self.stride = stride
        if group not in ("C4", "D2", "K4"):
            raise ValueError(
                f"Unknown group '{group}': expected 'C4', 'D2', or 'K4' "
                "('D2' and 'K4' are aliases for the rectangle/Klein-four symmetry group)."
            )
        self.group = group

        # Reject, rather than silently ignore, every setting the group-conv loop cannot
        # honour: each of these would otherwise return a branch that is quietly not
        # invariant (or quietly not padded). See the class docstring for the measurements.
        if stride != 1:
            raise NotImplementedError(
                f"ConvBranch_Equivariant does not support stride={stride}: a strided "
                "sampling grid is only group-symmetric for particular kernel/image size "
                "combinations, so the invariance would hold or fail silently depending "
                "on the kernel shape. Use stride=1."
            )
        if not np.isscalar(dilation) and len(set(dilation)) > 1:
            raise NotImplementedError(
                f"ConvBranch_Equivariant does not support anisotropic dilation={dilation}: "
                "rotating the kernel does not rotate the dilation, so the transformed "
                "footprints are not a group orbit. Use a scalar (isotropic) dilation."
            )
        if padding_size not in (0, None) or padding_mode not in (None, "zeros"):
            raise NotImplementedError(
                f"ConvBranch_Equivariant does not support padding (got padding_size="
                f"{padding_size}, padding_mode={padding_mode}): the per-group-element "
                "F.conv2d loop does not reimplement nn.Conv2d's padding_mode handling. "
                "Padding is also unwanted for the correlator reading, since padded sites "
                "carry value 0 rather than +/-1."
            )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_shape,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.number_of_filters,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if mask is not None:
            assert tuple(mask.squeeze().shape) == tuple(kernel_shape), "Mask does not match kernel shape."
            # Match the mask to the parameters' device/dtype before registering it; see
            # the same note in ConvBranch_twoLayer. This branch additionally *uses* the
            # mask at construction time (just below), i.e. before the net is moved to
            # the training device, so a mask handed in already on cuda would otherwise
            # meet CPU weights and raise "Expected all tensors to be on the same device".
            mask = mask.to(device=self.conv1.weight.device, dtype=self.conv1.weight.dtype)
        self.register_buffer("mask", mask)

        if mask is not None:
            # Zero the masked-off entries of the stored parameter once, at construction.
            # forward() masks out of place (W * mask), so masked entries receive exactly
            # zero gradient and, once zero, stay zero forever: AdamW's update on a
            # permanently-zero gradient is -lr * weight_decay * w, which cannot move a
            # weight away from 0. Without this the entries keep their init values for the
            # whole run even though they never affect the output, which would (a) show up
            # as bright cells in plots.plot_weights, whose shared colour scale is taken
            # from the data, and (b) be charged by the optional cf.weight_penalty L1 in
            # train.py, which sums over every conv1 parameter. ConvBranch_twoLayer masks
            # in place instead (conv1.weight.data *= mask each forward), so its stored
            # weights are already zero there; this keeps the two branches comparable.
            with torch.no_grad():
                self.conv1.weight.mul_(self.mask)

    def _group_transformed_weights(self, W_masked):
        """Return the list of transformed conv1 weights for one full sweep of self.group."""
        if self.group == "C4":
            return [torch.rot90(W_masked, k=r, dims=(2, 3)) for r in range(4)]
        else:  # "D2" / "K4": Klein four-group of shape-preserving flips
            return [
                W_masked,                                 # identity
                torch.flip(W_masked, dims=[2]),            # vertical mirror
                torch.flip(W_masked, dims=[3]),            # horizontal mirror
                torch.flip(W_masked, dims=[2, 3]),         # 180-deg rotation
            ]

    def forward(self, x):
        W_masked = self.conv1.weight
        if self.mask is not None:
            W_masked = W_masked * self.mask  # broadcast over (out,in) channels; mask is (kh,kw)

        outs = []
        for W in self._group_transformed_weights(W_masked):
            f = F.conv2d(x, W, bias=self.conv1.bias, stride=self.stride, dilation=self.dilation, padding=0)
            f = F.relu(f)
            f = self.conv2(f)  # shared 1x1 conv (same params every group element)
            f = torch.mean(f, dim=(2, 3))  # GAP -> (B, number_of_filters)
            outs.append(f)
        z = torch.stack(outs, dim=0).mean(dim=0)  # mean over the group -> (B, number_of_filters)
        return z


class SmallModel(nn.Module):
    def __init__(self, sizes, device):
        super(SmallModel, self).__init__()

        self.sizes = sizes
        self.length = len(self.sizes) - 1
        self.activation = F.relu
        self.device = device

        self.hiddens = nn.ModuleList()
        for k in range(self.length):
            layer = nn.Linear(self.sizes[k], self.sizes[k + 1])
            self.hiddens.append(layer)

    def forward(self, x):
        h = x
        for k in range(self.length):
            h = self.hiddens[k](h)
            if k != self.length - 1:
                h = self.activation(h)
        return h

    def collectParameters(self):
        all_param_list = []
        for k in range(self.length):
            for x in self.hiddens[k].parameters():
                all_param_list.append(x.view(-1))
        return torch.cat(all_param_list)



class PhaseCNN(nn.Module):
    """"
    Simple CNN, non-Tetris, for comparison of phase transition indicator.
    """
    def __init__(self, in_channels, kernel_shape, output_dim=1):
        super(PhaseCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=kernel_shape, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_shape, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_shape, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))  # Output will always be [batch, 128, 1, 1]

        # self.fc1 = nn.Linear(128 * 37, 256)  # 300 → 150 → 75 → 37
        self.fc1 = nn.Linear(128 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.global_pool(x)       # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)

        # print("x.shape", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation for regression
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    
class ResNet18(nn.Module):
    def __init__(self, in_channels, kernel_shape, output_dim=1):
        super(ResNet18, self).__init__()
        del kernel_shape  # accepted for interface compatibility; ResNet18 uses fixed 3×3 kernels
        self._cur_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self._cur_channels, out_channels, s))
            self._cur_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)