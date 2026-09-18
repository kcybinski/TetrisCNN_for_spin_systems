"""Group A/B: tests for tetriscnn.plots.plot_branch_fits (the cf.fit_branches
feature). No dataset files needed -- snapshots are synthetic spin arrays, exactly
like tests/test_interpret.py uses for tetriscnn.interpret.fit_activation_to_correlators
itself.

plot_branch_fits generalizes Figure5.ipynb's hand-picked-branch correlator
regression figure: it first calls get_active_branches to find which branches
cleared the learning-rate noise floor, then BFA-fits and plots exactly those.
"""
import numpy as np
import pytest

from tetriscnn.utils import AttrDict
from tetriscnn.pattern_generation import pattern_correlator
from tetriscnn.plots import plot_branch_fits

# Kernel specs follow cf.kernels entries: [shape, n_filters, dilation, mask, stride].
SINGLE_SITE = [(1, 1), 1, 1, None, 1]
DOMINO_V = [(2, 1), 1, 1, None, 1]
DIAG_2x2 = [(2, 2), 1, 1, [[0, 1], [1, 0]], 1]  # one of "smallkernels"'s masked branches


def _random_spins(rng, B, C, H, W):
    return rng.choice(np.array([-1.0, 1.0]), size=(B, C, H, W))


def _branch_fits_cf(**overrides):
    cf = AttrDict()
    cf.model = "tetriscnn"
    cf.dataset = "Paris_Ising"
    cf.equivariant = False
    cf.kernels = [SINGLE_SITE, DOMINO_V]
    cf.logdir = "unused"  # save_fig is monkeypatched out below
    cf.update(overrides)
    return cf


def _capture_save(monkeypatch):
    import tetriscnn.plots as plots_mod
    captured = {}

    def _fake_save_fig(fig, folder_path, name):
        captured["called"] = True
        captured["n_axes"] = len(fig.axes)

    monkeypatch.setattr(plots_mod, "save_fig", _fake_save_fig)
    return captured


class TestPlotBranchFits:
    def test_only_active_branch_is_fit_and_plotted(self, monkeypatch):
        rng = np.random.default_rng(0)
        snaps = _random_spins(rng, B=300, C=1, H=6, W=6)
        cf = _branch_fits_cf()

        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(1, 1))
        z0 = 6.41 * C_site - 2.96       # branch 0: real signal, large amplitude
        z1 = rng.normal(scale=1e-7, size=z0.shape)  # branch 1: pinned at the noise floor
        z = np.stack([z0, z1], axis=1)

        captured = _capture_save(monkeypatch)
        fits = plot_branch_fits(cf, snaps, z, "branch_fits.png", final_lr=1e-5, floor_margin=10.0)

        assert captured.get("called") is True
        assert set(fits.keys()) == {0}
        assert fits[0]["r_squared"] == pytest.approx(1.0, abs=1e-6)
        assert captured["n_axes"] == 1  # one panel per active branch

    def test_no_active_branch_skips_without_crashing(self, monkeypatch, capsys):
        rng = np.random.default_rng(1)
        snaps = _random_spins(rng, B=100, C=1, H=6, W=6)
        cf = _branch_fits_cf()
        z = rng.normal(scale=1e-7, size=(100, 2))  # both branches at the floor

        captured = _capture_save(monkeypatch)
        fits = plot_branch_fits(cf, snaps, z, "branch_fits.png", final_lr=1e-5, floor_margin=10.0)

        assert fits is None
        assert captured.get("called") is None
        assert "no branch cleared the activity floor" in capsys.readouterr().out

    def test_non_tetriscnn_model_is_a_no_op(self, monkeypatch, capsys):
        rng = np.random.default_rng(2)
        snaps = _random_spins(rng, B=50, C=1, H=6, W=6)
        cf = _branch_fits_cf(model="ResNet18")
        z = rng.normal(size=(50, 2))

        captured = _capture_save(monkeypatch)
        fits = plot_branch_fits(cf, snaps, z, "branch_fits.png", final_lr=1e-5)

        assert fits is None
        assert captured.get("called") is None
        assert "no bottleneck branches" in capsys.readouterr().out

    def test_masked_kernel_pattern_labels_render_without_a_system_latex(self, monkeypatch):
        """Regression test: tetriscnn.interpret always builds its pattern labels with
        mask_to_latex_pattern(full_latex=True) (real-LaTeX macros: \\blacksquare,
        \\square, \\substack), because its only prior consumer was the paper-figure
        notebooks (which set text.usetex=True via paper.mplstyle). plot_branch_fits
        runs from the ordinary main.py pipeline instead, with no system LaTeX
        assumed, and matplotlib's built-in mathtext renderer does not know \\square
        or \\substack -- it used to raise a ValueError at figure-save time for any
        branch with a masked footprint (an unmasked single-row pattern like
        SINGLE_SITE/DOMINO_V never reached \\square, so it accidentally passed even
        before the fix). DIAG_2x2 has a real \\square in its own kernel-pattern
        label AND a 2-row \\substack among its correlator terms, exercising both."""
        rng = np.random.default_rng(4)
        snaps = _random_spins(rng, B=200, C=1, H=6, W=6)
        cf = _branch_fits_cf(kernels=[SINGLE_SITE, DIAG_2x2])

        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(1, 1))
        z0 = rng.normal(scale=1e-7, size=(200,))  # branch 0: inactive
        z1 = 2.0 * C_site + 0.5                   # branch 1 (DIAG_2x2): active
        z = np.stack([z0, z1], axis=1)

        captured = _capture_save(monkeypatch)
        fits = plot_branch_fits(cf, snaps, z, "branch_fits.png", final_lr=1e-5, floor_margin=10.0)

        assert captured.get("called") is True
        assert set(fits.keys()) == {1}

    def test_defaults_final_lr_to_cf_learning_rate(self, monkeypatch):
        """When no scheduler ran, metrics has no 'learning_rate' key; callers pass
        final_lr=None and this must fall back to cf.learning_rate rather than crash."""
        rng = np.random.default_rng(3)
        snaps = _random_spins(rng, B=200, C=1, H=6, W=6)
        cf = _branch_fits_cf(learning_rate=1e-5)

        C_site = pattern_correlator(snaps[:, 0:1, :, :], [(0, 0)], window_shape=(1, 1))
        z0 = 3.0 * C_site + 1.0
        z1 = rng.normal(scale=1e-7, size=z0.shape)
        z = np.stack([z0, z1], axis=1)

        captured = _capture_save(monkeypatch)
        fits = plot_branch_fits(cf, snaps, z, "branch_fits.png")  # final_lr omitted

        assert captured.get("called") is True
        assert set(fits.keys()) == {0}
