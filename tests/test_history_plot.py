"""Group A: pure-function tests for tetriscnn.plots.plot_history,
get_active_branches, and cf.enabled_metrics on plot_lambda/plot_lbc. No data
files, no training -- everything here runs on synthetic metrics/activations.

Regression coverage for three release-cleanup fixes:
  * plot_history used to always draw a fixed 4x2 grid, with the bottom two panels
    entirely dead (their content was commented out), regardless of cf.enabled_metrics.
    It is now adaptive: the panel set (and hence the figure grid) tracks
    cf.enabled_metrics, and a "pt" panel (phase indicator vs. epoch) is drawn when
    requested for a regression run instead of being unreachable dead code.
  * get_active_branches operationalizes "active branch" (appendices/LR_Scheduling.tex)
    as mean|z_k| clearing a multiple of the run's final learning rate.
  * plot_lambda's "pt_deriv"/"pt2_deriv" (imshow heatmap) branch used `Normalize`
    and `make_axes_locatable` without importing either -- any cf.enabled_metrics
    selection that reached it raised NameError at plot time. Both are now
    imported at the top of tetriscnn/plots.py.
"""
import numpy as np
import pytest

from tetriscnn.utils import AttrDict
from tetriscnn.plots import get_active_branches, plot_history, plot_lambda, plot_lbc


def _history_cf(**overrides):
    cf = AttrDict()
    cf.model = "tetriscnn"
    cf.task = "classification"
    cf.loss_str = "CEL"
    cf.goodness_str = "acc"
    cf.logdir = "unused"  # save_fig is monkeypatched out in these tests
    cf.kernels = [
        [(1, 1), 1, 1, None, 1],
        [(2, 1), 1, 1, None, 1],
    ]
    cf.update(overrides)
    return cf


def _synthetic_metrics(cf, n_epochs=5, with_pt=False):
    metrics = {
        f"train_{cf.loss_str}": list(np.linspace(1.0, 0.1, n_epochs)),
        f"val_{cf.loss_str}": list(np.linspace(1.1, 0.2, n_epochs)),
        "train_l1": list(np.linspace(0.5, 0.05, n_epochs)),
        "val_l1": list(np.linspace(0.5, 0.05, n_epochs)),
        f"train_{cf.goodness_str}": list(np.linspace(0.5, 0.95, n_epochs)),
        f"val_{cf.goodness_str}": list(np.linspace(0.5, 0.9, n_epochs)),
    }
    for k in range(len(cf.kernels)):
        metrics[f"z_{k}"] = list(np.linspace(0.01, 0.5, n_epochs) * (k + 1))
    if cf.task == "regression":
        metrics["train_r2agg"] = list(np.linspace(0.0, 0.9, n_epochs))
        metrics["val_r2agg"] = list(np.linspace(0.0, 0.85, n_epochs))
        if with_pt:
            metrics["pt"] = [[0.1 * i, 0.5 + 0.01 * i] for i in range(n_epochs)]
    return metrics


class TestGetActiveBranches:
    def test_separates_floor_from_signal(self):
        z = np.array([
            [1e-6, 0.50],
            [1e-6, 0.55],
            [-1e-6, 0.45],
        ])
        active_mask, mean_abs_z = get_active_branches(z, final_lr=1e-4, floor_margin=10.0)
        assert active_mask.tolist() == [False, True]
        assert mean_abs_z[1] == pytest.approx(0.5, abs=0.05)

    def test_floor_margin_is_configurable(self):
        z = np.array([[0.01, 0.5], [0.01, 0.5]])
        # floor = 1 * 1e-2 = 1e-2: branch 0's mean|z| (0.01) does not clear it.
        active_lo, _ = get_active_branches(z, final_lr=1e-2, floor_margin=1.0)
        assert active_lo.tolist() == [False, True]
        # A much smaller margin lets branch 0 clear the (now tiny) floor too.
        active_hi, _ = get_active_branches(z, final_lr=1e-2, floor_margin=0.5)
        assert active_hi.tolist() == [True, True]


class TestPlotHistoryAdaptive:
    def _capture_axes(self, monkeypatch, cf, metrics, file_name="history.png"):
        import tetriscnn.plots as plots_mod
        captured = {}

        def _fake_save_fig(fig, folder_path, name):
            captured["n_axes"] = len(fig.axes)
            captured["off_axes"] = sum(1 for ax in fig.axes if not ax.axison)

        monkeypatch.setattr(plots_mod, "save_fig", _fake_save_fig)
        plot_history(metrics, cf, file_name)
        return captured

    def test_default_metrics_fill_grid_with_no_dead_panels(self, monkeypatch):
        """Default {'z','loss','goodness'} on a tetriscnn run draws exactly 4 real
        panels (loss, l1, goodness, z) in a 2x2 grid -- no empty bottom panels."""
        cf = _history_cf(enabled_metrics={"z", "loss", "goodness"})
        metrics = _synthetic_metrics(cf)
        captured = self._capture_axes(monkeypatch, cf, metrics)
        assert captured["n_axes"] == 4
        assert captured.get("off_axes", 0) == 0

    def test_single_metric_selection_shrinks_the_grid(self, monkeypatch):
        cf = _history_cf(enabled_metrics={"loss"})
        metrics = _synthetic_metrics(cf)
        captured = self._capture_axes(monkeypatch, cf, metrics)
        # "loss" alone still pulls in the l1 companion panel for a tetriscnn model.
        assert captured["n_axes"] == 2

    def test_missing_enabled_metrics_falls_back_to_default_set(self, monkeypatch):
        cf = _history_cf()  # no cf.enabled_metrics at all
        metrics = _synthetic_metrics(cf)
        captured = self._capture_axes(monkeypatch, cf, metrics)
        assert captured["n_axes"] == 4

    def test_pt_panel_only_drawn_for_regression_when_requested(self, monkeypatch):
        cf = _history_cf(task="regression", label_param="delta",
                          enabled_metrics={"z", "loss", "goodness", "pt"})
        metrics = _synthetic_metrics(cf, with_pt=True)
        captured = self._capture_axes(monkeypatch, cf, metrics)
        # loss + l1 + goodness + z + pt = 5 panels -> 2x3 grid, one cell left off.
        assert captured["n_axes"] == 6
        assert captured.get("off_axes", 0) == 1

    def test_pt_not_requested_never_adds_a_panel(self, monkeypatch):
        cf = _history_cf(task="regression", label_param="delta",
                          enabled_metrics={"z", "loss", "goodness"})
        metrics = _synthetic_metrics(cf, with_pt=True)  # data present but not requested
        captured = self._capture_axes(monkeypatch, cf, metrics)
        assert captured["n_axes"] == 4

    def test_no_applicable_metrics_skips_the_plot_without_crashing(self, monkeypatch, capsys):
        cf = _history_cf(enabled_metrics={"pt"})  # classification: "pt" never applies
        metrics = _synthetic_metrics(cf)
        import tetriscnn.plots as plots_mod
        called = {"save_fig": False}
        monkeypatch.setattr(plots_mod, "save_fig", lambda *a, **k: called.__setitem__("save_fig", True))
        plot_history(metrics, cf, "history.png")
        assert called["save_fig"] is False
        assert "skipping" in capsys.readouterr().out


def _lambda_cf(**overrides):
    cf = AttrDict()
    cf.lambdas = [-1, 0, 1]
    cf.experiment_name = "lambdamax"
    cf.task = "regression"
    cf.label_param = "delta"
    cf.goodness_str = "r2agg"
    cf.loss_str = "MSE"
    cf.seeds = [1, 2, 3]
    cf.kernels = [[(1, 1), 1, 1, None, 1], [(2, 1), 1, 1, None, 1], [(1, 1), 1, 1, None, 1]]
    cf.logdir = "unused"
    cf.update(overrides)
    return cf


def _lambda_metrics(rng, n_seeds, n_kernels, n_grid=5, with_pt2=False):
    """Metrics shaped like train.py's actual output: 'pt'/'pt2' are
    [deriv (array over the tuning-parameter grid), peak (scalar)] per seed --
    see PhaseDataset.get_phase_indicator, which np.gradient()s over the grid."""
    m = {
        "z": np.abs(rng.standard_normal((n_seeds, n_kernels))).tolist(),
        "val_MSE": rng.random(n_seeds).tolist(),
        "val_r2agg": rng.random(n_seeds).tolist(),
        "pt": [[rng.standard_normal(n_grid).tolist(), float(rng.random())] for _ in range(n_seeds)],
        "epochs": rng.integers(50, 250, n_seeds).tolist(),
        "net2_norm": rng.random(n_seeds).tolist(),
    }
    if with_pt2:
        m["pt2"] = [[rng.standard_normal(n_grid).tolist(), float(rng.random())] for _ in range(n_seeds)]
    return m


class TestEnabledMetricsBeyondTheDefault:
    """cf.enabled_metrics is documented (docs/CONFIGURATION.md) as accepting a much
    larger set than the default {"z", "loss", "goodness"}; this locks that in."""

    def _capture(self, monkeypatch):
        import tetriscnn.plots as plots_mod
        captured = {}
        monkeypatch.setattr(plots_mod, "save_fig",
                             lambda fig, folder, name: captured.update(n_axes=len(fig.axes)))
        return captured

    def test_plot_lambda_full_metric_set(self, monkeypatch):
        rng = np.random.default_rng(0)
        cf = _lambda_cf()
        all_metrics = {str(lam): _lambda_metrics(rng, 3, 3) for lam in cf.lambdas}
        full_set = {"z", "goodness", "loss", "pt", "pt_deriv", "pt_deriv_ste", "net2_norm", "epochs"}
        captured = self._capture(monkeypatch)
        plot_lambda(all_metrics, cf, "lambda.png", enabled_metrics=full_set)
        assert captured["n_axes"] >= len(full_set)  # pt_deriv panels add a colorbar axis each

    def test_plot_lambda_deltaomega_pt2_set(self, monkeypatch):
        rng = np.random.default_rng(1)
        cf = _lambda_cf(label_param="deltaomega")
        all_metrics = {str(lam): _lambda_metrics(rng, 3, 3, with_pt2=True) for lam in cf.lambdas}
        full_set = {"z", "goodness", "loss", "pt", "pt_deriv", "pt_deriv_ste",
                    "pt2", "pt2_deriv", "pt2_deriv_ste", "net2_norm", "epochs"}
        captured = self._capture(monkeypatch)
        plot_lambda(all_metrics, cf, "lambda2.png", enabled_metrics=full_set)
        assert captured["n_axes"] >= len(full_set)

    def test_plot_lbc_z_acc_loss(self, monkeypatch, tmp_path):
        cf = _lambda_cf(dataset="Paris_Ising", lam=3, task="lbc",
                         goodness_str="acc", loss_str="CEL", logdir=str(tmp_path))
        cf.remake_lambda_plot = False

        class FakeProcessor:
            unique_labels = np.array([0.0, 1.0, 2.0])  # -> 2 partitions

        class FakeDataset:
            processor = FakeProcessor()

        cf.val_dataset = FakeDataset()
        metrics_per_seed = {
            1: {
                "z": [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]],
                "val_acc": [0.5, 0.6], "train_acc": [0.6, 0.7],
                "val_CEL": [0.3, 0.2], "train_CEL": [0.2, 0.1],
            },
        }
        captured = self._capture(monkeypatch)
        plot_lbc(metrics_per_seed, cf, "lbc.png", enabled_metrics={"z", "acc", "loss"})
        assert captured["n_axes"] == 3
