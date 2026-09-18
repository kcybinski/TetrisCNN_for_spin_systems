"""Group C: tetriscnn.experiments.attach_sample_weights.

Importing tetriscnn.experiments pulls in tetriscnn.models, tetriscnn.datasets, and
tetriscnn.plots (-> matplotlib/seaborn/statsmodels/plotly). Measured cold: ~3s warm,
up to ~17s on a fully cold filesystem cache (first Python invocation in the conda env
after a reboot) -- there is nothing accidentally-heavy in the module itself (no PySR/
Julia), it is just a lot of ordinary scientific-Python import weight. That's an
acceptable cost for one test module (not gated as `slow` here), but is why every
other test file in this suite avoids importing tetriscnn.experiments or
tetriscnn.plots.

NOTE the formula was recently changed to inverse-frequency:
    weight = min_count_across_times / count_at_this_sample's_time
so under-represented time points are up-weighted toward 1 (never above 1), and the
best-represented time point(s) get weight 1.
"""
import pytest
import torch

from tetriscnn.experiments import attach_sample_weights
from tetriscnn.utils import AttrDict


class _StubDataset:
    """Minimal stand-in for a PhaseDataset: attach_sample_weights only reads .times
    and writes .sample_weights, so we don't need a real PhaseDataset here."""
    def __init__(self, times):
        self.times = torch.tensor(times, dtype=torch.float32)


def _cf_with_times(times, use_weighted_loss=True):
    cf = AttrDict()
    cf.use_weighted_loss = use_weighted_loss
    cf.train_dataset = _StubDataset(times)
    return cf


class TestAttachSampleWeights:
    def test_inverse_frequency_weights(self):
        # 100 samples at t=1, 200 at t=2, 50 at t=3 -> min_count = 50
        times = [1.0] * 100 + [2.0] * 200 + [3.0] * 50
        cf = _cf_with_times(times)

        attach_sample_weights(cf, verbose=False)

        weights = cf.train_dataset.sample_weights
        assert weights.shape == (350,)
        assert weights.dtype == torch.float32

        # t=1 -> 50/100 = 0.5 ; t=2 -> 50/200 = 0.25 ; t=3 -> 50/50 = 1.0
        assert torch.allclose(weights[:100], torch.full((100,), 0.5))
        assert torch.allclose(weights[100:300], torch.full((200,), 0.25))
        assert torch.allclose(weights[300:], torch.full((50,), 1.0))

    def test_most_underrepresented_time_gets_weight_one(self):
        times = [1.0] * 10 + [2.0] * 40
        cf = _cf_with_times(times)
        attach_sample_weights(cf, verbose=False)
        weights = cf.train_dataset.sample_weights
        assert weights.max().item() == pytest.approx(1.0)
        assert weights[:10].tolist() == pytest.approx([1.0] * 10)

    def test_equal_counts_give_uniform_weight_one(self):
        times = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
        cf = _cf_with_times(times)
        attach_sample_weights(cf, verbose=False)
        weights = cf.train_dataset.sample_weights
        assert torch.allclose(weights, torch.ones(60))

    def test_noop_when_use_weighted_loss_false(self):
        cf = _cf_with_times([1.0, 2.0, 2.0], use_weighted_loss=False)
        attach_sample_weights(cf, verbose=False)
        assert not hasattr(cf.train_dataset, "sample_weights")

    def test_noop_when_use_weighted_loss_key_absent(self):
        cf = AttrDict()
        cf.train_dataset = _StubDataset([1.0, 2.0, 2.0])
        # cf.use_weighted_loss is never set at all; the function checks membership
        # via `"use_weighted_loss" in cf.keys()`, NOT hasattr/getattr, so this is
        # safe from the AttrDict hasattr/getattr pitfall documented elsewhere.
        attach_sample_weights(cf, verbose=False)
        assert not hasattr(cf.train_dataset, "sample_weights")

    def test_verbose_does_not_raise(self, capsys):
        cf = _cf_with_times([1.0] * 5 + [2.0] * 5)
        attach_sample_weights(cf, verbose=True)
        captured = capsys.readouterr()
        assert "Weighted Loss" in captured.out
