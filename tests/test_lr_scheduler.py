"""Group: tetriscnn.train.build_lr_scheduler / LR_SCHEDULER_DEFAULTS.

Characterizes the learning-rate-scheduler configuration mechanism introduced to
declutter main.py::setup_experiment(): only the *active* cf.lr_scheduler_type's
parameters need to be set by the caller, the other four types' parameters are filled
in from LR_SCHEDULER_DEFAULTS on demand. All five scheduler types must keep
constructing the exact torch.optim.lr_scheduler class they did before this
refactor, with the same parameters, and an explicit cf.<param> must always win over
its default (this is what makes old, fully-populated saved config.json files behave
identically to before).

No real dataset/model is needed here -- build_lr_scheduler only touches
`cf` and an `optimizer`, plus `len(train_loader)` for OneCycleLR, so a plain
nn.Linear/AdamW pair and a stand-in list are enough.
"""
import pytest
import torch

from tetriscnn.utils import AttrDict
from tetriscnn.train import build_lr_scheduler, LR_SCHEDULER_DEFAULTS


def _cf(scheduler_type, epochs=42):
    cf = AttrDict()
    cf.epochs = epochs
    cf.lr_scheduler_type = scheduler_type
    return cf


def _optimizer():
    net = torch.nn.Linear(4, 2)
    return torch.optim.AdamW(net.parameters(), lr=1e-2)


# len() is all OneCycleLR's steps_per_epoch derivation needs from this.
_TRAIN_LOADER_STUB = list(range(7))


class TestAllFiveSchedulerTypesBuild:
    def test_reduce_on_plateau(self):
        cf = _cf("reduce_on_plateau")
        cf.lr_reduce_factor = 0.5
        cf.lr_reduce_patience = 5
        cf.min_lr = 1e-9
        sched = build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched.factor == 0.5
        assert sched.patience == 5
        assert sched.min_lrs[0] == 1e-9

    def test_cosine_annealing(self):
        cf = _cf("cosine_annealing")
        sched = build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert sched.T_max == 42
        assert sched.eta_min == LR_SCHEDULER_DEFAULTS["cosine_annealing"]["min_lr"]

    def test_step(self):
        cf = _cf("step")
        sched = build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)
        assert sched.step_size == LR_SCHEDULER_DEFAULTS["step"]["lr_step_size"]
        assert sched.gamma == LR_SCHEDULER_DEFAULTS["step"]["lr_step_gamma"]

    def test_exponential(self):
        cf = _cf("exponential")
        sched = build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)
        assert isinstance(sched, torch.optim.lr_scheduler.ExponentialLR)
        assert sched.gamma == LR_SCHEDULER_DEFAULTS["exponential"]["lr_exp_gamma"]

    def test_onecycle(self):
        cf = _cf("onecycle")
        sched = build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)
        assert isinstance(sched, torch.optim.lr_scheduler.OneCycleLR)
        d = LR_SCHEDULER_DEFAULTS["onecycle"]
        expected_div_factor = d["onecycle_max_lr"] / d["onecycle_initial_lr"]
        expected_final_div_factor = d["onecycle_initial_lr"] / d["onecycle_final_lr"]
        # OneCycleLR doesn't re-expose div_factor directly; check it took effect via
        # the initial LR it programmed into the optimizer's param groups.
        initial_lr = sched.optimizer.param_groups[0]["initial_lr"]
        assert initial_lr == pytest.approx(d["onecycle_max_lr"] / expected_div_factor)
        assert expected_div_factor == 100.0
        assert expected_final_div_factor == 1e4

    def test_unknown_type_raises(self):
        cf = _cf("not_a_real_scheduler")
        with pytest.raises(ValueError, match="Unknown lr_scheduler_type"):
            build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)


class TestDefaultsOnlyFillMissingKeys:
    def test_explicit_value_overrides_default(self):
        cf = _cf("exponential")
        cf.lr_exp_gamma = 0.123  # explicit; must NOT be clobbered by the 0.85 default
        sched = build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)
        assert sched.gamma == 0.123

    def test_defaults_applied_are_visible_on_cf_afterwards(self):
        # build_lr_scheduler uses cf.setdefault, so a caller that inspects/serializes
        # cf afterwards (e.g. train() -> save_json(cf, ..., "config.json")) sees the
        # concrete values that were actually used, not missing keys.
        cf = _cf("step")
        assert "lr_step_size" not in cf.keys()
        build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)
        assert cf.lr_step_size == LR_SCHEDULER_DEFAULTS["step"]["lr_step_size"]
        assert cf.lr_step_gamma == LR_SCHEDULER_DEFAULTS["step"]["lr_step_gamma"]

    def test_min_lr_no_longer_collides_across_types(self):
        # Regression test for the historical bug: setup_experiment() used to write a
        # bare `cf.min_lr = 1e-9` for reduce_on_plateau immediately followed by
        # `cf.min_lr = 1e-8` for cosine_annealing, so the second silently won even
        # when reduce_on_plateau was the active type. Each type now only pulls its
        # own default when cf.min_lr isn't already set, so building each type in
        # isolation gets that type's own intended default.
        cf_plateau = _cf("reduce_on_plateau")
        build_lr_scheduler(cf_plateau, _optimizer(), _TRAIN_LOADER_STUB)
        assert cf_plateau.min_lr == LR_SCHEDULER_DEFAULTS["reduce_on_plateau"]["min_lr"] == 1e-9

        cf_cosine = _cf("cosine_annealing")
        build_lr_scheduler(cf_cosine, _optimizer(), _TRAIN_LOADER_STUB)
        assert cf_cosine.min_lr == LR_SCHEDULER_DEFAULTS["cosine_annealing"]["min_lr"] == 1e-8


class TestBackwardCompatWithOldSavedConfigs:
    """An old config.json saved by the pre-refactor code had ALL scheduler types'
    flat keys already baked in (setup_experiment used to set every one of them
    unconditionally), including the buggy min_lr=1e-8 for reduce_on_plateau (the
    cosine_annealing assignment that ran right after it in the old code). Loading
    such a config the way sr_toolbox.py does (AttrDict() + cf.update(load_json(...)))
    and building a scheduler from it must reproduce that exact historical value,
    not silently "fix" it to the new default -- old runs must stay reproducible.
    """

    def test_old_fully_populated_config_preserves_its_own_min_lr(self):
        cf = AttrDict()
        cf.update({
            "epochs": 200,
            "use_lr_scheduler": True,
            "lr_scheduler_type": "reduce_on_plateau",
            "lr_reduce_factor": 0.5,
            "lr_reduce_patience": 5,
            "min_lr": 1e-8,  # the historical "silently winning" value, not the new 1e-9 default
            "lr_step_size": 30,
            "lr_step_gamma": 0.5,
            "lr_exp_gamma": 0.85,
            "onecycle_initial_lr": 1e-4,
            "onecycle_max_lr": 1e-2,
            "onecycle_final_lr": 1e-8,
            "onecycle_pct_start": 0.3,
        })
        sched = build_lr_scheduler(cf, _optimizer(), _TRAIN_LOADER_STUB)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert sched.min_lrs[0] == 1e-8

    def test_old_config_missing_use_lr_scheduler_key_never_builds_a_scheduler(self):
        # Mirrors train.py's own gate: `if 'use_lr_scheduler' in cf.keys() and
        # cf.use_lr_scheduler:`. Some old saved configs predate the scheduler
        # feature entirely and have no such key -- build_lr_scheduler is simply
        # never called for them, so this documents the gate condition itself.
        cf = AttrDict()
        cf.update({"lr_scheduler": "reduce_on_plateau", "min_lr": 1e-9})  # old key name
        assert "use_lr_scheduler" not in cf.keys()
        should_build = "use_lr_scheduler" in cf.keys() and cf.get("use_lr_scheduler", False)
        assert not should_build
