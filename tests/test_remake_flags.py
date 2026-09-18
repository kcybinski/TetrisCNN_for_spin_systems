"""The remake path: re-plotting a recorded run without retraining it.

Every figure in the manuscript is drawn from runs recorded under `Plots_data/`, and
the remake flags are how those plots are regenerated from a checkpoint rather than by
repeating the training. That path is easy to break without noticing, because ordinary
training never exercises it: it loads its whole configuration back from the run's own
`config.json`, including keys that current code no longer sets.

These tests run the real remake against a copy of a recorded run, so nothing in
`Plots_data/` is written to.
"""
import shutil

import pytest

from tests.conftest import REPO_ROOT, data_available

RECORDED_RUN = REPO_ROOT / "Plots_data" / "Ising_C4_smallkernels_lam3_seed42_p2"

pytestmark = pytest.mark.skipif(
    not (RECORDED_RUN / "config.json").exists(),
    reason="recorded run not present in Plots_data/",
)

REMAKE_FLAGS = ("remake_history_and_pt_plots", "remake_lambda_plot", "remake_sweep_plot")

PLOT_CONFIG = {"enabled_metrics": {"z", "loss", "goodness"}, "fit_branch_number": 1}


@pytest.fixture
def run_copy(tmp_path):
    """A throwaway copy of one recorded run, laid out as a seed-level folder."""
    dest = tmp_path / "seed_42"
    shutil.copytree(RECORDED_RUN, dest)
    return dest


def _flags(active):
    flags = {name: False for name in REMAKE_FLAGS}
    flags[active] = True
    return flags


def test_config_round_trips_out_of_a_recorded_run(run_copy):
    from tetriscnn.utils import load_or_init_config

    cf = load_or_init_config(_flags("remake_history_and_pt_plots"), PLOT_CONFIG, run_copy)

    # The whole point of the remake path: the run describes itself, so the plots are
    # redrawn under the configuration that produced them rather than today's defaults.
    assert cf.dataset == "Paris_Ising"
    assert cf.task == "partition"
    assert cf.logdir == run_copy
    # Kernel specs gained a trailing stride element after some runs were recorded;
    # loading has to pad the older ones back up or the branch drawing fails.
    assert all(len(spec) == 5 for spec in cf.kernels)
    # SR is driven entirely by sr_toolbox.py and must stay off on this path.
    assert cf.run_sr is False and cf.fit_sr is False


@pytest.mark.parametrize("flag", REMAKE_FLAGS)
@pytest.mark.skipif(not data_available(), reason="snapshot data not present")
@pytest.mark.slow
def test_remake_flag_runs(run_copy, flag):
    from tetriscnn import experiments

    flags = _flags(flag)
    run_dirs = experiments.find_run_directories(run_copy) or [run_copy]
    experiments.remake_plots_for_runs(run_dirs, flags, PLOT_CONFIG, run_copy)


def test_remake_flags_are_mutually_exclusive(run_copy):
    from tetriscnn.utils import load_or_init_config

    both = {name: False for name in REMAKE_FLAGS}
    both["remake_history_and_pt_plots"] = True
    both["remake_lambda_plot"] = True
    with pytest.raises(AssertionError):
        load_or_init_config(both, PLOT_CONFIG, run_copy)


def test_should_train_is_false_under_every_remake_flag():
    from tetriscnn.utils import AttrDict, should_train

    for flag in REMAKE_FLAGS:
        cf = AttrDict(); cf.update(_flags(flag))
        assert should_train(cf) is False, f"{flag} must not retrain"

    cf = AttrDict(); cf.update({name: False for name in REMAKE_FLAGS})
    assert should_train(cf) is True
