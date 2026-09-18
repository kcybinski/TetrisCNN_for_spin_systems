"""Group B: the path/sweep system (tetriscnn/utils.py).

This is the highest-value file in the suite: the refactor is porting a sweep-aware
save/load path system onto `main`, and these tests pin down exact path strings so
backward compatibility can be verified mechanically.

A note on AttrDict (see test_utils_pure.py::TestAttrDict for the root cause):
`build_logdir_path` uses `hasattr(cf, 'label_param')`, `hasattr(cf, 'seeds')`,
`hasattr(cf, 'partition_index')` and `getattr(cf, 'model', 'tetriscnn')`. AttrDict's
missing-key error is AttributeError, so all of these gracefully treat a truly-absent
key as absent/default, matching normal Python attribute-access semantics. Every
`_cf(...)` helper below still sets these keys explicitly (that remains the realistic
shape of a config produced by `setup_experiment()`), but
`test_build_logdir_path_model_key_absent_defaults_to_tetriscnn` demonstrates that
omitting 'model' no longer crashes.
"""
from pathlib import Path

import pytest

from tetriscnn.utils import (
    AttrDict,
    build_logdir_path,
    get_experiment_level_path,
    get_sweepable_params,
)


def _cf(**overrides):
    """Minimal AttrDict with every key build_logdir_path's hasattr/getattr checks."""
    cf = AttrDict()
    cf.experiment_name = "singlerun"
    cf.dataset = "Paris_Ising"
    cf.task = "regression"
    cf.label_param = "delta"
    cf.model = "tetriscnn"
    cf.kernel_set = "smallkernels"
    cf.lambdas = [-1]
    cf.seeds = [42]
    cf.seed = 42
    cf.partition_index = None
    cf.lam = -1
    cf.update(overrides)
    return cf


# ---------------------------------------------------------------------------
# get_sweepable_params
# ---------------------------------------------------------------------------

class TestGetSweepableParams:
    def test_only_lists_with_len_gt_1_count(self):
        cf = _cf()
        cf.samples_per_pt_cap = [100, 200, 300]      # sweepable (len 3)
        cf.even_split = [True]                        # NOT sweepable (len 1)
        cf.hidden_size = 32                            # NOT sweepable (not a list)
        cf.some_pair = [1, 2]                          # sweepable (len 2)
        result = get_sweepable_params(cf)
        assert set(result.keys()) == {"samples_per_pt_cap", "some_pair"}
        assert result["samples_per_pt_cap"] == [100, 200, 300]

    def test_default_exclusions_are_honored(self):
        cf = _cf()
        cf.lambdas = [-5, -4, -3]           # excluded by default
        cf.seeds = [42, 43]                 # excluded by default
        cf.kernels = [[1], [2]]             # excluded by default
        cf.penalty_params = [1, 2, 3]       # excluded by default
        cf.filter_t_values = [1.0, 2.0]     # excluded by default
        cf.unique_labels = [0, 1]           # excluded by default
        cf.samples_per_pt_cap = [1, 2]      # NOT excluded -> should appear
        result = get_sweepable_params(cf)
        assert set(result.keys()) == {"samples_per_pt_cap"}

    def test_result_ordered_by_abbreviated_name(self):
        cf = _cf()
        # abbreviations: samples_per_pt_cap->spc, hidden_size->hs, even_split->es
        cf.samples_per_pt_cap = [1, 2]
        cf.hidden_size = [8, 16]
        cf.even_split = [True, False]
        result = get_sweepable_params(cf)
        # alphabetical by abbreviation: es < hs < spc
        assert list(result.keys()) == ["even_split", "hidden_size", "samples_per_pt_cap"]

    def test_custom_exclude_is_unioned_with_defaults_not_a_replacement(self):
        # Passing exclude=[] does NOT remove the default exclusions (lambdas/seeds/
        # kernels/penalty_params/filter_t_values/unique_labels) -- there is no way
        # to un-exclude those via this API. Passing a name DOES add it on top.
        cf = _cf()
        cf.lambdas = [-5, -4, -3]
        cf.samples_per_pt_cap = [1, 2]
        # exclude=[] "override attempt": lambdas still excluded (default), spc stays
        assert set(get_sweepable_params(cf, exclude=[]).keys()) == {"samples_per_pt_cap"}
        # explicitly excluding samples_per_pt_cap on top of the defaults empties it
        assert get_sweepable_params(cf, exclude=["samples_per_pt_cap"]) == {}


# ---------------------------------------------------------------------------
# build_logdir_path -- the full matrix
# ---------------------------------------------------------------------------

class TestBuildLogdirPathMatrix:
    def test_singlerun_no_sweep_single_seed(self):
        cf = _cf()
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_smallkernels_[-1]/singlerun_-1"

    def test_singlerun_no_sweep_multi_seed(self):
        cf = _cf(seeds=[42, 43], seed=42)
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_smallkernels_[-1]/singlerun_-1/seed_42"

    def test_lambdamax_hierarchical_single_param_sweep(self):
        cf = _cf(experiment_name="lambdamax", lam=4)
        got = build_logdir_path(cf, {"samples_per_pt_cap": 100})
        assert got == \
            "logs/lambdamax_Paris_Ising_regression_delta_smallkernels_[-1]/lambdamax_4/spc=100"

    def test_lambdamax_hierarchical_multi_param_sweep_multi_seed(self):
        cf = _cf(experiment_name="lambdamax", lam=4, seeds=[42, 43], seed=43)
        got = build_logdir_path(cf, {"samples_per_pt_cap": 100, "even_split": True})
        # hierarchical levels sorted by abbreviation: es < spc
        assert got == (
            "logs/lambdamax_Paris_Ising_regression_delta_smallkernels_[-1]/"
            "lambdamax_4/es=True/spc=100/seed_43"
        )

    def test_lambdatot_partition_index_suffix(self):
        cf = _cf(experiment_name="lambdatot", lam=2, partition_index=2)
        got = build_logdir_path(cf, None)
        assert got == \
            "logs/lambdatot_Paris_Ising_regression_delta_smallkernels_[-1]/lambdatot_2/partition_2"

    def test_weighted_loss_flat_sweep_single_seed(self):
        cf = _cf(experiment_name="weighted_loss", lam=4)
        got = build_logdir_path(
            cf, {"even_split": True, "samples_per_pt_cap": None, "use_weighted_loss": False}
        )
        assert got == (
            "logs/weighted_loss_Paris_Ising_regression_delta_smallkernels_[-1]/"
            "weighted_loss_4/es=True-spc=None-wl=False"
        )

    def test_weighted_loss_flat_sweep_multi_seed(self):
        cf = _cf(experiment_name="weighted_loss", lam=4, seeds=[42, 43, 44], seed=44)
        got = build_logdir_path(
            cf, {"even_split": False, "samples_per_pt_cap": None, "use_weighted_loss": True}
        )
        assert got == (
            "logs/weighted_loss_Paris_Ising_regression_delta_smallkernels_[-1]/"
            "weighted_loss_4/es=False-spc=None-wl=True/seed_44"
        )

    def test_non_tetriscnn_model_no_sweep(self):
        # model != "tetriscnn" -> base_name drops kernel_set/lambdas, adds model name
        cf = _cf(model="PhaseCNN")
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_PhaseCNN/singlerun_-1"

    def test_non_tetriscnn_model_hierarchical_sweep_multiseed_partition(self):
        cf = _cf(
            experiment_name="lambdamax", model="PhaseCNN", lam=1,
            seeds=[1, 2], seed=2, partition_index=1,
        )
        got = build_logdir_path(cf, {"samples_per_pt_cap": 50})
        assert got == (
            "logs/lambdamax_Paris_Ising_regression_delta_PhaseCNN/"
            "lambdamax_1/spc=50/seed_2/partition_1"
        )

    def test_label_param_none_omits_label_suffix(self):
        cf = _cf(label_param=None)
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_smallkernels_[-1]/singlerun_-1"

    def test_build_logdir_path_model_key_absent_defaults_to_tetriscnn(self):
        # build_logdir_path does `getattr(cf, 'model', 'tetriscnn')` intending to
        # default to "tetriscnn" when cf has no model field. AttrDict.__getattr__ now
        # raises AttributeError (not KeyError) for missing keys, so Python's getattr()
        # correctly catches it and applies the default -- a config missing 'model' is
        # treated the same as one with model="tetriscnn".
        cf = _cf()
        del cf["model"]
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_smallkernels_[-1]/singlerun_-1"


class TestEquivariantGroupInPath:
    """An equivariant run and a non-equivariant one can agree on experiment_name,
    dataset, task, label_param, kernel_set, lambdas, lam and seed while training
    genuinely different architectures (different branch type, and 5 or 6 branches
    instead of 10). build_logdir_path tags the symmetry group so the two do not
    resolve to the same folder and overwrite each other.
    """

    def test_non_equivariant_path_is_untagged(self):
        # The equivariant flag absent entirely (older configs) must not change paths.
        cf = _cf()
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_smallkernels_[-1]/singlerun_-1"

    def test_explicit_false_is_untagged(self):
        cf = _cf(equivariant=False, equivariant_group="C4")
        # group is set but unused, because equivariant is False
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_smallkernels_[-1]/singlerun_-1"

    def test_c4_is_tagged(self):
        cf = _cf(equivariant=True, equivariant_group="C4")
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_smallkernels_C4_[-1]/singlerun_-1"

    def test_d2_is_tagged(self):
        cf = _cf(equivariant=True, equivariant_group="D2")
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_smallkernels_D2_[-1]/singlerun_-1"

    def test_group_defaults_to_c4_when_absent(self):
        # Matches the default in ShapeAdaptiveConvNet/train.py/sr_toolbox.py.
        cf = _cf(equivariant=True)
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_smallkernels_C4_[-1]/singlerun_-1"

    def test_all_three_variants_are_mutually_distinct(self):
        """The actual collision this guards against."""
        paths = {
            build_logdir_path(_cf(), None),
            build_logdir_path(_cf(equivariant=True, equivariant_group="C4"), None),
            build_logdir_path(_cf(equivariant=True, equivariant_group="D2"), None),
        }
        assert len(paths) == 3

    def test_k4_alias_is_distinct_from_d2(self):
        # K4 is an alias for the same group as D2, but the kernel lists are built from
        # the string, so keep the folders separate rather than silently equating them.
        c_d2 = build_logdir_path(_cf(equivariant=True, equivariant_group="D2"), None)
        c_k4 = build_logdir_path(_cf(equivariant=True, equivariant_group="K4"), None)
        assert c_d2 != c_k4

    def test_tag_survives_sweep_seed_and_partition_levels(self):
        cf = _cf(
            experiment_name="lambdamax", lam=1, seeds=[42, 43], seed=43,
            partition_index=2, equivariant=True, equivariant_group="C4",
        )
        got = build_logdir_path(cf, {"samples_per_pt_cap": 50})
        assert got == (
            "logs/lambdamax_Paris_Ising_regression_delta_smallkernels_C4_[-1]/"
            "lambdamax_1/spc=50/seed_43/partition_2"
        )

    def test_non_tetriscnn_model_is_never_tagged(self):
        # Equivariance is a tetriscnn-branch concept; PhaseCNN/ResNet ignore the flag,
        # so tagging their folders would be misleading.
        cf = _cf(model="PhaseCNN", equivariant=True, equivariant_group="C4")
        assert build_logdir_path(cf, None) == \
            "logs/singlerun_Paris_Ising_regression_delta_PhaseCNN/singlerun_-1"


class TestBackwardCompatibilityKnownDivergence:
    """The pre-sweep-era path format was:

        logs/{experiment}_{dataset}_{task}_{label_param}_{kernel_set}_{lambdas}/
            {experiment}_{lam}/seed_{seed}

    build_logdir_path must still reproduce this exactly for a no-sweep, multi-seed
    config (verified in TestBuildLogdirPathMatrix.test_singlerun_no_sweep_multi_seed
    already, but re-asserted here explicitly against the format string as the
    "spec"). The ONE known divergence is the single-seed case.
    """

    def _pre_sweep_era_format(self, cf, seed):
        return str(Path(
            "logs",
            f"{cf.experiment_name}_{cf.dataset}_{cf.task}_{cf.label_param}_{cf.kernel_set}_{cf.lambdas}",
            f"{cf.experiment_name}_{cf.lam}",
            f"seed_{seed}",
        ))

    def test_multi_seed_reproduces_pre_sweep_era_format_exactly(self):
        cf = _cf(seeds=[42, 43, 44], seed=43)
        expected = self._pre_sweep_era_format(cf, seed=43)
        assert build_logdir_path(cf, None) == expected

    def test_single_seed_omits_seed_level_KNOWN_DIVERGENCE(self):
        """KNOWN, INTENTIONAL-LOOKING BUT UNCONFIRMED DIVERGENCE from the pre-sweep
        path format.

        Source: tetriscnn/utils.py, build_logdir_path, ~line 174:
            if hasattr(cf, 'seeds') and len(cf.seeds) > 1:
                path_parts.append(f"seed_{cf.seed}")

        For a SINGLE seed (len(cf.seeds) == 1), the new code OMITS the `seed_{seed}`
        path component entirely, whereas the documented pre-sweep-era format always
        included it. This means:
          - old single-seed runs live at   .../{experiment}_{lam}/seed_{seed}/
          - new single-seed runs live at    .../{experiment}_{lam}/
        This is a real behavior change a human should sign off on before the refactor
        ports this path system to `main`: any code that globs for `seed_*` under a
        single-seed run's lambda directory will find nothing post-refactor.
        """
        cf = _cf(seeds=[42], seed=42)
        pre_sweep_format = self._pre_sweep_era_format(cf, seed=42)
        actual = build_logdir_path(cf, None)

        assert actual != pre_sweep_format
        assert actual == pre_sweep_format.rsplit("/seed_42", 1)[0]
        assert "seed_42" not in actual


# ---------------------------------------------------------------------------
# get_experiment_level_path
# ---------------------------------------------------------------------------

class TestGetExperimentLevelPath:
    def test_walks_up_to_the_level_containing_metrics_per_lambda_json(self, tmp_path):
        # metrics_per_lambda.json placed ONE level above the seed dir (at the
        # lambda level), not two -- distinguishes "walk up until found" from the
        # naive "always 2 levels up" fallback.
        seed_dir = tmp_path / "logs" / "expA" / "lambdamax_4" / "seed_42"
        seed_dir.mkdir(parents=True)
        lambda_level = seed_dir.parent
        (lambda_level / "metrics_per_lambda.json").write_text("{}")

        result = get_experiment_level_path(seed_dir)
        assert result == lambda_level
        assert result != seed_dir.parent.parent  # would be the naive-fallback answer

    def test_walks_up_past_extra_sweep_levels(self, tmp_path):
        seed_dir = tmp_path / "logs" / "expA" / "lambdamax_4" / "spc=100" / "seed_42"
        seed_dir.mkdir(parents=True)
        experiment_level = tmp_path / "logs" / "expA"
        (experiment_level / "metrics_per_lambda.json").write_text("{}")

        result = get_experiment_level_path(seed_dir)
        assert result == experiment_level

    def test_recognizes_remake_variant_filename(self, tmp_path):
        seed_dir = tmp_path / "logs" / "expA" / "lambdamax_4" / "seed_42"
        seed_dir.mkdir(parents=True)
        (seed_dir.parent / "metrics_per_lambda_remake.json").write_text("{}")

        result = get_experiment_level_path(seed_dir)
        assert result == seed_dir.parent

    def test_fallback_two_levels_up_when_no_metrics_file_exists(self, tmp_path):
        seed_dir = tmp_path / "logs" / "expA" / "lambdamax_4" / "seed_42"
        seed_dir.mkdir(parents=True)
        # no metrics_per_lambda*.json anywhere

        result = get_experiment_level_path(seed_dir)
        assert result == seed_dir.parent.parent  # explicit "up two levels" fallback
        assert result == tmp_path / "logs" / "expA"
