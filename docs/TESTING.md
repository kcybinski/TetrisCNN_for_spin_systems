# The test suite

`tests/` is a characterization suite: a behavior-preservation safety net built while
`tetriscnn` was refactored, not a spec of intended behavior. Where current behavior
looks wrong, the tests still pin it down (with a comment or marker saying so) rather
than silently asserting what it "should" do. `tests/conftest.py` states this policy and
pins the working directory to the repository root so results do not depend on where
`pytest` was invoked from.

```bash
pytest                                    # the full suite
pytest -m "not slow"                      # skip the real-training-loop tests
python scripts/golden_run.py --check      # numerical fingerprint of a reference run
python scripts/golden_bfa_selection.py --check   # fingerprint of the branch-fit selector
```

Three pytest markers, declared in `pytest.ini`, gate tests that a fresh clone or a
CPU-only machine cannot run: `data` (needs the `datasets/`/`data/` snapshot tree),
`slow` (runs a real training loop), and `accelerator` (needs a non-CPU device). Gated
tests skip cleanly rather than failing, so `pytest` passes end to end immediately
after `git clone`, before any data is fetched.

## What each file tests

**`test_utils_pure.py`** — pure-function characterization of small utilities in
`tetriscnn/utils.py` (no data files, no training, whole file well under 5s). Pins the
current numeric behavior of these functions so a refactor can be checked against them
float-for-float, including `AttrDict`'s missing-key behavior (`AttributeError`, not
`KeyError`), which several other modules rely on via `hasattr`/`getattr`.

**`test_paths.py`** — the sweep-aware save/load path system in `tetriscnn/utils.py`
(`build_logdir_path` and friends). The highest-value file in the suite: it pins exact
path strings so backward compatibility with runs already on disk under `Plots_data/`
and `logs/` can be verified mechanically, not by inspection.

**`test_weighted_loss.py`** — `tetriscnn.experiments.attach_sample_weights` (Group C).
Also the one file that pays the cost of importing `tetriscnn.experiments`, which pulls
in `tetriscnn.models`, `tetriscnn.datasets` and `tetriscnn.plots`
(matplotlib/seaborn/statsmodels); every other test file avoids that import
specifically to stay fast.

**`test_smoke_data.py`** — data-dependent smoke tests (Group D). Every test needs the
experimental snapshot tree; the whole module is skipped cleanly when it is absent.

**`test_device.py`** — device-placement characterization for `tetriscnn/models.py`.
Targets a class of bug invisible on a CPU-only machine: a module whose buffers end up on
a different device than its parameters, which builds and trains fine until something
touches the buffer before the whole net is `.to(DEVICE)`'d, then fails with a
CUDA/MPS device-mismatch `RuntimeError`. `ConvBranch_Equivariant` is the specific
motivating case, since it uses its mask at construction time.

**`test_interpret.py`** — `tetriscnn.interpret.fit_activation_to_correlators`, the
primary interpretability route (regressing a branch activation onto the spin
correlators of its filter footprint). The core guarantee: when the activation is a known
linear combination of correlators, OLS recovers those coefficients at $R^2=1$. Also
locks in the correlator-term enumeration (single- and cross-basis counts) and confirms
the module carries no torch/PySR import cost.

**`test_branch_fits.py`** — `tetriscnn.plots.plot_branch_fits`, i.e. `cf.fit_branches`.
Generalizes `Figure5.ipynb`'s hand-picked-branch correlator regression figure: it
first calls `get_active_branches` to find which branches cleared the learning-rate noise
floor, then fits and plots exactly those. Uses synthetic spin arrays, like
`test_interpret.py`, so no dataset files are needed.

**`test_history_plot.py`** — pure-function tests for `tetriscnn.plots.plot_history`,
`get_active_branches`, and `cf.enabled_metrics` handling in `plot_lambda`/`plot_lbc`.
Regression coverage for three release-cleanup fixes, including making the training-curve
grid adaptive to `cf.enabled_metrics` instead of a fixed 4x2 grid with dead panels.

**`test_lr_scheduler.py`** — `tetriscnn.train.build_lr_scheduler` and
`LR_SCHEDULER_DEFAULTS`. Characterizes the mechanism that declutters
`main.py::setup_experiment()`: only the *active* `cf.lr_scheduler_type`'s parameters need
to be set by the caller, and all five scheduler types must keep constructing the exact
`torch.optim.lr_scheduler` class, with the same parameters, they did before the refactor.

**`test_json_arrays.py`** — how numeric arrays are written to and read back from
`config.json`. `save_json()` used to record numpy arrays via `str()`, so most runs under
`Plots_data/` store fields like `unique_labels` as a printed array rather than a JSON
list; three separate readers had grown their own fix. These tests pin the shared
replacement reader against every recorded run (so the consolidation is provably a no-op
on existing data) and pin the writer so new runs do not reintroduce the problem.

**`test_remake_flags.py`** — the remake path (`tetriscnn.experiments`): re-plotting a
recorded run from its checkpoint and `config.json` without retraining. Every manuscript figure is drawn this
way, and the path is easy to break silently because ordinary training never exercises
it. Runs the real remake logic against a copy of a recorded run, so nothing under
`Plots_data/` is written to.

**`test_simulated_datasets.py`** — the simulated datasets (ILGT, 1D TFIM, XXZ) TetrisCNN
was originally developed on, kept working alongside the experimental route. Checks each
is reachable through `create_datasets()`, that the per-snapshot tuning-parameter array
the phase-indicator machinery reads is populated on the pre-split path, and that the
likely user-error failure modes report what to do about them.

**`test_register_dataset.py`** — the `register_dataset()` hook that lets a user-defined
data source plug into `create_datasets()`. Uses a tiny in-memory processor (no files),
and checks that a registered name is dispatched to its factory with the task-derived
label settings, that the pooled and pre-split processor shapes both work, and that an
unregistered name still fails with the list of choices. Also covers the config defaults
(`apply_config_defaults`): a config naming only `dataset` and `task` builds datasets and
trains, explicit values are never overwritten, missing required keys are named, and an
unset `penalty_params` is the only default that warns. Finally, it checks the
`SnapshotProcessor` base class on an in-memory format: 0/1 to ±1 mapping, bases stacked
as channels, regression labels, `_phase_label` for classification (and the error
without it), and learning by confusion. The worked, user-facing version is
`notebooks/BYODataset_Tutorial.ipynb`.

**`test_optional_sr_boundary.py`** — the line between the main install and the optional
symbolic-regression extra. Parses every library module and every notebook and asserts
none of them imports PySR, plotly, or `tetriscnn.symbolic_regression` at module level,
so a plain installation runs all of them. Written after exactly that broke: the SR
plotting move stranded the synthetic spin-configuration generators inside
`symbolic_regression.py`, and `Figure6.ipynb` imported three of them for its
out-of-distribution probes, which made a main-text figure need PySR. They now live in
`tetriscnn/synthetic_configs.py`. Static analysis only, so it costs nothing and needs
neither PySR nor data.

**`test_equivariant.py`** — data-free characterization of the C4-equivariant branch
(`tetriscnn.models.ConvBranch_Equivariant`) and its wiring into `ShapeAdaptiveConvNet`
and `set_kernels`: the rotation-averaged convolution of Eq. (6) in the manuscript, with
implementation details in the class docstring. Must
stay fast (a few seconds).

**`test_mixed_equivariance.py`** — the per-branch `branch_groups` mechanism
(`cf.kernel_set='smallkernels_mixed'`) and the `equivariant_rebate` penalty multiplier
that lets equivariant and non-equivariant branches coexist in one net. The overriding
requirement is that the default path (no `branch_groups`) is provably untouched, so the
mixed feature cannot leak into ordinary runs.

## The golden-run scripts

`scripts/golden_run.py` and `scripts/golden_bfa_selection.py` sit alongside `tests/` but
run outside pytest, as a `--record`/`--check` pair rather than assertions. Each fixes a
seed, runs something deterministic (a small real training run for the former, a
best-subset correlator selection on synthetic snapshots for the latter), and fingerprints
the numeric outcome into `tests/golden/{baseline,bfa_selection}.json`. `--check` re-runs
and diffs against that fingerprint, catching a silent numerical drift that no individual
unit assertion would; `--record` regenerates it after an intentional change. Both are
part of the Installation verification in the top-level README.
