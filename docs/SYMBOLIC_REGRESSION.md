# Symbolic regression (optional)

> **Optional.** Nothing else in the repository needs this route or its dependencies.
> Training, the primary regression-based interpretation, the figure notebooks and the
> tests all work without it.

Symbolic regression (PySR) is an ***optional***, **secondary, post-hoc** interpretability route,
run only after a model has finished training and only against its saved `net1.pt`/
`net2.pt`. The **primary** interpretability route is a (multi)linear regression of a
branch's activation onto its spin correlators, implemented by
`fit_activation_to_correlators()` in [`tetriscnn/interpret.py`](../tetriscnn/interpret.py)
(the code behind the manuscript's closed-form correlator equations, e.g. Ising
`z[■] = −4.626·C^Z[■] + 0.954`, Eq. (12)); SR exists mainly to demonstrate, via the
out-of-distribution metrics below, that a closed-form equation extracted
in-distribution does *not* reliably generalize, which is itself the finding App. I of
the manuscript reports. **SR is not run from `main.py` and has no `cf.*` fields of
its own** (older documentation describing `cf.run_sr`/`cf.fit_sr`/`cf.sr_mode` on the
training-time `cf` is stale); it is driven entirely by
[`sr_toolbox.py`](../sr_toolbox.py) and a single `SRConfig` dataclass.

## Installation

PySR, plotly for the SR plots, and kaleido (which plotly uses to save them as PNG/PDF)
are optional dependencies, not part of `environment.yml` or `requirements.txt`. Add
them with

```bash
pip install -r requirements-sr.txt      # or: pip install -e ".[sr]"
```

Without PySR, importing `tetriscnn.symbolic_regression` raises an `ImportError` saying
so. Nothing outside `sr_toolbox.py` and `tetriscnn/sr_experiments.py` needs either
package. PySR downloads and precompiles its own Julia runtime the first time it is
imported, so that first import is slow; no manual Julia install is needed. Kaleido
in turn uses a Chrome/Chromium installation; if it reports none, run
`plotly_get_chrome` once.

## Import order

PySR wraps `SymbolicRegression.jl` and boots a Julia
runtime via `juliacall` on first import; that boot must claim the process's
threading/signal setup before `torch` does, or the two runtimes can deadlock. Any
script driving SR (`sr_toolbox.py`, or a notebook doing the same) must import
`tetriscnn.symbolic_regression` (or `pysr` directly) **before** importing `torch`,
`tetriscnn.models`, or anything that transitively imports either.

## Running `sr_toolbox.py`

Configuration lives in the file's `__main__` block, edited directly (no CLI, same
convention as `main.py`):

```python
remake_flags = {
    "fit_sr": True,          # fit a new PySR model from NN activations
    "remake_sr_plot": True,  # remake SR plots from an existing saved PySR model
}
sr_config = SRConfig(sr_mode="raw", top_k=3, gen_samples=2500, num_flips=4, ...)
basepath = Path("logs/.../seed_42")   # or an SR subfolder; None opens a directory picker
```

`sr_toolbox.py` holds only this configuration; it hands it to `run_sr_toolbox()` in
[`tetriscnn/sr_experiments.py`](../tetriscnn/sr_experiments.py), which auto-detects which
of two scenarios `basepath` names:

| Scenario | `basepath` contains | `sr_config` source | Behavior |
|---|---|---|---|
| **A: SR subfolder** | `sr_config.json` (e.g. `seed_42/SR_default/`) | Loaded from that folder's own `sr_config.json` | NN weights are loaded from the *parent* seed folder; results are written back into the exact subfolder selected (`sr_folder_name_override` is set to that folder's own name, so a freshly computed path can never disagree with where the user pointed). |
| **B: seed folder** | `net1.pt` + `net2.pt` (e.g. `seed_42/`) | The in-file `sr_config` variable | The target SR subfolder name is computed fresh via `build_sr_folder_path()` (below). |

Any other selection prints an error explaining the two valid shapes.

`fit_sr=True, remake_sr_plot=False` fits a new model (creating the SR subfolder if
needed) and does not require one to already exist; `remake_sr_plot=True,
fit_sr=False` requires `model_sr_raw.pkl` to already be present in the target folder,
and errors out (naming what to set instead) if it is not.

## The `SRConfig` dataclass

Every field below lives in `tetriscnn/symbolic_regression.py`. `SRConfig.save()`/
`.load()` serialize to `sr_config.json` alongside the SR results (the two ergonomic
methods `add_binary_operator()`/`add_unary_operator()` mutate an existing config
in place rather than requiring the whole operator dict to be retyped); `.load()`
tolerates and drops any key that no longer matches a current field, so an
`sr_config.json` recorded by an older version of the class keeps loading.

**Mode and feature selection**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `sr_mode` | `str` | `"raw"` | The only implemented mode (per-sample activations). `"averaged"` is an obsolete string kept only for backward-compatible `.load()`; the function that implemented it no longer exists. |
| `top_k` | `int` | `3` | Number of branches, ranked by mean `\|activation\|` on the training split, selected as SR input features. **Distinct from** the `top_k` parameter of `save_sr_hof_with_avg()` (default `5`), which instead controls how many top-ranked Hall-of-Fame equations get overlaid on the snapshot-average plot; the two are unrelated despite the shared name. |
| `classification_sr_mode` | `str` | `'logit_diff'` | How a binary classifier's `(N, 2)` logits collapse to one scalar SR target. `'logit_diff'` (Mode A) uses `logit_0 - logit_1`. `'single_logit'` (Mode B, binary only) auto-picks whichever logit correlates most positively with the mean top-k activation, recording the choice in the internal `_picked_logit_idx` field (not itself user-set). |

**Operator sets**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `binary_operators` | `list[str]` | `["+", "*"]` | The binary primitives PySR may compose. |
| `unary_operators` | `list[str]` | `["neg", "square", "cube", "quart(x) = x^4"]` | The unary primitives; `exp` is supported by the codebase but commented out of the default set. |
| `extra_sympy_mappings` | `dict` | `{"quart": lambda x: x**4}` | Maps a custom operator string (like `quart`) to a callable sympy can use when rendering/evaluating the fitted expression. |

**Constraints**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `complexity_of_variables` | `int` | `2` | Per-token cost of using a free variable (a branch activation); raising it discourages equations that reference many distinct branches. `set_complexity_of_variables()` is the ergonomic setter. |
| `complexity_of_constants` | `int` | `1` | Per-token cost of a free numeric constant; raising it (`set_complexity_of_constants()`) discourages equations with many independently-fit constants. |
| `constraints` | `dict` | `{"*": (-1, 1), "square": 3, "cube": 3, "quart": 3}` | PySR structural constraints. `"*": (-1, 1)` forces the right argument of multiplication to be a constant (no variable-times-variable terms); the unary entries cap nesting depth. |
| `nested_constraints` | `dict` | Forbids `square`/`cube`/`quart` from nesting inside one another | Prevents degenerate compositions like `square(cube(x))`. |
| `complexity_of_operators` | `dict` | `{"+": 1, "*": 1, "square": 2, "cube": 2, "quart": 3}` | Per-operator complexity cost, feeding the parsimony-vs-fit trade-off PySR optimizes. |

**Optimization parameters**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `maxdepth` | `int` | `15` | Maximum expression tree depth. |
| `maxsize` | `int` | `25` | Maximum expression tree size (node count). |
| `parsimony` | `float` | `0.01` | Base complexity penalty weight in PySR's loss. |
| `adaptive_parsimony_scaling` | `int` | `2000` | How aggressively PySR adapts the parsimony weight over the search. |
| `weight_optimize` | `float` | `0.001` | Probability weight PySR gives to constant-optimization mutations. |
| `elementwise_loss` | `str` | `"L1DistLoss()"` | Mean-absolute-error loss, chosen for robustness (over L2) to the noisy experimental activations being fit. |
| `model_selection` | `str` | `"best"` | PySR's own internal equation-selection criterion (distinct from this codebase's own Borda-count `optimized_picked`, below). |

**PySR execution**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `niterations` | `int` | `15000` | Search budget; the docstring notes `2000` is usually already sufficient for a quick look. |
| `populations` | `int` | `25` | Number of parallel evolutionary populations. |
| `procs` | `int` | `0` | Worker processes (`0` lets PySR choose). |
| `turbo` | `bool` | `False` | PySR's experimental fast-evaluation mode. |
| `batching` | `bool` | `True` | Whether PySR evaluates candidate equations on mini-batches rather than the full dataset each generation. |
| `batch_size` | `int` or `None` | `None` | Batch size when `batching=True`; `None` lets PySR auto-determine it. |

**Generalization (OOD) dataset**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `gen_samples` | `int` | `2500` | Samples generated per synthetic configuration type (paramagnetic/ferromagnetic/antiferromagnetic; see the Pipeline below). |
| `num_flips` | `int` | `4` | Spin flips applied to the near-uniform (ferro) or checkerboard (antiferro) base configuration to add controlled disorder. |
| `gen_seed` | `int` or `None` | `2137` | Seed for the OOD probe generation. These configurations are otherwise redrawn on every evaluation, so a fixed seed is what makes the Hall-of-Fame's OOD columns (and hence the Borda-count `optimized_picked` choice) reproducible across separate runs; `None` draws fresh configurations each time. |

**TensorBoard, naming, and equation selection**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `use_tensorboard` | `bool` | `True` | Opt-in TensorBoard logging of the PySR search. |
| `sr_folder_name_override` | `str` or `None` | `None` | Forces a specific SR output folder name, bypassing `build_sr_folder_path()`'s auto-generated name; useful to keep experiments that share the same "significant" parameters (below) from colliding on disk. |
| `selected_equation_idx` | `int`, `list[int \| None]`, or `None` | `None` | Manually overrides which Hall-of-Fame equation the 3D generalization plots highlight. An `int` applies to every output; a list gives one override per output dimension; `None` defers to the automatic pick. |
| `optimized_picking_metrics` | `list[(str, bool)]` or `None` | `None` (resolved by `get_optimized_picking_metrics()` from the task type) | The Borda-count leaderboard used to choose the recommended equation; see "Equation selection" below. |

**Technical**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `precision` | `int` | `64` | Floating-point precision PySR fits in. |
| `temp_equation_file` | `bool` | `False` | Whether PySR writes its equation search log to a temp file. |
| `delete_tempfiles` | `bool` | `True` | Whether PySR cleans up its own temp files after a run. |

## SR folder naming

`build_sr_folder_path()` (`tetriscnn/utils.py`) names the output folder
`SR_default` when every `SRConfig` field is at its default, or
`SR_{param}={value}_{param}={value}...` for each field `get_significant_params()`
finds to differ from a fresh `SRConfig()` (currently tracked: `sr_mode`, `top_k`,
`classification_sr_mode`, the operator sets, `maxdepth`, `maxsize`,
`elementwise_loss`, `complexity_of_constants`, `complexity_of_variables`, and
`gen_samples`), unless `sr_folder_name_override` is set, which wins outright.

## Pipeline (`run_sr_raw_mode()`)

1. **Extract activations.** Push the train split (and, separately, the val split)
   through `net1` → `net2` once to record bottleneck activations `z` and task-head
   predictions.
2. **Feature selection.** Rank branches by mean `\|activation\|` on the training
   split; keep the top `top_k` as `sr_z_mask`. SR variable names encode the actual
   branch index (`a5`, `a2`, `a7`, ...), not a re-numbered 0..k-1 sequence.
3. **Fit PySR.** `fit_symbolic_regression()` fits `z[:, sr_z_mask] → y_pred`, where
   `y_pred` is the **network's own prediction**, not the ground-truth physics label:
   SR is explaining what the readout head does with the bottleneck, not re-deriving
   the physics from scratch. Saved as `model_sr_raw.pkl`.
4. **Generate OOD probes.** Three families of *physical* spin configurations the
   network never trained on are synthesized directly in spin space, via
   `generate_all_generalization_configs()`:
   - `paramagnetic`: every spin i.i.d. `±1` (fully disordered).
   - `ferromagnetic`: near-uniform (all `+1` or all `-1`) with `num_flips` random flips.
   - `antiferromagnetic`: checkerboard, with `num_flips` flips.

   For the two-channel `Paris_XY_XZ` dataset, the two channels are built as
   complementary orders (Z-ferro pairs with X-para, Z-para with X-ferro, and so on).

   The three per-family generators themselves live in `tetriscnn/synthetic_configs.py`,
   not here: they are plain numpy draws and need no PySR, so `Figure6.ipynb` can reuse
   the same probes on a plain installation. `generate_all_generalization_configs()`
   stays in this module because it reads its sample count, flip count and seed from
   `SRConfig`.
5. **Score and pick.** `save_sr_hof_with_avg()` evaluates every Hall-of-Fame equation
   on train, val, and all three OOD probe sets, and picks a recommended equation by
   Borda count (below).
6. **Plot.** Snapshot-average overlays and 3D activation-vs-prediction plots
   (`plot_3d_augmentation()`, in `tetriscnn/symbolic_regression.py` with the rest of the
   SR plotting) are written per output and per OOD type.

## Equation selection (Borda count)

Two different "picked" equations are recorded per output, and they can disagree:

- **`SR_picked`**: PySR's own choice, the equation with the highest internal `score`
  in `model_sr.equations_`, a training-time complexity/loss trade-off with no
  knowledge of validation or OOD behavior.
- **`optimized_picked`**: this codebase's recommended choice, computed by Borda-count
  rank aggregation over `get_optimized_picking_metrics()`'s leaderboard list (each a
  `(metric_name, higher_is_better)` pair). Every equation is ranked on every listed
  metric (`NaN` always ranks last); an equation's score is the sum of its per-metric
  ranks, lowest total wins, ties broken by highest `r2_val`. The default metric list
  deliberately mixes in-distribution validation metrics (`r2_val`, `nrmse_val`,
  `rho_val`) with the three OOD metrics (`rho_ferro`, `rho_antiferro`, `rho_para`,
  and their NRMSE counterparts), so an equation that fits validation well but
  collapses out-of-distribution is penalized relative to one that holds up on both.

**Reading the result:** in-distribution `r2_val` alone is misleading, since it only
shows the equation matches the network on the training distribution. Always read it
alongside the OOD columns; a large gap between in- and out-of-distribution
performance is itself the finding to report (this is the concrete conclusion behind
App. I's claim that SR is an "expected-form extractor, not a discoverer": many
distinct symbolic forms fit the activations about equally well in-distribution, and
which one PySR returns is dictated by the operator/complexity priors above, not by a
unique underlying expression).

## Output files (per SR subfolder)

```
seed_{seed}/{SR_folder}/            # SR_default/, SR_topk=5_.../, or an override name
├── sr_config.json                  # serialized SRConfig
├── model_sr_raw.pkl                # fitted PySR model (pickle)
├── equations/
│   ├── hof_{label}.csv / .md       # per-output Hall of Fame, all per-split metrics
│   └── hof_all_outputs.csv         # concatenated HOF across every output
└── plots/
    └── *.png / *.pdf / *.html      # snapshot-average + 3D OOD generalization plots
```

The exact HOF column set (`SRConfig.get_hof_columns()`) differs between regression
and classification tasks; both include per-split R²/MSE/NRMSE/Spearman-ρ metrics on
train, val, and each of the three OOD probe types, plus `SR_picked`/`optimized_picked`
boolean flags.

## Configuration recipes

Adapt these in `sr_toolbox.py`, after `sr_config` is created.

Add operators:

```python
sr_config.add_binary_operator("/", complexity=2, constraint=(3, 5))
sr_config.add_unary_operator("exp", complexity=4, constraint=5, nested_constraints={"exp": 0})
sr_config.add_unary_operator("sin", complexity=3)
```

Override the auto-generated SR folder name, to keep experiments that share the same
significant parameters from colliding on disk:

```python
sr_config.sr_folder_name_override = "SR_div_exp_no_constraints"
```

Change which metrics drive the automatic ("optimized") equation choice, as a list of
`(metric_name, higher_is_better)` pairs ranked by Borda count (see "Equation
selection" above). Available metrics: `r2_val`, `nrmse_val`, `rho_val`, `rho_ferro`,
`rho_antiferro`, `rho_para`, `nrmse_ferro`, `nrmse_antiferro`, `nrmse_para`.

```python
sr_config.optimized_picking_metrics = [
    ("r2_val", True), ("nrmse_val", False), ("rho_val", True),
    ("rho_ferro", True), ("rho_antiferro", True), ("rho_para", True),
    ("nrmse_ferro", False), ("nrmse_antiferro", False), ("nrmse_para", False),
]
```

Penalise free variables more, so equations use fewer distinct activations:

```python
sr_config.set_complexity_of_variables(3)  # default is 2
```

Choose which Hall-of-Fame equation the 3D plots use on a remake (`None` keeps PySR's
own pick or the Borda-count "optimized" pick):

```python
sr_config.selected_equation_idx = 0       # the same rank-0 equation for every output
sr_config.selected_equation_idx = [0, 1]  # one choice per output
```
