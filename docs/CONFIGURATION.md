# Configuration reference

TetrisCNN has no command-line interface. Every experiment is defined by editing
`setup_experiment(cf)` near the top of [`main.py`](../main.py) and rerunning
`python main.py`. `cf` is an `AttrDict` (a `dict` subclass with attribute access,
defined in [`tetriscnn/utils.py`](../tetriscnn/utils.py)) that starts empty and is
populated field by field inside `setup_experiment()`. Nothing about `cf`'s shape is
declared in advance: any attribute assigned there becomes a configuration field.

A field left unset is filled in when it is first needed. `create_datasets()` and
`train()` both start by calling `apply_config_defaults(cf)`
([`tetriscnn/utils.py`](../tetriscnn/utils.py)), which copies every missing key from
`CONFIG_DEFAULTS`. General settings (epochs, batch size, learning rate, early stopping,
architecture, `even_split`, `normalize_labels`, ...) default to the values
`setup_experiment()` uses; settings that only some datasets read (`param_cutoffs`,
`filter_t_values`, `samples_per_pt_cap`, `label_subset_count`) default to "off". For
training it also derives `cf.kernels` from `cf.kernel_set` and the loss/metric names
from `cf.task`. The filled-in values are written to the run's `config.json`, so the
record stays complete. A few fields have no default:

- `cf.dataset` and `cf.task` are always required, and `cf.logdir` for training; a
  missing one raises `ValueError` naming it.
- `cf.penalty_params`, the L1 penalty ramp, falls back to `[10, -3, 3, 1]`
  ($\lambda_{\max} = 3$, the manuscript's value) with a `UserWarning`, because the
  sparsity strength is a modelling choice rather than a technicality.
- `cf.partition_index` (for `partition`/`lbc`) and `cf.lam` (for `lambdatot`) are still
  checked where they are used.

Other fields are read with `getattr(cf, ..., default)` at the point of use.

Once built, `cf` flows through the whole run:

1. The `__main__` block of `main.py` sets the remake flags, the plotting options and
   the folder to re-plot, and passes them with `setup_experiment` to
   `run_or_remake()` in [`tetriscnn/experiments.py`](../tetriscnn/experiments.py).
   That module holds everything that runs an experiment; `main.py` itself only
   describes one. `run_or_remake()` calls `load_or_init_config()`, which either
   returns a fresh `cf` (normal training) or reloads a `config.json` from disk (a
   remake; see [Section 9](#9-remake-flags-and-re-plotting)), then calls
   `setup_experiment(cf)` when not remaking.
2. `run_experiments(cf)` inspects `cf` for sweepable fields (any field whose value is
   a `list` of length greater than one) and drives one training run per point of the
   sweep, calling into `run_seeds()`, `run_lbc()`, or `run_partition()` depending on
   `cf.task` ([Section 8](#8-parameter-sweeps)).
3. Each individual run calls `create_datasets(cf)` ([`tetriscnn/datasets.py`](../tetriscnn/datasets.py)) to build
   the train/val split, then `train(cf, "metrics.json")`
   (`tetriscnn/train.py`), which reads dozens of `cf.<field>` values directly: the
   model, the optimizer, the scheduler, the loss, the L1 penalty.
4. `cf` (JSON-serializable fields only; see `save_json()`'s `_json_safe` helper) is
   written to `config.json` inside the run's log directory, so every run is
   reproducible from its own saved configuration, and remaking plots later reloads
   exactly this file.

Because `cf` is just a dict, any field can be swept by assigning it a list instead of
a scalar; the sweep machinery in `tetriscnn/utils.py` discovers this automatically
(see [Section 8](#8-parameter-sweeps)). The sections below mirror the grouping
`setup_experiment()` itself uses, followed by a dedicated section on symbolic
regression (configured separately, in `sr_toolbox.py`) and a set of worked examples.

---

## 1. Experiment type and task

| Field | Type | Default in `setup_experiment()` | Legal values | Purpose |
|---|---|---|---|---|
| `experiment_name` | `str` | `"singlerun"` | `"singlerun"`, `"lambdamax"`, `"lambdatot"`, `"weighted_loss"` | Selects which penalty-construction branch `run_experiments()` uses and how the log path is built. `"singlerun"`/`"lambdamax"` share the same area-keyed penalty formula and differ only in whether `cf.lambdas` holds one value or a sweep; `"lambdatot"` applies a single uniform penalty to every branch regardless of pattern size; `"weighted_loss"` is the data-splitting/loss-weighting ablation ([Section 8](#weighted-loss-experiment-scenario-table)). |
| `task` | `str` | `"classification"` | `"regression"`, `"classification"`, `"lbc"`, `"partition"` | Selects the loss, the goodness metric, and which of `run_seeds()` / `run_lbc()` / `run_partition()` drives the sweep. `"regression"` is the prediction-divergence-method (PDM) route; `"classification"` trains one classifier at a single, fixed split of the tuning parameter (see below); `"lbc"` and `"partition"` are the two learning-by-confusion routes described below. |
| `partition_index` | `int` or `None` | `None` | `None`, or an integer in `[0, no_partitions - 1]` | Which learning-by-confusion threshold to train on. Ignored outright for `task="regression"` (`create_datasets()` hardcodes it to `None` internally regardless of this field); auto-resolved for `task="classification"` on a dataset in `CLASSIFICATION_PARTITION_INDEX` (below), otherwise **must** be set explicitly for `task="classification"` on `"XXZ"`; overwritten by `run_lbc()` for `task="lbc"`, which sweeps every partition itself; **must** be a concrete integer for `task="partition"` (`run_experiments()` asserts `cf.partition_index is not None` before calling `run_partition()`). `no_partitions` (`= len(unique tuning values) - 1`) is dataset-dependent and can be read with `get_no_partitions(cf)` in `tetriscnn/datasets.py`. |

### `cf.task = "classification"`: which partition it actually trains on

A classifier needs two label classes from somewhere. Two different mechanisms supply
them, and which one applies depends entirely on the dataset:

- **The Paris datasets (and `"XXZ"`) carry no ground-truth phase label** — the
  transition location is exactly what the manuscript is trying to establish — so
  `"classification"` has to pick a threshold on the tuning-parameter grid and derive
  binary labels from it, mechanically identical to `task="partition"` at one fixed
  index. `create_datasets()` (`tetriscnn/datasets.py`) hardcodes that index for the
  four Paris datasets, via the manuscript's reported transition-flanking pair on each
  one's tuning-parameter grid:

  | Dataset | `CLASSIFICATION_PARTITION_INDEX` |
  |---|---|
  | `"Paris_XY_XZ"` | `2` |
  | `"Paris_XY_X"` | `2` |
  | `"Paris_Ising"` | `2` |
  | `"Paris_XY_Z"` | `4` |

  `"XXZ"` needs the same mechanism (`XXZDataProcessor` also raises `NotImplementedError`
  for discrete labels without learning-by-confusion) but has no equivalent canonical
  index, so `cf.partition_index` **must** be set explicitly for `cf.dataset = "XXZ"`;
  leaving it `None` raises `AssertionError` naming the dataset before any file is read.
- **`"ILGT"` and `"1D_TFIM_*"` carry a genuine ground-truth phase label of their own**
  (`ILGTDataProcessor`'s classification file; `TFIMDataProcessor`'s `class=` column),
  entirely independent of any threshold. `"classification"` on these two needs no
  partition index at all — `cf.partition_index` stays `None` throughout, exactly as it
  did before this mechanism existed. Do not set `cf.partition_index` expecting it to
  pick a split here; it is simply not read.

`CLASSIFICATION_PARTITION_INDEX` and the discriminating set `CLASSIFICATION_NEEDS_PARTITION`
both live at the top of `tetriscnn/datasets.py`.

**`lbc` vs. `partition`.** Both implement learning by confusion (train a binary
classifier per candidate transition threshold and look for a peak in accuracy at the
true transition), but `run_lbc()` (task `"lbc"`) loops over *every* partition
internally for each seed and produces one aggregated `lbc.png` per seed set, while
`run_partition()` (task `"partition"`) trains exactly the one partition named by
`cf.partition_index`, which is what a parameter sweep over `partition_index` (or an
external driver script) uses to spread the work across separate runs. `run_lbc()`
unconditionally sets `cf.partition_index = 0` as its first step and then overwrites it
again on every iteration of its internal loop, so whatever `cf.partition_index` was
set to before calling it is simply discarded; the convention documented in
`main.py`'s `setup_experiment()`
(leave it `None` for `task="lbc"`) exists for clarity, not because a different value
would raise an error.

`set_plotting_logging_strings(cf)`, called at the end of `setup_experiment()`, derives
three fields the user does not set directly: `cf.goodness_str` is forced to `"acc"`
for every task except `"regression"` (so a `goodness_str` set earlier for a
classification/lbc/partition run is silently overwritten), `cf.loss_str` becomes
`"MSE"` for regression and `"CEL"` (cross-entropy) otherwise, and
`cf.phase_indicator_str` is a LaTeX label used only in plots.

---

## 2. Dataset selection and splitting

### Where the data lives

`tetriscnn/dataprocessing.py`'s `get_data_root()` searches upward from the working
directory for a directory named `datasets/` (the public-release layout) and falls
back to the legacy name `data/` if `datasets/` is absent, so older working trees keep
running unmodified. Setting the environment variable `TETRISCNN_DATA_ROOT` overrides
the search entirely, which is how a dataset tree kept outside the repository (a
scratch disk, a shared mount) is pointed at:

```bash
export TETRISCNN_DATA_ROOT=/scratch/tetriscnn/datasets
```

A dataset a config asks for that is not present on disk raises
`DatasetNotAvailableError` (a `FileNotFoundError` subclass) naming the exact path it
looked in, rather than failing deeper inside a loader with an opaque message. See
[`datasets/README.md`](../datasets/README.md) for the full directory layout, the
provenance of each dataset, and integrity verification via
`scripts/dataset_manifest.py`.

### `cf.dataset`

| Value | Physical system | Grid | Basis / channels | Split |
|---|---|---|---|---|
| `"Paris_Ising"` | 2D transverse-field Ising, Rydberg array | 8×8 | Z only, 1 channel | Random or even split, computed here |
| `"Paris_XY_X"` | 2D dipolar XY, Rydberg array | 6×7 | X only, 1 channel | Random or even split, computed here |
| `"Paris_XY_Z"` | 2D dipolar XY, Rydberg array | 6×7 | Z only, 1 channel | Random or even split, computed here |
| `"Paris_XY_XZ"` | 2D dipolar XY, Rydberg array | 6×7 | X and Z stacked, 2 channels | Random or even split, computed here |
| `"ILGT"` | 2D Ising lattice gauge theory, simulated | 16×16 | 2 channels (link orientations) | Fixed train/test split shipped with the data |
| `"1D_TFIM_X"` / `"1D_TFIM_Y"` / `"1D_TFIM_Z"` | 1D transverse-field Ising chain, simulated | 150×1 | 1 channel, basis selects the measurement operator | Fixed train/test split shipped with the data |
| `"XXZ"` | 1D XXZ spin chain, simulated | 300×1 | 1 channel | Random or even split, computed here (see caveat below) |

Any other name must first be registered with
`tetriscnn.datasets.register_dataset(name, factory)`, which is how a new data source
is added without editing the package; the factory receives `cf` and returns a
processor honouring the same contract as the built-in ones described below.
[`notebooks/BYODataset_Tutorial.ipynb`](../notebooks/BYODataset_Tutorial.ipynb)
walks through it end to end. An unregistered name raises `ValueError` listing the
built-in choices and the registered ones. `create_datasets()`
(`tetriscnn/datasets.py`) dispatches on `cf.dataset` to build the right
`*DataProcessor` (`tetriscnn/dataprocessing.py`), which loads the raw snapshots and
returns `(data, label)` samples; `create_datasets()` then either performs the
train/val split itself (the `Paris_*` and `XXZ` processors, which expose a flat
`samples` list) or takes the processor's own fixed split (`ILGT` and `1D_TFIM_*`,
which expose `train_samples`/`val_samples` directly, since those datasets ship a
canonical partition and re-splitting them would not be meaningful). `even_split` is a
no-op (with a printed note) on the fixed-split datasets.

**Writing a processor for new data.** Data stored as a set of snapshots per value of
the tuning parameter, which covers the Rydberg data and most sweeps, needs only a
subclass of `tetriscnn.dataprocessing.SnapshotProcessor`. The base class builds
`samples`/`sample_times`, stacks several measurement bases into channels (with the
pairing modes of the next subsection), maps stored 0/1 values to ±1
(`data_format="-11"`), applies `label_dict` and `filter_t_values`, and assigns labels
by task: the tuning parameter for regression, 0/1 around a threshold for learning by
confusion. A subclass passes `grid_size`, `label_param` and a `{basis: [files]}` dict
to `super().__init__()` and implements

- `_load_basis_files(file_list)`, returning `{t: array}` with one snapshot per row;
- `_to_image(snapshot)`, optionally, when a row needs more than a reshape to
  `grid_size` (`ParisDataProcessor`, the Rydberg format, also rotates it to the atom
  layout);
- `_phase_label(t)`, optionally, returning 0 or 1, if the phase at each `t` is known.
  Without it, `task="classification"` raises `NotImplementedError` and learning by
  confusion is the way to get discrete labels, as for the experimental data.

`IsingDataProcessor` and `XYDataProcessor` are built this way, through
`ParisDataProcessor`.

A bare `"1D_TFIM"` is not a legal value; the basis suffix is required (`_X`, `_Y`, or
`_Z`). Internally `create_datasets()` strips the `"1D_TFIM"` prefix and uppercases
whatever remains, defaulting to `"Z"` only if the suffix is empty.

**XXZ discrete-label caveat.** `XXZDataProcessor.__init__` raises
`NotImplementedError` only when it receives `discrete_labels=True` **without**
`learning_by_confusion=True` — i.e. bare `task="classification"` on `"XXZ"` without an
explicit `cf.partition_index` (see above). `task="lbc"` and `task="partition"` both set
`learning_by_confusion=True` before reaching the processor and work normally, deriving
binary labels by thresholding the anisotropy `Jz`, exactly as for the Paris datasets.
The XXZ dataset is also not distributed with the repository (see
[`datasets/README.md`](../datasets/README.md)) and raises `DatasetNotAvailableError`
if requested without it.

### `cf.phase_path`

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `phase_path` | `str` | not set (`getattr(cf, "phase_path", "FM_PM")` in `create_datasets()`) | `"FM_PM"`, `"AFM_PM"` | Only read for `cf.dataset` starting with `"1D_TFIM"`. Selects which of the two simulated sweeps to load: a ferromagnetic-to-paramagnetic approach or an antiferromagnetic-to-paramagnetic one (see `datasets/README.md`). Ignored by every other dataset. |

### `cf.label_param`

`setup_experiment()` sets `cf.label_param = "delta"` when `cf.task == "regression"`
and `cf.label_param = "t"` for every other task (the tuning axis a
classification/lbc/partition run trains against is always raw acquisition time or
sweep index; only regression needs a physically meaningful continuous label). The
legal values for regression are dataset-dependent and enforced by asserts inside each
processor:

| Dataset | Legal `label_param` (regression) | Notes |
|---|---|---|
| `Paris_Ising` | `"t"`, `"delta"`, `"omega"`, `"deltaomega"` | `"deltaomega"` is a two-component label `(delta, omega)`; the only dataset with a two-output regression head. |
| `Paris_XY_X` / `Paris_XY_Z` | `"t"`, `"delta"` | No `omega` field for the dipolar XY model. |
| `Paris_XY_XZ` | `"t"`, `"delta"` | `"delta"` averages the X- and Z-basis light shifts at each time point; `"t"` uses the common acquisition time directly. |
| `ILGT` | `"beta"` | The inverse temperature; the only regression label. |
| `1D_TFIM_*` | `"g"` | The transverse field; the only regression label. |
| `XXZ` | `"Jz"` | The anisotropy; the only regression label. |

`cf.normalize_labels` (bool, default `True`, only meaningful for regression) rescales
labels to `[0, 1]` using the training split's own min/max, which the validation split
then reuses (never its own min/max) to avoid leaking validation-set range into
normalization.

### Splitting and sample balance

These fields all describe the Rydberg experimental data's central practical problem:
snapshot counts vary by up to an order of magnitude across the acquisition-time sweep
(see the data-preparation appendix of the manuscript), so a naive random split
over-represents whichever time points happen to have the most repetitions.

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `data_fraction` | `float` | `1` | `0`–`1` | Fraction of samples kept before the 70/30 train/val split, on the **random-split path only**. **Ignored when `even_split=True`.** An earlier version subset by keeping only the first *N* time points, which truncates the sweep in time and can delete an entire phase (and the transition itself) rather than thinning the data; `samples_per_pt_cap` is the even-split-safe replacement. |
| `even_split` | `bool` | `True` | `True`, `False`, or a list for a sweep | When `True`, samples are grouped by their acquisition time (or sweep index for the simulated single-parameter datasets) before an independent 70/30 split is taken *within each time point*, guaranteeing every sweep point is represented in both train and val regardless of how lopsided the raw counts are. When `False`, the split is a single random 70/30 draw over the whole pooled dataset. No-op (with a printed note) on `ILGT`/`1D_TFIM_*`, which ship their own split. |
| `samples_per_pt_cap` | `int`, `dict`, or `None` | `None` | `None` (no cap), a positive int, a `{time_point: cap}` dict, or a list of any of these for a sweep | Caps the number of samples kept per time point **before** the even-split division, thinning every point evenly so the shape of the sweep (and the transition) survives. An int caps every time point at `min(value, that point's own count)`; a dict caps only the listed time points (others pass through uncapped), useful for matching another dataset's per-time-point counts exactly (e.g. reproducing XZ's `min(n_X(t), n_Z(t))` pairing count on a single-basis run). Only has an effect when `even_split=True`; a bare int cap is otherwise ignored (there is no time-point grouping to cap within). |
| `use_weighted_loss` | `bool` | not set in `setup_experiment()` (only assigned inside the `weighted_loss` experiment block); read as `cf.use_weighted_loss` and defaults falsy if absent | `True`, `False`, or a list for a sweep | When `True`, `attach_sample_weights()` (`main.py`) attaches an inverse-frequency per-sample weight (`min_count_across_timepoints / count_at_that_timepoint`) to the training dataset, and `train()` uses it to reweight cross-entropy (classification/lbc/partition) or MSE (regression) so under-represented time points are not drowned out by common ones. Only meaningful in combination with `experiment_name="weighted_loss"`; see the scenario table below. |
| `label_subset_count` | `int` or `None` | `None` | Must stay `None` | Superseded by `samples_per_pt_cap`; `PhaseDataset._process_snapshot_info()` raises `NotImplementedError` if this is set to anything else. Kept only so old `config.json` files that recorded `None` still load. |
| `filter_t_values` | `list[float]` or `None` | `None` | A list of exact acquisition times (in ns) to keep, or `None` for no filtering | Only read by the `Paris_*` processors (`ParisDataProcessor._append_sample`); silently has no effect on `ILGT`/`1D_TFIM_*`/`XXZ`. Useful for restricting a run to a subset of the sweep, e.g. debugging on a handful of time points. |
| `param_cutoffs` | `dict` | `{"delta": 6, "omega": 0}` | `{"delta": int, "omega": int}` | Only consulted when `label_param` is `"delta"`, `"omega"`, or `"deltaomega"` on an Ising or XY dataset, where it restricts the domain the phase-indicator derivative is computed over (`PhaseDataset.get_phase_indicator()`) to avoid a numerical divergence near the sweep boundary. The main.py comment notes `delta=11` diverges for the XZ dataset; `6` is the safe default there. Irrelevant for `label_param="t"` or any single-parameter simulated dataset. |

### Multi-basis snapshot pairing (`Paris_XY_XZ` only)

The X- and Z-basis snapshots of `Paris_XY_XZ` come from two *independent*
experimental runs, so no snapshot of one basis has a physical partner in the other;
whichever convention stacks them into a two-channel image is therefore a free choice.
These three fields, read via `getattr` (none are set by `setup_experiment()`, so
every pre-existing config that predates this ablation loads unchanged), control that
convention and only affect `Paris_XY_XZ` (single-basis datasets have nothing to pair
and silently ignore all three).

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `pairing_mode` | `str` | not set (defaults to `"index"`) | `"index"`, `"repair_fixed"`, `"repair_resample"`, `"cross_time"` | `"index"` is the historical convention (first *m* rows of each basis, in file order) and is bit-for-bit backward compatible. `"repair_fixed"`/`"repair_resample"` re-pair a fixed or freshly resampled subset for a pairing-robustness ablation. `"cross_time"` is a null control: every non-anchor basis is read at a deliberately *wrong* acquisition time (a fixed-point-free derangement), so a network that still finds the transition under this mode is reading only per-channel marginals. |
| `pairing_seed` | `int` or `None` | not set (defaults to `None`) | Any int, or `None` | `None` disables randomization entirely (every mode degrades to `"index"` behavior) regardless of `pairing_mode`; this is a deliberate safety default, not an oversight. Set to a concrete int to activate `repair_fixed`/`repair_resample`/`cross_time`. |
| `pairing_subset_seed` | `int` | not set (defaults to `0`) | Any int | Only used by `"repair_fixed"`: fixes *which* snapshots survive truncation to *m* per basis, independent of `pairing_seed`, so the ablation can vary the pairing without also varying which snapshots are even eligible. |

A non-default `pairing_mode` gets its own log subtree (`_pm=<mode>` appended to the
base experiment name by `build_logdir_path()`), so a pairing ablation never
overwrites the default-pairing logs for the same nominal configuration.

---

## 3. Model and architecture

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `model` | `str` | `"tetriscnn"` | `"tetriscnn"`, `"PhaseCNN"`, `"ResNet18"` | `"tetriscnn"` is `ShapeAdaptiveConvNet` + `SmallModel`, the interpretable architecture this repository is about (Net1 = feature extractor with the correlator-reading bottleneck, Net2 = a small MLP readout). `"PhaseCNN"` and `"ResNet18"` are ordinary, non-interpretable CNN baselines trained through the same loop; they have no bottleneck, so the branch/sparsity/L1 machinery below does not apply to them and `net2` is `None` for both. They are **not** part of the manuscript's method-comparison appendix (App. E uses the discriminator backbone from Malyshev et al. instead) — they predate that comparison and are kept only as a generic non-interpretable-CNN reference point. |
| `cnn_kernel_shape` | `tuple[int, int]` | `(3, 3)` | Any 2-tuple | Only consumed by `PhaseCNN` (its three stacked conv layers all use this kernel shape). `ResNet18` accepts the argument for interface compatibility but discards it (`del kernel_shape`) since it hardcodes 3×3 kernels; `tetriscnn` never reads it. |

**Verified working (2026-09).** `ResNet18` trains end-to-end on both `Paris_Ising`
(8×8) and the rectangular `Paris_XY_*` (6×7) at realistic batch sizes. Its
`BatchNorm2d` layers do inherit the standard PyTorch constraint that every training
batch have more than one sample, though: a `(dataset size) % batch_size == 1` last
batch raises `ValueError: Expected more than 1 value per channel...` — reachable with
a small dataset (e.g. a tight `samples_per_pt_cap`) and an unlucky `batch_size`, not a
dataset-specific failure. `PhaseCNN` trains fine on the square `Paris_Ising` grid, but
its three stacked `MaxPool2d(2)` layers halve the spatial size three times in a row,
which does not survive a 6×7 input (`Paris_XY_*`): the third pooling collapses a
spatial dimension to `0` and training raises `RuntimeError: ... Output size is too
small` at the very first backward pass. Use `PhaseCNN` only with `Paris_Ising`, or
adapt its pooling depth to the grid before pointing it at the XY datasets.
| `hidden_size` | `int` | `32` | Any positive int | Only read when `model == "tetriscnn"`. Number of channels in each branch's first convolution (`ConvBranch_twoLayer`/`ConvBranch_Equivariant`'s `conv1`), before the 1×1 second conv collapses back to a scalar. Controls per-branch expressivity, independent of the number of branches (set by `kernel_set`). |
| `init` | `str` | `"kaiming"` | Only `"kaiming"` is implemented | Weight-initialization scheme for `ShapeAdaptiveConvNet`. `ConvBranch_twoLayer` (the ordinary, non-equivariant branch) prints "init is unused for standard convolutions" and ignores it entirely; the setting only has an effect on `ConvBranch_Equivariant` branches (`cf.equivariant=True`, or a mixed kernel set). |

### `cf.kernel_set`

Selects the list of `[kernel_shape, no_filters, dilation, mask, stride]` specs built
by `set_kernels(cf)` (`tetriscnn/utils.py`); each spec becomes one branch of
`ShapeAdaptiveConvNet`. The branch's pattern *area* (its non-zero mask count, or
`height × width` when unmasked) is what the L1 penalty is keyed on
([Section 6](#6-regularization-and-the-l1-penalty)), and is exactly the order of the
spin correlator the branch's activation is, by construction, a linear function of
(the Boolean-Fourier argument behind the whole architecture; see Sec. II C and
App. B 1 of the manuscript, where this count is the pattern cardinality $|P|$ of
Eq. (11)).

| Value | Branches | 2D or 1D | Dataset compatibility | Notes |
|---|---|---|---|---|
| `"smallkernels"` | 10 (non-equivariant) | 2D | Any 2D dataset | The manuscript's primary kernel set: the 1×1 site, the two dominoes `(2,1)`/`(1,2)`, and six masked/unmasked 2×2 patterns spanning every 3- and 4-point sub-shape of a plaquette. |
| `"defaultkernels"` | 9 | 2D | Any 2D dataset | An older, less systematic set (includes dilated dominoes and a 3×3 corner); kept for backward compatibility with early runs. |
| `"bigkernels"` | 4 or 5, dataset-dependent | 2D | `Paris_Ising`, `Paris_XY_*`, `ILGT` | Branches on `cf.dataset`: `1×1, 2×2, 4×4, 8×8` for Ising, `1×1, 2×2, 4×4, 6×7` for XY (the last spans the full lattice), and an extra `16×16` for ILGT. Raises `ValueError` for any other dataset (including the 1D chains). |
| `"bigkernels_stride2"` | Same counts as `bigkernels` | 2D | Same as `bigkernels` | Same shapes, but the `2×2` branch reads with stride 2, thinning its receptive field's overlap. |
| `"chainkernels"` | 5 | 1D (one column) | `1D_TFIM_*`, `XXZ` | The 1-, 2- (nearest- and, via dilation, next-nearest-neighbour), 3-, and 4-site chain segments. Required for any dataset whose snapshots are stored as `(1, N, 1)` or `(1, N, 1)`-shaped one-column images; a 2D kernel set's horizontally extended patterns (e.g. `(1,2)`) do not fit a single column. |
| `"smallkernels_mixed"` | 20 (10 non-equivariant + 10 orbit-averaged) | 2D | Any 2D dataset | Puts the ordinary 10-branch `smallkernels` list **and** its canonical equivariant representatives side by side in one bottleneck, so the L1 penalty chooses between an orientation-resolved and a symmetrized reading of the same correlator content. Reads `cf.mixed_group` (below); mutually exclusive with `cf.equivariant=True` (`set_kernels()` raises `ValueError` if both are set, since that would make every branch equivariant and destroy the mixture). |

Any other string, or a dataset `"bigkernels"`/`"bigkernels_stride2"` does not
recognize, raises `ValueError` naming the available sets.

`set_kernels()` always appends a default `stride=1` to any kernel spec shorter than
five elements, so hand-written or legacy 4-element specs (without a stride) keep
working. `validate_kernels_fit()`, called from `train.py` right before model
construction, checks every branch's *effective* receptive field (accounting for
dilation: `(k - 1) * dilation + 1`) against the snapshot's actual height/width and
raises `ValueError` naming the offending branch(es) if any exceed it, rather than
letting `torch`'s own `conv2d` fail deeper with a message naming neither the branch
nor the dataset. If the snapshot is one column wide, the error specifically suggests
`kernel_set="chainkernels"`.

### Equivariant branches

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `equivariant` | `bool` | `False` | `True`, `False` | Builds `ConvBranch_Equivariant` (a hand-rolled, weight-tied group convolution) instead of `ConvBranch_twoLayer` for every branch. **Requires `kernel_set == "smallkernels"`** (`set_kernels()` raises `NotImplementedError` otherwise, since only `smallkernels` has worked-out canonical orbit representatives) and is mutually exclusive with `kernel_set == "smallkernels_mixed"` (raises `ValueError`; that set carries its own per-branch equivariance via `cf.branch_groups` instead). |
| `equivariant_group` | `str` | `"C4"` | `"C4"`, `"D2"`, `"K4"` (`"D2"`/`"K4"` are aliases for the same Klein four-group) | Only consulted when `equivariant=True`. `"C4"` (the four 90° rotations) is the natural symmetry of the square 8×8 Ising lattice and collapses `smallkernels`'s 10 branches to 5. `"D2"`/`"K4"` (180° rotation plus both mirrors, i.e. the symmetry of a non-square rectangle) suits the 6×7 XY lattice and collapses to 6 branches, since the flip-only group does not mix the two dominoes into one orbit the way C4 does. An unrecognized value raises `ValueError`. |

The manuscript's results use `equivariant=True` with `equivariant_group="C4"` for both
the Ising and the XY data; this is what the paper calls the *rotationally invariant*
TetrisCNN, and `equivariant=False` is its *unconstrained* variant (App. F 3). D2/K4 is
an available alternative for the rectangular XY lattice, not used in the paper.

Turning `equivariant` on changes the branch *count*, so an equivariant and a
non-equivariant run are architecturally different models even when every other field
matches; `build_logdir_path()` appends `_{equivariant_group}` to the log path
specifically to keep the two from silently overwriting each other's logs.
Interpreting an equivariant branch's activation also requires passing
`group=cf.equivariant_group` to `interpret.fit_activation_to_correlators()`, since the
branch is linear in the *orbit-averaged* correlators, not the plain sub-pattern ones
a non-equivariant branch reads.

Two further fields are consumed only by `kernel_set == "smallkernels_mixed"` and are
not set by `setup_experiment()` at all (both read via `getattr` with a default):

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `mixed_group` | `str` | not set (defaults to `"C4"` inside `set_kernels()`) | `"C4"`, `"D2"`, `"K4"` | Which symmetry group the *equivariant half* of the mixed bottleneck uses. Unlike `equivariant_group`, this is read only when `kernel_set == "smallkernels_mixed"`. |
| `equivariant_rebate` | `float` or `"inverse_orbit"` or `None` | not set (`getattr(cf, "equivariant_rebate", None)` in `train.py`) | `None` (no rebate, rebate factor 1), a float `r` (uniform multiplier on every equivariant branch's penalty; `r < 1` discounts it, `r > 1` surcharges it), or the string `"inverse_orbit"` (charges each equivariant branch `1 / branch_orbit_sizes[k]`, i.e. per non-equivariant pattern it stands in for) | Only meaningful with a mixed kernel set (`get_branch_penalties()` raises `ValueError` if given without `cf.branch_groups`, which only `set_kernels()`'s `smallkernels_mixed` branch populates). Lets the L1 penalty compensate for (or amplify) the fact that one symmetrized branch carries the same correlator information that several non-equivariant branches would otherwise each pay a penalty for separately. |

`cf.branch_groups` and `cf.branch_orbit_sizes` are **outputs** of `set_kernels()`,
not inputs: for every kernel set except `"smallkernels_mixed"` they are set to `None`
(restoring ordinary `cf.equivariant`-driven behavior); for `"smallkernels_mixed"` they
record, per branch, which symmetry group (if any) it belongs to and how many
non-equivariant patterns it stands in for. Setting either by hand before calling
`set_kernels(cf)` has no effect, since the function unconditionally overwrites both.

---

## 4. Training hyperparameters

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `epochs` | `int` | `250` | Positive int | Maximum training epochs; early stopping (below) usually ends a run sooner. |
| `learning_rate` | `float` | `1e-2` | Positive float | Base learning rate for the `AdamW` optimizer (decoupled weight decay), before any scheduler modifies it. The LR-annealing rationale for `Section 5` below assumes `AdamW`'s decoupled decay specifically: suppressed-branch oscillation amplitude scales with the learning rate, so annealing it is what sharpens sparsity late in training. |
| `weight_decay` | `float` | `1e-5` | Non-negative float | `AdamW`'s L2 weight decay, applied to *all* parameters of `net1` and `net2` together. Distinct from `weight_penalty` below, which is an L1 penalty on branch weights specifically. |
| `patience` | `int` | `10` | Positive int | Epochs of non-improvement (post-warmup) before `EarlyStopper` stops training. |
| `early_stop_warmup` | `int` | `150` | Non-negative int | Epochs at the start of training during which early stopping is disabled (the best validation objective is still tracked, but training never stops during this window). A large warmup lets the network settle into its penalty-shaped representation before monitoring begins; `150` out of `250` total epochs means early stopping only has a 100-epoch window to act, by design. |
| `early_stop_min_delta` | `float` | `1e-5` | Non-negative float | Minimum decrease in the validation objective, relative to the best seen so far, that counts as improvement and resets `EarlyStopper`'s patience counter. |
| `init` | `str` | `"kaiming"` | See [Section 3](#3-model-and-architecture) | Listed again here because it is set among the training hyperparameters in `setup_experiment()`, but documented fully in Section 3. |
| `num_workers` | `int` | `0` | Non-negative int | `DataLoader` worker processes for the training loader. Forced to `0` on an MPS (Apple Silicon) device regardless of this setting, since MPS does not support multiprocessing dataloading; a warning is printed when this override fires. |
| `pin_memory` | `bool` | `False` | `True`, `False` | `DataLoader` pinned-memory flag. Likewise forced to `False` on MPS. |
| `batch_size` | `int` | `64` | Positive int | Micro-batch size for the actual forward/backward step. |
| `VRAM_batch_size` | `int` | `1024` | Positive int, typically ≥ `batch_size` | Two-tier batching: this many samples are moved onto the accelerator at once (amortizing host-to-device transfer), then processed in `batch_size`-sized micro-batches. `use_vram_batching` in `train.py` is only `True` when `VRAM_batch_size > batch_size`; setting it equal to or below `batch_size` disables the two-tier path and falls back to ordinary per-batch transfer. |
| `save_models` | `bool` | `True` | `True`, `False` | Whether `net1.pt`/`net2.pt` are written to the run's log directory at the end of training. Disabling this saves disk space but makes the run unrecoverable for symbolic regression or manual inspection later. |
| `save_histories` | `bool` | `True` | `True`, `False` | Whether `metrics.json` records a full per-epoch time series (loss, goodness, every branch's `z_k`, snapshot averages, phase-indicator arrays) or only the final epoch's values. Cross-seed/cross-lambda aggregation (`update_metrics_per_seed()`/`update_metrics_per_partition()`) asserts on the shape it expects, so flipping this between runs that are later aggregated together will fail loudly rather than silently mixing conventions. |
| `save_final_values` | `bool` | `True` when `experiment_name == "singlerun"` and `task != "lbc"`, else `False` | Derived, but overridable after the fact | When `True`, `train()` writes `fit_metrics.json` (the very last batch's inputs, bottleneck activations, network outputs, and every branch's raw `conv1` weights) alongside `metrics.json`. Required for `cf.fit_branches = True` to have anything to read ([Section 9](#9-remake-flags-and-re-plotting)). |

---

## 5. Learning-rate scheduling

Gated by `cf.use_lr_scheduler` (`bool`, default `True`). `setup_experiment()` only
sets the scheduler type and the parameters that type actually needs;
`build_lr_scheduler()` (`tetriscnn/train.py`) fills in everything else the *active*
`lr_scheduler_type` requires from the module-level `LR_SCHEDULER_DEFAULTS` dict via
`cf.setdefault(key, value)`, so an explicit value always wins and switching scheduler
types is a one-line change rather than requiring every other type's parameters to be
pre-populated.

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `use_lr_scheduler` | `bool` | `True` | `True`, `False` | Master switch; `False` trains at a constant `learning_rate` throughout. |
| `lr_scheduler_type` | `str` | `"reduce_on_plateau"` | `"reduce_on_plateau"`, `"cosine_annealing"`, `"step"`, `"exponential"`, `"onecycle"` | Which `torch.optim.lr_scheduler` class `build_lr_scheduler()` constructs. |

`setup_experiment()` explicitly sets the `"reduce_on_plateau"` parameters (the type it
actually uses); the other four types' parameters below are only ever supplied by
`LR_SCHEDULER_DEFAULTS` unless the config is edited to set them explicitly.

| Scheduler type | Fields | Defaults (`setup_experiment()` / `LR_SCHEDULER_DEFAULTS`) | Purpose |
|---|---|---|---|
| `reduce_on_plateau` | `lr_reduce_factor`, `lr_reduce_patience`, `min_lr` | `0.5`, `5`, `1e-9` (all set explicitly in `setup_experiment()`) | Halves the LR whenever validation loss fails to improve for `lr_reduce_patience` epochs, down to a floor of `min_lr`. The adaptive default: explores at a high LR, then anneals as validation loss plateaus. |
| `cosine_annealing` | `min_lr` | `1e-8` (from `LR_SCHEDULER_DEFAULTS`) | Smoothly anneals from `learning_rate` to `min_lr` over `cf.epochs`, via `CosineAnnealingLR(T_max=cf.epochs, eta_min=min_lr)`. |
| `step` | `lr_step_size`, `lr_step_gamma` | `30`, `0.5` | Multiplies the LR by `lr_step_gamma` every `lr_step_size` epochs. |
| `exponential` | `lr_exp_gamma` | `0.85` | Multiplies the LR by `lr_exp_gamma` every epoch. |
| `onecycle` | `onecycle_initial_lr`, `onecycle_max_lr`, `onecycle_final_lr`, `onecycle_pct_start` | `1e-4`, `1e-2`, `1e-8`, `0.3` | Ramps LR up from `onecycle_initial_lr` to `onecycle_max_lr` over the first `onecycle_pct_start` fraction of training, then back down to `onecycle_final_lr`. **Steps once per batch, not once per epoch** (`_step()` in `train.py` calls `scheduler.step()` directly whenever `lr_scheduler_type == "onecycle"`), unlike every other type here, which steps once per epoch. |

**The `min_lr` dual-default is intentional, not a bug.** A *fresh* `reduce_on_plateau`
run that does not set `cf.min_lr` explicitly gets `1e-9` from
`LR_SCHEDULER_DEFAULTS["reduce_on_plateau"]`; a `config.json` loaded from an older
run keeps whatever `min_lr` it was originally recorded with, since `cf.setdefault`
only fills in a field that is genuinely absent. Do not "fix" an old run's `min_lr` to
match a newer default; doing so changes what a remake is measuring relative to the
original training run.

---

## 6. Regularization and the L1 penalty

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `lambdas` | `list[float]` | `[3]` | Any list of numbers; length 1 for `singlerun`/`weighted_loss`, length > 1 for a `lambdamax`/`lambdatot` sweep | The one architectural knob this whole codebase is organized around. For `experiment_name` in `{"lambdamax", "singlerun", "weighted_loss"}`, each entry is `λmax`, the top of a log-uniform penalty schedule (below); for `"lambdatot"` each entry is the base-10 exponent of a single uniform penalty applied to every branch. Excluded from the generic sweep-discovery mechanism ([Section 8](#8-parameter-sweeps)) since `run_experiments()` always loops over it explicitly as the outermost axis. |
| `weight_penalty` | `float` or `None` | `None` | Any float, or `None` to disable | An *experimental*, off-by-default extra: when set, adds `weight_penalty * sum(abs(all branch conv1 weights))` to the loss, an L1 penalty on the branches' first-layer *weights* themselves, layered on top of the activation penalty below. Read in `train.py`'s `_step()`. |

`penalty_params` (a 4-element `[base, exp_min, exp_max, n_pen]` list) is **not** set
directly in `setup_experiment()`; `run_experiments()` assigns it once per lambda
iteration from `cf.lam` (the current element of `cf.lambdas`):

```python
if cf.experiment_name in ("lambdamax", "singlerun", "weighted_loss"):
    cf.penalty_params = [10, -3, cf.lam, 1]     # cf.lam IS λmax here
elif cf.experiment_name == "lambdatot":
    cf.penalty_params = None                     # uniform 10**cf.lam penalty instead
```

**`exp_min = -3` matches the manuscript's stated `λmin = 10^-3` and must stay `-3`.**
The four call sites in `tetriscnn/experiments.py` (`run_experiments()`'s no-sweep and sweep branches,
each setting `cf.penalty_params` twice — once outside the sweep loop for backward
compatibility, once inside it) all hardcode this second list element; there is no
single shared constant, so changing it means editing all four. Do not lower it back
toward `-5` to chase a particular published figure: the canonical runs the manuscript
quotes (Figs. 5, 6, and the main text) were trained at `λmin = 10^-3`, and a run
trained at a different `λmin` is measuring a different penalty schedule, not
reproducing the same one at higher resolution. If a figure ever needs the *specific*
`λmin = 10^-5` runs that predate this convention (an earlier internal draft of Fig. 4
used them), point `basepath` at that run's own saved logs instead of changing this
default.

`get_branch_penalties()` (`tetriscnn/utils.py`) turns `penalty_params` into one
scalar per branch:

```python
penalty_base = np.logspace(base=10, start=-3, stop=λmax, num=max_kernel_area)
λ_k = penalty_base[area_of_branch_k - 1]
```

**The penalty is keyed by kernel *area*, not by branch index.** `area` is the number
of active sites in the branch's pattern (`np.count_nonzero(mask)` when masked, else
`height × width`), which is exactly the order of the spin correlator that branch's
activation encodes by construction. Two branches of the same area therefore share the
same base penalty regardless of shape or orientation (e.g. `(2,1)` and `(1,2)` are
both area 2 and get the identical base `λ`). On top of the area-keyed base, two
secondary scalings apply: the penalty scales *linearly* with `dilation` (for area > 1
only), and by `n_pen ** filter_index` across multiple filters of the same shape (only
when `n_pen != 1`; every configuration in this codebase currently uses `n_pen = 1`,
so this term is inert in practice).

For the `lambdatot` experiment, `train.py` bypasses `get_branch_penalties()` entirely
and uses `penalties = [10**cf.lam] * len(cf.kernels)`, a single value shared by every
branch irrespective of area; this is the "total penalty" ablation referenced by the
experiment's name, contrasted against `lambdamax`'s area-graded schedule.

The L1 term itself, `Σ_k λ_k |z_k|`, is computed by `l1_regularization()` and applied
to bottleneck **activations**, not weights, so `|z_k|` is directly usable as a
feature-importance readout: a branch survives only when its task-signal gradient
outweighs its own `λ_k`. See the manuscript's `\seclab~ss:interpretable_Tetris` for the
full physics argument (nested patterns, increasing penalties, Rashomon effect).

---

## 7. Reproducibility

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `seeds` | `list[int]` | `[46]` | Any list of ints | One full training run per seed, aggregated afterward (mean/spread across seeds in the sweep plots). `set_seeds(seed_no)` seeds `torch`, `torch.cuda`, and `numpy` together for each seed in turn. Excluded from the generic sweep mechanism; `run_seeds()`/`run_lbc()`/`run_partition()` always loop over it explicitly, one level inside the `lambdas` loop. |

The train/val split itself uses an **independent, fixed** seed (`torch.Generator().manual_seed(42)`
inside `create_datasets()`), deliberately decoupled from `cf.seeds`, so that sweeping
seeds varies only weight initialization and training stochasticity, never which
samples ended up in which split.

---

## 8. Parameter sweeps

Any `cf` field can be swept by assigning it a `list` of length greater than one
instead of a scalar. `get_sweepable_params(cf, exclude=...)` (`tetriscnn/utils.py`)
discovers these automatically by scanning `cf`'s top-level items; a handful of
fields are always excluded from this discovery, whether or not they happen to be
list-valued, because they are either driven explicitly by other loops
(`lambdas`, `seeds`) or are themselves internal per-branch bookkeeping rather than a
genuine sweep axis (`kernels`, `penalty_params`, `filter_t_values`,
`unique_labels`, `branch_groups`, `branch_orbit_sizes`).

`run_experiments()` builds the Cartesian product of every remaining sweepable field
(via `itertools.product`) and runs one training call per combination, **except** for
`experiment_name == "weighted_loss"`, which instead `zip`s the sweepable lists in
parallel (scenario `i` uses element `i` of each list), and raises `ValueError` if the
lists are not all the same length.

### Weighted-loss experiment scenario table

The one first-class use of the parallel-sweep path: `even_split`, `samples_per_pt_cap`,
and `use_weighted_loss` are each given a 4-element list, and `run_experiments()` runs
the four scenarios in lockstep rather than the (nonsensical) 8-combination cross
product.

**Write the four lists as one dict keyed by scenario, not as three separately-typed
raw lists.** Three same-length lists have to be kept aligned by *position* across
three separate lines — scenario (c) is index 2 of `even_split`, index 2 of
`samples_per_pt_cap`, and index 2 of `use_weighted_loss`, and nothing checks that a
fourth line inserted between two of them keeps that alignment. That is exactly the
error-prone "repeated sweep" bookkeeping the `zip`-based mechanism exists to avoid in
the first place, so writing the lists that way defeats the point:

```python
scenarios = {
    "(a)": {"even_split": True,  "samples_per_pt_cap": None, "use_weighted_loss": False},
    "(b)": {"even_split": True,  "samples_per_pt_cap": 500,  "use_weighted_loss": False},
    "(c)": {"even_split": False, "samples_per_pt_cap": None, "use_weighted_loss": False},
    "(d)": {"even_split": False, "samples_per_pt_cap": None, "use_weighted_loss": True},
}

cf.experiment_name    = "weighted_loss"
cf.even_split         = [scenarios[s]["even_split"] for s in scenarios]
cf.samples_per_pt_cap = [scenarios[s]["samples_per_pt_cap"] for s in scenarios]
cf.use_weighted_loss  = [scenarios[s]["use_weighted_loss"] for s in scenarios]
cf.lambdas = [4]
cf.seeds   = [42, 123, 456, 789, 101112, 131415]
cf.visualize_sweep = True
```

This produces *exactly* the same three lists `run_experiments()` reads
(`cf.even_split == [True, True, False, False]`, etc.) — the dict is purely a
readability device, not a different mechanism. `setup_experiment()`'s "Weighted-loss
experiment" comment block in `main.py` keeps this pattern commented out, ready to
uncomment for a weighted-loss run.

| Scenario | `even_split` | `samples_per_pt_cap` | `use_weighted_loss` | Meaning |
|---|---|---|---|---|
| (a) | `True` | `None` | `False` | Even split, uncapped, ordinary loss |
| (b) | `True` | `500` | `False` | Even split, capped at 500 samples/time point, ordinary loss |
| (c) | `False` | `None` | `False` | Random split, uncapped, ordinary loss (the naive baseline) |
| (d) | `False` | `None` | `True` | Random split, uncapped, inverse-frequency weighted loss |

`attach_sample_weights()` (`tetriscnn/experiments.py`) computes the weight
`w = min_count_across_timepoints / count_at_that_timepoint` per sample and attaches it
to `cf.train_dataset`; `train()` then uses weighted cross-entropy or weighted MSE
(whichever the task calls for) whenever `use_weighted_loss` is `True` for the current
scenario.

### Naming and abbreviation

Sweep points get a compact folder-name component from `PARAM_ABBREVIATIONS`
(`tetriscnn/utils.py`); any field not listed here falls back to its own full name:

| Field | Abbreviation | Field | Abbreviation |
|---|---|---|---|
| `samples_per_pt_cap` | `spc` | `onecycle_initial_lr` | `oilr` |
| `learning_rate` | `lr` | `onecycle_max_lr` | `omlr` |
| `min_lr` | `minlr` | `onecycle_final_lr` | `oflr` |
| `top_k` | `topk` | `onecycle_pct_start` | `ops` |
| `hidden_size` | `hs` | `even_split` | `es` |
| `batch_size` | `bs` | `use_weighted_loss` | `wl` |
| `weight_decay` | `wd` | `pairing_seed` | `ps` |
| `patience` | `pat` | `pairing_mode` | `pm` |
| `epochs` | `ep` | `pairing_subset_seed` | `pss` |
| `lr_reduce_factor` | `lrf` | `lr_step_size` | `lss` |
| `lr_reduce_patience` | `lrp` | `lr_step_gamma` | `lsg` |
| | | `lr_exp_gamma` | `leg` |

`build_logdir_path()` uses hierarchical naming (one path component per swept
parameter, sorted by abbreviation) for every experiment except `weighted_loss`,
which flattens all swept parameters into one `-`-joined path component instead
(matching the symbolic-regression folder-naming convention); see
[Section 10](#10-output-and-log-directory-structure) for full path examples.

### `cf.visualize_sweep`

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `visualize_sweep` | `bool` | `False` | `True`, `False` | When `True` (and exactly one parameter is swept with exactly one lambda value), `run_experiments()` generates a line plot of metrics vs. the swept parameter after training completes. With more than one swept parameter or more than one lambda, a note is printed instead and no plot is made. Independently, `remake_flags["remake_sweep_plot"] = True` regenerates this plot from already-completed logs without retraining (this also forces `cf.visualize_sweep = True` internally). |

---

## 9. Remake flags and re-plotting

The `__main__` block of `main.py` defines two dicts, and the `basepath` to re-plot,
before handing all three to `tetriscnn.experiments.run_or_remake()`:

```python
remake_flags = {
    "remake_lambda_plot": False,
    "remake_history_and_pt_plots": False,
    "remake_sweep_plot": False,
}
plot_config = {
    "enabled_metrics": {"z", "loss", "goodness"},
    "fit_branches": False,
}
```

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `remake_history_and_pt_plots` | `bool` | `False` in the checked-in `__main__` block | `True`, `False` | Re-plots each individual run's training-history and phase-transition-indicator figures directly from its own saved `config.json` and `metrics.json`, without retraining. Mutually exclusive with `remake_lambda_plot` (asserted in `load_or_init_config()`). |
| `remake_lambda_plot` | `bool` | `False` | `True`, `False` | Regenerates only the cross-lambda aggregate plot from `metrics_per_lambda.json` (or, after a partial remake, `metrics_per_lambda_remake.json`), without touching per-run plots. |
| `remake_sweep_plot` | `bool` | `False` | `True`, `False` | Regenerates the single-parameter sweep plot ([Section 8](#8-parameter-sweeps)) from existing logs, bypassing `run_experiments()` entirely. |

**This default must stay `False`.** All three remake flags gate the same branch in
`run_or_remake()`: `if not any(remake_flags.values()): setup_experiment(cf)`.
With `remake_history_and_pt_plots` (or either of the other two) `True`, that call is
skipped entirely — `main.py` reloads an *existing* run's `config.json` and re-draws its
plots instead of training anything, regardless of whatever `setup_experiment()` would
otherwise have set up. Leaving it `True` after a one-off remake silently turns the very
next `python main.py` invocation into another no-op remake of the same old run rather
than the new training setup a reader edited into `setup_experiment()`, with no error to
signal it. Flip it to `True` only for the one remake invocation you mean it for
(`basepath` set to, or about to be picked as, the run/folder to re-plot), then set it
back to `False` before touching `setup_experiment()` again.

Setting any of the three to `True` makes `main.py` skip calling `setup_experiment(cf)`
altogether; instead `load_or_init_config()` reloads the first `config.json` found
under the chosen `basepath` (see below) and layers `remake_flags`/`plot_config` on
top of the loaded fields (`cf.update(remake_flags); cf.update(lambda_plot_config)`).
Symbolic regression is always forced off on this path (`cf.run_sr = cf.fit_sr = False`);
SR is managed exclusively by `sr_toolbox.py` ([Section 11](#11-symbolic-regression)).

**`plot_config` is not remake-only.** `load_or_init_config()` calls
`cf.update(lambda_plot_config)` unconditionally, on every invocation, not just when a
remake flag is set — so both fields below are also live during an ordinary training
run, not only when re-plotting an old one:

| Field | Type | Default | Legal values | Purpose |
|---|---|---|---|---|
| `enabled_metrics` | `set[str]` | `{"z", "loss", "goodness"}` | Any subset of `"z"`, `"loss"`, `"goodness"`, `"pt"`, `"pt_deriv"`, `"pt_deriv_ste"`, `"net2_norm"`, `"epochs"` (plus `"pt2"`/`"pt2_deriv"`/`"pt2_deriv_ste"` when `cf.label_param == "deltaomega"`); `plot_lbc()` additionally recognizes `"acc"` in place of `"goodness"` | Restricts which panels are drawn, in three places: the per-run training-history figure (`plot_history()`, `"loss"` also gates the L1 companion panel and `"pt"` only draws when the run is a regression task that tracked it), and the cross-lambda / cross-partition aggregate plots (`plot_lambda()`/`plot_lbc()`). Every plot in all three sizes its figure grid to however many of its own applicable panels are selected — asking for fewer never leaves an empty panel behind. |
| `fit_branches` | `bool` | `False` | `True`, `False` | When `True` (and `cf.save_final_values` was `True` at training time, so `fit_metrics.json` exists), BFA-fits and plots every **active** branch of the trained run — see below. |

**`fit_branches` replaces the old `fit_branch_number`.** The previous field asked the
user to name one branch index up front, whose raw first-layer `conv1` weights would
then be visualized as a heatmap by `plot_weights()`; that function was never actually
called from anywhere in the pipeline (dead code), so setting `fit_branch_number` had no
effect regardless of its value. `fit_branches` is a plain on/off switch instead: when
`True`, `single_seed_plots()` (`tetriscnn/plots.py`) calls
`plot_branch_fits(cf, fit_metrics["all_final_x"], fit_metrics["all_final_z"],
"branch_fits.png", final_lr=...)`, which

1. calls `get_active_branches(z, final_lr, floor_margin=1e3)` to find which branches
   are **active** rather than settled on the learning-rate-set residual floor — the
   same notion App. D of the manuscript defines ("Origin of the activation floor" and
   "Active versus deactivated branches") (`z_floor = O(alpha_t)` to leading order, independent of the branch's own
   `λ_k`; a branch counts as active once its mean `|z_k|` clears `floor_margin` times
   the run's final learning rate, read from `metrics["learning_rate"][-1]` when a
   scheduler ran, else the fixed `cf.learning_rate`);
2. regresses each active branch's activation onto the correlators of its own filter
   footprint (`tetriscnn.interpret.fit_activation_to_correlators`, the same routine
   `Figure5.ipynb` uses for the manuscript's Fig. 5); and
3. draws one panel per active branch — the fitted coefficient for every candidate
   correlator, with the best/forward-selected terms outlined and the rest faded — in
   the same visual language as `Figure5.ipynb`'s figure, just generalized to
   whichever branches *this* run's own activations mark as active, rather than the
   manuscript notebook's hand-picked `panel_branches`. Each panel is titled with the
   fitted equation, $z[P] \approx c_0 + \sum_j c_j\, C[P_j]$ (written $C_{\mathrm{rot}}$
   for an equivariant branch, whose features are rotation-averaged), followed by its
   $R^2$; a small box inside the panel gives the branch index, mean $|z|$ and the number
   of terms kept.

Called directly, `plot_branch_fits()` takes a few more options. `max_bars=None` draws
the complete coefficient spectrum of the exact fit, with every bar alike, and titles
the panel with the full equation (at most six terms are written, the largest ones,
followed by an ellipsis). The default `max_bars=5` is the pruned view: the title is the
unpenalized refit on the forward-selected terms. `branches=[...]` plots the given
branches instead of the active ones, `channel_names=[...]` names the measurement bases
in the term labels, and `show_dont_save=True` displays the figure instead of writing it
to `cf.logdir` (`plot_history()` accepts the same flag). `BYODataset_Tutorial.ipynb`
uses both views.

If no branch clears the activity floor, `plot_branch_fits()` prints a note and returns
`None` without writing a file, rather than plotting nothing meaningful. Because the
correlator-pattern labels `tetriscnn.interpret` builds assume a real LaTeX installation
(`mask_to_latex_pattern(full_latex=True)`, matching the paper-figure notebooks' own
`text.usetex=True`), `plot_branch_fits()` converts them to matplotlib's built-in
mathtext-safe glyphs itself, so `cf.fit_branches = True` needs no system LaTeX — see
`tetriscnn.plots._mathtext_safe_pattern_label()`. The equation titles use the same
fallback.

**Selecting `basepath`.** When any remake flag is set and `basepath` is `None` (or
does not exist), a directory picker (`pick_directory()`) opens (a Tk dialog outside a
notebook, an `ipyfilechooser` widget inside one). Two shapes of selection are handled
differently:

- **A seed-level folder** (contains `net1.pt` + `net2.pt` directly): `cf.seeds` is
  narrowed to that one seed, parsed from the folder name (`seed_<n>`), and
  `cf._remake_seed_logdir` is stashed so `resolve_logdir()` writes back to the exact
  folder selected rather than reconstructing a path from `cf` fields (which would
  only agree with the real location if the run tree was never moved).
- **A parent folder** holding several runs: `find_run_directories()` collects every
  leaf run (any directory with `net1.pt` + `net2.pt` + `metrics.json`, so an
  `lbc`/`partition` run's individual `partition_{i}` folders are each their own run),
  and `find_seed_directories_with_models()` collapses those onto their `seed_{n}`
  parents for the purposes of `cf.seeds` and the aggregate plots. Each run is then
  re-plotted from its *own* `config.json` (`remake_plots_for_runs()`), rather than
  reconstructing a sweep from one representative config and re-running it once per
  combination, which is wrong whenever the on-disk runs don't span a clean cartesian
  grid (they never do, once any tree has been pruned or partially rerun).

---

## 10. Output and log directory structure

`build_logdir_path()` (`tetriscnn/utils.py`) assembles the log path from `cf` fields;
`resolve_logdir()` wraps it so a remake prefers the actual on-disk location over a
freshly reconstructed path (see above).

**No sweep:**
```
logs/{experiment}_{dataset}_{task}_{label_param}_{kernel_set}[_{equivariant_group}][_pm={pairing_mode}]_{lambdas}/
└── {experiment}_{lam}/
    └── seed_{seed}/              # only present when len(cf.seeds) > 1
        ├── config.json           # the full cf as of training time
        ├── metrics.json          # training history (or final-epoch values only)
        ├── fit_metrics.json      # final-batch snapshot, if save_final_values=True
        ├── net1.pt / net2.pt     # trained weights, if save_models=True
        ├── plots/
        └── {SR_folder}/          # only created later, by sr_toolbox.py
```

`model` replaces `kernel_set` in the base name whenever `cf.model != "tetriscnn"`
(`PhaseCNN`/`ResNet18` baselines have no kernel set to name). `[_{equivariant_group}]`
appears only when `cf.equivariant` is `True`; `[_pm={pairing_mode}]` appears only for
a non-default pairing convention. An `lbc`/`partition` run additionally suffixes
`/partition_{i}` onto the seed-level path.

**Hierarchical sweep** (every experiment except `weighted_loss`): one extra path
component per swept parameter, sorted by abbreviation:
```
logs/.../{experiment}_{lam}/spc=100/seed_42/
logs/.../{experiment}_{lam}/spc=200/seed_42/
```

**Flat sweep** (`weighted_loss` only): every swept parameter folded into one
`-`-joined component:
```
logs/weighted_loss_{...}/weighted_loss_{lam}/es=True-spc=None-wl=False/seed_42/
logs/weighted_loss_{...}/weighted_loss_{lam}/es=True-spc=500-wl=False/seed_42/
```

At the experiment level (one directory above the outermost `lambdamax_{lam}` /
`singlerun_{lam}` folders), `run_experiments()` writes `metrics_per_lambda.json`
(or `metrics_per_lambda_remake.json` on a remake) aggregating every seed/partition's
final metrics, which the cross-lambda plots read.

Symbolic regression output lives one level further down, inside a run's own seed
folder, and is documented in [`SYMBOLIC_REGRESSION.md`](SYMBOLIC_REGRESSION.md#output-files-per-sr-subfolder).

---

## 11. Symbolic regression

Symbolic regression (PySR) is an optional, post-hoc route, run separately from
training by [`sr_toolbox.py`](../sr_toolbox.py) and configured through its own
`SRConfig` rather than through `cf`. It is documented in
**[`SYMBOLIC_REGRESSION.md`](SYMBOLIC_REGRESSION.md)**: installation, the two ways of
running `sr_toolbox.py`, every `SRConfig` field, the pipeline, how equations are
picked, the output files, and configuration recipes.

---

## 12. Worked configurations

Each block below is a complete `setup_experiment()` body for a common goal, adapted
from the commented-out alternatives already present in `main.py`, checked against the
current code.

### A single interpretable run (the checked-in default)

`setup_experiment()`'s own defaults, unedited: one seed, one λmax, the primary
10-branch kernel set, classification on the two-basis XY dataset at the manuscript's
hardcoded transition-flanking index (`cf.partition_index` resolves to `2`, see
[`cf.task = "classification"`](#cftask--classification-which-partition-it-actually-trains-on)
above).

```python
cf.experiment_name = "singlerun"
cf.task = "classification"
cf.dataset = "Paris_XY_XZ"
cf.kernel_set = "smallkernels"
cf.equivariant = False
cf.lambdas = [3]
cf.seeds = [46]
cf.even_split = True
cf.epochs = 250
```

For the regression (PDM) route on the same physical system instead:

```python
cf.experiment_name = "singlerun"
cf.task = "regression"
cf.dataset = "Paris_Ising"
cf.label_param = "delta"
cf.normalize_labels = True
cf.goodness_str = "r2agg"
cf.kernel_set = "smallkernels"
cf.equivariant = False
cf.lambdas = [4]
cf.seeds = [42, 43, 44]
cf.even_split = True
cf.epochs = 250
```

Afterward, `tetriscnn/interpret.py`'s `fit_activation_to_correlators()` is the next
step: regress the dominant branch's `z_k` onto its own correlator dictionary to read
off a closed-form physical readout, before reaching for symbolic regression at all.
Setting `cf.fit_branches = True` in `plot_config` ([Section 9](#9-remake-flags-and-re-plotting))
does exactly this automatically for every active branch, right after training.

### A λmax sweep

Sweeping `cf.lambdas` (rather than giving it one value) is what demonstrates the
Rashomon-effect phenomenon central to the manuscript: accuracy staying essentially
flat while the surviving branch set simplifies as `λmax` rises.

```python
cf.experiment_name = "lambdamax"
cf.task = "regression"
cf.dataset = "Paris_Ising"
cf.label_param = "delta"
cf.kernel_set = "smallkernels"
cf.seeds = [42]
cf.lambdas = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
```

Symbolic regression is a separate step run afterward against each resulting
`seed_{seed}/` folder via `sr_toolbox.py` (Section 11); it is never configured on
this training-time `cf`.

### Classification with learning by confusion

```python
cf.experiment_name = "lambdamax"
cf.task = "lbc"
cf.dataset = "Paris_XY_XZ"
cf.kernel_set = "bigkernels"
cf.seeds = [42, 123]
cf.lambdas = [2, 4, 8]
cf.partition_index = None   # required for task="lbc"; run_lbc() sweeps it internally
```

`cf.label_param` is set to `"t"` automatically by `setup_experiment()`'s
`else` branch, since only `task="regression"` needs a physical label; `run_lbc()`
discovers the number of available partitions from the dataset itself
(`no_partitions = len(unique tuning values) - 1`) and trains one model per partition
for every seed, then aggregates into a single W-shaped accuracy-vs-threshold plot.

### A run on the simulated 1D TFIM data

Demonstrates the codebase on its original proof-of-concept data rather than the
experimental Rydberg snapshots; useful for a quick end-to-end check without the
experimental dataset tree. `chainkernels` is required, since the 2D kernel sets
carry horizontally extended patterns that do not fit a one-column chain snapshot.

```python
cf.experiment_name = "singlerun"
cf.task = "classification"
cf.dataset = "1D_TFIM_Z"
cf.phase_path = "FM_PM"      # or "AFM_PM" for the antiferromagnetic sweep
cf.kernel_set = "chainkernels"
cf.equivariant = False
cf.lambdas = [4]
cf.seeds = [42]
```

For the regression (PDM) route on the same data instead, set `cf.task = "regression"`
and `cf.label_param = "g"` (the transverse field is the only regression label TFIM
exposes); `cf.even_split` has no effect either way, since `1D_TFIM_*` ships its own
fixed train/test split.

### The weighted-loss data-balancing comparison

Reproduces the appendix's even-split / sample-capping / weighted-loss ablation in one
sweep, using the parallel (`zip`) iteration `experiment_name="weighted_loss"` enables:

```python
scenarios = {
    "(a)": {"even_split": True,  "samples_per_pt_cap": None, "use_weighted_loss": False},
    "(b)": {"even_split": True,  "samples_per_pt_cap": 500,  "use_weighted_loss": False},
    "(c)": {"even_split": False, "samples_per_pt_cap": None, "use_weighted_loss": False},
    "(d)": {"even_split": False, "samples_per_pt_cap": None, "use_weighted_loss": True},
}

cf.experiment_name = "weighted_loss"
cf.task = "regression"
cf.dataset = "Paris_Ising"    # or "Paris_XY_Z", "Paris_XY_XZ"
cf.kernel_set = "smallkernels"

cf.even_split         = [scenarios[s]["even_split"] for s in scenarios]
cf.samples_per_pt_cap = [scenarios[s]["samples_per_pt_cap"] for s in scenarios]
cf.use_weighted_loss  = [scenarios[s]["use_weighted_loss"] for s in scenarios]

cf.lambdas = [4]
cf.seeds = [42, 123, 456, 789, 101112, 131415]
cf.visualize_sweep = True
```

All three swept lists must have equal length, since `run_experiments()` zips them
rather than taking their cartesian product for this one experiment type; a length
mismatch raises `ValueError` before any training starts. See
[Section 8](#weighted-loss-experiment-scenario-table) for why the dict form (not
three raw parallel lists) is the pattern to write this as, and for what each of the
four scenarios means physically.
