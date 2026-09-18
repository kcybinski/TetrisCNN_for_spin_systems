import copy
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import itertools
from tqdm import tqdm
import os
import re
import sys
from typing import TYPE_CHECKING
import sklearn

# ("1.10.0" -> (1, 10)) so the comparison is numeric, not lexicographic.
_SKLEARN_VERSION = tuple(
    int(part) for part in sklearn.__version__.split(".")[:2] if part.isdigit()
)

if TYPE_CHECKING:
    # Annotations only. Importing symbolic_regression for real would boot PySR's
    # Julia runtime, which must happen before torch loads (see sr_toolbox.py).
    from tetriscnn.symbolic_regression import SRConfig

sns.set_theme()
sns.color_palette("Paired")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Parameter abbreviations for compact folder names
PARAM_ABBREVIATIONS = {
    "samples_per_pt_cap": "spc",
    "learning_rate": "lr",
    "min_lr": "minlr",
    "top_k": "topk",
    "hidden_size": "hs",
    "batch_size": "bs",
    "weight_decay": "wd",
    "patience": "pat",
    "epochs": "ep",
    "lr_reduce_factor": "lrf",
    "lr_reduce_patience": "lrp",
    "lr_step_size": "lss",
    "lr_step_gamma": "lsg",
    "lr_exp_gamma": "leg",
    "onecycle_initial_lr": "oilr",
    "onecycle_max_lr": "omlr",
    "onecycle_final_lr": "oflr",
    "onecycle_pct_start": "ops",
    "even_split": "es",
    "use_weighted_loss": "wl",
    "pairing_seed": "ps",
    "pairing_mode": "pm",
    "pairing_subset_seed": "pss",
}

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, name):
        # dict.__getitem__ raises KeyError on a missing key, but hasattr()/
        # getattr(obj, name, default) only catch AttributeError -- with __getattr__
        # bound directly to dict.__getitem__, a missing key crashed both instead of
        # falling back. Re-raise as AttributeError so attribute-style access on a
        # missing key behaves like normal Python attribute access.
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as e:
            raise AttributeError(name) from e


# Values create_datasets() and train() fall back to for any config key left unset, so a
# config only has to name what matters for its own run (a new dataset has no use for
# the Rydberg-specific keys, say). General knobs take main.py's setup_experiment()
# values; knobs that only some datasets read default to "off". Keys are filled in on
# cf itself, so the config.json saved with a run records what was actually used.
CONFIG_DEFAULTS = {
    # data selection and splitting (create_datasets)
    "data_fraction": 1,
    "even_split": True,
    "samples_per_pt_cap": None,
    "filter_t_values": None,                          # Rydberg datasets only
    "param_cutoffs": {"delta": None, "omega": None},  # Rydberg regression only
    "label_subset_count": None,
    "partition_index": None,
    "label_param": "t",                               # the tuning parameter a regression targets
    "normalize_labels": True,                         # regression only
    # architecture (train)
    "model": "tetriscnn",
    "kernel_set": "smallkernels",
    "equivariant": False,
    "hidden_size": 32,
    "init": "kaiming",
    "weight_penalty": None,
    # sparsity
    "experiment_name": "singlerun",
    # optimisation
    "epochs": 250,
    "batch_size": 64,
    "learning_rate": 1e-2,
    "weight_decay": 1e-5,
    "use_lr_scheduler": False,
    "lr_scheduler_type": "reduce_on_plateau",
    "patience": 10,
    "early_stop_warmup": 150,
    "early_stop_min_delta": 1e-5,
    "num_workers": 0,
    "pin_memory": False,
    "seed": 42,
    # what is written to cf.logdir
    "save_models": True,
    "save_histories": True,
    "save_final_values": False,
}

#: The L1 penalty ramp used when cf.penalty_params is unset: lambda_max = 3, the value
#: behind the manuscript's interpretability results. Unlike the defaults above it is a
#: modelling choice, so train() warns when it has to fall back to it.
DEFAULT_PENALTY_PARAMS = [10, -3, 3, 1]



def logistic_penalty_kwargs(l1: bool = True) -> dict:
    """Keyword arguments choosing an L1 or L2 penalty for `LogisticRegression`.

    scikit-learn 1.8 deprecated `penalty=` in favour of `l1_ratio=` (1.0 for pure
    L1, 0.0 for pure L2) and will remove it in 1.10. The two spellings cannot
    simply be swapped, because on scikit-learn < 1.8 `l1_ratio` is accepted and
    then **ignored** for every solver except saga/elasticnet: it warns and fits an
    L2 model. Passing the new spelling unconditionally would therefore silently
    turn the L1 boundary fits of Figs. 6 and 22-23 into L2 ones on older
    installations, which is exactly the kind of change no warning would catch.

    So the spelling is chosen by version. Both produce the same fit: checked on
    scikit-learn 1.7.2 against 1.9.1 with solver="liblinear", the coefficients
    agree to five significant figures (the residual difference is solver
    tolerance), with an identical support and identical accuracy.
    """
    if _SKLEARN_VERSION >= (1, 8):
        return {"l1_ratio": 1.0 if l1 else 0.0}
    return {"penalty": "l1" if l1 else "l2"}


def apply_config_defaults(cf, for_training=False):
    """Fill every unset key of ``cf`` from CONFIG_DEFAULTS, in place.

    ``cf.dataset`` and ``cf.task`` have no sensible default and must be set. With
    ``for_training`` the training-only derived keys are completed as well: the kernel
    list (from ``cf.kernel_set``), the loss/metric names, and the penalty ramp, which
    falls back to DEFAULT_PENALTY_PARAMS with a warning. ``cf.logdir`` is then required.
    """
    missing = [key for key in ("dataset", "task") if key not in cf]
    if for_training:
        missing += [key for key in ("logdir",) if key not in cf]
    if missing:
        raise ValueError(f"cf is missing required key(s) {missing}; see docs/CONFIGURATION.md.")

    for key, value in CONFIG_DEFAULTS.items():
        if key not in cf:
            cf[key] = copy.deepcopy(value)
    if cf.task == "regression":
        cf.setdefault("goodness_str", "r2agg")
    if not for_training:
        return cf

    if "kernels" not in cf and cf.model == "tetriscnn":
        set_kernels(cf)
    if "loss_str" not in cf or "goodness_str" not in cf:
        set_plotting_logging_strings(cf)
    if "penalty_params" not in cf and cf.experiment_name != "lambdatot":
        warnings.warn(
            f"cf.penalty_params is not set; using {DEFAULT_PENALTY_PARAMS} "
            f"(lambda_max = {DEFAULT_PENALTY_PARAMS[2]}). Set it explicitly to choose "
            f"the sparsity strength.", stacklevel=3)
        cf.penalty_params = list(DEFAULT_PENALTY_PARAMS)
    return cf


def create_path(folder_path, overwrite=False):
    """
    Creates a path to save a file in a folder.
    If the *parent experiment folder* exists, adds a suffix to it.
    """
    base_folder = Path(folder_path)
    experiment_folder = base_folder.parent / base_folder.name  # initial path

    if not overwrite:
        counter = 1
        while experiment_folder.exists():
            print(f"Folder {experiment_folder} exists. Adding suffix.")
            experiment_folder = base_folder.parent.with_name(
                f"{base_folder.parent.name}_{counter}"
            ) / base_folder.name
            counter += 1

    experiment_folder.mkdir(parents=True, exist_ok=True)
    return experiment_folder


def get_sweepable_params(cf, exclude=None):
    """
    Extracts sweepable parameters from config.

    A parameter is sweepable if it's a list with len > 1.

    Args:
        cf: Configuration object (AttrDict)
        exclude: List of parameter names to exclude (default: ['lambdas', 'seeds', 'kernels'])

    Returns:
        OrderedDict of {param_name: list_of_values} sorted alphabetically by abbreviated name
    """
    from collections import OrderedDict

    # Default exclusions
    # branch_groups / branch_orbit_sizes describe the per-branch structure of a single
    # model (one entry per branch, 15 of them for "smallkernels_mixed"), exactly like
    # `kernels`. Without them here, reloading a mixed-set config.json makes both look
    # like swept axes and run_experiments takes their 15x15 cartesian product, re-running
    # the same run 225 times while overwriting cf.branch_groups with a single element.
    default_exclude = ['lambdas', 'seeds', 'kernels', 'penalty_params', 'filter_t_values',
                       'unique_labels', 'branch_groups', 'branch_orbit_sizes']
    if exclude is None:
        exclude = default_exclude
    else:
        exclude = list(set(default_exclude + list(exclude)))

    sweepable = {}
    for key, value in cf.items():
        # Check if it's a list with more than one element
        if isinstance(value, list) and len(value) > 1:
            if key not in exclude:
                sweepable[key] = value

    # Sort by abbreviated name for consistent ordering
    sorted_items = sorted(
        sweepable.items(),
        key=lambda item: PARAM_ABBREVIATIONS.get(item[0], item[0])
    )

    return OrderedDict(sorted_items)


def build_logdir_path(cf, param_values=None):
    """
    Constructs log directory path dynamically.

    For weighted_loss experiment: uses flat naming (like SR)
        logs/{base_name}/{lambda_level}/{param1=val1-param2=val2}/seed_{seed}

    For other experiments: uses hierarchical naming
        logs/{base_name}/{lambda_level}/{param1=val1}/{param2=val2}/seed_{seed}

    Args:
        cf: Configuration object with experiment parameters
        param_values: Dict of current parameter values for sweep (None for no sweep)

    Returns:
        str: Full log directory path
    """
    # Base experiment name
    base_name = f"{cf.experiment_name}_{cf.dataset}_{cf.task}"

    # Add label_param only if it exists (not for lbc task)
    if hasattr(cf, 'label_param') and cf.label_param is not None:
        base_name += f"_{cf.label_param}"

    if getattr(cf, 'model', 'tetriscnn') != "tetriscnn":
        base_name += f"_{cf.model}"
    else:
        base_name += f"_{cf.kernel_set}"
        # Tag the symmetry group for equivariant runs. An equivariant and a
        # non-equivariant run can otherwise agree on every single path component
        # while training different architectures (different branch type, and a
        # different number of branches: 5 for C4 and 6 for D2/K4 against 10 for
        # smallkernels), so without this they silently overwrite each other's logs.
        # Non-equivariant paths are unaffected, keeping older runs discoverable.
        if getattr(cf, 'equivariant', False):
            base_name += f"_{getattr(cf, 'equivariant_group', 'C4')}"
        base_name += f"_{cf.lambdas}"

    # Pairing-robustness ablation: a non-default snapshot-pairing convention changes
    # what the *data* is, not just the fit, so each convention gets its own log
    # subtree. Without this the arms collide: pairing_mode is a scalar within an arm,
    # so it never reaches the sweep levels below and repair_resample would silently
    # overwrite repair_fixed. Deliberately a no-op for the default "index" pairing,
    # so every pre-ablation log path is unchanged.
    pairing_mode = getattr(cf, 'pairing_mode', 'index')
    if pairing_mode != "index":
        base_name += f"_pm={pairing_mode}"

    # Lambda level
    lambda_level = f"{cf.experiment_name}_{cf.lam}"

    # Build path components
    path_parts = ["logs", base_name, lambda_level]

    # Add parameter sweep levels
    if param_values:
        # For weighted_loss experiment: use flat naming (like SR)
        if cf.experiment_name == "weighted_loss":
            # Build flat parameter string: param1=val1-param2=val2-...
            param_strs = []
            for param_name in sorted(param_values.keys(),
                                    key=lambda k: PARAM_ABBREVIATIONS.get(k, k)):
                abbrev = PARAM_ABBREVIATIONS.get(param_name, param_name)
                value = param_values[param_name]
                param_strs.append(f"{abbrev}={value}")

            # Add as single path component
            path_parts.append("-".join(param_strs))
        else:
            # For other experiments: use hierarchical naming
            for param_name in sorted(param_values.keys(),
                                    key=lambda k: PARAM_ABBREVIATIONS.get(k, k)):
                abbrev = PARAM_ABBREVIATIONS.get(param_name, param_name)
                value = param_values[param_name]
                path_parts.append(f"{abbrev}={value}")

    # Add seed level (only if multiple seeds)
    if hasattr(cf, 'seeds') and len(cf.seeds) > 1:
        path_parts.append(f"seed_{cf.seed}")

    # Build the path
    logdir = str(Path(*path_parts))

    # Add partition index suffix if applicable
    if hasattr(cf, 'partition_index') and cf.partition_index is not None:
        logdir = str(Path(logdir) / f"partition_{cf.partition_index}")

    return logdir


def build_sr_folder_path(
    sr_folder_base: Path,
    sr_config: 'SRConfig'
) -> Path:
    """Construct SR subfolder path with parameter tracking.

    Uses flat naming when SR config differs from defaults:
        SR_mode=averaged_topk=5_maxd=15/

    Uses simple "SR/" when all defaults or when override is specified.

    Args:
        sr_folder_base: Base path (e.g., Path(cf.logdir))
        sr_config: SRConfig instance

    Returns:
        Path: SR folder path
    """
    sr_folder_base = Path(sr_folder_base)

    # Check for manual override
    if sr_config.sr_folder_name_override is not None:
        return sr_folder_base / sr_config.sr_folder_name_override

    significant_params = sr_config.get_significant_params()

    if not significant_params:
        return sr_folder_base / "SR_default"

    param_strs = [
        f"{param_name}={value}"
        for param_name, value in sorted(significant_params.items())
    ]

    folder_name = "SR_" + "_".join(param_strs)
    return sr_folder_base / folder_name


def get_sr_folder_name(sr_config: 'SRConfig') -> str:
    """Get just the SR folder name without base path (for TensorBoard naming).

    Returns folder name like "SR" or "SR_mode=averaged_topk=5_maxd=15"

    Args:
        sr_config: SRConfig instance

    Returns:
        str: SR folder name
    """
    # Check for manual override
    if sr_config.sr_folder_name_override is not None:
        return sr_config.sr_folder_name_override

    significant_params = sr_config.get_significant_params()

    if not significant_params:
        return "SR_default"

    param_strs = [
        f"{param_name}={value}"
        for param_name, value in sorted(significant_params.items())
    ]

    return "SR_" + "_".join(param_strs)


def find_existing_runs(basepath, remake_flags=None):
    """
    Discovers existing run folders when remaking plots/SR.

    Handles variable folder hierarchy depth by searching for config.json files.

    Args:
        basepath: Base directory to search in (Path or str)
        remake_flags: Dict of remake flags (for potential filtering)

    Returns:
        list of Path: Directories containing config.json (seed-level directories)
    """
    basepath = Path(basepath)

    # Find all config.json files recursively
    config_files = list(basepath.glob("**/config.json"))

    # Return parent directories (seed-level folders)
    seed_dirs = [config_file.parent for config_file in config_files]

    return seed_dirs


def get_experiment_level_path(seed_path):
    """
    Navigates from seed level to experiment base level.

    Finds the directory level containing metrics_per_lambda.json by going up
    the directory tree.

    Args:
        seed_path: Path to seed-level directory (Path or str)

    Returns:
        Path: Experiment base level path
    """
    seed_path = Path(seed_path)
    current_path = seed_path

    # Go up until we find metrics_per_lambda.json or hit logs/ directory
    while current_path.name != "logs" and current_path.parent != current_path:
        # Check if metrics_per_lambda.json exists at this level
        if (current_path / "metrics_per_lambda.json").exists():
            return current_path
        if (current_path / "metrics_per_lambda_remake.json").exists():
            return current_path

        current_path = current_path.parent

    # If not found, go up 2 levels from seed path (backward compatibility)
    # This handles the case where metrics_per_lambda.json doesn't exist yet
    experiment_level = seed_path.parent.parent
    return experiment_level




def save_json(obj, folder_path, file_name):
    """
    Saves obj to path as a json file.
    """
    def _json_safe(value):
        # numpy arrays and scalars become native JSON. They used to fall through to
        # str(), which recorded arrays as their printed form ("[   0.  250. ...]") and
        # left every reader to parse that back; see parse_array_field() for the reader
        # side, which still has to accept runs recorded that way.
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, list):
            return [float(x) if isinstance(x, np.floating) else x for x in value]
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        if not isinstance(value, (str, int, float, bool, type(None))):
            return str(value)
        return value

    obj_to_save = {
        k: _json_safe(v)
        for k, v in obj.items()
    }
    folder = Path(folder_path)
    load_path = folder.joinpath(file_name)
    with open(load_path, "w") as file:
        json.dump(obj_to_save, file)

def parse_array_field(raw):
    """Read back a numeric array recorded in a config.json, whatever form it took.

    Until save_json() learned to write numpy arrays as JSON lists, it recorded them
    as their printed form, so fields such as ``unique_labels`` in most runs under
    ``Plots_data/`` and ``App_*_data/`` hold a string like ``"[   0.  250. ...]"``
    (newlines included) rather than a list. This accepts either, and also the
    two-dimensional printed form ``"[[a b]\n [c d]]"`` that a ``deltaomega`` label
    table produces, keeping its shape.

    Returns:
        np.ndarray of float.
    """
    if not isinstance(raw, str):
        return np.asarray(raw, dtype=float)
    body = raw.strip().strip("[]").strip()
    rows = [row for row in re.split(r"\]\s*\[", body) if row.strip()]
    parsed = [np.array(row.replace("[", " ").replace("]", " ").split(), dtype=float)
              for row in rows]
    if not parsed:
        return np.array([], dtype=float)
    return parsed[0] if len(parsed) == 1 else np.vstack(parsed)


def load_json(folder_path, file_name):
    """
    Loads a json file from path.
    """
    folder = Path(folder_path)
    load_path = folder.joinpath(file_name)
    with open(load_path) as file:
        return json.load(file)


def set_seeds(seed_no):
    """"
    Sets the seed for reproducibility.
    NB: seed is accessible globally.
    """
    global seed
    seed = seed_no
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EarlyStopper:
    """
    Early stopping class with warmup period.

    During warmup, early stopping is disabled to allow the network to learn
    without premature termination. After warmup, normal early stopping behavior resumes.

    Note: Expects to be called with epoch number (0-indexed) for proper warmup tracking.
    """
    def __init__(self, patience=1, min_delta=0, verbose=True, relative=False, pbar=None, warmup_epochs=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_obj = float('inf')
        self.verbose = verbose
        self.relative = relative
        self.warmup_epochs = warmup_epochs

    def early_stop(self, validation_obj, epoch=None):
        """
        Check if training should stop early.

        Args:
            validation_obj: Validation loss/objective to monitor
            epoch: Current epoch number (0-indexed). If None, uses internal counter (legacy behavior).

        Returns:
            True if training should stop, False otherwise
        """
        # Determine current epoch
        if epoch is not None:
            current_epoch = epoch
        else:
            # Legacy behavior: use internal counter (will be 1-indexed after first call)
            if not hasattr(self, '_internal_epoch'):
                self._internal_epoch = 0
            self._internal_epoch += 1
            current_epoch = self._internal_epoch - 1  # Convert to 0-indexed

        # During warmup, never stop early (epochs 0 to warmup_epochs-1)
        if current_epoch < self.warmup_epochs:
            # Still track the best validation objective during warmup
            if validation_obj < self.min_validation_obj:
                self.min_validation_obj = validation_obj
                self.counter = 0
            return False

        # After warmup, normal early stopping logic
        if np.isnan(validation_obj):
            print("Validation objective is NaN. Stopping early.")
            return True
        difference = validation_obj - self.min_validation_obj
        if self.relative:
            difference /= self.min_validation_obj
        if validation_obj < self.min_validation_obj:
            # if self.verbose:
                # tqdm.write(f"Validation objective decreased ({self.min_validation_obj:.2e} --> {validation_obj:.2e}).")
            self.min_validation_obj = validation_obj
            self.counter = 0
        elif difference >= self.min_delta:
            self.counter += 1
            if self.verbose:
                tqdm.write(f"Validation objective increased ({self.min_validation_obj:.2e} --> {validation_obj:.2e}). Counter: {self.counter} out of {self.patience}.")
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        """Reset counter and internal epoch tracking."""
        self.counter = 0
        if hasattr(self, '_internal_epoch'):
            self._internal_epoch = 0
    
##############################################
# TetrisCNN specific functions
##############################################

def set_kernels(cf):
    # Equivariant branches are only defined for "smallkernels", the one set for which
    # canonical group representatives have been worked out. Every other set would build
    # ConvBranch_Equivariant over its ordinary kernel list and silently return something
    # that is not what the flag promises: "defaultkernels" carries both (2,1) and (1,2),
    # which are the same C4 orbit and so become duplicate branches, plus dilated entries;
    # "bigkernels_stride2" carries stride-2 entries, which ConvBranch_Equivariant rejects;
    # "bigkernels" carries a (6,7) kernel whose rotation does not fit a (6,7) image.
    if getattr(cf, "equivariant", False) and cf.kernel_set == "smallkernels_mixed":
        raise ValueError(
            "cf.kernel_set='smallkernels_mixed' already carries per-branch equivariance "
            "in cf.branch_groups, so cf.equivariant must stay False. Setting both would "
            "make every branch equivariant and silently destroy the mixture."
        )

    if getattr(cf, "equivariant", False) and cf.kernel_set != "smallkernels":
        raise NotImplementedError(
            f"cf.equivariant=True is only implemented and tested for "
            f"cf.kernel_set='smallkernels', not '{cf.kernel_set}'. Only smallkernels "
            "defines canonical group representatives (5 for C4, 6 for D2/K4, covering "
            "the same 10 patterns as the non-equivariant list)."
        )
    if getattr(cf, "equivariant", False) and getattr(cf, "equivariant_group", "C4") not in ("C4", "D2", "K4"):
        raise ValueError(
            f"Unknown cf.equivariant_group '{cf.equivariant_group}': expected 'C4', "
            "'D2', or 'K4' ('D2' and 'K4' are aliases for the rectangle/Klein-four "
            "symmetry group)."
        )

    # Per-branch group assignment, consumed by ShapeAdaptiveConvNet(branch_groups=...)
    # and get_branch_penalties(branch_groups=...). Only "smallkernels_mixed" sets it to
    # a real list; every other kernel set is uniform and leaves it None, which restores
    # the ordinary cf.equivariant-driven behaviour. Assigned unconditionally so a swept
    # cf reused across kernel sets cannot carry a stale mixture forward.
    cf.branch_groups = None
    cf.branch_orbit_sizes = None

    if cf.kernel_set == "defaultkernels":
        cf.kernels = [              # DEFAULT Kacper's experimental paper set
            [(1,1), 1, 1, None],    # kernel size, no_filters, dilation, mask.
            [(2,1), 1, 1, None],
            [(1,2), 1, 1, None],
            [(2,2), 1, 1, None],
            [(2,1), 1, 2, None],
            [(1,2), 1, 2, None],
            [(3,1), 1, 1, None],
            [(1,3), 1, 1, None],
            [(3,3), 1, 1, None],
        ]
    elif cf.kernel_set == "bigkernels":
        if cf.dataset == "Paris_Ising":
            cf.kernels = [      # BIG KERNELS       
                [(1,1), 1, 1, None],    # kernel size, no_filters, dilation, mask.
                [(2,2), 1, 1, None],
                [(4,4), 1, 1, None],
                [(8,8), 1, 1, None],      # Paris_Ising
            ]
        elif cf.dataset.__contains__('Paris_XY'):
            cf.kernels = [      # BIG KERNELS       
                [(1,1), 1, 1, None],    # kernel size, no_filters, dilation, mask.
                [(2,2), 1, 1, None],
                [(4,4), 1, 1, None],
                [(6,7), 1, 1, None],      # Paris_Ising
            ]
        elif cf.dataset == "ILGT":
            cf.kernels = [      # BIG KERNELS       
                [(1,1), 1, 1, None],    # kernel size, no_filters, dilation, mask.
                [(2,2), 1, 1, None],
                [(4,4), 1, 1, None],
                [(8,8), 1, 1, None],      
                [(16,16), 1, 1, None],  
            ]
    elif cf.kernel_set == "bigkernels_stride2":
        if cf.dataset == "Paris_Ising":
            cf.kernels = [      # BIG KERNELS       
                [(1,1), 1, 1, None, 1],    # kernel size, no_filters, dilation, mask, stride
                [(2,2), 1, 1, None, 2],
                [(4,4), 1, 1, None, 1],
                [(8,8), 1, 1, None, 1],      # Paris_Ising
            ]
        elif cf.dataset.__contains__('Paris_XY'):
            cf.kernels = [      # BIG KERNELS       
                [(1,1), 1, 1, None, 1],    # kernel size, no_filters, dilation, mask, stride
                [(2,2), 1, 1, None, 2],
                [(4,4), 1, 1, None, 1],
                [(6,7), 1, 1, None, 1],      # Paris_Ising
            ]
        elif cf.dataset == "ILGT":
            cf.kernels = [      # BIG KERNELS       
                [(1,1), 1, 1, None, 1],    # kernel size, no_filters, dilation, mask.
                [(2,2), 1, 1, None, 2],
                [(4,4), 1, 1, None, 1],
                [(8,8), 1, 1, None, 1],      
                [(16,16), 1, 1, None, 1],  
            ]
    elif cf.kernel_set == "smallkernels":
        equivariant_group = getattr(cf, "equivariant_group", "C4")
        if getattr(cf, "equivariant", False) and equivariant_group in ("D2", "K4"):
            # Canonical D2/K4 (Klein four-group = rectangle symmetry) representatives.
            # Unlike C4, the flip-only group does not mix (2,1) and (1,2), so the
            # vertical and horizontal dominoes are separate orbits and both survive
            # as canonical branches (6 total, vs. 5 for C4). The diagonal mask
            # [[1,0],[0,1]] and the L-tromino mask [[1,1],[1,0]] each still have a
            # D2 orbit of the same size as their C4 orbit (2 and 4 respectively,
            # verified by direct enumeration of flip(dims=[2]), flip(dims=[3]),
            # flip(dims=[2,3])), so one canonical representative each still covers
            # the whole orbit (see models.ConvBranch_Equivariant).
            cf.kernels = [       # SMALL KERNELS (D2/K4-equivariant, canonical reps)
                [(1,1), 1, 1, None],    # kernel size, no_filters, dilation, mask.
                [(2,1), 1, 1, None],
                [(1,2), 1, 1, None],
                [(2,2), 1, 1, [[1,0],[0,1]] ],
                [(2,2), 1, 1, [[1,1],[1,0]] ],
                [(2,2), 1, 1, None],
            ]
        elif getattr(cf, "equivariant", False):
            # Canonical C4 representatives: each branch's ConvBranch_Equivariant
            # rotation-orbit reproduces the corresponding 2 (diagonal masks) or 4
            # (L-tromino masks, (2,1)/(1,2)) branches of the non-equivariant list
            # below, collapsing 10 branches into 5 (see models.ConvBranch_Equivariant).
            cf.kernels = [       # SMALL KERNELS (C4-equivariant, canonical reps)
                [(1,1), 1, 1, None],    # kernel size, no_filters, dilation, mask.
                [(2,1), 1, 1, None],
                [(2,2), 1, 1, [[1,0],[0,1]] ],
                [(2,2), 1, 1, [[1,1],[1,0]] ],
                [(2,2), 1, 1, None],
            ]
        else:
            cf.kernels = [       # SMALL KERNELS
                [(1,1), 1, 1, None],    # kernel size, no_filters, dilation, mask.
                [(2,1), 1, 1, None],
                [(1,2), 1, 1, None],
                [(2,2), 1, 1, [[1,0],[0,1]] ],
                [(2,2), 1, 1, [[0,1],[1,0]] ],
                [(2,2), 1, 1, [[1,1],[1,0]] ],
                [(2,2), 1, 1, [[1,1],[0,1]] ],
                [(2,2), 1, 1, [[0,1],[1,1]] ],
                [(2,2), 1, 1, [[1,0],[1,1]] ],
                [(2,2), 1, 1, None],
            ]

    elif cf.kernel_set == "chainkernels":
        # SMALL KERNELS for 1D chains (the simulated 1D TFIM and XXZ datasets), whose
        # snapshots are stored as one-column (N, 1) images. The 2D sets above all carry
        # horizontally extended patterns such as (1,2), which do not fit a single column,
        # so a chain gets its own list: the site, the nearest-neighbour bond, its
        # next-nearest-neighbour counterpart via dilation, and three- and four-site
        # segments. As in 2D, the branch of pattern area A encodes the A-point
        # correlator and is penalised by lambda(A).
        cf.kernels = [
            [(1,1), 1, 1, None],    # kernel size, no_filters, dilation, mask.
            [(2,1), 1, 1, None],
            [(2,1), 1, 2, None],
            [(3,1), 1, 1, None],
            [(4,1), 1, 1, None],
        ]

    elif cf.kernel_set == "smallkernels_mixed":
        # MIXED set: the full 10-branch non-equivariant "smallkernels" list AND the
        # canonical equivariant representatives, side by side in one bottleneck. Every
        # pattern is therefore present twice over -- once as individual orientations
        # (branches 0-9) and once orbit-averaged (branches 10+) -- so the L1 penalty has
        # to choose between an orientation-resolved and a symmetrised reading of the
        # SAME correlator content. That competition is the whole point of the set.
        #
        # `orbit_sizes` records how many non-equivariant branches each equivariant
        # representative stands in for. It is what the "inverse_orbit" rebate divides
        # by, and it sums to 10 for both groups -- the two halves cover exactly the same
        # ten patterns, which is what makes the comparison fair.
        group = getattr(cf, "mixed_group", "C4")
        non_equivariant = [
            [(1,1), 1, 1, None],
            [(2,1), 1, 1, None],
            [(1,2), 1, 1, None],
            [(2,2), 1, 1, [[1,0],[0,1]] ],
            [(2,2), 1, 1, [[0,1],[1,0]] ],
            [(2,2), 1, 1, [[1,1],[1,0]] ],
            [(2,2), 1, 1, [[1,1],[0,1]] ],
            [(2,2), 1, 1, [[0,1],[1,1]] ],
            [(2,2), 1, 1, [[1,0],[1,1]] ],
            [(2,2), 1, 1, None],
        ]
        if group == "C4":
            equivariant = [
                [(1,1), 1, 1, None],
                [(2,1), 1, 1, None],
                [(2,2), 1, 1, [[1,0],[0,1]] ],
                [(2,2), 1, 1, [[1,1],[1,0]] ],
                [(2,2), 1, 1, None],
            ]
            orbit_sizes = [1, 2, 2, 4, 1]
        elif group in ("D2", "K4"):
            equivariant = [
                [(1,1), 1, 1, None],
                [(2,1), 1, 1, None],
                [(1,2), 1, 1, None],
                [(2,2), 1, 1, [[1,0],[0,1]] ],
                [(2,2), 1, 1, [[1,1],[1,0]] ],
                [(2,2), 1, 1, None],
            ]
            orbit_sizes = [1, 1, 1, 2, 4, 1]
        else:
            raise ValueError(
                f"Unknown cf.mixed_group '{group}': expected 'C4', 'D2' or 'K4'."
            )
        assert sum(orbit_sizes) == len(non_equivariant), (
            "The equivariant representatives must cover exactly the non-equivariant "
            f"patterns: orbit sizes sum to {sum(orbit_sizes)}, not {len(non_equivariant)}."
        )
        cf.kernels = non_equivariant + equivariant
        cf.branch_groups = [None] * len(non_equivariant) + [group] * len(equivariant)
        cf.branch_orbit_sizes = [1] * len(non_equivariant) + orbit_sizes

    if "kernels" not in cf.keys() or cf.kernels is None:
        # "bigkernels"/"bigkernels_stride2" branch on cf.dataset and only cover the
        # Rydberg and ILGT lattices, so an unlisted dataset would otherwise fall through
        # with no kernels at all (or, on a swept cf, with the previous set still attached).
        raise ValueError(
            f"No kernel list is defined for cf.kernel_set={cf.kernel_set!r} with "
            f"cf.dataset={getattr(cf, 'dataset', None)!r}. Available sets: "
            "'smallkernels', 'defaultkernels', 'bigkernels', 'bigkernels_stride2', "
            "'smallkernels_mixed' (2D lattices), and 'chainkernels' (1D chains)."
        )

    for k in range(len(cf.kernels)):
        if len(cf.kernels[k]) < 5: # TODO: this is a quickfix
                cf.kernels[k].append(1) # add stride if not specified
    # return cf


def validate_kernels_fit(kernels, input_shape, kernel_set=None):
    """Check every branch's receptive field against the snapshot it will read.

    A convolution whose kernel is larger than its input raises a torch RuntimeError
    deep inside the forward pass, naming neither the branch nor the dataset. Checking
    up front lets the message say which branch does not fit and what to do about it,
    which matters most for the 1D datasets: their snapshots are (N, 1) columns, so any
    horizontally extended pattern in a 2D kernel set is out of bounds.

    Args:
        kernels: cf.kernels, i.e. [[(h, w), no_filters, dilation, mask, stride], ...].
        input_shape: the snapshot's (height, width), without the channel axis.
        kernel_set: cf.kernel_set, quoted in the error message when given.
    """
    height, width = int(input_shape[0]), int(input_shape[1])
    offenders = []
    for k, spec in enumerate(kernels):
        (kh, kw), dilation = spec[0], spec[2]
        # A dilated kernel spans (k - 1) * d + 1 sites, not k.
        eff_h, eff_w = (kh - 1) * dilation + 1, (kw - 1) * dilation + 1
        if eff_h > height or eff_w > width:
            offenders.append(f"branch {k}: {kh}x{kw} (dilation {dilation}, spanning "
                             f"{eff_h}x{eff_w})")
    if offenders:
        set_str = f" of cf.kernel_set={kernel_set!r}" if kernel_set else ""
        hint = (" Snapshots one site wide are 1D chains; use "
                "cf.kernel_set='chainkernels' for them.") if width == 1 else ""
        raise ValueError(
            f"These branches{set_str} do not fit a {height}x{width} snapshot: "
            + "; ".join(offenders) + "." + hint
        )

def should_train(cf):
    return not cf.remake_history_and_pt_plots and not cf.remake_lambda_plot and not cf.remake_sweep_plot

def mask_to_latex_pattern(mask, equation_env=True, full_latex=False):
    """
    Convert a mask to a LaTeX pattern representation.
    """
    mask = np.array(mask)
    width, height = mask.shape

    if full_latex:
        filled_char = r"\square"
        empty_char = r"\blacksquare"
    else:
        filled_char = "\u25A1"  # filled square character
        empty_char = "\u25A0"   # empty square character

    base_pattern = []
    for i in range(height):
        row = ""  # Initialize an empty string to build the row
        for j in range(width):
            if mask[j,i] != 1: # NB: j, i 
                row += filled_char  # filled square character
            else:
                row += empty_char  # empty square character
        base_pattern.append(row)

    if height > 1:
        latex_pattern = "\\substack{{{0}}}".format(r' \\ '.join(base_pattern))
    else:
        latex_pattern = base_pattern[0]

    if width < 6 and height < 6:
        if equation_env:
            return rf"${latex_pattern}$"
        else:
            return latex_pattern
    else:
        return f"[{width}x{height}]"
    

def kernel_to_latex_pattern(pattern_spec, equation_env=True, full_latex=False):
    """
    Convert a kernel specification to a LaTeX pattern representation.
    """
    dimensions, _, dilation, mask, stride = pattern_spec
    width, height = dimensions

    if full_latex:
        filled_char = r"\square"
        empty_char = r"\blacksquare"
    else:
        filled_char = "\u25A1"  # filled square character
        empty_char = "\u25A0"   # empty square character

    # Calculate the effective width and height considering dilation
    expanded_width = width + (width - 1) * (dilation - 1)
    expanded_height = height + (height - 1) * (dilation - 1)
    
    if mask is None:
        mask = np.ones((expanded_width, expanded_height))
    else:
        mask = np.array(mask)

    base_pattern = []
    for i in range(expanded_height):
        row = ""  # Initialize an empty string to build the row
        for j in range(expanded_width):
            if j % dilation != 0 or i % dilation != 0 or mask[j,i] != 1: # NB: j, i 
                row += filled_char
            else:
                row += empty_char
        base_pattern.append(row)
    # print(f"Base pattern for kernel {pattern_spec}:")
    # pprint.pprint(base_pattern)
    if height > 1:
        # latex_pattern = f"\\substack{{{r'\\'.join(base_pattern)}}}"
        joined = r'\\'.join(base_pattern)
        latex_pattern = f"\\substack{{{joined}}}"
    else:
        latex_pattern = base_pattern[0]

    if expanded_width < 9 and expanded_height < 9:
        if equation_env:
            return rf"${latex_pattern}$"
        else:
            return latex_pattern
    else:
        return f"[{width}x{height}], dil={dilation}"


def l1_regularization(z, penalties):
    r""""
    Returns $\sum_{k=1}^K |z_k| \lambda_k$, where $z_k$ is the $k-$th bottleneck activation,
    and $\lambda_k$ is the corresponding penalty for this branch. l1 regularization encourages
    sparsity in the bottleneck, and penalties encourage simplicity.
    """
    assert z.shape[1] == len(penalties), "Number of penalties must match the number of branches."
    branches = torch.abs(z)
    branches *= penalties # TODO double check
    return torch.sum( branches, dim=1 ) # sum over the branches


def set_plotting_logging_strings(cf):
    """
    Sets strings for plotting and logging based on task and goodness function.
    """  
    if cf.task != "regression":
        cf.goodness_str = "acc"
    cf.phase_indicator_str = r"$\bar{y}$" if cf.task == "classification" else r"${\partial\hat{\gamma}}/{\partial\gamma}$"
    cf.loss_str = "CEL" if cf.task != "regression" else "MSE"



def normalize01(y, y_min, y_max):
    """
    Normalize to [0, 1] range.
    """
    return (y - y_min) / (y_max - y_min)

def denormalize01(y, y_min, y_max):
    """
    Denormalize from [0, 1] range to original range.
    """
    return y * (y_max - y_min) + y_min


def r2_agg(y_pred, y_true, multioutput='variance_weighted'):
    """
    Compute aggregated R² score: first aggregates predictions by unique y_true values,
    then computes R² on the aggregated data. Supports variance-weighted multioutput.

    This function handles cases where multiple samples share the same y_true value
    (e.g., repeated measurements at the same parameter value). It aggregates
    predictions for each unique y_true value, then computes R².

    Args:
        y_true: tensor [N, D] or [N, 1] or [N] - ground truth values
        y_pred: tensor [N, D] or [N, 1] or [N] - predicted values
        multioutput: str - 'variance_weighted' (default), 'uniform_average', or 'raw_values'
            - 'variance_weighted': R² scores averaged weighted by variance of each output
            - 'uniform_average': R² scores averaged with uniform weight
            - 'raw_values': Returns individual R² score for each output dimension

    Returns:
        R² score (scalar for weighted/uniform, tensor [D] for raw_values)

    Notes:
        Implements variance weighting consistent with scikit-learn and torcheval.
        For single output, variance weighting has no effect (returns same as uniform).
    """
    # Ensure at least 2D tensors [N, D]
    if y_true.dim() == 1:
        y_true = y_true.view(-1, 1)
    if y_pred.dim() == 1:
        y_pred = y_pred.view(-1, 1)

    assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"

    N, D = y_true.shape

    # Compute per-output aggregated R² scores
    r2_scores = []
    variances = []

    for i in range(D):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        # Get unique target values and aggregate predictions
        unique_targets = torch.unique(y_true_i)
        avg_preds = torch.stack([
            y_pred_i[y_true_i == t].mean() for t in unique_targets
        ])

        # Compute R² on aggregated data
        ss_res = torch.sum((unique_targets - avg_preds) ** 2)
        ss_tot = torch.sum((unique_targets - unique_targets.mean()) ** 2)
        r2_i = 1 - ss_res / (ss_tot + 1e-10)  # Add epsilon to avoid division by zero

        r2_scores.append(r2_i)

        # Variance of unique true values for this output (for weighting)
        var_i = torch.var(unique_targets, unbiased=False)  # Population variance
        variances.append(var_i)

    r2_scores = torch.stack(r2_scores)
    variances = torch.stack(variances)

    if multioutput == 'raw_values':
        return r2_scores
    elif multioutput == 'uniform_average':
        return torch.mean(r2_scores)
    elif multioutput == 'variance_weighted':
        # Variance-weighted average: R² = Σ(var(y_true_i) × R²_i) / Σ(var(y_true_i))
        total_var = torch.sum(variances)
        if total_var > 1e-10:  # Avoid division by zero
            weighted_r2 = torch.sum(r2_scores * variances) / total_var
        else:
            # Fall back to uniform average if all variances are zero
            weighted_r2 = torch.mean(r2_scores)
        return weighted_r2
    else:
        raise ValueError(f"Unknown multioutput option: {multioutput}. Use 'variance_weighted', 'uniform_average', or 'raw_values'.")




def get_branch_penalties(lambda_params, kernels, branch_groups=None,
                        equivariant_rebate=None, branch_orbit_sizes=None):
    """"
    Calculates the penalties (lambdas) for network branches, based on kernel parameters.
    Large kernels (areas) are penalized more; i.e. have larger lambda values, exponentially.
    Kernels with more dilation are penalized more, linearly.
    
    lambdas[0]: base of exponential, default 10
    lambdas[1]: min exponent, default -4
    lambdas[2]: max exponent, default 0
    lambdas[3]: n_pen, default 1. 
    This is penalty scaling for additional kernels of the same area. 
    Models trained with this are marked as `VarLambdas`.
    For model without this, set n_pen to 1.

    branch_groups / equivariant_rebate / branch_orbit_sizes (all optional, all None by
    default -> the returned penalties are byte-identical to the historical behaviour):
    used only by the mixed "smallkernels_mixed" set, where some branches are equivariant
    and some are not. `branch_groups[k]` is None for an ordinary branch or a group name
    for an equivariant one, and `equivariant_rebate` multiplies the penalty of every
    equivariant branch:

      float r        -- uniform multiplier. r < 1 discounts the symmetrised branch,
                        r > 1 surcharges it. r = 1 is the neutral baseline: note that
                        neutral is NOT the same as no advantage, since one equivariant
                        branch carries the information that |orbit| non-equivariant
                        branches would each have to pay lambda for separately.
      "inverse_orbit" -- r_k = 1 / branch_orbit_sizes[k], i.e. the equivariant branch is
                        charged per non-equivariant pattern it stands in for.

    The rebate is applied on top of the area-keyed lambda, which is itself group
    invariant, so it is the ONLY thing that distinguishes the two branch types.
    """
    max_kernel_area = max([k[0][0] * k[0][1] for k in kernels]) # e.g. 3x1=3 for TFIM

    penalty_base = np.logspace (
        base = float(lambda_params[0]),
        start = float(lambda_params[1]),
        stop = float(lambda_params[2]),
        num = int(max_kernel_area),
    )
    n_pen = float(lambda_params[3])

    if equivariant_rebate is not None and branch_groups is None:
        raise ValueError(
            "equivariant_rebate was given without branch_groups, so there is no way to "
            "tell which branches are equivariant. Pass cf.branch_groups as well."
        )
    if branch_groups is not None and len(branch_groups) != len(kernels):
        raise ValueError(
            f"branch_groups has {len(branch_groups)} entries but there are "
            f"{len(kernels)} kernels; they must correspond one-to-one."
        )
    if equivariant_rebate == "inverse_orbit" and branch_orbit_sizes is None:
        raise ValueError(
            "equivariant_rebate='inverse_orbit' needs branch_orbit_sizes to divide by; "
            "cf.branch_orbit_sizes is set by set_kernels for 'smallkernels_mixed'."
        )

    penalties = []
    for b_idx, (k_shape, k_filt_num, k_dilation, k_mask, k_stride) in enumerate(kernels):
        # "area" is the number of sites the pattern reads, |P| in Eq. (11) of the manuscript.
        if k_mask:
            k_area = np.count_nonzero(np.array(k_mask))
        else:
            k_area = k_shape[0] * k_shape[1]

        l = penalty_base[k_area - 1] # e.g. area 1 => 10^{lambda_params[1]} = 10^{\lambda_{min}}
        # TODO: what if areas are not consecutive, e.g. from 2x1=2 to 2x2=4? 
        
        # Equivariant rebate/surcharge, mixed sets only (rebate == 1 for every branch
        # when branch_groups is None, so the default path is untouched).
        rebate = 1.0
        if branch_groups is not None and branch_groups[b_idx] is not None:
            if equivariant_rebate == "inverse_orbit":
                rebate = 1.0 / float(branch_orbit_sizes[b_idx])
            elif equivariant_rebate is not None:
                rebate = float(equivariant_rebate)

        for fnum in range(k_filt_num):
            factor = k_dilation if k_area > 1 else 1 # linear scaling with dilation
            scale = n_pen**fnum if n_pen != 1 else 1 # exponential scaling with number of filters, only if n_pen != 1
            penalties.append(l * scale * factor * rebate)
    return np.array(penalties).flatten()






def update_metrics_per_seed(cf, metrics, metrics_per_seed):
    if cf.model == "tetriscnn":
        index = cf.patience if cf.epochs > cf.patience else 1
        
        if cf.task == "regression":                  
            MVUL_out = np.array(metrics["MVUL_out"])[-index]
            metrics_per_seed["MVUL_out"] = MVUL_out.tolist()

            # try:
                # new code
            if cf.label_param == "deltaomega":
                metrics_per_seed["pt"].append( metrics["pt"][-index] )
                metrics_per_seed["pt2"].append( metrics["pt2"][-index] )
            else:
                metrics_per_seed["pt"].append( metrics["pt"][-index] )
            # except: 
            #     # TEMPORARY, for old data without pt saved

        if cf.save_histories:
            assert len(np.array(metrics[f'z_0'])) > 1 , "Saved data does not have right dimension; history not saved?"
            z = np.array([ metrics[f'z_{i}'][-index] for i in range(len(cf.kernels)) ]) 
        else:
            assert len(np.array(metrics[f'z_0'])) == 1, "Saved data does not have right dimension; history was saved."
            z = np.array([ metrics[f'z_{i}'] for i in range(len(cf.kernels)) ]) # if we did not save histories, there is only one value
        assert z.ndim == 1 and z.shape[0] == len(cf.kernels), f"z has wrong shape: {z.shape}, expected ({len(cf.kernels)}, )"
        metrics_per_seed["z"].append(z.tolist()) # one value per kernel

        metrics_per_seed["net2_norm"].append(metrics["net2_norm"])
        metrics_per_seed[f"val_{cf.loss_str}"].append(metrics[f"val_{cf.loss_str}"][-index])
        metrics_per_seed[f"val_{cf.goodness_str}"].append(metrics[f"val_{cf.goodness_str}"][-index])
        metrics_per_seed["epochs"].append(len(metrics[f"val_{cf.loss_str}"]))

def update_metrics_per_partition(cf, metrics, metrics_per_partition):
    if cf.model == "tetriscnn":
        index = cf.patience if cf.epochs > cf.patience else 1
        if cf.save_histories:
            assert len(np.array(metrics[f'z_0'])) > 1 , "Saved data does not have right dimension; history not saved?"
            z = np.array([ metrics[f'z_{i}'][-index] for i in range(len(cf.kernels)) ])
        else:
            assert len(np.array(metrics[f'z_0'])) == 1, "Saved data does not have right dimension; history was saved."
            z = np.array([ metrics[f'z_{i}'] for i in range(len(cf.kernels)) ]) # if we did not save histories, there is only one value
        assert z.ndim == 1 and z.shape[0] == len(cf.kernels), f"z has wrong shape: {z.shape}, expected ({len(cf.kernels)}, )"
        metrics_per_partition["z"].append(z.tolist()) # one value per kernel
        metrics_per_partition[f"val_{cf.loss_str}"].append(metrics[f"val_{cf.loss_str}"][-index])
        metrics_per_partition[f"train_{cf.loss_str}"].append(metrics[f"train_{cf.loss_str}"][-index])
        metrics_per_partition[f"val_{cf.goodness_str}"].append(metrics[f"val_{cf.goodness_str}"][-index])
        metrics_per_partition[f"train_{cf.goodness_str}"].append(metrics[f"train_{cf.goodness_str}"][-index])
    else:  # non-tetriscnn model — no bottleneck z, just loss and goodness
        index = cf.patience if cf.epochs > cf.patience else 1
        metrics_per_partition[f"val_{cf.loss_str}"].append(metrics[f"val_{cf.loss_str}"][-index])
        metrics_per_partition[f"train_{cf.loss_str}"].append(metrics[f"train_{cf.loss_str}"][-index])
        metrics_per_partition[f"val_{cf.goodness_str}"].append(metrics[f"val_{cf.goodness_str}"][-index])
        metrics_per_partition[f"train_{cf.goodness_str}"].append(metrics[f"train_{cf.goodness_str}"][-index])



##############################################
# CONFIG
##############################################

def load_or_init_config(remake_flags, lambda_plot_config, basepath):
    cf = AttrDict() # Describes all the 'knobs' of an experiment

    if any(remake_flags.values()): # if remaking plots, load config from file
        # Use find_existing_runs to handle variable folder hierarchy depth
        seed_dirs = find_existing_runs(basepath, remake_flags)

        if not seed_dirs:
            raise FileNotFoundError(f"No config.json files found in {basepath}")

        # Use the first seed directory found
        cfg_path = seed_dirs[0] / "config.json"

        cf.update(load_json(cfg_path.parent, cfg_path.name))  # update fields using file

        cf.update(remake_flags)
        cf.logdir = Path(cfg_path).parent  # go one level up to seed folder

        # SR is fully managed by sr_toolbox.py — always off here
        cf.run_sr = False
        cf.fit_sr = False

        if remake_flags['remake_sweep_plot']:
            cf.visualize_sweep = True  # enable sweep visualization
        else:
            if not 'visualize_sweep' in cf.keys():
                cf.visualize_sweep = False

        if "kernels" in cf.keys():
            # Compatibility fix for kernels without stride specified
            for k in range(len(cf.kernels)):
                if len(cf.kernels[k]) < 5: # TODO: this is a quickfix
                        cf.kernels[k].append(1) # add stride if not specified

        print(f"\nLOADING CONFIG FROM {cfg_path}:\n")

    cf.update(remake_flags)
    cf.update(lambda_plot_config)

    assert not (cf.remake_lambda_plot and cf.remake_history_and_pt_plots), "Either remake all plots or lambdaplot."

    return cf

def find_lbc_transition(accuracies, partitions):
    # A valley is lower than its neighbors
    valleys = [i for i in range(1, len(accuracies)-1) 
               if accuracies[i] < accuracies[i-1] and accuracies[i] < accuracies[i+1]]
    
    if len(valleys) < 2:
        return 0, 0 

    # Pick the two deepest valleys (the 'dips' of the W)
    v_sorted = sorted(valleys, key=lambda x: accuracies[x])
    v1, v2 = sorted([v_sorted[0], v_sorted[1]])
    
    # The peak is the highest point between those two deepest valleys
    peak_idx = np.argmax(accuracies[v1:v2]) + v1
    return partitions[peak_idx], peak_idx

def is_notebook():
    """Returns True if running in a Jupyter Notebook/Lab environment."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return None      # Standard Python Interpreter

def pick_directory(start_path=os.getcwd(), title="Select Directory"):
    """
    Unified file picker. 
    - In Jupyter: Returns an ipyfilechooser object (Needs to be displayed).
    - In Standard Python: Opens a popup and returns the file path string.
    """
    if is_notebook() and is_notebook() is not None:
        # --- JUPYTER MODE ---
        from ipyfilechooser import FileChooser
        fc = FileChooser(start_path)
        fc.title = title
        return fc
    else:
        # --- STANDARD PYTHON MODE ---
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw() # Hide the main window
        
        # Make the window jump to the front
        root.attributes('-topmost', True)
        root.focus_force()
        
        file_path = filedialog.askdirectory(
            initialdir=start_path,
            title=title
        )
        if file_path == "":
            print("No directory selected.")
            sys.exit(0)
        
        root.destroy()
        return file_path

    

REMAKE_FLAGS = ("remake_history_and_pt_plots", "remake_lambda_plot", "remake_sweep_plot")


def is_remaking(cf):
    """True if any remake flag is set, i.e. we are re-plotting an existing run."""
    return any(getattr(cf, flag, False) for flag in REMAKE_FLAGS)


def resolve_logdir(cf, param_values=None, seed=None):
    """
    Log directory to read/write for the current (seed, param) point.

    While training this is just build_logdir_path(), which reconstructs the path
    from cf fields. That reconstruction only resolves to the real folder if the
    run still lives where it was originally trained -- and it does not reproduce
    paths at all for runs whose logdir was set explicitly by a script (e.g. the
    scripts/*_survey.py helpers). So when remaking plots we prefer the actual
    on-disk location stashed by tetriscnn.experiments.run_or_remake() at
    basepath-selection time.
    """
    if is_remaking(cf):
        stashed = getattr(cf, "_remake_seed_logdir", None)
        per_seed = getattr(cf, "_remake_seed_logdirs", None)
        if stashed is None and per_seed:
            key = cf.seed if seed is None else seed
            if key in per_seed:
                stashed = per_seed[key]
        if stashed is not None:
            return str(_with_partition_subdir(Path(stashed), cf))
    return build_logdir_path(cf, param_values)


def _with_partition_subdir(seed_dir, cf):
    """
    Descend into the run's partition_{i} subfolder if it exists on disk.

    The stashed remake paths are seed-level folders. For an "lbc" run each
    partition lives in its own partition_{i} subfolder underneath, matching the
    suffix build_logdir_path() adds. A "partition" run written by the survey
    scripts, though, keeps metrics.json in the seed folder itself even though
    cf.partition_index is set -- so test the directory rather than the config.
    """
    partition_index = getattr(cf, "partition_index", None)
    if partition_index is None or seed_dir.name.startswith("partition_"):
        return seed_dir
    candidate = seed_dir / f"partition_{partition_index}"
    return candidate if candidate.is_dir() else seed_dir


# LaTeX rendering of each supported symmetry group. "K4" is an alias for "D2"
# (the Klein four-group / rectangle symmetry group), so both render the same way.
GROUP_LATEX = {
    "C4": r"C_4",
    "D2": r"D_2",
    "K4": r"D_2",
}


def branch_group(cf, branch_index):
    """
    Symmetry group of branch `branch_index`, or None if it is not equivariant.

    Two ways a branch can be equivariant, and they are mutually exclusive by
    construction (set_kernels raises if both are set):
      - cf.branch_groups[k] -- per-branch, used by the mixed "smallkernels_mixed"
        set where only the back half of the bottleneck is symmetrised;
      - cf.equivariant -- uniform, every branch carries cf.equivariant_group.
    """
    branch_groups = getattr(cf, "branch_groups", None)
    if branch_groups is not None:
        if branch_index < len(branch_groups):
            return branch_groups[branch_index]
        return None
    if getattr(cf, "equivariant", False):
        return getattr(cf, "equivariant_group", "C4")
    return None


def branch_label(cf, branch_index, equation_env=True, full_latex=False):
    """
    Plot label for one branch: its kernel pattern, plus a symmetry-group signature
    when the branch is equivariant.

    An equivariant branch is not the same object as a non-equivariant branch of the
    same footprint -- it reads the orbit-averaged correlator rather than that one
    orientation -- so a bare pattern glyph is ambiguous. This matters most for the
    mixed set, where the SAME pattern appears twice in one bottleneck, once each way,
    and the two would otherwise share a legend entry.
    """
    spec = cf.kernels[branch_index]
    label = kernel_to_latex_pattern(spec, equation_env=equation_env, full_latex=full_latex)

    group = branch_group(cf, branch_index)
    if group is None:
        return label

    group_tex = GROUP_LATEX.get(group, str(group))

    # kernel_to_latex_pattern falls back to a plain (non-math) "[WxH], dil=D" string
    # for kernels too large to draw; math-mode markup would render literally there,
    # so tag those in plain text instead.
    if not label.startswith("$") and "\\" not in label:
        # return f"{label} [{group}]"
        return f"{label}"

    # Bracketed suffix rather than a superscript: at legend font size a superscripted
    # group name is too small to read, and these labels sit in paper figures.
    if label.startswith("$") and label.endswith("$"):
        # return f"${label[1:-1]}\\,[{group_tex}]$"
        return f"${label[1:-1]}$"
    # return f"{label}\\,[{group_tex}]"
    return f"{label}"


# Colormaps for branch curves. Non-equivariant branches keep the familiar tab10
# ordering; equivariant ones are drawn from Set2, whose pastel-ish hues are broadly
# orthogonal to tab10's saturated ones, so the two halves of a mixed bottleneck stay
# tellable apart at a glance even where the legend glyphs are near-identical.
BRANCH_CMAP_PLAIN = "tab10"
BRANCH_CMAP_EQUIVARIANT = "Accent" # Was Set2


def branch_colors(cf):
    """
    One color per branch, or None if the default color cycle should be left alone.

    Returns None for an all-plain kernel set, so every existing non-equivariant
    figure keeps exactly the colors it had (the active cycle is seaborn's, not
    tab10, and silently switching it would change published figures).

    When any branch is equivariant, colors are assigned explicitly: plain branches
    walk tab10 in order, equivariant ones walk Set2 in order. Each family is
    indexed within itself, so the k-th plain branch is the k-th tab10 color no
    matter how the two kinds are interleaved.
    """
    import matplotlib.pyplot as plt

    groups = [branch_group(cf, k) for k in range(len(cf.kernels))]
    if not any(g is not None for g in groups):
        return None

    plain_cmap = plt.get_cmap(BRANCH_CMAP_PLAIN).colors
    equi_cmap = plt.get_cmap(BRANCH_CMAP_EQUIVARIANT).colors

    colors = []
    n_plain = n_equi = 0
    for group in groups:
        if group is None:
            colors.append(plain_cmap[n_plain % len(plain_cmap)])
            n_plain += 1
        else:
            colors.append(equi_cmap[n_equi % len(equi_cmap)])
            n_equi += 1
    return colors


def branch_color(cf, branch_index):
    """Color for one branch, or None to fall back to the default cycle."""
    colors = branch_colors(cf)
    return None if colors is None else colors[branch_index]
