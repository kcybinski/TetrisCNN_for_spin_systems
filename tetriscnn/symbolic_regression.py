# Julia must be initialized before torch is imported, so this import comes first.
try:
    from pysr import PySRRegressor, TensorBoardLoggerSpec
except ImportError as exc:  # PySR (like plotly below) is an optional dependency
    raise ImportError(
        "Symbolic regression needs PySR, which is not part of the main install. "
        "Install it with `pip install -r requirements-sr.txt` (or `pip install -e \".[sr]\"`)."
    ) from exc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pickle
import plotly.graph_objects as go
import sympy as sp
import re
from plotly.io import to_html
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import torch
import seaborn as sns

from tetriscnn.utils import DEVICE
# The synthetic OOD probes are plain numpy/torch and carry no PySR dependency, so
# they live in their own module and are re-exported here for existing callers.
from tetriscnn.synthetic_configs import (
    generate_paramagnetic_configs,
    generate_ferromagnetic_configs,
    generate_antiferromagnetic_configs,
)
sns.set_theme()
sns.color_palette("Paired")

def convert_floats_to_ints(latex_eq):
    """
    Convert floats to integers in a LaTeX equation string.
    """
    return re.sub(r"(?<!\d)(\d+)\.\\", r"\1\\", latex_eq)




def round_expr(expr: Any, n_digits: int = 3):
    """Safely round a sympy expression to a few significant digits.

    Falls back to returning the expression unchanged if rounding fails.
    """
    try:
        return sp.N(expr, n_digits)
    except Exception:
        return expr

@dataclass
class SRConfig:
    """Comprehensive symbolic regression configuration.

    Centralizes all SR parameters previously spread across:
    - setup_experiment() in main.py
    - run_sr() in main.py
    - run_sr_raw_mode() in main.py
    - initialize_recommended_sr_cfg() in symbolic_regression.py

    SR here is a secondary, post-hoc interpretability route decoupled from
    training (see run_sr_raw_mode() and sr_toolbox.py's module docstring):
    it fits on the training split only, then reports both in-distribution
    (val) and out-of-distribution (synthetic generalization-config) metrics
    to show how much a fitted equation's accuracy degrades outside the
    training distribution.

    Field groups (see inline ``# === ... ===`` comments below for exact
    boundaries):
    - Mode & feature selection: ``sr_mode``, ``top_k``, ``classification_sr_mode``.
    - Operator sets: ``binary_operators``, ``unary_operators``, ``extra_sympy_mappings``
      (the primitives PySR is allowed to combine into equations).
    - Constraints: ``complexity_of_variables/constants/operators``, ``constraints``,
      ``nested_constraints`` (bounds on equation shape/depth per operator).
    - Optimization parameters: ``maxdepth``, ``maxsize``, ``parsimony``,
      ``adaptive_parsimony_scaling``, ``weight_optimize``, ``elementwise_loss``,
      ``model_selection`` (PySR's evolutionary search hyperparameters).
    - PySR execution: ``niterations``, ``populations``, ``procs``, ``turbo``,
      ``batching``, ``batch_size`` (how the search is run, not what it searches for).
    - Generalization dataset: ``gen_samples``, ``num_flips`` (size/perturbation of
      the synthetic para/ferro/antiferromagnetic OOD probe configs, see
      generate_all_generalization_configs()).
    - TensorBoard & naming: ``use_tensorboard``, ``sr_folder_name_override``,
      ``selected_equation_idx`` (logging and output-path/plot bookkeeping).
    - Optimized picking metrics: ``optimized_picking_metrics``, the Borda-count
      leaderboard used to choose the "optimized_picked" equation (see
      get_optimized_picking_metrics() and save_sr_hof_with_avg()).
    - Technical: ``precision``, ``temp_equation_file``, ``delete_tempfiles``
      (PySR/Julia-side plumbing).

    Use ``save()``/``load()`` to persist/restore a config as JSON alongside SR
    results; ``load()`` tolerates and drops unknown keys (e.g. from fields
    retired in a later version of this class) so old ``sr_config.json`` files
    keep loading.
    """

    # === Mode & Feature Selection ===
    sr_mode: str = "raw"  # "raw" (default) or "averaged" (obsolete)
    top_k: int = 3  # Number of top-importance features to select
    classification_sr_mode: str = 'logit_diff'  # 'logit_diff' (Mode A) or 'single_logit' (Mode B)
    _picked_logit_idx: Optional[int] = None  # set at runtime by Mode B; not serialized

    # === Operator Sets ===
    binary_operators: List[str] = field(default_factory=lambda: ["+", "*"])
    unary_operators: List[str] = field(default_factory=lambda: [
        "neg", "square", "cube", "quart(x) = x^4", 
        # "exp"
    ])
    extra_sympy_mappings: Dict[str, Any] = field(default_factory=lambda: {
        "quart": lambda x: x**4
    })

    # === Constraints ===
    complexity_of_variables: int = 2
    complexity_of_constants: int = 1
    constraints: Dict[str, Any] = field(default_factory=lambda: {
        "*": (-1, 1),  # Right arg must be constant
        "square": 3, "cube": 3, "quart": 3,
    })
    nested_constraints: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        # "exp": {"exp": 0, "square": 0, "cube": 0, "quart": 0},
        "square": {"square": 0, "cube": 0, "quart": 0},
        "cube": {"square": 0, "cube": 0, "quart": 0},
        "quart": {"square": 0, "cube": 0, "quart": 0},
    })
    complexity_of_operators: Dict[str, int] = field(default_factory=lambda: {
        "+": 1, "*": 1, "square": 2, "cube": 2, "quart": 3, 
        # "exp": 5
    })

    # === Optimization Parameters ===
    maxdepth: int = 15
    maxsize: int = 25
    parsimony: float = 0.01
    adaptive_parsimony_scaling: int = 2000
    weight_optimize: float = 0.001
    elementwise_loss: str = "L1DistLoss()"  # L1 loss (MAE) - default for robustness
    model_selection: str = "best"

    # === PySR Execution ===
    niterations: int = 15000 # NOTE: 2000 is usually just fine
    populations: int = 25 # 20-25 is usually fine
    procs: int = 0
    turbo: bool = False
    batching: bool = True
    batch_size: Optional[int] = None  # If None, PySR auto-determines

    # === Generalization Dataset ===
    gen_samples: int = 2500  # Samples per config type (para/ferro/antiferro)
    num_flips: int = 4  # Spin flips for ferro/antiferro perturbations
    # Seed for the OOD probe configurations. These are redrawn on every
    # evaluation, so without a fixed seed the OOD columns of the Hall of Fame
    # (r2_para/ferro/antiferro, and hence the Borda `optimized_picked` row) are
    # not reproducible across runs. Set to None for a fresh draw each time.
    gen_seed: Optional[int] = 2137

    # === TensorBoard & Naming ===
    use_tensorboard: bool = True  # Opt-in logging
    sr_folder_name_override: Optional[str] = None  # Override auto-generated SR folder name
    # Override equation pick for plots (None = use PySR's own pick)
    # int = same override for all outputs; List[Optional[int]] = per-output overrides
    selected_equation_idx: Optional[Union[int, List[Optional[int]]]] = None

    # === Optimized Picking Metrics ===
    # List of (metric_name, higher_is_better) tuples used for Borda count ranking.
    # Equations are ranked on each metric; the equation with lowest total Borda rank wins.
    # NaN values are always ranked last. Tie-break: r2_val (highest wins).
    optimized_picking_metrics: List[Tuple[str, bool]] = None

    # === Technical ===
    precision: int = 64
    temp_equation_file: bool = False
    delete_tempfiles: bool = True

            

    def to_base_params(self) -> Dict[str, Any]:
        """Convert to base_params dict for PySR compatibility.

        Returns dict matching initialize_recommended_sr_cfg() output.
        """
        return {
            "binary_operators": self.binary_operators,
            "unary_operators": self.unary_operators,
            "extra_sympy_mappings": self.extra_sympy_mappings,
            "complexity_of_variables": self.complexity_of_variables,
            "complexity_of_constants": self.complexity_of_constants,
            "constraints": self.constraints,
            "nested_constraints": self.nested_constraints,
            "complexity_of_operators": self.complexity_of_operators,
            "elementwise_loss": self.elementwise_loss,
            "model_selection": self.model_selection,
            "maxdepth": self.maxdepth,
            "maxsize": self.maxsize,
            "weight_optimize": self.weight_optimize,
            "parsimony": self.parsimony,
            "adaptive_parsimony_scaling": self.adaptive_parsimony_scaling,
            "temp_equation_file": self.temp_equation_file,
            "delete_tempfiles": self.delete_tempfiles,
            "precision": self.precision,
            "turbo": self.turbo,
        }

    def get_significant_params(self) -> Dict[str, Any]:
        """Extract params that differ from defaults for folder naming.

        Tracks significant parameter deviations from defaults including:
        - Mode and feature selection (sr_mode, top_k)
        - Operator sets (binary, unary)
        - Complexity constraints (maxdepth, maxsize)
        - Loss function changes
        - Generalization dataset size
        """
        defaults = SRConfig()
        significant = {}

        # Mode and feature selection
        if self.sr_mode != defaults.sr_mode:
            significant['mode'] = self.sr_mode
        if self.top_k != defaults.top_k:
            significant['topk'] = self.top_k
        if self.classification_sr_mode != defaults.classification_sr_mode:
            significant['clf_sr'] = self.classification_sr_mode

        # Operator changes
        if set(self.binary_operators) != set(defaults.binary_operators):
            significant['binops'] = "_".join(sorted(self.binary_operators))
        if set(self.unary_operators) != set(defaults.unary_operators):
            # Shorten operator names for folder naming
            unary_names = []
            for op in self.unary_operators:
                # Extract just the operator name (e.g., "quart(x) = x^4" → "quart")
                op_name = op.split("(")[0]
                unary_names.append(op_name)
            significant['unops'] = "_".join(sorted(unary_names))

        # Complexity and structure
        if self.maxdepth != defaults.maxdepth:
            significant['maxd'] = self.maxdepth
        if self.maxsize != defaults.maxsize:
            significant['maxs'] = self.maxsize

        # Loss function
        if self.elementwise_loss != defaults.elementwise_loss:
            # Shorten common loss names
            loss_abbrev = {
                "L1DistLoss()": "L1",
                "L2DistLoss()": "L2",
            }.get(self.elementwise_loss, "custom")
            significant['loss'] = loss_abbrev

        # Complexity of constants and variables
        if self.complexity_of_constants != defaults.complexity_of_constants:
            significant['coc'] = self.complexity_of_constants
        if self.complexity_of_variables != defaults.complexity_of_variables:
            significant['cov'] = self.complexity_of_variables

        # Generalization dataset
        if self.gen_samples != defaults.gen_samples:
            significant['gen'] = self.gen_samples

        return significant

    def set_complexity_of_constants(self, value: int) -> None:
        """Set the complexity penalty for numeric constants.

        Higher values discourage free constants in equations, promoting
        more parsimonious closed-form expressions. Default is 1.

        Reflected in folder naming as 'coc={value}'.

        Example:
            sr_config.set_complexity_of_constants(3)  # discourage free constants
        """
        self.complexity_of_constants = value

    def set_complexity_of_variables(self, value: int) -> None:
        """Set the complexity penalty for input variables.

        Higher values discourage using many distinct variables, promoting
        simpler equations with fewer features. Default is 2.

        Reflected in folder naming as 'cov={value}'.

        Example:
            sr_config.set_complexity_of_variables(1)  # treat variables as cheap as constants
        """
        self.complexity_of_variables = value

    def get_hof_columns(self, is_classification: bool, has_val: bool, has_gen: bool) -> List[str]:
        """Return the ordered HOF CSV column list for this task type."""
        base = ['rank', 'output_label', 'complexity', 'equation_latex', 'SR_picked', 'optimized_picked']
        if is_classification:
            # r2/mse measure logit_diff fit quality; accuracy/bce measure decision-boundary recovery
            train_cols = ['r2_train', 'mse_train', 'r2_true', 'mse_true',
                          'accuracy_train', 'accuracy_agg_train', 'r2_agg_train', 'bce_train']
            val_cols   = (['r2_val', 'mse_val', 'nrmse_val', 'rho_val',
                           'accuracy_val', 'accuracy_agg_val', 'r2_agg_val', 'bce_val']
                          if has_val else [])
            gen_cols   = (['r2_para', 'mse_para', 'r2_ferro', 'mse_ferro',
                           'r2_antiferro', 'mse_antiferro', 'r2_gen_combined', 'mse_gen_combined',
                           'nrmse_ferro', 'rho_ferro', 'accuracy_ferro',
                           'nrmse_antiferro', 'rho_antiferro', 'accuracy_antiferro',
                           'nrmse_para', 'rho_para', 'accuracy_para',
                           'delta_nrmse_mean', 'delta_rho_mean']
                          if has_gen else [])
        else:
            train_cols = ['r2_train', 'mse_train', 'r2_agg_train', 'mse_agg_train',
                          'r2_true', 'mse_true', 'r2_agg_true', 'mse_agg_true']
            val_cols   = (['r2_val', 'mse_val', 'r2_agg_val', 'mse_agg_val',
                           'r2_val_true', 'mse_val_true', 'r2_agg_val_true', 'mse_agg_val_true']
                          if has_val else [])
            gen_cols   = (['r2_para', 'mse_para', 'r2_ferro', 'mse_ferro',
                           'r2_antiferro', 'mse_antiferro', 'r2_gen_combined', 'mse_gen_combined',
                           'nrmse_val', 'nrmse_ferro', 'rho_ferro', 'nrmse_antiferro', 'rho_antiferro',
                           'nrmse_para', 'rho_para', 'delta_nrmse_mean', 'delta_rho_mean']
                          if has_gen else [])
        return base + train_cols + val_cols + gen_cols

    def get_optimized_picking_metrics(self, is_classification: bool) -> List[Tuple[str, bool]]:
        """Return Borda-count picking metrics for equation selection."""
        if self.optimized_picking_metrics is not None:
            return self.optimized_picking_metrics
        if is_classification:
            return [
                ("accuracy_val",       True),
                ("accuracy_agg_val",   True),
                ("bce_val",            False),  # lower is better
                ("rho_ferro",          True),
                ("rho_antiferro",      True),
                ("rho_para",           True),
                ("nrmse_ferro",        False),
                ("nrmse_antiferro",    False),
                ("nrmse_para",         False),
            ]
        return [
            ("r2_val",          True),
            ("nrmse_val",       False),
            ("rho_val",         True),
            ("rho_ferro",       True),
            ("rho_antiferro",   True),
            ("rho_para",        True),
            ("nrmse_ferro",     False),
            ("nrmse_antiferro", False),
            ("nrmse_para",      False),
        ]

    def add_unary_operator(
        self,
        operator: str,
        complexity: int = 2,
        sympy_mapping: Optional[Any] = None,
        constraint: Optional[int] = None,
        nested_constraints: Optional[Dict[str, int]] = None
    ) -> None:
        """Add a unary operator to the existing set.

        Example:
            sr_config.add_unary_operator("sin", complexity=3)
            sr_config.add_unary_operator(
                "inv(x) = 1/x",
                complexity=3,
                sympy_mapping=lambda x: 1/x,
                constraint=3,
                nested_constraints={"inv": 0, "exp": 0}
            )
        """
        if operator not in self.unary_operators:
            self.unary_operators.append(operator)

        # Extract operator name for complexity/constraints
        op_name = operator.split("(")[0]
        self.complexity_of_operators[op_name] = complexity

        if sympy_mapping is not None:
            self.extra_sympy_mappings[op_name] = sympy_mapping

        if constraint is not None:
            self.constraints[op_name] = constraint

        if nested_constraints is not None:
            self.nested_constraints[op_name] = nested_constraints

    def add_binary_operator(
        self,
        operator: str,
        complexity: int = 1,
        constraint: Optional[Union[int, Tuple[int, int]]] = None,
        nested_constraints: Optional[Dict[str, int]] = None
    ) -> None:
        """Add a binary operator to the existing set.

        Example:
            sr_config.add_binary_operator("-", complexity=1)
            sr_config.add_binary_operator(
                "/",
                complexity=2,
                constraint=(3, 3),
                nested_constraints={"/": 0, "exp": 0}
            )
        """
        if operator not in self.binary_operators:
            self.binary_operators.append(operator)

        self.complexity_of_operators[operator] = complexity

        if constraint is not None:
            self.constraints[operator] = constraint

        if nested_constraints is not None:
            self.nested_constraints[operator] = nested_constraints

    def save(self, filepath: Union[str, Path]) -> None:
        """Save config to JSON file.

        Note: Lambda functions in extra_sympy_mappings cannot be serialized.
        They will be saved as strings indicating the operator name.
        """
        from dataclasses import asdict
        import json

        config_dict = {k: v for k, v in asdict(self).items() if not k.startswith('_')}

        # Handle non-serializable lambda functions
        if 'extra_sympy_mappings' in config_dict:
            config_dict['extra_sympy_mappings'] = {
                k: f"<lambda for {k}>"
                for k in config_dict['extra_sympy_mappings'].keys()
            }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SRConfig':
        """Load config from JSON file.

        Note: Lambda functions in extra_sympy_mappings need to be reconstructed
        based on operator names. Standard operators (quart, inv) are automatically
        restored.

        Forward-compatible with on-disk configs from older code: any key that no
        longer matches an SRConfig field (e.g. the removed ``use_separate_splits``)
        is dropped with a one-line warning instead of raising a TypeError, so
        existing ``sr_config.json`` files keep loading after fields are retired.
        """
        import json
        from dataclasses import fields

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        # Drop unknown/removed keys (e.g. legacy 'use_separate_splits') for
        # backward compatibility with sr_config.json files written by older code.
        valid_field_names = {f.name for f in fields(cls)}
        unknown_keys = [k for k in config_dict if k not in valid_field_names]
        if unknown_keys:
            print(f"[SRConfig.load] Ignoring unknown/removed config keys from {filepath}: {unknown_keys}")
            for k in unknown_keys:
                del config_dict[k]

        # Reconstruct standard lambda mappings
        if 'extra_sympy_mappings' in config_dict:
            mappings = {}
            for op_name, _ in config_dict['extra_sympy_mappings'].items():
                if op_name == 'quart':
                    mappings['quart'] = lambda x: x**4
                elif op_name == 'inv':
                    mappings['inv'] = lambda x: 1/x
                # Add more standard operators as needed
            config_dict['extra_sympy_mappings'] = mappings

        return cls(**config_dict)


def generate_all_generalization_configs(
    grid_shape: Tuple[int, int],
    sr_config: SRConfig,
    device: torch.device,
    n_channels: int = 1,
) -> Dict[str, torch.Tensor]:
    """Generate all three types of generalization (out-of-distribution) configurations.

    These synthetic configs are the OOD probe used throughout this module (see
    run_sr_raw_mode(), save_sr_hof_with_avg()) to test whether a fitted SR
    equation generalizes beyond the training distribution of experimental
    snapshots, or merely interpolates within it. They are idealized
    zero-temperature-like analogs of the three phases (paramagnetic,
    ferromagnetic, antiferromagnetic), not draws from the true experimental
    distribution: real Rydberg snapshots carry projective-measurement noise
    and finite-time dynamics that these synthetic configs do not reproduce.
    Treat OOD metrics computed on them as a lower bound on how far a fitted
    equation's applicability extends, not as a faithful resample of the data.

    For single-channel models (n_channels=1), each tensor has shape
    (n_samples, 1, H, W).

    For two-channel models (n_channels=2), channel ordering follows the
    XY_XZ dataset convention (channel 0 = X-basis, channel 1 = Z-basis).
    The Z-channel carries the primary magnetic order; the X-channel carries
    the complementary (transverse) signal:
        - ferromagnetic   Z → paramagnetic   X  (ordered  + disordered)
        - antiferromagnetic Z → paramagnetic X  (ordered  + disordered)
        - paramagnetic    Z → ferromagnetic  X  (disordered + ordered)

    For n_channels > 2, extra channels are filled with paramagnetic noise.

    Args:
        grid_shape: (height, width) of spin lattice
        sr_config: SRConfig instance with gen_samples, num_flips, gen_seed
        device: PyTorch device
        n_channels: Number of input channels (default 1, use 2 for XZ datasets)

    Returns:
        Dict with keys 'paramagnetic', 'ferromagnetic', 'antiferromagnetic',
        each containing a tensor of spin configurations with shape
        (n_samples, n_channels, H, W)
    """
    n = sr_config.gen_samples
    nf = sr_config.num_flips
    seed = getattr(sr_config, "gen_seed", None)
    # One generator threaded through every draw below, so the whole probe set is
    # fixed by sr_config.gen_seed and the OOD metrics are reproducible.
    rng = np.random.default_rng(seed)
    print(f"[SR Gen] Generating {n} samples per config type "
          f"(n_channels={n_channels}, gen_seed={seed})")

    z_para  = generate_paramagnetic_configs(grid_shape, n, device, rng)
    z_ferro = generate_ferromagnetic_configs(grid_shape, n, nf, device, rng)
    z_anti  = generate_antiferromagnetic_configs(grid_shape, n, nf, device, rng)

    if n_channels == 1:
        return {
            'paramagnetic':     z_para,
            'ferromagnetic':    z_ferro,
            'antiferromagnetic': z_anti,
        }

    # Two-channel: prepend the cross-type X-channel
    x_for_ferro = generate_paramagnetic_configs(grid_shape, n, device, rng)   # X ~ para
    x_for_anti  = generate_paramagnetic_configs(grid_shape, n, device, rng)   # X ~ para
    x_for_para  = generate_ferromagnetic_configs(grid_shape, n, nf, device, rng)  # X ~ ferro

    configs_2ch = {
        'paramagnetic':      torch.cat([x_for_para,  z_para],  dim=1),
        'ferromagnetic':     torch.cat([x_for_ferro, z_ferro], dim=1),
        'antiferromagnetic': torch.cat([x_for_anti,  z_anti],  dim=1),
    }

    if n_channels == 2:
        return configs_2ch

    # Fallback for n_channels > 2: fill extra channels with paramagnetic noise
    result = {}
    for key, tensor_2ch in configs_2ch.items():
        extra = [
            generate_paramagnetic_configs(grid_shape, n, device, rng)
            for _ in range(n_channels - 2)
        ]
        result[key] = torch.cat([tensor_2ch] + extra, dim=1)
    return result


def run_sr_raw_mode(
    cf,
    net1,
    net2,
    train_dataset,
    val_dataset,
    sr_config: SRConfig,
    n_outputs: int,
) -> None:
    """Run symbolic regression on raw per-sample activations.

    This is the top-level entry point for the "raw" SR mode (the only supported
    ``sr_config.sr_mode``; "averaged" is obsolete). It extracts bottleneck
    activations ``z`` and task-head predictions from ``net1``/``net2`` on three
    kinds of data, fits (or loads) a PySR model mapping the top-``k`` activations
    to the task-head output, and writes Hall-of-Fame equations, metrics, and
    plots to the SR folder resolved by :func:`tetriscnn.utils.build_sr_folder_path`.

    Data sources:
    - Training split (``train_dataset``): SR equations are fit here, and only here.
    - Validation split (``val_dataset``): in-distribution generalization check
      (``r2_val``, ``nrmse_val``, ``rho_val``, ...).
    - Synthetic generalization configs (paramagnetic/ferromagnetic/antiferromagnetic,
      see :func:`generate_all_generalization_configs`): out-of-distribution check
      (``r2_para``/``r2_ferro``/``r2_antiferro``, ``rho_*``, ...). This OOD signal is
      the primary reason SR exists in this codebase: it is used to show that a
      closed-form equation extracted from in-distribution data does not reliably
      generalize, not to claim SR as a robust discovery method (see
      docs/SYMBOLIC_REGRESSION.md).

    Args:
        cf: Experiment configuration object (must have ``.logdir``, ``.kernels``,
            and ``.fit_sr``/``.remake_sr_plot`` flags set by the caller).
        net1: Trained feature extractor network (produces bottleneck activations z).
        net2: Trained readout network (produces task-head predictions from z).
        train_dataset: Training dataset; SR equations are fit exclusively on this split.
        val_dataset: Validation dataset; used only for evaluation metrics, never for fitting.
        sr_config: SR configuration (SRConfig dataclass).
        n_outputs: Number of output dimensions.
    """
    from pathlib import Path
    from torch.utils.data import DataLoader
    import pickle

    # Determine SR folder path with parameter tracking
    from tetriscnn.utils import build_sr_folder_path
    sr_folder = build_sr_folder_path(Path(cf.logdir), sr_config)
    sr_folder.mkdir(exist_ok=True, parents=True)

    # Save SR config (using SRConfig's custom save method to handle lambda functions)
    sr_config.save(sr_folder / "sr_config.json")

    # 1. Extract training data
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset),
        shuffle=False, num_workers=0, pin_memory=False
    )

    with torch.no_grad():
        for x_train, y_train in train_loader:
            x_train = x_train.to(DEVICE)
            z_train = net1(x_train)
            y_pred_train = net2(z_train)

            x_train_np = x_train.cpu().numpy()
            z_train_np = z_train.cpu().numpy()
            y_true_train_np = y_train.cpu().numpy()
            y_pred_train_np = y_pred_train.cpu().numpy()

    # 2. Extract validation data
    val_loader = DataLoader(
        val_dataset, batch_size=len(val_dataset),
        shuffle=False, num_workers=0, pin_memory=False
    )

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(DEVICE)
            z_val = net1(x_val)
            y_pred_val = net2(z_val)

            z_val_np = z_val.cpu().numpy()
            y_true_val_np = y_val.cpu().numpy()
            y_pred_val_np = y_pred_val.cpu().numpy()

    # 3. Generate and process generalization configs
    grid_shape = (train_dataset.data.shape[2], train_dataset.data.shape[3])
    n_channels = train_dataset.data.shape[1]
    gen_configs = generate_all_generalization_configs(grid_shape, sr_config, DEVICE, n_channels=n_channels)

    gen_data = {}
    for config_type, x_gen in gen_configs.items():
        with torch.no_grad():
            z_gen = net1(x_gen)
            y_pred_gen = net2(z_gen)

            gen_data[config_type] = {
                'x': x_gen.cpu().numpy(),
                'z': z_gen.cpu().numpy(),
                'y_pred': y_pred_gen.cpu().numpy(),
            }

    print(f"[SR Split] Train: {len(train_dataset)} samples (fitting)")
    print(f"[SR Split] Val: {len(val_dataset)} samples (metrics)")
    print(f"[SR Split] Gen: {sr_config.gen_samples} × 3 types (metrics)")

    # Feature selection on training data (computed before classification transform so Mode B can use it)
    branch_activations = np.array([
        np.abs(z_train_np[:, k].mean()) for k in range(len(cf.kernels))
    ])
    branch_importance_order = np.argsort(-branch_activations)
    sr_z_mask = branch_importance_order[:sr_config.top_k]

    # Classification: collapse (N,2) logits → (N,1) scalar SR target (Mode A or Mode B)
    is_classification = getattr(train_dataset, 'discrete_labels', False)
    if is_classification:
        y_pred_train_np, y_pred_val_np, gen_data = _prepare_classification_sr_target(
            sr_config, branch_importance_order,
            z_train_np, y_pred_train_np, y_pred_val_np, gen_data,
        )

    z_train_selected = z_train_np[:, sr_z_mask]
    z_val_selected = z_val_np[:, sr_z_mask]

    print(f"[SR RAW] Selected top-{sr_config.top_k} features: {sr_z_mask}")

    # Check if model exists
    model_sr_path = sr_folder / "model_sr_raw.pkl"

    if model_sr_path.exists():
        if not cf.get("remake_sr_plot", False):
            cf.fit_sr = False
            print(f"[SR RAW] Loading existing model from {model_sr_path}")
    else:
        if not cf.get("remake_sr_plot", False):
            print(f"[ERROR] SR model not found at {model_sr_path}")
            return
        cf.fit_sr = True

    # Variable names encode actual branch indices (a5, a2, a7 → renders as a_{5}, a_{2}, a_{7} in LaTeX)
    var_names = [f"a{idx}" for idx in sr_z_mask]

    # Fit or load SR model
    if cf.fit_sr:
        print(f"[SR RAW] Fitting SR model on {z_train_selected.shape[0]} training samples...")

        model_sr = fit_symbolic_regression(
            z_train_selected, y_pred_train_np,
            sr_config=sr_config,
            fit=True,
            variable_names=var_names,
            exp_config=cf,
        )

        # Save fitted model
        try:
            with open(model_sr_path, "wb") as f:
                pickle.dump(model_sr, f)
            print(f"[SR RAW] Saved model to {model_sr_path}")
        except Exception as e:
            print(f"[WARN] Could not save SR model: {e}")
    else:
        # Load pre-fitted model
        try:
            with open(model_sr_path, "rb") as f:
                model_sr = pickle.load(f)
            print(f"[SR RAW] Loaded model from {model_sr_path}")
        except Exception as e:
            print(f"[ERROR] Could not load SR model: {e}")
            return
        # Patch variable names in case model was saved with old naming (z1, z2, ...)
        _old_names = list(model_sr.feature_names_in_) if hasattr(model_sr, "feature_names_in_") else []
        if _old_names != var_names:
            if hasattr(model_sr, "feature_names_in_"):
                model_sr.feature_names_in_ = np.array(var_names)
            # Rewrite sympy symbols in cached expressions
            _sym_map = {sp.Symbol(old): sp.Symbol(new) for old, new in zip(_old_names, var_names)}
            if _sym_map:
                eq_dfs = model_sr.equations_ if isinstance(model_sr.equations_, list) else [model_sr.equations_]
                for eq_df in eq_dfs:
                    if "sympy_format" in eq_df.columns:
                        eq_df["sympy_format"] = eq_df["sympy_format"].apply(
                            lambda expr: expr.subs(_sym_map) if hasattr(expr, "subs") else expr
                        )

    # Output labels
    out_labels = []
    if hasattr(train_dataset, 'label_param'):
        if train_dataset.label_param == "deltaomega":
            out_labels = ["delta", "omega"]
        else:
            out_labels = [train_dataset.label_param]
    else:
        out_labels = [f"output_{i}" for i in range(n_outputs)]

    # Classification: set output label based on mode
    if is_classification:
        if sr_config.classification_sr_mode == 'single_logit':
            _idx = getattr(sr_config, '_picked_logit_idx', 0)
            out_labels = [f"logit_{_idx}"]
        else:
            out_labels = ["logit_diff"]

    # Save metrics and plots
    # TODO (Phase 2.5): Replace save_sr_hof_with_avg with save_sr_hof_with_multi_split_metrics
    # For now, use validation data for backward compatibility
    try:
        # Prepare data for HOF saving with new signature
        # Select features according to feature_ranking (top_k)
        z_train_selected = z_train_np[:, sr_z_mask]
        z_val_selected = z_val_np[:, sr_z_mask]

        # Create gen_data_selected with only selected features
        gen_data_selected = {}
        for gen_type, gen_dict in gen_data.items():
            gen_data_selected[gen_type] = {
                'z': gen_dict['z'][:, sr_z_mask],
                'y_pred': gen_dict['y_pred'],
            }

        optimized_eq_idx_per_output, optimized_gen_accuracy_per_output = save_sr_hof_with_avg(
            model_sr=model_sr,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            X_train=z_train_selected,
            y_train_nn_pred=y_pred_train_np,
            y_train_true=y_true_train_np,
            X_val=z_val_selected,
            y_val_nn_pred=y_pred_val_np,
            y_val_true=y_true_val_np,
            gen_data=gen_data_selected,
            sr_folder=sr_folder,
            output_labels=out_labels,
            sr_config=sr_config,
            kernels=cf.kernels,
            sr_z_mask=sr_z_mask,
        )
        print(f"[SR RAW] Saved Hall of Fame to {sr_folder}")
    except Exception as e:
        print(f"[WARN] Could not save SR Hall of Fame: {e}")
        import traceback
        traceback.print_exc()
        optimized_eq_idx_per_output = {}
        optimized_gen_accuracy_per_output = {}

    # Generate 3D augmentation plots for each generalization type
    # Create three separate plots for para/ferro/antiferro
    for gen_type, gen_data_dict in gen_data.items():
        z_gen_selected = gen_data_dict['z'][:, sr_z_mask]
        y_gen_pred = gen_data_dict['y_pred']

        try:
            # Look up OOD accuracy for this gen_type from the optimized equation
            _gen_abbrev = {'paramagnetic': 'para', 'ferromagnetic': 'ferro', 'antiferromagnetic': 'antiferro'}.get(gen_type, gen_type)
            _ood_accuracy_for_plot = {
                out_idx: _acc_dict.get(_gen_abbrev, float("nan"))
                for out_idx, _acc_dict in optimized_gen_accuracy_per_output.items()
            }

            plot_3d_augmentation(
                X_real=z_val_selected,
                X_synthetic=z_gen_selected,
                y_real=y_pred_val_np,
                y_synthetic=y_gen_pred,
                selected_idx=sr_z_mask,
                model_sr_folder=sr_folder,
                fname_prefix=f"SR_raw_{gen_type}",
                random_snapshots=None,  # No longer using random snapshots
                output_labels=out_labels,
                selected_equation_idx=sr_config.selected_equation_idx,
                optimized_equation_idx=optimized_eq_idx_per_output,
                ood_accuracy=_ood_accuracy_for_plot,
                kernels=cf.kernels,
            )
            print(f"[SR RAW] Saved {gen_type} 3D plot to {sr_folder}")
        except Exception as e:
            print(f"[WARN] Could not generate {gen_type} visualization: {e}")
            import traceback
            traceback.print_exc()


def fit_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    sr_config: 'SRConfig',
    fit: bool = True,
    variable_names: Optional[List[str]] = None,
    exp_config: Optional[Any] = None
) -> 'PySRRegressor':
    """Instantiate and optionally fit a PySRRegressor.

    Thin wrapper around PySRRegressor construction: builds its kwargs from
    sr_config (operators, constraints, search hyperparameters; see
    SRConfig.to_base_params()) and, if fit=True, runs PySR's evolutionary
    equation search on (X, y). Optional TensorBoard logging is controlled by
    sr_config.use_tensorboard (requires exp_config to derive a log directory).

    This function fits directly on whatever (X, y) it is given; it performs
    no train/val splitting itself, no feature selection, and no equation
    ranking or generalization evaluation, those responsibilities belong to
    the caller (run_sr_raw_mode() selects features and the split; the
    resulting hall of fame is later scored by save_sr_hof_with_avg(), whose
    "optimized_picked" column is the recommended equation, not necessarily
    the one PySR itself ranks highest).

    Limitations: the equation search is stochastic (repeated fits on the same
    data can return different equations) and, depending on
    sr_config.niterations/maxsize, can take anywhere from minutes to hours.
    With multi-output y, PySR fits one equation set per output column
    (model.equations_ becomes a list); with single-output y it is a single
    DataFrame.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector or matrix (n_samples,) or (n_samples, n_outputs)
        sr_config: SRConfig with all optimization and operator parameters
        fit: Whether to fit the model (True) or just instantiate (False)
        variable_names: Optional list of variable names for equations
        exp_config: Optional experiment config for logging directory

    Returns:
        Fitted or unfitted PySRRegressor instance
    """
    kwargs = sr_config.to_base_params()
    kwargs.update({
        "niterations": int(sr_config.niterations),
        "populations": int(sr_config.populations),
        "procs": int(sr_config.procs),
        "turbo": bool(sr_config.turbo),
    })
    if sr_config.procs > 0:
        kwargs["parallelism"] = "multiprocessing"
    elif sr_config.turbo:
        kwargs["parallelism"] = "multithreading"
    else:
        kwargs["parallelism"] = "serial"
    if sr_config.batching:
        kwargs["batching"] = True
        if sr_config.batch_size is not None:
            kwargs["batch_size"] = int(sr_config.batch_size)
        else:
            kwargs["batch_size"] = min(512, max(32, X.shape[0] // 4)) 

    if sr_config.use_tensorboard and exp_config is not None:
        from tetriscnn.utils import get_sr_folder_name
        sr_folder_name = get_sr_folder_name(sr_config)
        logdir_path = Path(exp_config.logdir)
        # Use the seed-level folder path relative to the logs/ root so TensorBoard
        # shows the full experiment hierarchy (e.g. lambdamax_4/spc=100/seed_42)
        try:
            logs_root = logdir_path
            while logs_root.name != "logs" and logs_root.parent != logs_root:
                logs_root = logs_root.parent
            experiment_name = str(logdir_path.relative_to(logs_root))
        except ValueError:
            experiment_name = logdir_path.name
        log_dir = f"PySR_logs/{experiment_name}/{sr_folder_name}"
        kwargs["logger_spec"] = TensorBoardLoggerSpec(log_dir=log_dir, log_interval=10)
        print(f"[SR TensorBoard] Logging to {log_dir}")

    model = PySRRegressor(**kwargs)
    if fit:
        if variable_names is not None:
            model.fit(X, y, variable_names=variable_names)
        else:
            model.fit(X, y)
    return model




def _prepare_classification_sr_target(
    sr_config: 'SRConfig',
    branch_importance_order: np.ndarray,
    z_train_np: np.ndarray,
    y_pred_train_np: np.ndarray,
    y_pred_val_np: np.ndarray,
    gen_data: dict,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Collapse (N,2) classification logits to a scalar SR target.

    PySR fits scalar-valued equations, so binary-classification logits (shape
    (N,2)) must be reduced to a single column before fitting. Two modes,
    selected via sr_config.classification_sr_mode:

    - Mode A ('logit_diff', default): target = logit_0 - logit_1. Always
      applicable to any binary classification head.
    - Mode B ('single_logit'): picks whichever logit column is more positively
      correlated (on the training split) with the mean top-k bottleneck
      activation, and fits SR to that raw logit alone. Only meaningful for
      binary classification tasks; the pick is a training-data correlation
      heuristic and can be unstable if both logits correlate similarly with
      the activations (corr0 ≈ corr1) or on small training splits.

    Side effects: sets sr_config._picked_logit_idx (Mode B only; not
    serialized by SRConfig.save()) and mutates gen_data's 'y_pred' entries
    in place.

    Args:
        sr_config: SRConfig; classification_sr_mode selects the mode, top_k
            bounds how many activations enter the Mode B correlation average.
        branch_importance_order: Kernel indices sorted by descending mean
            |activation| on the training split (see run_sr_raw_mode()).
        z_train_np: Training bottleneck activations, shape (N_train, n_kernels).
        y_pred_train_np: Training task-head logits, shape (N_train, 2).
        y_pred_val_np: Validation task-head logits, shape (N_val, 2).
        gen_data: Dict of {config_type: {'z': ..., 'y_pred': (N,2) logits}}
            for the synthetic generalization configs (see
            generate_all_generalization_configs()).

    Returns:
        (y_pred_train_np, y_pred_val_np, gen_data) with y_pred arrays reshaped
        to (N,1) instead of (N,2); gen_data is the same dict, mutated in place.
    """
    if sr_config.classification_sr_mode == 'single_logit':
        z_mean_top_k = z_train_np[:, branch_importance_order[:sr_config.top_k]].mean(axis=1)
        corr0 = float(np.corrcoef(z_mean_top_k, y_pred_train_np[:, 0])[0, 1])
        corr1 = float(np.corrcoef(z_mean_top_k, y_pred_train_np[:, 1])[0, 1])
        logit_idx = 0 if corr0 >= corr1 else 1
        sr_config._picked_logit_idx = logit_idx
        print(f"[SR Mode B] Selected logit_{logit_idx} (corr0={corr0:.3f}, corr1={corr1:.3f})")
        y_pred_train_np = y_pred_train_np[:, logit_idx].reshape(-1, 1)
        y_pred_val_np   = y_pred_val_np[:, logit_idx].reshape(-1, 1)
        for _gtype in gen_data:
            gen_data[_gtype]['y_pred'] = gen_data[_gtype]['y_pred'][:, logit_idx].reshape(-1, 1)
    else:
        # Mode A (default): logit_diff = logit_0 - logit_1
        y_pred_train_np = (y_pred_train_np[:, 0] - y_pred_train_np[:, 1]).reshape(-1, 1)
        y_pred_val_np   = (y_pred_val_np[:, 0]   - y_pred_val_np[:, 1]).reshape(-1, 1)
        for _gtype in gen_data:
            _gd = gen_data[_gtype]['y_pred']
            gen_data[_gtype]['y_pred'] = (_gd[:, 0] - _gd[:, 1]).reshape(-1, 1)
        print("[SR] Classification Mode A: using logit_diff = logit_0 - logit_1 as SR target")
    return y_pred_train_np, y_pred_val_np, gen_data


def _predict_single_equation(
    model_sr,
    X: np.ndarray,
    eq_idx: int,
    output_idx: Optional[int] = None,
):
    """Predict with a specific PySR equation using lambda_format (simple and robust)."""
    eq_set = model_sr.equations_[output_idx] if isinstance(model_sr.equations_, list) else model_sr.equations_
    func = eq_set.lambda_format[eq_idx]
    pred = np.asarray(func(X))
    if pred.ndim == 0:
        pred = np.full(shape=(X.shape[0],), fill_value=float(pred))
    return pred.reshape(-1)


def save_sr_hof_with_avg(
    model_sr,
    train_dataset,
    val_dataset,
    X_train: np.ndarray,
    y_train_nn_pred: np.ndarray,
    y_train_true: np.ndarray,
    X_val: Union[np.ndarray, None] = None,
    y_val_nn_pred: Union[np.ndarray, None] = None,
    y_val_true: Union[np.ndarray, None] = None,
    gen_data: Union[dict, None] = None,
    sr_folder: Path = None,
    output_labels: List[str] = None,
    top_k: int = 5,
    sr_config: Optional['SRConfig'] = None,
    kernels: Optional[list] = None,
    sr_z_mask: Optional[np.ndarray] = None,
):
    """Save Hall-of-Fame equations with comprehensive train/val/generalization metrics.

    This function evaluates SR equations on multiple datasets:
    - Training data (real): r2_train, mse_train, r2_agg_train, mse_agg_train
    - Validation data (real): r2_val, mse_val, r2_agg_val, mse_agg_val
    - Ground truth labels (real): r2_true, mse_true, r2_agg_true, mse_agg_true
    - Paramagnetic generalization (synthetic): r2_para, mse_para
    - Ferromagnetic generalization (synthetic): r2_ferro, mse_ferro
    - Antiferromagnetic generalization (synthetic): r2_antiferro, mse_antiferro
    - Combined generalization (synthetic): r2_gen_combined, mse_gen_combined

    NOTE: _agg metrics (using snapshot_average) are ONLY computed for real data
    (train/val/true) which has temporal structure. Synthetic generalization data
    has no temporal structure, so NO _agg metrics.

    Equation selection — two columns are written per equation, and they can pick
    different equations:
    - ``SR_picked``: PySR's own choice (the equation with the highest internal
      ``score`` in ``model_sr.equations_``), a training-time complexity/loss
      trade-off that knows nothing about validation or OOD behavior.
    - ``optimized_picked``: this function's own choice, and the one this codebase
      recommends using. It is chosen by Borda count over ``sr_config``'s
      ``optimized_picking_metrics`` (default from
      ``SRConfig.get_optimized_picking_metrics()``): for each metric in that list,
      every equation is ranked (1 = best on that metric, NaN always ranked last),
      and an equation's total score is the sum of its per-metric ranks; the
      equation with the lowest total score wins, ties broken by highest ``r2_val``.
      The default metric list deliberately mixes in-distribution validation
      metrics (``r2_val``, ``nrmse_val``, ``rho_val``) with the OOD generalization
      metrics (``rho_ferro``/``rho_antiferro``/``rho_para``,
      ``nrmse_ferro``/``nrmse_antiferro``/``nrmse_para``), so an equation that
      fits validation data well but collapses out-of-distribution is penalized
      relative to one that holds up on both.
    - Reading the result: **in-distribution ``r2_val`` alone is misleading.** A
      high ``r2_val`` only shows the equation matches the network on the training
      distribution; it says nothing about whether the equation is the "true"
      closed form the network implements. Always read ``r2_val``/``nrmse_val``/
      ``rho_val`` alongside the OOD columns (``rho_ferro``/``rho_antiferro``/
      ``rho_para``, ``r2_para``/``r2_ferro``/``r2_antiferro``,
      ``delta_nrmse_mean``/``delta_rho_mean``); a large gap between in- and
      out-of-distribution performance is itself the finding (see
      docs/SYMBOLIC_REGRESSION.md).

    Parameters:
    -----------
    model_sr : PySRRegressor
        Trained symbolic regression model
    train_dataset : PhaseDataset
        Training dataset object with snapshot_average() method
    val_dataset : PhaseDataset
        Validation dataset object with snapshot_average() method
    X_train : np.ndarray
        Training bottleneck activations
    y_train_nn_pred : np.ndarray
        Training neural network predictions (what SR was fitted to)
    y_train_true : np.ndarray
        Training ground truth physics labels (e.g., δ, ω)
    X_val : np.ndarray, optional
        Validation bottleneck activations
    y_val_nn_pred : np.ndarray, optional
        Validation neural network predictions
    y_val_true : np.ndarray, optional
        Validation ground truth physics labels
    gen_data : dict, optional
        Dict with keys 'paramagnetic', 'ferromagnetic', 'antiferromagnetic',
        each containing {'z': bottleneck activations, 'y_pred': NN predictions}
    sr_folder : Path
        Folder to save equations and plots
    output_labels : List[str]
        Labels for output dimensions
    top_k : int
        Number of top-ranked equations to overlay on the snapshot-average plot
        (via _write_hof_and_plot()). Unrelated to sr_config.top_k, which controls
        how many bottleneck activations are selected as SR input features.
    sr_config : SRConfig, optional
        Supplies get_optimized_picking_metrics() and get_hof_columns() for the
        Borda-count selection and CSV/markdown column layout described above.
        Falls back to SRConfig() defaults if None.
    kernels : list, optional
        cf.kernels; used with sr_z_mask to render the activation legend
        ("a_{idx} = a(<pattern>)") in the markdown Hall-of-Fame tables.
    sr_z_mask : np.ndarray, optional
        Indices into kernels selected as SR features (top-k by activation
        magnitude); required alongside kernels to build the activation legend.

    Returns:
    --------
    Tuple[Dict[int, Optional[int]], Dict[int, Dict[str, float]]]
        (optimized_eq_idx_per_output, optimized_gen_accuracy_per_output):
        - optimized_eq_idx_per_output: for each output index, the PySR
          equations_ row index of the "optimized_picked" equation (None if no
          equations were available for that output). Consumed by the caller to
          highlight the recommended equation in the 3D augmentation plots.
        - optimized_gen_accuracy_per_output: for each output index, a dict of
          {'ferro'/'antiferro'/'para': accuracy} for the optimized equation
          (classification tasks only; NaN otherwise).
        CSV/markdown Hall-of-Fame files and snapshot-average plots are also
        written to sr_folder as a side effect.
    """

    equations_folder = sr_folder / "equations"
    plots_folder = sr_folder / "plots"
    equations_folder.mkdir(parents=True, exist_ok=True)
    plots_folder.mkdir(parents=True, exist_ok=True)

    # Ensure training data has proper shape
    if y_train_nn_pred.ndim == 1:
        y_train_nn_pred = y_train_nn_pred.reshape(-1, 1)
    if y_train_true.ndim == 1:
        y_train_true = y_train_true.reshape(-1, 1)

    # Ensure validation data has proper shape (if provided)
    if y_val_nn_pred is not None and y_val_nn_pred.ndim == 1:
        y_val_nn_pred = y_val_nn_pred.reshape(-1, 1)
    if y_val_true is not None and y_val_true.ndim == 1:
        y_val_true = y_val_true.reshape(-1, 1)

    equations_list = model_sr.equations_
    multi_output = isinstance(equations_list, list)
    n_outputs = y_train_nn_pred.shape[1]

    if not multi_output:
        equations_list = [equations_list]

    aggregated_rows = []
    optimized_eq_idx_per_output: Dict[int, Optional[int]] = {}
    optimized_gen_accuracy_per_output: Dict[int, Dict[str, float]] = {}

    for out_idx in range(n_outputs):
        eq_df = equations_list[out_idx]
        if eq_df is None or len(eq_df) == 0:
            print(f"[WARN] No equations for output {out_idx} – skipping.")
            continue

        # Determine which equation index PySR picked (highest score in equations_ DataFrame)
        try:
            eq_df_for_pick = model_sr.equations_[out_idx] if multi_output else model_sr.equations_
            picked_idx = eq_df_for_pick.index[
                eq_df_for_pick["score"] == eq_df_for_pick["score"].max()
            ][0]
        except Exception:
            picked_idx = None

        # Apply override if sr_config.selected_equation_idx is provided
        if sr_config is not None and sr_config.selected_equation_idx is not None:
            override = sr_config.selected_equation_idx
            if isinstance(override, list):
                effective_override = override[out_idx] if out_idx < len(override) else None
            else:
                effective_override = override
            if effective_override is not None:
                picked_idx = effective_override

        eq_records = []
        for rank, (eq_idx, row) in enumerate(eq_df.iterrows()):
            # Initialize result dictionary
            res_dict = {
                "rank": rank,
                "output_idx": out_idx,
                "output_label": output_labels[out_idx] if out_idx < len(output_labels) else f"output_{out_idx}",
                "complexity": row.get("complexity", None),
                "SR_picked": (eq_idx == picked_idx) if picked_idx is not None else False,
            }

            # Get equation latex
            try:
                eq_set = model_sr.equations_[out_idx] if multi_output else model_sr.equations_
                expr = eq_set.sympy_format[eq_idx]
                eq_latex = convert_floats_to_ints(sp.latex(round_expr(expr)))
                res_dict["equation_latex"] = eq_latex
            except Exception as e:
                print(f"[WARN] Could not extract equation latex for {eq_idx}: {e}")
                res_dict["equation_latex"] = "N/A"

            # ========== EVALUATE ON TRAINING DATA (real, has temporal structure) ==========
            try:
                pred_eq_train = _predict_single_equation(
                    model_sr, X_train, eq_idx=int(eq_idx),
                    output_idx=out_idx if multi_output else None,
                )
                y_train_nn_pred_vec = y_train_nn_pred[:, out_idx]
                y_train_true_vec = y_train_true[:, out_idx]

                # Metrics vs NN predictions
                res_dict["r2_train"] = r2_score(y_train_nn_pred_vec, pred_eq_train)
                res_dict["mse_train"] = float(np.mean((y_train_nn_pred_vec - pred_eq_train) ** 2))

                # Metrics vs ground truth (regression only: for classification, y_true is one-hot so r2 is meaningless)
                _is_clf = getattr(train_dataset, 'discrete_labels', False)
                if not _is_clf:
                    res_dict["r2_true"] = r2_score(y_train_true_vec, pred_eq_train)
                    res_dict["mse_true"] = float(np.mean((y_train_true_vec - pred_eq_train) ** 2))

                # Aggregated metrics — task-dependent
                if not _is_clf:
                    # Regression: snapshot-averaged r2/mse
                    if y_train_nn_pred.squeeze().ndim == 1:
                        avg_train_nn_pred, _ = train_dataset.snapshot_average(y_train_nn_pred_vec, output=True)
                        avg_train_true, _    = train_dataset.snapshot_average(y_train_true_vec, output=True)
                        avg_pred_train, _    = train_dataset.snapshot_average(pred_eq_train, output=True)
                    else:
                        y_tmp = np.zeros(shape=(y_train_nn_pred_vec.shape[0], n_outputs))
                        y_tmp[:, out_idx] = y_train_nn_pred_vec
                        avg_train_nn_pred, _ = train_dataset.snapshot_average(y_tmp, output=True)
                        avg_train_nn_pred    = avg_train_nn_pred[:, out_idx]

                        y_tmp[:] = 0
                        y_tmp[:, out_idx] = y_train_true_vec
                        avg_train_true, _ = train_dataset.snapshot_average(y_tmp, output=True)
                        avg_train_true    = avg_train_true[:, out_idx]

                        y_tmp[:] = 0
                        y_tmp[:, out_idx] = pred_eq_train
                        avg_pred_train, _ = train_dataset.snapshot_average(y_tmp, output=True)
                        avg_pred_train    = avg_pred_train[:, out_idx]
                        del y_tmp

                    res_dict["r2_agg_train"] = r2_score(avg_train_nn_pred, avg_pred_train, multioutput='variance_weighted')
                    res_dict["mse_agg_train"] = float(np.mean((avg_train_nn_pred - avg_pred_train) ** 2))

                    res_dict["r2_agg_true"] = r2_score(avg_train_true, avg_pred_train, multioutput='variance_weighted')
                    res_dict["mse_agg_true"] = float(np.mean((avg_train_true - avg_pred_train) ** 2))

                    # Store for potential aggregation plots
                    res_dict["avg_pred"] = avg_pred_train
                    res_dict["avg_true"] = avg_train_true
                else:
                    # Classification Mode A: sign-agreement accuracy + per-timepoint accuracy_agg + BCE + r2_agg
                    _sign_agree_train = (np.sign(y_train_nn_pred_vec) == np.sign(pred_eq_train)).astype(np.float32)
                    res_dict["accuracy_train"] = float(np.mean(_sign_agree_train))
                    _means_tp, _ = train_dataset.snapshot_average(_sign_agree_train, output=False)
                    res_dict["accuracy_agg_train"] = float(np.mean(_means_tp))
                    # BCE: label from sign of NN logit_diff; prob from sigmoid of SR output
                    _lbl = (y_train_nn_pred_vec > 0).astype(np.float32)
                    _prob = 1.0 / (1.0 + np.exp(-np.clip(pred_eq_train, -30, 30)))
                    res_dict["bce_train"] = float(-np.mean(_lbl * np.log(_prob + 1e-7) + (1 - _lbl) * np.log(1 - _prob + 1e-7)))
                    # r2_agg: R² on per-timepoint means of logit_diff (trend fidelity across phase diagram)
                    _avg_nn_train, _ = train_dataset.snapshot_average(y_train_nn_pred_vec, output=False)
                    _avg_pred_tr, _ = train_dataset.snapshot_average(pred_eq_train, output=False)
                    res_dict["r2_agg_train"] = r2_score(_avg_nn_train, _avg_pred_tr)

            except Exception as e:
                print(f"[WARN] Could not evaluate equation {eq_idx} on training data: {e}")
                res_dict.update({"r2_train": np.nan, "mse_train": np.nan, "r2_true": np.nan, "mse_true": np.nan,
                                  "r2_agg_train": np.nan, "mse_agg_train": np.nan, "r2_agg_true": np.nan, "mse_agg_true": np.nan,
                                  "accuracy_train": np.nan, "accuracy_agg_train": np.nan, "bce_train": np.nan})

            # ========== EVALUATE ON VALIDATION DATA (real, has temporal structure) ==========
            if X_val is not None and y_val_nn_pred is not None:
                try:
                    pred_eq_val = _predict_single_equation(
                        model_sr, X_val, eq_idx=int(eq_idx),
                        output_idx=out_idx if multi_output else None,
                    )
                    y_val_nn_pred_vec = y_val_nn_pred[:, out_idx]
                    y_val_true_vec = y_val_true[:, out_idx] if y_val_true is not None else None

                    # Metrics vs NN predictions
                    res_dict["r2_val"] = r2_score(y_val_nn_pred_vec, pred_eq_val)
                    res_dict["mse_val"] = float(np.mean((y_val_nn_pred_vec - pred_eq_val) ** 2))

                    # NRMSE and Spearman ρ (in-distribution baseline for delta aggregates)
                    _range_val = y_val_nn_pred_vec.max() - y_val_nn_pred_vec.min()
                    if _range_val > 1e-8:
                        res_dict["nrmse_val"] = float(
                            np.sqrt(np.mean((y_val_nn_pred_vec - pred_eq_val) ** 2)) / _range_val
                        )
                    else:
                        print(f"[WARN] y_val range near zero for output {out_idx}, eq {eq_idx} — setting nrmse_val=NaN")
                        res_dict["nrmse_val"] = float("nan")
                    if np.std(pred_eq_val) < 1e-8 or np.std(y_val_nn_pred_vec) < 1e-8:
                        res_dict["rho_val"] = float("nan")
                    else:
                        res_dict["rho_val"] = float(spearmanr(y_val_nn_pred_vec, pred_eq_val).statistic)

                    # Aggregated metrics — task-dependent
                    _is_clf = getattr(val_dataset, 'discrete_labels', False)
                    if not _is_clf:
                        # Regression: snapshot-averaged r2/mse
                        if y_val_nn_pred.squeeze().ndim == 1:
                            avg_val_nn_pred, _ = val_dataset.snapshot_average(y_val_nn_pred_vec, output=True)
                            avg_pred_val, _    = val_dataset.snapshot_average(pred_eq_val, output=True)
                        else:
                            y_tmp = np.zeros(shape=(y_val_nn_pred_vec.shape[0], n_outputs))
                            y_tmp[:, out_idx] = y_val_nn_pred_vec
                            avg_val_nn_pred, _ = val_dataset.snapshot_average(y_tmp, output=True)
                            avg_val_nn_pred    = avg_val_nn_pred[:, out_idx]

                            y_tmp[:] = 0
                            y_tmp[:, out_idx] = pred_eq_val
                            avg_pred_val, _ = val_dataset.snapshot_average(y_tmp, output=True)
                            avg_pred_val    = avg_pred_val[:, out_idx]
                            del y_tmp

                        res_dict["r2_agg_val"] = r2_score(avg_val_nn_pred, avg_pred_val, multioutput='variance_weighted')
                        res_dict["mse_agg_val"] = float(np.mean((avg_val_nn_pred - avg_pred_val) ** 2))

                        # Aggregate val ground truth (needed for _val_true metrics and plot)
                        if y_val_true_vec is not None:
                            if y_val_nn_pred.squeeze().ndim == 1:
                                avg_val_true, _ = val_dataset.snapshot_average(y_val_true_vec, output=True)
                            else:
                                y_tmp2 = np.zeros(shape=(y_val_true_vec.shape[0], n_outputs))
                                y_tmp2[:, out_idx] = y_val_true_vec
                                avg_val_true, _ = val_dataset.snapshot_average(y_tmp2, output=True)
                                avg_val_true    = avg_val_true[:, out_idx]
                                del y_tmp2

                            res_dict["r2_val_true"]      = r2_score(y_val_true_vec, pred_eq_val)
                            res_dict["mse_val_true"]     = float(np.mean((y_val_true_vec - pred_eq_val) ** 2))
                            res_dict["r2_agg_val_true"]  = r2_score(avg_val_true, avg_pred_val, multioutput='variance_weighted')
                            res_dict["mse_agg_val_true"] = float(np.mean((avg_val_true - avg_pred_val) ** 2))

                            # Store for plotting helper
                            res_dict["avg_val_true"]    = avg_val_true
                            res_dict["avg_val_nn_pred"] = avg_val_nn_pred
                            res_dict["avg_pred_val"]    = avg_pred_val
                    else:
                        # Classification Mode A: sign-agreement accuracy + per-timepoint accuracy_agg + BCE + r2_agg
                        _sign_agree_val = (np.sign(y_val_nn_pred_vec) == np.sign(pred_eq_val)).astype(np.float32)
                        res_dict["accuracy_val"] = float(np.mean(_sign_agree_val))
                        _means_tp_val, _ = val_dataset.snapshot_average(_sign_agree_val, output=False)
                        res_dict["accuracy_agg_val"] = float(np.mean(_means_tp_val))
                        _lbl_val = (y_val_nn_pred_vec > 0).astype(np.float32)
                        _prob_val = 1.0 / (1.0 + np.exp(-np.clip(pred_eq_val, -30, 30)))
                        res_dict["bce_val"] = float(-np.mean(_lbl_val * np.log(_prob_val + 1e-7) + (1 - _lbl_val) * np.log(1 - _prob_val + 1e-7)))

                        # r2_agg: R² on per-timepoint means of logit_diff (trend fidelity across phase diagram)
                        _avg_val_nn, _ = val_dataset.snapshot_average(y_val_nn_pred_vec, output=False)
                        _avg_pred_v, _ = val_dataset.snapshot_average(pred_eq_val, output=False)
                        res_dict["r2_agg_val"] = r2_score(_avg_val_nn, _avg_pred_v)

                        # Store per-timepoint averages for plotting helper
                        # avg_val_true omitted: one-hot labels have no meaningful continuous average
                        res_dict["avg_val_nn_pred"] = _avg_val_nn
                        res_dict["avg_pred_val"]    = _avg_pred_v

                except Exception as e:
                    print(f"[WARN] Could not evaluate equation {eq_idx} on validation data: {e}")
                    res_dict.update({
                        "r2_val": np.nan, "mse_val": np.nan, "r2_agg_val": np.nan, "mse_agg_val": np.nan,
                        "r2_val_true": np.nan, "mse_val_true": np.nan,
                        "r2_agg_val_true": np.nan, "mse_agg_val_true": np.nan,
                        "accuracy_val": np.nan, "accuracy_agg_val": np.nan, "bce_val": np.nan,
                    })

            # ========== EVALUATE ON GENERALIZATION DATA (synthetic, NO temporal structure → NO _agg) ==========
            if gen_data is not None:
                # Create combined generalization dataset first
                gen_X_combined = []
                gen_y_combined = []
                for gen_type in ['paramagnetic', 'ferromagnetic', 'antiferromagnetic']:
                    if gen_type in gen_data:
                        gen_X_combined.append(gen_data[gen_type]['z'])
                        gen_y_combined.append(gen_data[gen_type]['y_pred'])

                has_gen_combined = len(gen_X_combined) > 0
                if has_gen_combined:
                    gen_X_combined = np.concatenate(gen_X_combined)
                    gen_y_combined = np.concatenate(gen_y_combined)

                # Evaluate on each generalization type separately
                for gen_type in ['paramagnetic', 'ferromagnetic', 'antiferromagnetic']:
                    if gen_type not in gen_data:
                        continue

                    abbrev = {'paramagnetic': 'para', 'ferromagnetic': 'ferro', 'antiferromagnetic': 'antiferro'}[gen_type]

                    try:
                        X_gen = gen_data[gen_type]['z']
                        y_gen_nn_pred = gen_data[gen_type]['y_pred']

                        pred_eq_gen = _predict_single_equation(
                            model_sr, X_gen, eq_idx=int(eq_idx),
                            output_idx=out_idx if multi_output else None,
                        )

                        if y_gen_nn_pred.ndim == 1:
                            y_gen_nn_pred_vec = y_gen_nn_pred
                        else:
                            y_gen_nn_pred_vec = y_gen_nn_pred[:, out_idx]

                        # Metrics vs NN predictions (NO _agg - synthetic has no temporal structure)
                        res_dict[f"r2_{abbrev}"] = r2_score(y_gen_nn_pred_vec, pred_eq_gen)
                        res_dict[f"mse_{abbrev}"] = float(np.mean((y_gen_nn_pred_vec - pred_eq_gen) ** 2))

                        # NRMSE and Spearman ρ per OOD phase
                        _range_gen = y_gen_nn_pred_vec.max() - y_gen_nn_pred_vec.min()
                        if _range_gen > 1e-8:
                            res_dict[f"nrmse_{abbrev}"] = float(
                                np.sqrt(np.mean((y_gen_nn_pred_vec - pred_eq_gen) ** 2)) / _range_gen
                            )
                        else:
                            print(f"[WARN] {gen_type} NN pred range near zero for output {out_idx}, eq {eq_idx} — setting nrmse_{abbrev}=NaN")
                            res_dict[f"nrmse_{abbrev}"] = float("nan")
                        if np.std(pred_eq_gen) < 1e-8 or np.std(y_gen_nn_pred_vec) < 1e-8:
                            res_dict[f"rho_{abbrev}"] = float("nan")
                        else:
                            res_dict[f"rho_{abbrev}"] = float(spearmanr(y_gen_nn_pred_vec, pred_eq_gen).statistic)

                        # Classification Mode A: sign-agreement accuracy on OOD data
                        if getattr(train_dataset, 'discrete_labels', False):
                            _sign_agree_gen = (np.sign(y_gen_nn_pred_vec) == np.sign(pred_eq_gen)).astype(np.float32)
                            res_dict[f"accuracy_{abbrev}"] = float(np.mean(_sign_agree_gen))

                    except Exception as e:
                        print(f"[WARN] Could not evaluate equation {eq_idx} on {gen_type} data: {e}")
                        res_dict[f"r2_{abbrev}"] = np.nan
                        res_dict[f"mse_{abbrev}"] = np.nan
                        res_dict[f"nrmse_{abbrev}"] = np.nan
                        res_dict[f"rho_{abbrev}"] = np.nan
                        res_dict[f"accuracy_{abbrev}"] = np.nan

                # Evaluate on combined generalization dataset
                if has_gen_combined:
                    try:
                        pred_eq_gen_combined = _predict_single_equation(
                            model_sr, gen_X_combined, eq_idx=int(eq_idx),
                            output_idx=out_idx if multi_output else None,
                        )

                        if gen_y_combined.ndim == 1:
                            gen_y_combined_vec = gen_y_combined
                        else:
                            gen_y_combined_vec = gen_y_combined[:, out_idx]

                        res_dict["r2_gen_combined"] = r2_score(gen_y_combined_vec, pred_eq_gen_combined)
                        res_dict["mse_gen_combined"] = float(np.mean((gen_y_combined_vec - pred_eq_gen_combined) ** 2))

                    except Exception as e:
                        print(f"[WARN] Could not evaluate equation {eq_idx} on combined generalization data: {e}")
                        res_dict["r2_gen_combined"] = np.nan
                        res_dict["mse_gen_combined"] = np.nan

            # Aggregate delta metrics across OOD phases (ratio of OOD to in-distribution NRMSE/rho)
            _phase_abbrevs = ['ferro', 'antiferro', 'para']
            _nrmse_val = res_dict.get("nrmse_val", float("nan"))
            _rho_val = res_dict.get("rho_val", float("nan"))

            _delta_nrmse_vals = []
            _delta_rho_vals = []
            for _abbrev in _phase_abbrevs:
                _nrmse_phase = res_dict.get(f"nrmse_{_abbrev}", float("nan"))
                _rho_phase = res_dict.get(f"rho_{_abbrev}", float("nan"))
                if not np.isnan(_nrmse_val) and not np.isnan(_nrmse_phase) and _nrmse_val > 1e-8:
                    _delta_nrmse_vals.append(_nrmse_phase / _nrmse_val)
                else:
                    _delta_nrmse_vals.append(float("nan"))
                if not np.isnan(_rho_val) and not np.isnan(_rho_phase):
                    _delta_rho_vals.append(_rho_phase / _rho_val)
                else:
                    _delta_rho_vals.append(float("nan"))

            _any_nrmse = not all(np.isnan(v) for v in _delta_nrmse_vals)
            _any_rho = not all(np.isnan(v) for v in _delta_rho_vals)
            res_dict["delta_nrmse_mean"] = float(np.nanmean(_delta_nrmse_vals)) if _any_nrmse else float("nan")
            res_dict["delta_rho_mean"] = float(np.nanmean(_delta_rho_vals)) if _any_rho else float("nan")

            eq_records.append(res_dict)

        if not eq_records:
            continue

        # Sort by training R² (primary) and complexity (secondary)
        eq_records.sort(key=lambda r: (-r.get("r2_train", -np.inf), r["complexity"] if r["complexity"] is not None else np.inf))

        # Compute optimized_picked via Borda count rank aggregation.
        # Each metric contributes one leaderboard; equations accumulate rank positions (1 = best).
        # Lower total Borda score = better overall. NaN values ranked last on each leaderboard.
        # Tie-break: highest r2_val.
        _is_clf_pick = getattr(train_dataset, 'discrete_labels', False)
        _picking_metrics = (
            sr_config.get_optimized_picking_metrics(_is_clf_pick)
            if sr_config is not None
            else SRConfig().get_optimized_picking_metrics(_is_clf_pick)
        )
        n_eq = len(eq_records)
        _borda_scores = [0] * n_eq

        for _metric, _higher_is_better in _picking_metrics:
            _vals = [rec.get(_metric, float("nan")) for rec in eq_records]
            _valid_idx = [i for i, v in enumerate(_vals) if not np.isnan(v)]
            _invalid_idx = [i for i, v in enumerate(_vals) if np.isnan(v)]
            _valid_idx_sorted = sorted(
                _valid_idx,
                key=lambda i: _vals[i],
                reverse=_higher_is_better,
            )
            for _rank_pos, _i in enumerate(_valid_idx_sorted):
                _borda_scores[_i] += (_rank_pos + 1)
            for _i in _invalid_idx:
                _borda_scores[_i] += n_eq  # NaN → last place on this leaderboard

        # Pick lowest Borda score; tie-break by r2_val descending
        _indexed_scores = sorted(
            enumerate(_borda_scores),
            key=lambda x: (x[1], -eq_records[x[0]].get("r2_val", float("-inf"))),
        )
        _opt_record_idx = _indexed_scores[0][0] if n_eq > 0 else None

        for i, rec in enumerate(eq_records):
            rec["optimized_picked"] = (i == _opt_record_idx)

        # Recover the PySR equation index of the optimized pick for use in 3D plots
        if _opt_record_idx is not None:
            _opt_rank = eq_records[_opt_record_idx]["rank"]
            _opt_eq_idx_pysr = None
            for _r, (_eidx, _) in enumerate(eq_df.iterrows()):
                if _r == _opt_rank:
                    _opt_eq_idx_pysr = int(_eidx)
                    break
            optimized_eq_idx_per_output[out_idx] = _opt_eq_idx_pysr
            # Extract OOD accuracy for the optimized equation (classification only)
            _opt_rec = eq_records[_opt_record_idx]
            optimized_gen_accuracy_per_output[out_idx] = {
                abbrev: _opt_rec.get(f"accuracy_{abbrev}", float("nan"))
                for abbrev in ("ferro", "antiferro", "para")
            }
        else:
            optimized_eq_idx_per_output[out_idx] = None
            optimized_gen_accuracy_per_output[out_idx] = {}

        # Build task-aware column list via SRConfig
        _is_clf_cols = getattr(train_dataset, 'discrete_labels', False)
        _sr_cfg_for_cols = sr_config if sr_config is not None else SRConfig()
        all_columns = _sr_cfg_for_cols.get_hof_columns(
            is_classification=_is_clf_cols,
            has_val=(X_val is not None),
            has_gen=(gen_data is not None),
        )
        df_out = pd.DataFrame([{col: rec.get(col, np.nan) for col in all_columns} for rec in eq_records])

        label_slug = output_labels[out_idx] if out_idx < len(output_labels) else f"output_{out_idx}"
        csv_path = equations_folder / f"hof_{label_slug}.csv"
        md_path = equations_folder / f"hof_{label_slug}.md"
        df_out.to_csv(csv_path, index=False)

        # Create markdown table with all columns dynamically
        md_header_cols = ["rank", "output_label", "complexity"] + all_columns[3:]  # Skip base columns that are already included
        md_header = "| " + " | ".join(md_header_cols) + " |\n"
        md_sep = "| " + " | ".join(["---"] * len(md_header_cols)) + " |\n"

        # Build pattern substitution map: a_{idx} → a(<pattern>) for markdown equations
        activation_legend = ""
        pattern_subs = {}  # maps "a_{idx}" → "a(<pattern_content>)"
        if kernels is not None and sr_z_mask is not None:
            from tetriscnn.utils import kernel_to_latex_pattern
            legend_parts = []
            for idx in sr_z_mask:
                if idx >= len(kernels):
                    continue
                pattern_content = kernel_to_latex_pattern(kernels[idx], equation_env=False)
                pattern_subs[f"a_{{{idx}}}"] = f"a({pattern_content})"
                legend_parts.append(f"$a_{{{idx}}}$ = $a({pattern_content})$")
            if legend_parts:
                activation_legend = "**Activations:** " + " &nbsp;&nbsp; ".join(legend_parts) + "\n\n"

        def _apply_pattern_subs(eq_latex: str) -> str:
            for old, new in pattern_subs.items():
                eq_latex = eq_latex.replace(old, new)
            return eq_latex

        md_rows = []
        for _, row in df_out.iterrows():
            row_vals = [str(row["rank"]), str(row["output_label"]), str(row["complexity"])]
            for col in all_columns[3:]:  # Skip rank, output_label, complexity (already handled)
                if col == "equation_latex":
                    eq = _apply_pattern_subs(str(row[col])) if pattern_subs else str(row[col])
                    row_vals.append(f"$$ {eq} $$")
                elif "r2_" in col:
                    row_vals.append(f"{row.get(col, np.nan):.4f}")
                elif "mse_" in col:
                    row_vals.append(f"{row.get(col, np.nan):.6f}")
                else:
                    row_vals.append(str(row.get(col, "")))
            md_rows.append("| " + " | ".join(row_vals) + " |")

        with open(md_path, "w") as f_md:
            f_md.write(f"# Hall of Fame for {label_slug}\n\n")
            f_md.write(activation_legend)
            f_md.write(md_header + md_sep + "\n".join(md_rows))

        aggregated_rows.extend(df_out.to_dict(orient="records"))

        # Delegate plotting to helper — extract val aggregates from first equation record
        if eq_records:
            if "avg_val_true" in eq_records[0]:
                avg_val_true_for_plot = eq_records[0]["avg_val_true"]
            else:
                avg_val_true_for_plot = None
            
            if "avg_val_nn_pred" in eq_records[0]:
                avg_val_nn_pred_for_plot = eq_records[0]["avg_val_nn_pred"]
            else:
                avg_val_nn_pred_for_plot = None

            _fig, _axs, _plot_paths, _final_label_slug, top_records = _write_hof_and_plot(
                plots_folder=plots_folder,
                out_idx=out_idx,
                output_labels=output_labels,
                eq_records=eq_records,
                avg_val_true=avg_val_true_for_plot,
                avg_val_nn_pred=avg_val_nn_pred_for_plot,
                sr_dataset=val_dataset,
                top_k=top_k,
                make_legend=True,
                latex_slugify=True,
                formats=["png", "pdf"],
            )

    if aggregated_rows:
        df_all = pd.DataFrame(aggregated_rows)
        df_all.to_csv(equations_folder / "hof_all_outputs.csv", index=False)

    return optimized_eq_idx_per_output, optimized_gen_accuracy_per_output


def _write_hof_and_plot(
    plots_folder: Path,
    out_idx: int,
    output_labels: List[str],
    eq_records: List[Dict[str, Any]],
    avg_val_true: np.ndarray,
    avg_val_nn_pred: np.ndarray,
    sr_dataset: Any,
    top_k: int = 5,
    make_legend: bool = True,
    latex_slugify: bool = True,
    formats: List[str] = ["png", "pdf"],
) -> Tuple[Any, Any, Dict[str, Path], str, List[Dict[str, Any]]]:
    """Write per-output HOF snapshot-average plots.

    Returns (fig, axs, plot_paths, final_label_slug, top_records).
    Note: CSV and markdown files are written by save_sr_hof_with_avg() before calling here.
    """
    label_slug = output_labels[out_idx] if out_idx < len(output_labels) else f"output_{out_idx}"

    # Prepare plotting data
    top_records = eq_records[: min(top_k, len(eq_records))]
    true_x = sr_dataset.unique_labels_unnormalized
    if getattr(true_x, "ndim", 1) > 1:
        true_x = true_x[:, out_idx]

    # Latex label wrapping for axis labels (filenames later sanitized)
    if latex_slugify and label_slug in ['delta', 'omega', 'beta']:
        label_for_axis = r"$\{}$".format(label_slug)
    else:
        if label_slug == "logit_diff":
            label_for_axis = r"$\Delta\,\text{Logits}$"
        elif label_slug.startswith("logit_"):
            _idx = label_slug.split("_")[-1]
            label_for_axis = rf"$\ell_{{{_idx}}}$"
        else:
            label_for_axis = label_slug

    has_times = hasattr(sr_dataset, 'times')
    if has_times:
        times_unique = np.array(list(sr_dataset.unique_times.cpu())) / 1e3  # ms -> μs
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        ax = axs[0]
        if avg_val_true is not None:
            ax.plot(times_unique, avg_val_true, "o:", label="True Labels (val)", color="black", linewidth=2)
        ax.plot(times_unique, avg_val_nn_pred, "x--", label="NN Prediction (val)", color="gray", linewidth=2)
        for rec in top_records:
            ax.plot(times_unique, rec["avg_pred_val"], "x--",
                    label=r"Eq {} $(R^2_{{val,agg}}={:.3f})$".format(rec['rank'], rec.get('r2_agg_val', float('nan'))))
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel(label_for_axis)
        ax.set_title(f"Snapshot average over time: {label_for_axis}")

        # --- RIGHT subplot: parity plot (NN avg predictions vs SR avg predictions per time point) ---
        ax = axs[1]
        if avg_val_nn_pred is not None and top_records:
            all_sr_preds = np.concatenate([rec["avg_pred_val"] for rec in top_records if "avg_pred_val" in rec])
            _pad = (avg_val_nn_pred.max() - avg_val_nn_pred.min()) * 0.05 or 0.5
            _lo = min(avg_val_nn_pred.min(), all_sr_preds.min()) - _pad
            _hi = max(avg_val_nn_pred.max(), all_sr_preds.max()) + _pad
            ax.plot([_lo, _hi], [_lo, _hi], 'k--', lw=1.5, label='Perfect agreement', zorder=0)
            for rec in top_records:
                if "avg_pred_val" in rec:
                    ax.scatter(
                        avg_val_nn_pred, rec["avg_pred_val"],
                        s=30, alpha=0.8, zorder=2,
                        label=r"Eq {} $(R^2_{{val,agg}}={:.3f})$".format(
                            rec['rank'], rec.get('r2_agg_val', float('nan'))
                        ),
                    )
        ax.set_xlabel(f"NN prediction ({label_for_axis})")
        ax.set_ylabel(f"SR prediction ({label_for_axis})")
        ax.set_title("Parity: NN vs SR (per time point)")
        if make_legend:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    else:
        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        axs = np.array([axs])
        ax = axs[0]
        # Single-panel fallback (no times): line plot vs tuning parameter
        if avg_val_true is not None:
            ax.plot(true_x, avg_val_true, "o:", label="True Labels (val)", color="black", linewidth=2)
        ax.plot(true_x, avg_val_nn_pred, "x--", label="NN Prediction (val)", color="gray", linewidth=2)
        for rec in top_records:
            ax.plot(true_x, rec["avg_pred_val"], "x--",
                    label=r"Eq {} $(R^2_{{val,agg}}={:.3f})$".format(rec['rank'], rec.get('r2_agg_val', float('nan'))))
        ax.set_xlabel(label_for_axis)
        ax.set_ylabel(r"$\hat{{{}}}$".format(label_for_axis[1:-1] if label_for_axis.startswith('$') and label_for_axis.endswith('$') else label_for_axis))
        ax.set_title(f"Snapshot average: {label_for_axis}")
        if make_legend:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

    for _ax in axs.flatten():
        _ax.grid(True)
    fig.tight_layout()

    # Sanitize slug for filenames
    final_label_slug = label_for_axis[2:-1] if label_for_axis.startswith('$') and label_for_axis.endswith('$') else label_for_axis
    if label_slug == "logit_diff" or label_slug.startswith("logit_"):
        final_label_slug = label_slug
    plot_base = plots_folder / f"hof_{final_label_slug}"
    plot_paths: Dict[str, Path] = {}
    for fmt in formats:
        out_path = plot_base.with_suffix(f".{fmt}")
        fig.savefig(out_path)
        plot_paths[fmt] = out_path
    plt.close()

    return fig, axs, plot_paths, final_label_slug, top_records


###############################################
# SR generalization plots (plotly/matplotlib)
###############################################
# Kept here rather than in tetriscnn/plots.py so that plotly, like PySR, is only needed
# for the optional symbolic-regression route (requirements-sr.txt).

def plot_3d_augmentation(
    X_real: np.ndarray,
    X_synthetic: np.ndarray,
    y_real: np.ndarray,
    y_synthetic: np.ndarray,
    selected_idx: np.ndarray,
    model_sr_folder: Path,
    fname_prefix: str = "SR_raw",
    sample_size: int = 5000,
    random_snapshots: Optional[list] = None,
    output_labels: Optional[List[str]] = None,
    selected_equation_idx: Optional[int] = None,
    optimized_equation_idx: Optional[Dict[int, Optional[int]]] = None,
    ood_accuracy: Optional[Dict[int, float]] = None,
    kernels: Optional[list] = None,
):
    """Visualize real vs synthetic data using Plotly; dispatches to 1D/2D/3D based on n_features."""

    n_features = X_real.shape[1]

    # Normalize y shape
    if y_real.ndim == 1:
        n_outputs = 1
        y_real = y_real.reshape(-1, 1)
        y_synthetic = y_synthetic.reshape(-1, 1)
    else:
        n_outputs = y_real.shape[1]

    if n_features > 3:
        print(f"[WARN] plot_3d_augmentation: n_features={n_features} > 3, skipping plot (PCA removed).")
        return

    # Build axis labels:
    #   latex_labels  — with $...$ for Plotly 2D axes and matplotlib (renders LaTeX)
    #   plain_labels  — no $...$ for Plotly 3D scenes and hover text (LaTeX unsupported there)
    if kernels is not None:
        from tetriscnn.utils import kernel_to_latex_pattern
        latex_labels = [
            f"$a({kernel_to_latex_pattern(kernels[idx], equation_env=False)})$"
            for idx in selected_idx if idx < len(kernels)
        ]
        plain_labels = [
            # Strip LaTeX commands for readable plain-text in Plotly 3D/hover
            re.sub(r'\\substack\{([^}]*)\}', lambda m: m.group(1).replace(r'\ ', ' '),
                   kernel_to_latex_pattern(kernels[idx], equation_env=False))
            .replace('{', '').replace('}', '')
            for idx in selected_idx if idx < len(kernels)
        ]
        plain_labels = [f"a({p})" for p in plain_labels]
    else:
        latex_labels = [f"$a_{{{idx}}}$" for idx in selected_idx]
        plain_labels = [f"a{idx}" for idx in selected_idx]
    hover_labels = plain_labels
    title_suffix = f"{n_features} selected activation{'s' if n_features > 1 else ''}"

    # Load SR model
    model_sr_path = model_sr_folder / "model_sr_raw.pkl"
    with open(model_sr_path, "rb") as f:
        model_sr = pickle.load(f)

    for output_idx in range(n_outputs):
        _eq_idx_for_plot = selected_equation_idx
        if _eq_idx_for_plot is None and optimized_equation_idx is not None:
            _eq_idx_for_plot = optimized_equation_idx.get(output_idx)
        _ood_acc_for_output = ood_accuracy.get(output_idx, float("nan")) if ood_accuracy else float("nan")

        _plot_single_output(
            X_real_nd=X_real,
            X_synthetic_nd=X_synthetic,
            y_real_plot=y_real[:, output_idx],
            y_synthetic_plot=y_synthetic[:, output_idx],
            model_sr=model_sr,
            labels=latex_labels,
            plain_labels=plain_labels,
            hover_labels=hover_labels,
            title_suffix=title_suffix,
            model_sr_folder=model_sr_folder,
            fname_prefix=fname_prefix,
            output_idx=output_idx,
            n_outputs=n_outputs,
            ndim=n_features,
            random_snapshots=random_snapshots,
            output_labels=output_labels,
            selected_equation_idx=_eq_idx_for_plot,
            ood_accuracy=_ood_acc_for_output,
        )


def _compute_sr_plot_data(model_sr, X_real, X_synthetic, y_real_plot, y_synthetic_plot,
                          selected_equation_idx, output_idx, ood_accuracy):
    """Shared SR prediction + metric computation for all dimensionality variants."""
    from scipy.stats import spearmanr as _spearmanr

    if selected_equation_idx is not None:
        y_pred_SR = _predict_single_equation(model_sr, X_real, selected_equation_idx, output_idx).reshape(-1, 1)
        y_pred_SR_syn = _predict_single_equation(model_sr, X_synthetic, selected_equation_idx, output_idx).reshape(-1, 1)
        _col = 0
    else:
        y_pred_SR = model_sr.predict(X_real)
        y_pred_SR_syn = model_sr.predict(X_synthetic)
        if y_pred_SR.ndim == 1: y_pred_SR = y_pred_SR.reshape(-1, 1)
        if y_pred_SR_syn.ndim == 1: y_pred_SR_syn = y_pred_SR_syn.reshape(-1, 1)
        _col = output_idx

    error_real = y_real_plot.squeeze() - y_pred_SR[:, _col].squeeze()
    error_syn  = y_synthetic_plot.squeeze() - y_pred_SR_syn[:, _col].squeeze()
    r2_real = r2_score(y_real_plot.squeeze(), y_pred_SR[:, _col].squeeze())
    r2_syn  = r2_score(y_synthetic_plot.squeeze(), y_pred_SR_syn[:, _col].squeeze())

    _y_real, _y_real_sr = y_real_plot.squeeze(), y_pred_SR[:, _col].squeeze()
    _y_syn,  _y_syn_sr  = y_synthetic_plot.squeeze(), y_pred_SR_syn[:, _col].squeeze()
    mse_real = float(np.mean((_y_real - _y_real_sr) ** 2))
    mse_syn  = float(np.mean((_y_syn  - _y_syn_sr)  ** 2))
    _rng_r, _rng_s = _y_real.max() - _y_real.min(), _y_syn.max() - _y_syn.min()
    nrmse_real = float(np.sqrt(mse_real) / _rng_r) if _rng_r > 1e-8 else float("nan")
    nrmse_syn  = float(np.sqrt(mse_syn)  / _rng_s) if _rng_s > 1e-8 else float("nan")
    delta_nrmse = (nrmse_syn / nrmse_real) if (not np.isnan(nrmse_real) and nrmse_real > 1e-8) else float("nan")
    rho_real = float(_spearmanr(_y_real, _y_real_sr).statistic)
    rho_syn  = float(_spearmanr(_y_syn,  _y_syn_sr).statistic)
    delta_rho = (rho_syn / rho_real) if (not np.isnan(rho_real) and abs(rho_real) > 1e-8) else float("nan")

    _acc = float("nan") if ood_accuracy is None else ood_accuracy
    return dict(
        y_pred_SR=y_pred_SR, y_pred_SR_syn=y_pred_SR_syn, _col=_col,
        error_real=error_real, error_syn=error_syn,
        r2_real=r2_real, r2_syn=r2_syn,
        mse_real=mse_real, mse_syn=mse_syn,
        nrmse_real=nrmse_real, nrmse_syn=nrmse_syn,
        delta_nrmse=delta_nrmse, delta_rho=delta_rho,
        _y_real=_y_real, _y_real_sr=_y_real_sr,
        _y_syn=_y_syn, _y_syn_sr=_y_syn_sr,
        ood_accuracy=_acc,
    )


def _sr_subplot_titles(d):
    """Build the six subplot title strings shared across dimensionalities."""
    _acc = d["ood_accuracy"]
    _acc_str = f", Acc={_acc:.3f}" if not np.isnan(_acc) else ""
    syn_title = f"Synthetic Data (\u0394_NRMSE={d['delta_nrmse']:.3f}, \u0394_\u03c1={d['delta_rho']:.3f}{_acc_str})"
    parity_real = (f"Parity: In-D (NN vs SR)<br>"
                   f"<sup>R²={d['r2_real']:.3f}, MSE={d['mse_real']:.4f}, NRMSE={d['nrmse_real']:.4f}{_acc_str}</sup>")
    parity_syn  = (f"Parity: OOD (NN vs SR)<br>"
                   f"<sup>R²={d['r2_syn']:.3f}, MSE={d['mse_syn']:.4f}, NRMSE={d['nrmse_syn']:.4f}{_acc_str}</sup>")
    return syn_title, parity_real, parity_syn


def _add_parity_rows(fig, d, cmin_real, cmax_real, cmin_syn, cmax_syn, row=3):
    """Add diagonal + error-colored parity scatter to rows 3 col 1/2."""
    _nn_r, _sr_r = d["_y_real"], d["_y_real_sr"]
    _nn_s, _sr_s = d["_y_syn"],  d["_y_syn_sr"]
    for col, (nn, sr, err, cmin, cmax, name) in enumerate([
        (_nn_r, _sr_r, d["error_real"], cmin_real, cmax_real, "Parity In-D"),
        (_nn_s, _sr_s, d["error_syn"],  cmin_syn,  cmax_syn,  "Parity OOD"),
    ], start=1):
        dmin, dmax = float(min(nn.min(), sr.min())), float(max(nn.max(), sr.max()))
        fig.add_trace(go.Scatter(x=[dmin, dmax], y=[dmin, dmax], mode='lines',
                                 line=dict(color='black', dash='dash', width=1),
                                 showlegend=False, hoverinfo='skip'), row=row, col=col)
        x_pos = 1.02 if col == 1 else 1.12
        fig.add_trace(go.Scatter(x=nn, y=sr, mode='markers',
                                 marker=dict(size=4, color=err.flatten(), colorscale='RdBu',
                                             cmin=cmin, cmax=cmax, showscale=True, opacity=0.8,
                                             colorbar=dict(len=0.25, y=0.12, x=x_pos, title="Error")),
                                 showlegend=False, name=name,
                                 hovertemplate="NN: %{x:.4f}<br>SR: %{y:.4f}<br>Error: %{marker.color:.4f}<extra></extra>"),
                      row=row, col=col)
    fig.update_xaxes(title_text="NN Prediction", row=row, col=1)
    fig.update_yaxes(title_text="SR Prediction", row=row, col=1)
    fig.update_xaxes(title_text="NN Prediction", row=row, col=2)
    fig.update_yaxes(title_text="SR Prediction", row=row, col=2)


def _save_matplotlib_3d(mpl_data: dict, base: Path):
    """Render the 3×2 SR plot grid in matplotlib with mathtext axis labels and save PNG/PDF.

    mpl_data keys:
        X_real, X_syn, y_real, y_syn, error_real, error_syn  — arrays (N, 3) / (N,)
        y_real_sr, y_syn_sr                                   — SR predictions (N,)
        labels                                                — list[str] with $...$ LaTeX
        titles                                                — 6-tuple of subplot title strings
        cmin_real, cmax_real, cmin_syn, cmax_syn             — float colorbar limits
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    X_r  = mpl_data["X_real"];  X_s  = mpl_data["X_syn"]
    y_r  = mpl_data["y_real"];  y_s  = mpl_data["y_syn"]
    e_r  = mpl_data["error_real"]; e_s = mpl_data["error_syn"]
    yr_sr = mpl_data["y_real_sr"]; ys_sr = mpl_data["y_syn_sr"]
    lbl  = mpl_data["labels"]
    ttls = mpl_data["titles"]   # (title_r_label, title_s_label, title_r_err, title_s_err, parity_r, parity_s)
    cmr, cxr = mpl_data["cmin_real"], mpl_data["cmax_real"]
    cms, cxs = mpl_data["cmin_syn"],  mpl_data["cmax_syn"]

    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Colormaps and norms
    norm_label_r = mcolors.Normalize(vmin=y_r.min(), vmax=y_r.max())
    norm_label_s = mcolors.Normalize(vmin=y_s.min(), vmax=y_s.max())
    norm_err_r   = mcolors.Normalize(vmin=cmr, vmax=cxr)
    norm_err_s   = mcolors.Normalize(vmin=cms, vmax=cxs)
    cmap_r = cm.viridis; cmap_s = cm.cividis; cmap_e = cm.RdBu

    specs_3d = [
        (0, 0, X_r, y_r.flatten(),  norm_label_r, cmap_r, ttls[0], "Predicted Label"),
        (0, 1, X_s, y_s.flatten(),  norm_label_s, cmap_s, ttls[1], ""),
        (1, 0, X_r, e_r.flatten(),  norm_err_r,   cmap_e, ttls[2], "y_pred − y_SR"),
        (1, 1, X_s, e_s.flatten(),  norm_err_s,   cmap_e, ttls[3], "y_synt − y_SR"),
    ]
    for row, col, X, c_vals, norm, cmap, title, cb_lbl in specs_3d:
        ax = fig.add_subplot(gs[row, col], projection='3d')
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=c_vals, cmap=cmap, norm=norm,
                        s=8, alpha=0.7)
        ax.set_xlabel(lbl[0], labelpad=8)
        ax.set_ylabel(lbl[1], labelpad=8)
        ax.set_zlabel(lbl[2], labelpad=8)
        ax.set_title(title, fontsize=8, pad=4)
        cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        cb.set_label(cb_lbl, fontsize=7)

    # Row 3: parity plots
    for col, (nn, sr, err, norm, title) in enumerate([
        (y_r.flatten(), yr_sr.flatten(), e_r.flatten(), norm_err_r, ttls[4]),
        (y_s.flatten(), ys_sr.flatten(), e_s.flatten(), norm_err_s, ttls[5]),
    ]):
        ax = fig.add_subplot(gs[2, col])
        dmin, dmax = min(nn.min(), sr.min()), max(nn.max(), sr.max())
        ax.plot([dmin, dmax], [dmin, dmax], 'k--', lw=1)
        sc = ax.scatter(nn, sr, c=err, cmap=cmap_e, norm=norm, s=6, alpha=0.7)
        ax.set_xlabel("NN Prediction"); ax.set_ylabel("SR Prediction")
        ax.set_title(title, fontsize=7)
        fig.colorbar(sc, ax=ax, shrink=0.6).set_label("Error", fontsize=7)

    for ext in (".png", ".pdf"):
        fig.savefig(str(base.with_suffix(ext)), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_sr_plot(fig, model_sr_folder, fname_prefix, output_labels, output_idx, n_outputs,
                  dim, title_suffix, coupled_cameras=False, mpl_data=None):
    """Save PNG/PDF (via matplotlib for 3D, plotly for 1D/2D) and HTML for 3D."""
    suffix = f"_output_{output_labels[output_idx]}" if n_outputs > 1 else ""
    dim_tag = f"_{dim}d_augmentation"
    base = model_sr_folder / f"{fname_prefix}{dim_tag}{suffix}"

    if dim == 3 and mpl_data is not None:
        _save_matplotlib_3d(mpl_data, base)
    else:
        fig.write_image(str(base.with_suffix(".png")), width=1600, height=1600)
        fig.write_image(str(base.with_suffix(".pdf")), width=1600, height=1600)

    if coupled_cameras:
        plot_div_id = "coupled_plot_div"
        plot_div = to_html(fig, full_html=False, include_plotlyjs='cdn', div_id=plot_div_id)
        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Coupled Visualization</title>
<style>body{{margin:0;padding:0}}.plot-container{{width:100%;height:100vh}}</style>
</head><body><div class="plot-container">{plot_div}</div>
<script>
function attachCoupling(){{
    var g=document.getElementById('{plot_div_id}');
    if(!g||!g.on){{setTimeout(attachCoupling,100);return;}}
    var sync=false;
    g.on('plotly_relayout',function(e){{
        if(sync)return;
        var upd={{}},src=null,prop=null,val=null;
        Object.keys(e).forEach(function(k){{
            var m=k.match(/^(scene\\d*)(\\.+)$/);
            if(m&&m[2].indexOf('camera')!==-1){{src=m[1];prop=m[2];val=e[k];}}
        }});
        if(src){{
            ['scene','scene2','scene3','scene4'].forEach(function(t){{
                if(t!==src)upd[t+prop]=val;
            }});
            if(Object.keys(upd).length){{sync=true;Plotly.relayout(g,upd).then(function(){{sync=false;}});}}
        }}
    }});
}}
attachCoupling();
</script></body></html>"""
        with open(base.with_suffix(".html"), "w", encoding="utf-8") as f:
            f.write(html_content)

    label = output_labels[output_idx] if output_labels and output_idx < len(output_labels) else output_idx
    print(f"[INFO] Saved {dim}D plots for output {label} to {base}")


def _plot_single_output(
    X_real_nd: np.ndarray,
    X_synthetic_nd: np.ndarray,
    y_real_plot: np.ndarray,
    y_synthetic_plot: np.ndarray,
    model_sr,
    labels: List[str],          # LaTeX labels with $...$ — used for 2D/1D plotly axes and matplotlib
    hover_labels: List[str],
    title_suffix: str,
    model_sr_folder: Path,
    fname_prefix: str,
    output_idx: int,
    n_outputs: int,
    plain_labels: Optional[List[str]] = None,  # Plain-text labels — used for 3D plotly scenes (no LaTeX support)
    ndim: int = 3,
    random_snapshots: Optional[list] = None,
    output_labels: Optional[List[str]] = None,
    selected_equation_idx: Optional[int] = None,
    ood_accuracy: float = float("nan"),
    shared_error_colorbar: bool = False,
    shared_xyz_bounds: bool = True,
):
    """Plot a single output in 1D/2D/3D embedding space with parity row (3×2 grid always)."""

    d = _compute_sr_plot_data(model_sr, X_real_nd, X_synthetic_nd, y_real_plot, y_synthetic_plot,
                               selected_equation_idx, output_idx, ood_accuracy)

    # Persist the parity data next to the plots. Downstream figure notebooks
    # (e.g. FiguresApp_I.ipynb) read these instead of re-running the model, so
    # they must be rewritten whenever the OOD probes are regenerated. Naming
    # follows _save_sr_plot(): the output suffix only appears for multi-output runs.
    _pkl_suffix = f"_output_{output_labels[output_idx]}" if (output_labels and n_outputs > 1) else ""
    _pkl_path = Path(model_sr_folder) / f"{fname_prefix}_sr_plot_data{_pkl_suffix}.pkl"
    with open(_pkl_path, "wb") as _f:
        pickle.dump(d, _f)

    error_real, error_syn = d["error_real"], d["error_syn"]

    cmax_real = float(np.abs(error_real).max()); cmin_real = -cmax_real
    cmax_syn  = float(np.abs(error_syn).max());  cmin_syn  = -cmax_syn
    if shared_error_colorbar:
        _cmax = max(cmax_real, cmax_syn); cmin_real = cmin_syn = -_cmax; cmax_real = cmax_syn = _cmax

    syn_title, parity_real_title, parity_syn_title = _sr_subplot_titles(d)

    # --- Figure layout: always 3×2 ---
    if ndim == 3:
        specs = [[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                 [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                 [{'type': 'xy'},        {'type': 'xy'}]]
    else:
        specs = [[{'type': 'xy'}, {'type': 'xy'}],
                 [{'type': 'xy'}, {'type': 'xy'}],
                 [{'type': 'xy'}, {'type': 'xy'}]]

    fig = make_subplots(
        rows=3, cols=2, specs=specs,
        subplot_titles=(
            f"Real Data (R²={d['r2_real']:.3f})", syn_title,
            "Error-colored Real Data", "Error-colored Synthetic Data",
            parity_real_title, parity_syn_title,
        ),
        horizontal_spacing=0.12, vertical_spacing=0.10,
    )

    # --- Rows 1 & 2: embedding subplots ---
    if ndim == 3:
        ht_label = (f"{hover_labels[0]}: %{{x:.4f}}<br>{hover_labels[1]}: %{{y:.4f}}<br>"
                    f"{hover_labels[2]}: %{{z:.4f}}<br>Label: %{{marker.color:.4f}}<extra></extra>")
        ht_error = (f"{hover_labels[0]}: %{{x:.4f}}<br>{hover_labels[1]}: %{{y:.4f}}<br>"
                    f"{hover_labels[2]}: %{{z:.4f}}<br>Error: %{{marker.color:.4f}}<extra></extra>")

        for (row, col), X, color, cs, cmin, cmax, cb_title, ht, name in [
            ((1, 1), X_real_nd,      y_real_plot.flatten(),  'Viridis', y_real_plot.min(),  y_real_plot.max(),  "Predicted Label", ht_label, "Real Data"),
            ((1, 2), X_synthetic_nd, y_synthetic_plot.flatten(), 'Cividis', y_synthetic_plot.min(), y_synthetic_plot.max(), "", ht_label, "Synthetic Data"),
            ((2, 1), X_real_nd,      error_real.flatten(),   'RdBu',    cmin_real,          cmax_real,          "y_pred - y_SR",   ht_error, "Real Error"),
            ((2, 2), X_synthetic_nd, error_syn.flatten(),    'RdBu',    cmin_syn,           cmax_syn,           "y_synt - y_SR",   ht_error, "Synthetic Error"),
        ]:
            fig.add_trace(go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2], mode='markers',
                marker=dict(size=4, color=color, colorscale=cs, showscale=True,
                            cmin=cmin, cmax=cmax, opacity=0.8,
                            colorbar=dict(len=0.3, y=0.83 if row == 1 else 0.50,
                                          x=1.02 if col == 1 else 1.12, title=cb_title)),
                name=name, showlegend=False, hovertemplate=ht,
            ), row=row, col=col)

        # Axis labels + optional shared bounds for 3D scenes
        _all_pts = np.concatenate([X_real_nd, X_synthetic_nd], axis=0)
        _xyz_ranges = [[float(_all_pts[:, k].min()), float(_all_pts[:, k].max())] for k in range(3)]
        for i in range(1, 3):
            for j in range(1, 3):
                _scene_labels = plain_labels if plain_labels is not None else labels
                scene_cfg = dict(xaxis_title=_scene_labels[0], yaxis_title=_scene_labels[1], zaxis_title=_scene_labels[2],
                                 camera=dict(eye=dict(x=2.0, y=2.0, z=1.8)))
                if shared_xyz_bounds:
                    scene_cfg["xaxis"] = dict(range=_xyz_ranges[0])
                    scene_cfg["yaxis"] = dict(range=_xyz_ranges[1])
                    scene_cfg["zaxis"] = dict(range=_xyz_ranges[2])
                fig.update_scenes(scene_cfg, row=i, col=j)

    elif ndim == 2:
        ht_label = (f"{hover_labels[0]}: %{{x:.4f}}<br>{hover_labels[1]}: %{{y:.4f}}<br>"
                    f"Label: %{{marker.color:.4f}}<extra></extra>")
        ht_error = (f"{hover_labels[0]}: %{{x:.4f}}<br>{hover_labels[1]}: %{{y:.4f}}<br>"
                    f"Error: %{{marker.color:.4f}}<extra></extra>")
        _shared_bounds = None
        if shared_xyz_bounds:
            _all = np.concatenate([X_real_nd, X_synthetic_nd], axis=0)
            _shared_bounds = [[float(_all[:, k].min()), float(_all[:, k].max())] for k in range(2)]

        # Colorbar anchors: col1→x≈0.46, col2→x≈1.02; row1→y≈0.83, row2→y≈0.50
        _cb_x = {1: 0.46, 2: 1.02}
        _cb_y = {1: 0.83, 2: 0.50}
        for (row, col), X, color, cs, cmin, cmax, cb_title, ht, name in [
            ((1, 1), X_real_nd,      y_real_plot.flatten(),      'Viridis', y_real_plot.min(),      y_real_plot.max(),      "Predicted Label", ht_label, "Real Data"),
            ((1, 2), X_synthetic_nd, y_synthetic_plot.flatten(), 'Cividis', y_synthetic_plot.min(), y_synthetic_plot.max(), "",                ht_label, "Synthetic Data"),
            ((2, 1), X_real_nd,      error_real.flatten(),       'RdBu',    cmin_real,              cmax_real,              "y_pred - y_SR",   ht_error, "Real Error"),
            ((2, 2), X_synthetic_nd, error_syn.flatten(),        'RdBu',    cmin_syn,               cmax_syn,               "y_synt - y_SR",   ht_error, "Synthetic Error"),
        ]:
            fig.add_trace(go.Scatter(
                x=X[:, 0], y=X[:, 1], mode='markers',
                marker=dict(size=4, color=color, colorscale=cs, showscale=True,
                            cmin=cmin, cmax=cmax, opacity=0.8,
                            colorbar=dict(title=cb_title, len=0.28,
                                          x=_cb_x[col], y=_cb_y[row])),
                name=name, showlegend=False, hovertemplate=ht,
            ), row=row, col=col)
            if _shared_bounds:
                fig.update_xaxes(range=_shared_bounds[0], row=row, col=col)
                fig.update_yaxes(range=_shared_bounds[1], row=row, col=col)
            fig.update_xaxes(title_text=labels[0], row=row, col=col)
            fig.update_yaxes(title_text=labels[1], row=row, col=col)

    else:  # ndim == 1
        x_label = labels[0] if labels else hover_labels[0]
        _shared_x = None
        if shared_xyz_bounds:
            _all_x = np.concatenate([X_real_nd[:, 0], X_synthetic_nd[:, 0]])
            _shared_x = [float(_all_x.min()), float(_all_x.max())]

        # Row 1: label-colored (NN scatter + SR line), Row 2: error-colored NN scatter
        for (row, col), X, y_nn, y_sr, err, cmin, cmax, name_nn in [
            ((1, 1), X_real_nd,      y_real_plot,      d["_y_real_sr"], error_real, cmin_real, cmax_real, "NN (real)"),
            ((1, 2), X_synthetic_nd, y_synthetic_plot, d["_y_syn_sr"],  error_syn,  cmin_syn,  cmax_syn,  "NN (synth)"),
            ((2, 1), X_real_nd,      y_real_plot,      d["_y_real_sr"], error_real, cmin_real, cmax_real, "NN (real)"),
            ((2, 2), X_synthetic_nd, y_synthetic_plot, d["_y_syn_sr"],  error_syn,  cmin_syn,  cmax_syn,  "NN (synth)"),
        ]:
            x = X[:, 0]
            sort_idx = np.argsort(x)
            is_label_row = (row == 1)
            marker_color = y_nn.flatten()[sort_idx] if is_label_row else err.flatten()[sort_idx]
            marker_cs    = 'Viridis' if is_label_row else 'RdBu'
            marker_cmin  = (y_nn.min() if col == 1 else y_synthetic_plot.min()) if is_label_row else cmin
            marker_cmax  = (y_nn.max() if col == 1 else y_synthetic_plot.max()) if is_label_row else cmax
            cb_title     = "Predicted Label" if is_label_row else "Error"
            ht_nn = f"{x_label}: %{{x:.4f}}<br>NN: %{{y:.4f}}<extra></extra>"

            fig.add_trace(go.Scatter(
                x=x[sort_idx], y=y_nn.flatten()[sort_idx], mode='markers',
                marker=dict(size=4, color=marker_color, colorscale=marker_cs, showscale=True,
                            cmin=marker_cmin, cmax=marker_cmax, opacity=0.7,
                            colorbar=dict(title=cb_title)),
                name=name_nn, showlegend=True,
                hovertemplate=ht_nn,
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=x[sort_idx], y=y_sr.flatten()[sort_idx], mode='lines',
                line=dict(color='black', width=2),
                name=name_nn.replace("NN", "SR"), showlegend=True,
                hovertemplate=f"{x_label}: %{{x:.4f}}<br>SR: %{{y:.4f}}<extra></extra>",
            ), row=row, col=col)
            if _shared_x:
                fig.update_xaxes(range=_shared_x, row=row, col=col)
            fig.update_xaxes(title_text=x_label, row=row, col=col)
            fig.update_yaxes(title_text="Output", row=row, col=col)

    # --- Row 3: parity (shared across all ndim) ---
    _add_parity_rows(fig, d, cmin_real, cmax_real, cmin_syn, cmax_syn, row=3)

    output_suffix = f" - Output {output_labels[output_idx]}" if n_outputs > 1 else ""
    width = 1600 if ndim == 3 else 1200
    fig.update_layout(
        title_text=f"{ndim}D Visualization ({title_suffix}){output_suffix}",
        height=1600, width=width,
        showlegend=(ndim == 1), margin=dict(r=200 if ndim == 3 else 150),
    )

    # For 3D: build matplotlib data bundle for LaTeX-rendered static export
    mpl_data = None
    if ndim == 3:
        syn_title_mpl, parity_real_mpl, parity_syn_mpl = _sr_subplot_titles(d)
        mpl_data = dict(
            X_real=X_real_nd, X_syn=X_synthetic_nd,
            y_real=y_real_plot, y_syn=y_synthetic_plot,
            error_real=error_real, error_syn=error_syn,
            y_real_sr=d["_y_real_sr"], y_syn_sr=d["_y_syn_sr"],
            labels=labels,  # LaTeX $...$ labels
            titles=(
                f"Real Data (R²={d['r2_real']:.3f})", syn_title_mpl,
                "Error-colored Real Data", "Error-colored Synthetic Data",
                parity_real_mpl, parity_syn_mpl,
            ),
            cmin_real=cmin_real, cmax_real=cmax_real,
            cmin_syn=cmin_syn,   cmax_syn=cmax_syn,
        )

    _save_sr_plot(fig, model_sr_folder, fname_prefix, output_labels, output_idx, n_outputs,
                  dim=ndim, title_suffix=title_suffix, coupled_cameras=(ndim == 3),
                  mpl_data=mpl_data)
