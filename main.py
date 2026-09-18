"""TetrisCNN training entry point.

Describe the experiment in setup_experiment() below and run `python main.py`. To
re-plot runs that already exist instead of training, set one of the remake flags in
the __main__ block. Every configuration field is documented in docs/CONFIGURATION.md.

The machinery that runs the experiment (sweeps, seeds, learning-by-confusion
partitions, re-plotting) lives in tetriscnn/experiments.py.
"""
from pathlib import Path  # noqa: F401  (for the commented-out basepath examples below)

from tetriscnn.experiments import run_or_remake
from tetriscnn.utils import set_kernels, set_plotting_logging_strings


def setup_experiment(cf):
    cf.data_fraction = 1
    cf.label_subset_count = None   # number between 0 and the max number of unique labels; None means all labels
    cf.param_cutoffs = {"delta" : 6, "omega" : 0}         # NB: for XZ, delta = 11 gives divergence

    # === Even Split Configuration ===
    # When enabled, ensures balanced sampling across all time points for train/val/test splits
    cf.even_split = True           # Enable even split mode (group by time point before splitting)
    cf.samples_per_pt_cap = None    # Cap on samples per time point (None = no cap, int = cap at min(value, min_samples_across_timepoints))
    # cf.samples_per_pt_cap = [2,3,4,5,6,7,8,9,10,15,20, 25, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350]    # Cap samples per time point

    cf.experiment_name = "singlerun"  # choose between "singlerun", "lambdamax", "lambdatot" or "weighted_loss"

    # === Weighted-loss experiment ===
    # Only active when cf.experiment_name == "weighted_loss": compares 4 data-splitting /
    # loss-weighting scenarios (even_split x samples_per_pt_cap x use_weighted_loss), run in
    # parallel via zip (see run_experiments() in tetriscnn/experiments.py; full scenario table and the
    # inverse-frequency weight formula are in docs/CONFIGURATION.md, "Weighted-loss
    # experiment"). Write it as a dict keyed by scenario, not as three raw parallel
    # lists: with raw lists, cf.even_split[i]/cf.samples_per_pt_cap[i]/
    # cf.use_weighted_loss[i] have to be kept aligned by POSITION across three separate
    # lines, which is exactly the error-prone "repeated sweep" bookkeeping this
    # zip-based mechanism exists to avoid. The dict form below produces the identical
    # lists, just derived from one readable source instead of typed three times:
    #   scenarios = {
    #       "(a)": {"even_split": True,  "samples_per_pt_cap": None, "use_weighted_loss": False},
    #       "(b)": {"even_split": True,  "samples_per_pt_cap": 500,  "use_weighted_loss": False},
    #       "(c)": {"even_split": False, "samples_per_pt_cap": None, "use_weighted_loss": False},
    #       "(d)": {"even_split": False, "samples_per_pt_cap": None, "use_weighted_loss": True},
    #   }
    #   cf.even_split         = [scenarios[s]["even_split"] for s in scenarios]
    #   cf.samples_per_pt_cap = [scenarios[s]["samples_per_pt_cap"] for s in scenarios]
    #   cf.use_weighted_loss  = [scenarios[s]["use_weighted_loss"] for s in scenarios]
    #   cf.lambdas = [4]; cf.seeds = [42, 123, 456, 789, 101112, 131415]
    #   cf.visualize_sweep = True

    cf.kernel_set = "smallkernels"  # bigkernels or smallkernels or default
    cf.equivariant = True
    # Symmetry group used when cf.equivariant is True (ignored otherwise). Choose
    # between "C4" (the four 90-degree rotations, natural for the square Ising
    # lattice) and "D2"/"K4" (aliases for the rectangle/Klein-four group: 180-degree
    # rotation plus the two mirrors, which suits the rectangular XY lattice). C4
    # collapses smallkernels to 5 branches, D2/K4 to 6, from the 10 non-equivariant
    # ones. Equivariant runs are only implemented for cf.kernel_set = "smallkernels".
    # The manuscript uses C4 for both datasets ("rotationally invariant TetrisCNN").
    cf.equivariant_group = "C4"

    # === Parameter Sweep Visualization ===
    # When enabled, creates a plot showing metrics vs swept parameter (for single-parameter sweeps only)
    cf.visualize_sweep = False

    # cf.lambdas = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]  # for lambdamax and lambdatot
    # cf.lambdas = [-5, -4, -3]  # for lambdamax and lambdatot
    # cf.lambdas = [1]  # for singlerun
    cf.lambdas = [3]  # for singlerun

    cf.seeds = [2137]
    # cf.seeds = [46]
    # cf.seeds = [42, 43]
    # cf.seeds = [42, 43, 44]
    # cf.seeds = [42, 43, 44, 45, 46] 

    cf.dataset = "Paris_XY_XZ"  # choose between "Paris_Ising", "Paris_XY_X", "Paris_XY_Z", "Paris_XY_XZ", "ILGT"
    cf.task = "classification"    # choose between "regression", "classification", "partition", "lbc"
    # For classification, this is auto-resolved by create_datasets(): the Paris datasets
    # get the manuscript's hardcoded transition-flanking index (see
    # tetriscnn.datasets.CLASSIFICATION_PARTITION_INDEX); for any other dataset it must
    # be set explicitly here, since no canonical transition point is known. For "lbc" /
    # "partition" it must always be set explicitly. See docs/CONFIGURATION.md.
    cf.partition_index = None

    if cf.task == "regression":
        cf.label_param = "delta"  # ISING: 't', 'delta', 'omega', 'deltaomega'; XY: 't' 'delta', ILGT: 'beta'
        cf.normalize_labels = True     # normalize labels to [0, 1] range, only for regression
        cf.goodness_str = "r2agg"  # choose between r2 and r2agg
    else:
        cf.label_param = "t"
        print(f"Label param set to 't' for non-regression tasks.")
    
    cf.save_models = True      # save trained model weights
    cf.save_histories = True    # plot history and phase transitions for each run
    cf.save_final_values = True if (cf.experiment_name == "singlerun" and cf.task != "lbc") else False  # plot final values across lambdas

    # cf.filter_t_values = [1200., 1400., 1600., 1800., 2000., 2200.,
    #     2400., 2600., 2800., 3000., 3200., 3400., 3600., 3800., 4000.,
    #     4200., 4400., 4600., 4800., 5000., 5200., 5400., 5600., 5800.,
    #     6000.]
    cf.filter_t_values = None 
        
    cf.epochs = 250
    cf.learning_rate = 1e-2       # Was 1e-3, good with no weight penalty
    cf.weight_decay = 1e-5        # L2 regularization on weights
    cf.patience = 10              # Early stopping patience, in epochs
    cf.early_stop_warmup = 150     # Early stopping warmup period (epochs, monitoring disabled)
    cf.early_stop_min_delta = 1e-5  # Minimum improvement in validation objective to reset patience
    cf.init = "kaiming"           # weight initialization method, only "kaiming" is implemented so far

    # "tetriscnn" is the interpretable architecture this repository is about. "PhaseCNN"
    # and "ResNet18" are ordinary (non-interpretable) CNN baselines, kept for the method
    # comparison: they train through the same loop but have no bottleneck to read
    # correlators off, so the branch/sparsity analysis does not apply to them.
    cf.model = "tetriscnn"  # choose between "tetriscnn", "PhaseCNN" and "ResNet18"
    cf.cnn_kernel_shape = (3, 3)    # PhaseCNN kernel — only used when model == "PhaseCNN"

    cf.num_workers = 0
    cf.pin_memory = False
    cf.batch_size = 64
    cf.VRAM_batch_size = 1024  # load this many samples onto GPU, then micro-batch at cf.batch_size

    if cf.model == "tetriscnn":
        cf.hidden_size = 32         # number of channels between conv1 and conv2, determines expressivity of the model.
        set_kernels(cf)

        # === Learning rate scheduling ===
        # Addresses validation loss volatility by reducing LR as training progresses:
        # initial exploration at a high LR, then convergence at a low one. Only
        # ReduceLROnPlateau is normally used, so only its parameters need to be set
        # here. Switching to "cosine_annealing" / "step" / "exponential" / "onecycle"
        # is a one-line change (cf.lr_scheduler_type below): the other types' own
        # parameters are filled in automatically from sensible defaults in
        # tetriscnn.train.LR_SCHEDULER_DEFAULTS (edit them there to override).
        cf.use_lr_scheduler = True
        cf.lr_scheduler_type = "reduce_on_plateau"  # "reduce_on_plateau" | "cosine_annealing" | "step" | "exponential" | "onecycle"
        cf.lr_reduce_factor = 0.5      # multiply LR by this when a plateau is detected (0.5 = halve LR)
        cf.lr_reduce_patience = 5      # epochs to wait before reducing LR
        cf.min_lr = 1e-9               # minimum learning rate (stop reducing below this)

        # Experimental: additional L1-regularization on branch conv1 weights, off by
        # default. Read by tetriscnn/train.py; set to a float to enable.
        cf.weight_penalty = None

    set_plotting_logging_strings(cf)


if __name__ == "__main__":

    # All False: train the experiment described by setup_experiment(). Set one to True
    # to re-plot runs already on disk under `basepath` instead, each from its own
    # config.json (see docs/CONFIGURATION.md, "Remake flags and re-plotting").
    remake_flags = {
        "remake_lambda_plot": False,
        "remake_history_and_pt_plots": True,
        "remake_sweep_plot": False,
    }

    plot_config = {
        "enabled_metrics": {"z", "loss", "goodness"},
        "fit_branches": True,   # BFA-fit and plot every active branch; see docs/CONFIGURATION.md
    }

    # basepath = Path("logs/singlerun_Paris_XY_Z_regression_delta_smallkernels_[-1]/singlerun_-1_w_all_smart_constraints")
    # basepath = Path("logs/singlerun_Paris_XY_Z_regression_delta_smallkernels_[-1]/singlerun_-1_baseline_real")
    # basepath = Path("logs/singlerun_Paris_Ising_regression_deltaomega_smallkernels_[4]")
    basepath = None
    # basepath = Path("logs/logs_after_bugfix/lambdamax_Paris_Ising_partition_t_smallkernels_C4_[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]/")

    run_or_remake(setup_experiment, remake_flags, plot_config, basepath)
