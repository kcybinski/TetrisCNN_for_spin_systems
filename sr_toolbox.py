"""Symbolic regression (optional) on an already trained TetrisCNN run.

Fits PySR to a trained network's bottleneck activations, or remakes the plots of an
earlier fit, without retraining. Needs the optional dependencies:
`pip install -r requirements-sr.txt`.

Edit the configuration below and run `python sr_toolbox.py`. Point `basepath` at a
run folder (containing net1.pt and net2.pt) to fit with the `sr_config` below, or at
one of its SR subfolders to reuse that subfolder's own sr_config.json.

The SR route, every SRConfig field, the output files and configuration recipes are
documented in docs/SYMBOLIC_REGRESSION.md. The code that runs it is in
tetriscnn/sr_experiments.py.
"""

# Keep this import first: PySR must start its Julia runtime before torch is loaded.
from tetriscnn.sr_experiments import SRConfig, run_sr_toolbox

from pathlib import Path


if __name__ == "__main__":

    remake_flags = {
        "fit_sr": True,         # fit a new PySR model from the network's activations
        "remake_sr_plot": True, # remake SR plots from an existing saved PySR model
    }

    # Used when basepath is a run folder. For an SR subfolder, its own sr_config.json is
    # used instead. See docs/SYMBOLIC_REGRESSION.md for every field and for recipes
    # (adding operators, overriding the folder name, changing the equation-picking metrics).
    sr_config = SRConfig(
        sr_mode="raw",
        top_k=3,
        gen_samples=2500,
        num_flips=4,
        use_tensorboard=True,
        classification_sr_mode="single_logit",  # "logit_diff" (default) or "single_logit" (binary classification only)
    )

    # None opens a directory picker.
    basepath = Path("logs/sl/small/lambdamax_Paris_XY_Z_partition_smallkernels_[-1]/lambdamax_-1/seed_44/partition_4")
    # An SR subfolder instead of a run folder:
    # basepath = Path("logs/singlerun_Paris_XY_Z_regression_delta_smallkernels_[-1]/singlerun_-1/seed_2137/SR_default")

    run_sr_toolbox(basepath, remake_flags, sr_config)
