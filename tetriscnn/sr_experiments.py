"""Symbolic-regression drivers: everything sr_toolbox.py runs around its configuration.

Optional: needs requirements-sr.txt. See docs/SYMBOLIC_REGRESSION.md.

``run_sr_toolbox`` is the entry point sr_toolbox.py calls. It inspects the selected
folder and dispatches to one of two scenarios:

* **A, an SR subfolder** (``seed_42/SR_default/``, contains ``sr_config.json``): the
  configuration is read from that file, the networks from the parent run folder,
  and results are written back into the selected subfolder.
* **B, a run folder** (``seed_42/``, contains ``net1.pt`` + ``net2.pt``): the
  ``SRConfig`` passed in is used, and the SR subfolder name is derived from it with
  ``build_sr_folder_path()``.
"""

# PySR boots a Julia runtime on import, which must happen before torch is loaded, so
# tetriscnn.symbolic_regression is imported before anything that imports torch.
from tetriscnn.symbolic_regression import run_sr_raw_mode, SRConfig

from pathlib import Path

import torch

from tetriscnn.datasets import create_datasets
from tetriscnn.models import ShapeAdaptiveConvNet, SmallModel
from tetriscnn.utils import AttrDict, DEVICE, build_sr_folder_path, load_json, pick_directory


def _is_seed_level_folder(path: Path) -> bool:
    """Returns True if path directly contains net1.pt and net2.pt."""
    return (path / "net1.pt").exists() and (path / "net2.pt").exists()


def _is_sr_subfolder(path: Path) -> bool:
    """Returns True if path directly contains sr_config.json (an SR subfolder)."""
    return (path / "sr_config.json").exists()


def _load_config_from_seed_folder(seed_folder: Path) -> AttrDict:
    """Load config.json from a seed-level folder and return a populated AttrDict.

    Applies the stride-compatibility fix for kernels.
    Sets cf.logdir to the seed folder path.
    """
    cfg_path = seed_folder / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json found in {seed_folder}")

    cf = AttrDict()
    cf.update(load_json(str(seed_folder), "config.json"))

    # Compatibility fix for kernels without stride specified (legacy runs)
    for k in range(len(cf.kernels)):
        if len(cf.kernels[k]) < 5:
            cf.kernels[k].append(1)

    cf.logdir = str(seed_folder)
    return cf


def _run_sr_for_seed(cf, seed_folder: Path, net1=None, net2=None):
    """Run SR for a single seed directory."""
    # Update cf to point to this seed
    original_logdir = cf.logdir if 'logdir' in cf.keys() else None
    original_seed_folder = cf.seed_folder if 'seed_folder' in cf.keys() else None
    cf.logdir = str(seed_folder)
    cf.seed_folder = str(seed_folder)

    # SR always fits on the train split and evaluates on val (see run_sr_raw_mode()).
    train_dataset, val_dataset = create_datasets(cf)

    # Load models if not provided
    if net1 is None or net2 is None:
        in_channels = train_dataset[0][0].shape[0]
        net2_dims = [len(cf.kernels), 32, 16, train_dataset.output_dim]

        net1_new = ShapeAdaptiveConvNet(
            in_channels=in_channels,
            kernels=cf.kernels,
            equivariant=cf.equivariant,
            equivariant_group=getattr(cf, "equivariant_group", "C4"),
            hidden_size=cf.hidden_size,
            init=cf.init,
            device=DEVICE
        ).to(DEVICE)

        net2_new = SmallModel(net2_dims, device=DEVICE).to(DEVICE)

        try:
            net1_new.load_state_dict(torch.load(seed_folder / "net1.pt", map_location=DEVICE))
            net2_new.load_state_dict(torch.load(seed_folder / "net2.pt", map_location=DEVICE))
            net1_new.eval()
            net2_new.eval()
        except FileNotFoundError:
            print(f"[ERROR] Model weights not found in {seed_folder}. Skipping SR.")
            # Restore original cf
            if original_logdir:
                cf.logdir = original_logdir
            if original_seed_folder:
                cf.seed_folder = original_seed_folder
            return

        net1 = net1_new
        net2 = net2_new

    # Dispatch based on sr_mode
    if cf.sr_config.sr_mode == "raw":
        run_sr_raw_mode(
            cf, net1, net2,
            train_dataset, val_dataset,
            cf.sr_config,
            train_dataset.output_dim
        )
    else:
        raise ValueError(f"Unknown sr_mode: {cf.sr_config.sr_mode}")


def _run_scenario_a(sr_folder: Path, fit_sr: bool, remake_sr_plot: bool):
    """Scenario A: user picked an SR subfolder (e.g. seed_42/SR_default/).

    Loads sr_config from the folder's sr_config.json.
    NN models are loaded from the parent (seed-level) folder.
    """
    seed_folder = sr_folder.parent

    if not _is_seed_level_folder(seed_folder):
        print(f"[ERROR] Parent of selected SR subfolder is not a seed folder: {seed_folder}")
        print("[ERROR] Expected net1.pt + net2.pt to be present there.")
        return

    cf = _load_config_from_seed_folder(seed_folder)
    cf.sr_config = SRConfig.load(sr_folder / "sr_config.json")

    # Force run_sr_raw_mode() to write into the exact folder the user picked,
    # not a freshly-computed path that might differ.
    cf.sr_config.sr_folder_name_override = sr_folder.name

    cf.fit_sr = fit_sr
    # fit_sr=True must bypass run_sr_raw_mode()'s "model not found" guard,
    # which only allows fitting when remake_sr_plot=True.
    cf.remake_sr_plot = remake_sr_plot or fit_sr

    if remake_sr_plot and not fit_sr:
        pkl_path = sr_folder / "model_sr_raw.pkl"
        if not pkl_path.exists():
            print(f"[ERROR] remake_sr_plot=True but no model_sr_raw.pkl found in {sr_folder}.")
            print("[ERROR] Set fit_sr=True to refit from scratch.")
            return

    print(f"[SR Toolbox] Scenario A — SR subfolder: {sr_folder}")
    _run_sr_for_seed(cf, seed_folder)


def _run_scenario_b(seed_folder: Path, fit_sr: bool, remake_sr_plot: bool, sr_config: SRConfig):
    """Scenario B: user picked a seed-level folder (e.g. seed_42/).

    Uses the sr_config passed in (the one configured in sr_toolbox.py).
    Computes the target SR subfolder via build_sr_folder_path().
    """
    cf = _load_config_from_seed_folder(seed_folder)
    cf.sr_config = sr_config
    cf.fit_sr = fit_sr
    # fit_sr=True must bypass run_sr_raw_mode()'s "model not found" guard,
    # which only allows fitting when remake_sr_plot=True.
    cf.remake_sr_plot = remake_sr_plot or fit_sr

    target_sr_folder = build_sr_folder_path(seed_folder, sr_config)

    if not target_sr_folder.exists():
        if remake_sr_plot and not fit_sr:
            print(f"[ERROR] remake_sr_plot=True but SR subfolder does not exist: {target_sr_folder}")
            print("[ERROR] Set fit_sr=True to create it.")
            return
        # fit_sr=True: subfolder will be created by run_sr_raw_mode
        print(f"[INFO] SR subfolder does not exist yet, will be created: {target_sr_folder}")
    else:
        if remake_sr_plot and not fit_sr:
            pkl_path = target_sr_folder / "model_sr_raw.pkl"
            if not pkl_path.exists():
                print(f"[ERROR] SR subfolder exists but no model_sr_raw.pkl found: {pkl_path}")
                print("[ERROR] Set fit_sr=True to refit.")
                return

    print(f"[SR Toolbox] Scenario B — seed folder: {seed_folder}")
    print(f"  Target SR subfolder: {target_sr_folder}")
    _run_sr_for_seed(cf, seed_folder)


def run_sr_toolbox(basepath, remake_flags: dict, sr_config: SRConfig):
    """Fit SR and/or remake its plots for one trained run.

    Args:
        basepath: a run folder (Scenario B) or one of its SR subfolders (Scenario A).
            None, or a path that does not exist, opens a directory picker.
        remake_flags: {"fit_sr": bool, "remake_sr_plot": bool}.
        sr_config: the SRConfig used in Scenario B (ignored in Scenario A, which reads
            the subfolder's own sr_config.json).
    """
    if basepath is None or not Path(str(basepath)).exists():
        basepath = Path(pick_directory(title="Select seed folder or SR subfolder"))
    else:
        basepath = Path(basepath)
    print(f"[SR Toolbox] Selected: {basepath}")

    fit_sr = remake_flags.get("fit_sr", False)
    remake_sr_plot = remake_flags.get("remake_sr_plot", False)

    if not fit_sr and not remake_sr_plot:
        print("[WARN] Both fit_sr and remake_sr_plot are False. Nothing to do.")
        return

    if _is_sr_subfolder(basepath):
        _run_scenario_a(basepath, fit_sr, remake_sr_plot)
    elif _is_seed_level_folder(basepath):
        _run_scenario_b(basepath, fit_sr, remake_sr_plot, sr_config)
    else:
        print(f"[ERROR] Could not determine scenario from selected folder: {basepath}")
        print("[ERROR] Please select either:")
        print("  - A seed-level folder containing net1.pt + net2.pt  (Scenario B)")
        print("  - An SR subfolder containing sr_config.json         (Scenario A)")
