import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score  # the regression goodness metric (cf.goodness_str == "r2")
from tqdm import tqdm
from datetime import datetime
import pprint
import gc

from tetriscnn.models import *
from tetriscnn.utils import *
from tetriscnn.datasets import *
from tetriscnn.plots import *


# Default parameters for each learning-rate scheduler type. setup_experiment() in
# main.py only has to set the parameters for the *active* cf.lr_scheduler_type;
# everything else is filled in here, so switching types is a one-line change there.
#
# Keys stay flat cf.<name> attributes (not nested in a dict/dataclass) so they:
#   (1) round-trip losslessly through save_json/load_json (save_json stringifies any
#       non-JSON-native value, so a dataclass instance on cf would NOT survive a
#       save/reload cycle -- see tetriscnn/utils.py:save_json's _json_safe helper);
#   (2) stay visible to the sweep machinery in tetriscnn/utils.py
#       (PARAM_ABBREVIATIONS / get_sweepable_params), which discovers sweepable
#       parameters by scanning cf's own top-level list-valued attributes.
#
# `min_lr` is deliberately given its own default under each of "reduce_on_plateau"
# and "cosine_annealing" rather than being assigned once at module scope: the old
# setup_experiment() wrote a bare `cf.min_lr = ...` twice in a row (once per
# scheduler), so the second write silently clobbered the first regardless of which
# scheduler was actually active. Applying defaults per-type via build_lr_scheduler()
# below (only filling in a key when it is not already set) removes that collision.
LR_SCHEDULER_DEFAULTS = {
    "reduce_on_plateau": {
        "lr_reduce_factor": 0.5,    # multiply LR by this when a plateau is detected
        "lr_reduce_patience": 5,    # epochs to wait before reducing LR
        "min_lr": 1e-9,             # stop reducing below this
    },
    "cosine_annealing": {
        "min_lr": 1e-8,             # LR at the end of the cosine cycle
    },
    "step": {
        "lr_step_size": 30,         # reduce LR every N epochs
        "lr_step_gamma": 0.5,       # multiply LR by this factor
    },
    "exponential": {
        "lr_exp_gamma": 0.85,       # multiply LR by this every epoch
    },
    "onecycle": {
        # Ramps LR up from initial_lr to max_lr, then down to final_lr;
        # div_factor/final_div_factor are derived from these in build_lr_scheduler().
        "onecycle_initial_lr": 1e-4,
        "onecycle_max_lr": 1e-2,
        "onecycle_final_lr": 1e-8,
        "onecycle_pct_start": 0.3,  # fraction of the cycle spent increasing LR
    },
}


def build_lr_scheduler(cf, optimizer, train_loader):
    """Builds the torch LR scheduler selected by cf.lr_scheduler_type.

    Any parameter the active type needs that isn't already set on cf (e.g. an old
    config.json saved with only one type's parameters, or a fresh setup_experiment()
    that only configured a different type) is filled in from LR_SCHEDULER_DEFAULTS.
    Explicit cf.<param> values (whether set by the caller or loaded from a saved
    config.json) always take precedence over these defaults.
    """
    scheduler_type = cf.lr_scheduler_type
    for key, value in LR_SCHEDULER_DEFAULTS.get(scheduler_type, {}).items():
        cf.setdefault(key, value)

    if scheduler_type == "cosine_annealing":
        # Cosine annealing: smoothly reduce LR from initial to min_lr
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cf.epochs, eta_min=cf.min_lr
        )
    elif scheduler_type == "reduce_on_plateau":
        # Reduce LR when validation loss plateaus (adaptive)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cf.lr_reduce_factor,
            patience=cf.lr_reduce_patience, min_lr=cf.min_lr
        )
    elif scheduler_type == "step":
        # Step decay: reduce LR at fixed intervals
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=cf.lr_step_size, gamma=cf.lr_step_gamma
        )
    elif scheduler_type == "exponential":
        # Exponential decay
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cf.lr_exp_gamma
        )
    elif scheduler_type == "onecycle":
        # OneCycleLR: cyclical learning rate with ramp-up and ramp-down
        # Computes div_factor and final_div_factor from user-provided initial, max, and final LRs
        div_factor = cf.onecycle_max_lr / cf.onecycle_initial_lr
        final_div_factor = cf.onecycle_initial_lr / cf.onecycle_final_lr

        # steps_per_epoch = number of batches per epoch
        steps_per_epoch = len(train_loader)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cf.onecycle_max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=cf.epochs,
            pct_start=cf.onecycle_pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )

        print(f"OneCycleLR: initial_lr={cf.onecycle_initial_lr:.2e}, max_lr={cf.onecycle_max_lr:.2e}, final_lr={cf.onecycle_final_lr:.2e}")
        print(f"  div_factor={div_factor:.2f}, final_div_factor={final_div_factor:.2f}")
    else:
        raise ValueError(f"Unknown lr_scheduler_type: {scheduler_type}")

    print(f"Using LR scheduler: {scheduler_type}")
    return scheduler


# @profile
def train(cf, filename):
    # Keys left unset fall back to tetriscnn.utils.CONFIG_DEFAULTS; cf.train_dataset,
    # cf.val_dataset (from create_datasets) and cf.logdir must be provided.
    apply_config_defaults(cf, for_training=True)
    set_seeds(cf.seed)

    cf.logdir = create_path(cf.logdir, overwrite=False)
    save_json(cf, cf.logdir, "config.json")

    print(f"\nTRAINING {cf.logdir}: --seed {cf.seed} --device {DEVICE}\n")
    pprint.pprint(cf)

    start_time = datetime.now()

    # MPS (Apple Silicon GPU) doesn't support multiprocessing with num_workers > 0
    # Force num_workers=0 and pin_memory=False for MPS to avoid errors
    num_workers = cf.num_workers
    pin_memory = cf.pin_memory
    if DEVICE.type == "mps":
        if num_workers > 0:
            print(f"Warning: MPS device detected. Setting num_workers=0 (was {num_workers}) to avoid multiprocessing issues.")
            num_workers = 0
        if pin_memory:
            print(f"Warning: MPS device detected. Setting pin_memory=False to avoid memory transfer issues.")
            pin_memory = False

    # Weighted loss (the weighted_loss experiment, scenario d) reweights each sample's
    # contribution to the loss by an importance weight attached to the dataset. It is
    # applied to whichever loss the task uses (cross-entropy or MSE), not only regression.
    use_weighted_loss = ("use_weighted_loss" in cf.keys() and cf.use_weighted_loss) \
        and getattr(cf.train_dataset, "sample_weights", None) is not None

    if use_weighted_loss:
        # Carry indices through the loader so weights stay aligned with the shuffled samples.
        sample_weights = cf.train_dataset.sample_weights.to(DEVICE)
        train_loader = DataLoader(IndexedDataset(cf.train_dataset), batch_size=cf.batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    else:
        sample_weights = None
        train_loader = DataLoader(cf.train_dataset, batch_size=cf.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(cf.val_dataset, batch_size=len(cf.val_dataset), shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    # batch_size = len(cf.val_dataset) is required to compute goodness_metric correclty

    # TASK-DEPENDENT SETTINGS; LOSS & METRIC (ACCURACY OR R2)
    per_sample_loss = None  # non-reducing loss used only on the weighted path
    if cf.task in ("classification", "lbc", "partition"):
        assert cf.train_dataset.discrete_labels, "Dataset must have discrete phase labels for classification task."
        goodness_function = lambda out, y: torch.mean((torch.argmax(out.data, axis=1) == torch.argmax(y, axis=1)).float())
        if use_weighted_loss:
            criterion = None
            per_sample_loss = nn.CrossEntropyLoss(reduction="none")
            print("[Weighted Loss] Using weighted cross-entropy loss (sample weights from dataset)")
        else:
            criterion = nn.CrossEntropyLoss()
    elif cf.task == "regression":
        assert not cf.train_dataset.discrete_labels, "Dataset must not have discrete phase labels for regression task."
        if use_weighted_loss:
            criterion = None  # weighted MSE computed inline per batch
            print("[Weighted Loss] Using weighted MSE loss (sample weights from dataset)")
        else:
            criterion = nn.MSELoss()

        if cf.goodness_str == "r2agg":
            goodness_function = r2_agg
        elif cf.goodness_str == "r2":
            goodness_function = r2_score
        else:
            raise ValueError(f"Unknown goodness metric: {cf.goodness_str}. Use 'r2' or 'r2agg'.")
    else:
        raise ValueError(f"Unknown task: {cf.task}. Use 'regression' or 'classification'.")
    
    # MODEL DEFINITION
    in_channels = cf.train_dataset[0][0].shape[0]

    if getattr(cf, 'model', 'tetriscnn') != "tetriscnn":
        model_cls = globals()[cf.model]
        net1 = model_cls(
            in_channels=in_channels,
            kernel_shape=cf.cnn_kernel_shape,
            output_dim=cf.train_dataset.output_dim,
        ).to(DEVICE)
        net2 = None
    else:
        # Branches whose receptive field overruns the snapshot fail deep inside
        # torch's conv2d with a message that names neither the branch nor the
        # dataset; check here so the error can say both.
        validate_kernels_fit(cf.kernels, cf.train_dataset[0][0].shape[1:],
                             kernel_set=getattr(cf, "kernel_set", None))

        net2_dims = [len(cf.kernels), 32, 16, cf.train_dataset.output_dim]
        # Net1: Feature extractor
        net1 = ShapeAdaptiveConvNet(
            in_channels=in_channels,
            kernels=cf.kernels,
            equivariant=cf.equivariant,
            equivariant_group=getattr(cf, "equivariant_group", "C4"),
            # None for every ordinary (uniform) run; a per-branch list only for the
            # mixed "smallkernels_mixed" set.
            branch_groups=getattr(cf, "branch_groups", None),
            hidden_size=cf.hidden_size,
            init=cf.init,
            device=DEVICE
        ).to(DEVICE)
        # Net2: Readout network
        net2 = SmallModel(net2_dims, device=DEVICE).to(DEVICE)

    # OPTIMIZER
    _params = list(net1.parameters()) + (list(net2.parameters()) if net2 is not None else [])
    optimizer = optim.AdamW(_params, lr=cf.learning_rate, weight_decay=cf.weight_decay)

    # LEARNING RATE SCHEDULER
    scheduler = None
    if 'use_lr_scheduler' in cf.keys() and cf.use_lr_scheduler:
        scheduler = build_lr_scheduler(cf, optimizer, train_loader)

    if net2 is not None:
        if cf.experiment_name in ["lambdamax", "singlerun", "weighted_loss"]:
            penalties = get_branch_penalties(
                cf.penalty_params, cf.kernels,
                branch_groups=getattr(cf, "branch_groups", None),
                equivariant_rebate=getattr(cf, "equivariant_rebate", None),
                branch_orbit_sizes=getattr(cf, "branch_orbit_sizes", None),
            )
        elif cf.experiment_name == "lambdatot":
            penalties = np.array([10**cf.lam for _ in range(len(cf.kernels)) ])
        else:
            raise ValueError(f"Unknown experiment_name {cf.experiment_name!r}; choose "
                             f"'singlerun', 'lambdamax', 'lambdatot' or 'weighted_loss'.")
        print(f"penalties: {penalties}")
        penalties = torch.tensor(penalties).float().to(DEVICE)
    else:
        penalties = None

    early_stopper = EarlyStopper(patience=cf.patience, min_delta=cf.early_stop_min_delta,
                                  warmup_epochs=cf.early_stop_warmup)

    metrics = {f"train_{cf.loss_str}": [], "train_l1": [], f"train_{cf.goodness_str}": [],
               f"val_{cf.loss_str}": [], "val_l1": [], f"val_{cf.goodness_str}": [], 
               "unique_labels" : [], "MVUL_out" : [], "MVUL_std_out" : [],
                "pt":[], "pt2":[], "train_time": [], "net2_norm": [], 
               "learning_rate": []}

    # Add learning rate tracking
    if 'use_lr_scheduler' in cf.keys() and cf.use_lr_scheduler:
        metrics["learning_rate"] = []
            
    if net2 is not None:
        for k in range(len(cf.kernels)):
            metrics[f"z_{k}"] = [] # bottleneck activations
            metrics[f"MVUL_{k}"] = []
            metrics[f"MVUL_std_{k}"] = []

    # Helper function for VRAM batching
    def create_vram_batches(dataloader, vram_batch_size, device):
        """
        Groups DataLoader batches into VRAM-sized chunks.
        Yields (vram_x, vram_y, vram_w) tuples with up to vram_batch_size samples on GPU;
        vram_w holds the per-sample weights when the weighted loss is active, else None.
        """
        vram_x_batch, vram_y_batch, vram_w_batch = [], [], []
        current_size = 0

        for batch in dataloader:
            if use_weighted_loss:
                x, y, idx = batch
                vram_w_batch.append(sample_weights[idx.to(device)])
            else:
                x, y = batch
            vram_x_batch.append(x)
            vram_y_batch.append(y)
            current_size += x.size(0)

            if current_size >= vram_batch_size:
                # Concatenate and move to GPU once
                vram_x = torch.cat(vram_x_batch, dim=0).to(device)
                vram_y = torch.cat(vram_y_batch, dim=0).to(device)
                vram_w = torch.cat(vram_w_batch, dim=0) if vram_w_batch else None
                yield vram_x, vram_y, vram_w

                vram_x_batch, vram_y_batch, vram_w_batch = [], [], []
                current_size = 0

        # Yield remaining samples
        if vram_x_batch:
            vram_x = torch.cat(vram_x_batch, dim=0).to(device)
            vram_y = torch.cat(vram_y_batch, dim=0).to(device)
            vram_w = torch.cat(vram_w_batch, dim=0) if vram_w_batch else None
            yield vram_x, vram_y, vram_w

    # Check if VRAM batching is enabled
    use_vram_batching = hasattr(cf, 'VRAM_batch_size') and cf.VRAM_batch_size > cf.batch_size

    # TRAINING LOOP
    print(f"\nTraining for {cf.epochs} epochs with {len(train_loader)} training batches and {len(val_loader)} validation batches.")
    if use_vram_batching:
        print(f"VRAM batching enabled: loading {cf.VRAM_batch_size} samples to GPU, processing in micro-batches of {cf.batch_size}")
    if cf.early_stop_warmup > 0:
        print(f"Early stopping warmup: {cf.early_stop_warmup} epochs (monitoring disabled)")
    print(f"Early stopping patience: {cf.patience} epochs")
    if scheduler is not None and cf.lr_scheduler_type == "onecycle":
        print(f"Note: OneCycleLR scheduler steps per batch, not per epoch")
    print()
    def _step(x, y, sample_w=None):
        """One forward+backward step. sample_w (per-sample weights) enables the weighted loss."""
        optimizer.zero_grad()

        if net2 is not None:
            z   = net1(x)
            out = net2(z)
            l1_per_sample = l1_regularization(z, penalties)   # (batch,): penalty per sample
            # The LOSS uses the per-sample MEAN, so the penalty is weighted against the
            # mean-reduced data-fit loss independently of batch size. (Previously this was
            # a sum over the batch, which made the effective penalty scale with batch size.)
            l1  = l1_per_sample.mean()
        else:
            z   = None
            out = net1(x)
            l1_per_sample = None
            l1  = torch.tensor(0.0, device=DEVICE)

        if criterion is not None:
            loss_base = criterion(out, y)
        elif per_sample_loss is not None:                    # weighted cross-entropy
            loss_base = (per_sample_loss(out, y) * sample_w).mean()
        elif sample_w is not None:                           # weighted MSE
            loss_base = ((out - y) ** 2 * sample_w.view(-1, 1)).mean()
        else:
            loss_base = nn.functional.mse_loss(out, y)      # VRAM path fallback (no weighted loss)

        loss = loss_base + l1
        if net2 is not None and cf.weight_penalty is not None:
            all_params = torch.cat([p.view(-1)
                for branch in net1.branches
                for p in branch.conv1.parameters()          # type: ignore
            ])
            loss += cf.weight_penalty * torch.sum(torch.abs(all_params))

        loss.backward()
        optimizer.step()
        if scheduler is not None and cf.lr_scheduler_type == "onecycle":
            scheduler.step()                                # type: ignore

        total_loss.add_(loss_base.detach())
        # The train_l1 METRIC (not the gradient) accumulates the per-sample penalty SUM,
        # so the epoch value total_l1 / len(dataset) is a per-sample average. This does not
        # affect the loss, which uses the mean above.
        if l1_per_sample is not None:
            total_l1.add_(l1_per_sample.sum().detach())
        train_outputs.append(out.detach())
        train_targets.append(y.detach())

    pbar = tqdm(range(cf.epochs))
    for epoch in pbar:
        net1.train()
        if net2 is not None:
            net2.train()

        total_loss = torch.tensor(0.0, device=DEVICE)
        total_l1   = torch.tensor(0.0, device=DEVICE)
        train_outputs, train_targets = [], [] # for goodness metric calculation

        if use_vram_batching:
            # Two-tier: load VRAM_batch_size samples to GPU, then micro-batch at cf.batch_size
            for vram_x, vram_y, vram_w in create_vram_batches(train_loader, cf.VRAM_batch_size, DEVICE):
                n = vram_x.size(0)
                for start in range(0, n, cf.batch_size):
                    x = vram_x[start : start + cf.batch_size]
                    y = vram_y[start : start + cf.batch_size].view(-1, cf.train_dataset.output_dim)
                    sample_w = vram_w[start : start + cf.batch_size] if vram_w is not None else None
                    _step(x, y, sample_w=sample_w)
        else:
            for batch in train_loader:
                if use_weighted_loss:
                    x, y, idx = batch
                    sample_w = sample_weights[idx.to(DEVICE)]
                else:
                    x, y = batch
                    sample_w = None
                x, y = x.to(DEVICE), y.to(DEVICE).view(-1, cf.train_dataset.output_dim)
                _step(x, y, sample_w=sample_w)

        # Concatenate all outputs and targets for the whole epoch
        train_outputs = torch.cat(train_outputs, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        metrics[f"train_{cf.loss_str}"].append( (total_loss / len(train_loader)).item() )
        metrics["train_l1"].append( (total_l1 / len(train_loader.dataset)).item() )
        metrics[f"train_{cf.goodness_str}"].append( goodness_function( train_outputs, train_targets ).item() )

        # Free memory
        del train_outputs, train_targets

        # VALIDATION
        net1.eval()
        if net2 is not None:
            net2.eval()
        with torch.no_grad(): # this enables larger batch size for validation; less memory
            x, y = next(iter(val_loader))
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1, cf.train_dataset.output_dim)

            if net2 is not None:
                z   = net1(x)
                out = net2(z)
                # Per-sample mean, matching the training loss and the mean-reduced
                # val_loss below (the val batch is the whole val set).
                l1  = torch.mean(l1_regularization(z, penalties))
            else:
                z   = None
                out = net1(x)
                l1  = torch.tensor(0.0, device=DEVICE)

            # Use standard MSE for validation even in weighted-loss training mode
            if criterion is not None:
                val_loss = criterion(out, y).detach()
            else:
                val_loss = nn.functional.mse_loss(out, y).detach()
            total_l1 = l1.detach()

            loss = (val_loss + total_l1).item()

            # METRICS FOR EACH EPOCH
            metrics[f"val_{cf.loss_str}"].append( val_loss.item() )
            metrics["val_l1"].append( total_l1.item() )  # already a per-sample mean
            metrics[f"val_{cf.goodness_str}"].append( goodness_function( out, y ).item() )

            if cf.save_histories:
                mvul, mvul_std = cf.val_dataset.snapshot_average(out)
                metrics["MVUL_out"].append( mvul.tolist() )
                metrics["MVUL_std_out"].append( mvul_std.tolist() )

                if cf.task == "regression":
                    if cf.label_param == "deltaomega":
                        metrics["pt"].append( [cf.val_dataset.get_phase_indicator( MVUL=mvul )[0][k].tolist() for k in ["deriv","peak"]] )
                        metrics["pt2"].append( [cf.val_dataset.get_phase_indicator( MVUL=mvul )[1][k].tolist() for k in ["deriv","peak"]] )
                    else:
                        metrics["pt"].append( [cf.val_dataset.get_phase_indicator( MVUL=mvul)[k].tolist() for k in ["deriv","peak"]] )

                if z is not None:
                    for k in range(len(cf.kernels)):  # Bottleneck activations and weights
                        metrics[f"z_{k}"].append( z[:, k].mean().cpu().numpy().tolist() ) # mean over batch

            # UPDATE LEARNING RATE SCHEDULER
            if scheduler is not None:
                # Track current learning rate BEFORE scheduler step
                # (this is the LR that was actually used for this epoch)
                current_lr = optimizer.param_groups[0]['lr']
                metrics["learning_rate"].append(current_lr)

                # Step the scheduler (except OneCycleLR which steps per batch)
                if cf.lr_scheduler_type == "reduce_on_plateau":
                    # ReduceLROnPlateau needs validation loss (base loss only, not including L1)
                    scheduler.step(metrics[f"val_{cf.loss_str}"][-1])
                elif cf.lr_scheduler_type != "onecycle":
                    # Other schedulers step based on epoch (not OneCycleLR)
                    scheduler.step()

            # PROGRESS BAR
            pbar.set_description(f"TRAINING")
            postfix_dict = {
                f'TRAIN {cf.loss_str}':     f"{metrics[f'train_{cf.loss_str}'][-1]:.2e}",
                f'TRAIN {cf.goodness_str}': f"{metrics[f'train_{cf.goodness_str}'][-1]:.2f}",
                f'VAL {cf.loss_str}':       f"{metrics[f'val_{cf.loss_str}'][-1]:.2e}",
                f'VAL {cf.goodness_str}':   f"{metrics[f'val_{cf.goodness_str}'][-1]:.2f}"
            }
            # Add learning rate to progress bar if scheduler is active
            if scheduler is not None:
                postfix_dict['LR'] = f"{current_lr:.2e}"
            pbar.set_postfix(postfix_dict)

            # if early_stopper.early_stop(metrics[f"val_{cf.loss_str}"][-1]): # NOTE: consider whether one should use the complete loss or just the MSE/CEL
            if early_stopper.early_stop(loss, epoch=epoch): # NOTE: consider whether one should use the complete loss or just the MSE/CEL
                tqdm.write(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Periodic garbage collection to free memory (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # METRICS FOR ONLY LAST EPOCH
    if net2 is not None:
        if not cf.save_histories:
            mvul, mvul_std = cf.val_dataset.snapshot_average(out)
            metrics["MVUL_out"], metrics["MVUL_std_out"] = mvul.tolist(), mvul_std.tolist()
            for k in range(len(cf.kernels)):  # Bottleneck activations and weights
                metrics[f"z_{k}"].append( z[:, k].mean().cpu().numpy().tolist() )  # type: ignore
        metrics["net2_norm"].append(torch.norm( torch.cat([p.view(-1) for p in net2.parameters()]) ).item() )

        # find the index of the kernel that has the largest bottleneck value
        max_k = np.argmax( [ abs(z[:, k].mean().cpu().numpy()) for k in range(len(cf.kernels)) ] )  # type: ignore
        print(f"Branch w. largest bottleneck activation: index {max_k}, kernel {cf.kernels[max_k]}")

        for k in range(len(cf.kernels)):  # Bottleneck activations and weights
            temp1, temp2 = cf.val_dataset.snapshot_average( z[:, k], output=False )  # type: ignore
            metrics[f"MVUL_{k}"], metrics[f"MVUL_std_{k}"] = temp1.tolist(), temp2.tolist()
    metrics["unique_labels"] = cf.val_dataset.unique_labels_unnormalized.tolist()

    metrics["train_time"].append(str(datetime.now() - start_time))

    if cf.save_final_values and hasattr(net1, 'branches'):
        fit_metrics = {}
        fit_metrics["all_final_x"] = x.cpu().numpy().tolist()
        fit_metrics["all_final_y"] = y.cpu().numpy().tolist()
        fit_metrics["all_final_out"] = out.cpu().numpy().tolist()
        fit_metrics["all_final_z"] = z.cpu().numpy().tolist()           # type: ignore
        for k in range(len(cf.kernels)):  # Bottleneck activations and weights
            fit_metrics[f"all_final_w_{k}"] = net1.branches[k].conv1.weight.cpu().detach().numpy().tolist() # type: ignore
        save_json(fit_metrics, cf.logdir, "fit_metrics.json")

    if "save_models" in cf.keys() and cf.save_models:
        torch.save(net1.state_dict(), os.path.join(cf.logdir, "net1.pt"))
        if net2 is not None:
            torch.save(net2.state_dict(), os.path.join(cf.logdir, "net2.pt"))

    print(f"\nTraining time: {metrics['train_time']}")

    save_json(metrics, cf.logdir, filename)

    # Final cleanup: free GPU memory
    del net1, net2, optimizer
    if scheduler is not None:
        del scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()