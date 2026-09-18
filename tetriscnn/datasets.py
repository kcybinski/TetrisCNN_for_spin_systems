import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tetriscnn.utils import *
from tetriscnn.dataprocessing import *

# Hardcoded partition index for cf.task == "classification" on the Paris (experimental)
# datasets: the index of the flanking pair in each dataset's sorted tuning-parameter
# grid that straddles the manuscript's reported transition location. There is no
# equivalent canonical index for these datasets otherwise, since Paris data carries no
# ground-truth phase label -- the partition threshold IS the classification target.
CLASSIFICATION_PARTITION_INDEX = {
    "Paris_XY_XZ": 2,
    "Paris_XY_X": 2,
    "Paris_Ising": 2,
    "Paris_XY_Z": 4,
}

# Datasets whose "classification" path additionally needs learning-by-confusion
# partitioning to mean anything: XXZDataProcessor raises NotImplementedError for
# discrete labels without it, exactly like ParisDataProcessor. ILGT and the 1D TFIM
# datasets are deliberately NOT here: those processors carry a genuine ground-truth
# phase label of their own (ILGTDataProcessor's classification file; TFIMDataProcessor's
# "class=" column), so classification there needs no partition index at all -- it is not
# "missing a canonical index", it simply isn't the same mechanism.
CLASSIFICATION_NEEDS_PARTITION = set(CLASSIFICATION_PARTITION_INDEX) | {"XXZ"}

# User-defined datasets, keyed by the cf.dataset name they answer to. Filled by
# register_dataset() so a new data source plugs into create_datasets() (and so into
# train() and main.py) without editing this module; see
# notebooks/BYODataset_Tutorial.ipynb for a worked example.
_DATASET_REGISTRY = {}


def register_dataset(name, factory):
    """Make ``cf.dataset = name`` load data through ``factory``.

    ``factory(cf, *, discrete_labels, label_param, learning_by_confusion,
    partition_index)`` must return a processor object, which create_datasets() then
    treats exactly like the built-in ones. The keyword arguments are what
    create_datasets() has already derived from ``cf.task``, so the factory does not
    have to repeat that logic. The processor must expose ``unique_labels`` and
    ``label_param``, plus either

    * ``samples`` and ``sample_times`` (one pool of ``(snapshot, label)`` pairs and
      the tuning-parameter value of each; the 70/30 train/val split, including
      ``cf.even_split``, is then done here), or
    * ``train_samples``/``val_samples`` and optionally ``train_times``/``val_times``
      (a split the data already comes with).

    Snapshots are ``float32`` tensors of shape ``(C, H, W)`` holding ``±1``; labels
    are ``float32`` scalar tensors (the class index for classification tasks, the
    target value for regression). Registering a name that is already taken replaces
    the earlier factory, which keeps re-running a notebook cell harmless.
    """
    if not callable(factory):
        raise TypeError(f"factory for dataset {name!r} must be callable, got {factory!r}")
    _DATASET_REGISTRY[name] = factory


def get_no_partitions(cf):
    cf.partition_index = 0
    _, temp_dataset = create_datasets(cf)   # deterministic; train-val split uses a separate seed.
    return getattr(temp_dataset.processor, "no_partitions", None)

def create_datasets(cf):
    """
    Returns train and validation datasets based on the configuration.
    Args:
        cf (object): Config object with following attributes:
            `dataset`: str, name of the dataset to load (e.g., "ILGT", "Paris_Ising", "Paris_XY", "1D_TFIM")
            `label_param`: str, label parameter for Paris datasets (e.g., "t", "delta", "omega")
            `normalize_labels`: bool, whether to normalize labels to [0, 1] range

    Returns:
        tuple: (train_dataset, val_dataset)

    Keys left unset on cf are filled from tetriscnn.utils.CONFIG_DEFAULTS.
    """
    even_split_requested = cf.get("even_split", False)   # before defaults fill it in
    apply_config_defaults(cf)
    if cf.task == "regression":
        discrete_labels = False
        label_param = cf.label_param
        normalize_labels = cf.normalize_labels
        label_subset_count = cf.label_subset_count
        learning_by_confusion = False
        partition_index = None
    else:
        discrete_labels = True
        label_param = "t"
        normalize_labels = False
        label_subset_count = None

        if cf.task == "classification" and cf.dataset in CLASSIFICATION_NEEDS_PARTITION:
            # A single classifier trained at one fixed split, so it needs a partition
            # index like "partition" does. The Paris datasets have a known
            # transition-flanking index on their tuning-parameter grid (above); XXZ
            # does not, so classification there falls back to the "partition" task's
            # cf.partition_index.
            if cf.dataset in CLASSIFICATION_PARTITION_INDEX:
                partition_index = CLASSIFICATION_PARTITION_INDEX[cf.dataset]
            else:
                assert cf.partition_index is not None, (
                    f"No canonical transition-flanking index is defined for "
                    f"dataset {cf.dataset!r}; classification falls back to the "
                    f"'partition' task there, so set cf.partition_index explicitly."
                )
                partition_index = cf.partition_index
            cf.partition_index = partition_index  # keep cf consistent (logdir naming, remake tooling)
            learning_by_confusion = True
        elif cf.task in ("lbc", "partition"):
            assert cf.partition_index is not None, "Partition index required for learning by confusion."
            learning_by_confusion = True
            partition_index = cf.partition_index
        else:
            learning_by_confusion = False
            partition_index = None


    # the processor loads the data, and returns samples.
    # samples is a list of (data, label) tuples.
    # the train-test split is then done here.
    # from the train and test samples, we create PhaseDataset instances.

    # Snapshot-pairing convention for multi-basis (XZ) datasets. Defaults reproduce the
    # historical index-order pairing bit-for-bit, so configs written before the
    # pairing-robustness ablation (which have none of these keys) load unchanged.
    pairing_kwargs = dict(
        pairing_mode=getattr(cf, "pairing_mode", "index"),
        pairing_seed=getattr(cf, "pairing_seed", None),
        pairing_subset_seed=getattr(cf, "pairing_subset_seed", 0),
    )

    if cf.dataset in _DATASET_REGISTRY:
        processor = _DATASET_REGISTRY[cf.dataset](
            cf, discrete_labels=discrete_labels, label_param=label_param,
            learning_by_confusion=learning_by_confusion, partition_index=partition_index)
    elif cf.dataset == "ILGT":
        processor = ILGTDataProcessor(discrete_labels=discrete_labels, label_param=label_param,
                                      learning_by_confusion=learning_by_confusion,
                                      partition_index=partition_index)
    elif cf.dataset.startswith("1D_TFIM"):
        # "1D_TFIM_Z" / "1D_TFIM_X" / "1D_TFIM_Y" pick the measurement basis; a bare
        # "1D_TFIM" defaults to Z. cf.phase_path selects which sweep to load
        # ("FM_PM" or "AFM_PM").
        suffix = cf.dataset[len("1D_TFIM"):].lstrip("_")
        basis_string = suffix.upper() if suffix else "Z"
        processor = TFIMDataProcessor(basis=basis_string,
                                      phase_path=getattr(cf, "phase_path", "FM_PM"),
                                      discrete_labels=discrete_labels, label_param=label_param,
                                      learning_by_confusion=learning_by_confusion,
                                      partition_index=partition_index)
    elif cf.dataset == "XXZ":
        processor = XXZDataProcessor(discrete_labels=discrete_labels, label_param=label_param,
                                     learning_by_confusion=learning_by_confusion,
                                     partition_index=partition_index)
    elif cf.dataset == "Paris_Ising":
        processor = IsingDataProcessor(data_format='-11', discrete_labels=discrete_labels, label_param=label_param, 
                                        filter_t_values=cf.filter_t_values,
                                        learning_by_confusion=learning_by_confusion, partition_index=partition_index)
    elif cf.dataset == "Paris_XY_X":
        processor = XYDataProcessor(basis="X", data_format='-11', discrete_labels=discrete_labels, label_param=label_param,
                                    filter_t_values=cf.filter_t_values,
                                     learning_by_confusion=learning_by_confusion, partition_index=partition_index,
                                     **pairing_kwargs)
    elif cf.dataset == "Paris_XY_Z":
        processor = XYDataProcessor(basis="Z", data_format='-11', discrete_labels=discrete_labels, label_param=label_param,
                                    filter_t_values=cf.filter_t_values,
                                     learning_by_confusion=learning_by_confusion, partition_index=partition_index,
                                     **pairing_kwargs)
    elif cf.dataset == "Paris_XY_XZ":
        processor = XYDataProcessor(basis="XZ", data_format='-11', discrete_labels=discrete_labels, label_param=label_param,
                                    filter_t_values=cf.filter_t_values,
                                     learning_by_confusion=learning_by_confusion, partition_index=partition_index,
                                     **pairing_kwargs)
    else:
        raise ValueError(
            f"Unknown dataset: {cf.dataset!r}. Choose one of 'Paris_Ising', "
            f"'Paris_XY_X', 'Paris_XY_Z', 'Paris_XY_XZ', 'ILGT', '1D_TFIM_X', "
            f"'1D_TFIM_Y', '1D_TFIM_Z', 'XXZ', or a name added with register_dataset() "
            f"(currently registered: {sorted(_DATASET_REGISTRY) or 'none'})."
        )
    
    cf.unique_labels = processor.unique_labels

    if hasattr(processor, 'train_samples') and hasattr(processor, 'val_samples'):
        # The simulated datasets (ILGT, 1D TFIM) ship their own train/test split, so
        # there is nothing to split here. They also carry the tuning-parameter value of
        # every snapshot, which is what the snapshot-average and phase-indicator
        # machinery downstream reads as its "time" axis; for these datasets the tuning
        # parameter is the label itself, so fall back to that if a processor omits it.
        train_samples = getattr(processor, "train_samples")
        val_samples = getattr(processor, "val_samples")
        train_times = getattr(processor, "train_times", None)
        val_times = getattr(processor, "val_times", None)
        if train_times is None:
            train_times = [float(lbl) for _, lbl in train_samples]
        if val_times is None:
            val_times = [float(lbl) for _, lbl in val_samples]

        if even_split_requested:
            print(f"[create_datasets] Note: even_split has no effect on {cf.dataset}, "
                  f"which ships a fixed train/test split.")
    else:
        # for the other datasets, we perform the train-val split here.
        samples = getattr(processor, "samples")
        g = torch.Generator().manual_seed(42) # use an isolated seed; do not want to average over different train-test splits

        # Check if even_split mode is enabled
        even_split = cf.even_split if hasattr(cf, 'even_split') else False
        samples_per_pt_cap = cf.samples_per_pt_cap if hasattr(cf, 'samples_per_pt_cap') else None

        sample_times = getattr(processor, "sample_times")
        if even_split and sample_times is not None:

            # Use original time values for grouping (works for regression, classification, and LBC)
            assert len(sample_times) == len(samples), \
                f"sample_times length ({len(sample_times)}) != samples length ({len(samples)})"
            
            # Group samples by their original time/sampling point values
            unique_time_points = sorted(set(sample_times))

            # Group samples by time point
            time_grouped_samples = {tp: [] for tp in unique_time_points}
            for sample, t_val in zip(samples, sample_times):
                time_grouped_samples[t_val].append(sample)

            # Print diagnostic info about samples per time point
            samples_per_tp = {tp: len(samples_list) for tp, samples_list in time_grouped_samples.items()}
            print(f"\n[Even Split] Found {len(unique_time_points)} time points")
            print(f"[Even Split] Samples per time point: min={min(samples_per_tp.values())}, max={max(samples_per_tp.values())}, median={np.median(list(samples_per_tp.values())):.1f}")
            
            # For LBC, show class distribution
            if discrete_labels and learning_by_confusion:
                labels_list = [sample[1].item() for sample in samples]
                class_0_count = sum(1 for l in labels_list if l == 0)
                class_1_count = sum(1 for l in labels_list if l == 1)
                print(f"[Even Split] LBC class distribution: class 0={class_0_count}, class 1={class_1_count}")

            # Determine sample cap per time point
            if samples_per_pt_cap is not None:
                if isinstance(samples_per_pt_cap, dict):
                    # Per-timepoint cap (e.g. to match another dataset's per-t snapshot
                    # counts exactly, such as XZ's min(n_X(t), n_Z(t)) pairing counts).
                    # A timepoint missing from the dict is left uncapped. Each entry is
                    # itself clamped to the actual count available, same as the scalar
                    # path below, so an over-large requested cap is a no-op rather than
                    # an error.
                    #
                    # Keys are coerced to float here because a dict saved into
                    # config.json and reloaded (e.g. by a post-hoc analysis script
                    # calling create_datasets(cf) again) comes back with JSON's string
                    # keys ("250.0"), which would otherwise silently fail to match the
                    # float time_grouped_samples keys and make the cap a no-op.
                    cap_by_t = {float(k): v for k, v in samples_per_pt_cap.items()}
                    applied = {}
                    for tp in time_grouped_samples:
                        requested = cap_by_t.get(float(tp))
                        if requested is None:
                            continue
                        actual_cap = min(requested, len(time_grouped_samples[tp]))
                        applied[tp] = actual_cap
                        if len(time_grouped_samples[tp]) > actual_cap:
                            idx_tp = torch.randperm(len(time_grouped_samples[tp]), generator=g)[:actual_cap]
                            time_grouped_samples[tp] = [time_grouped_samples[tp][i] for i in idx_tp]
                    print(f"[Even Split] Applying per-timepoint cap dict over {len(applied)} time points "
                          f"(of {len(samples_per_pt_cap)} requested): {applied}")
                else:
                    min_samples = min(len(samples_list) for samples_list in time_grouped_samples.values())
                    actual_cap = min(samples_per_pt_cap, min_samples)
                    print(f"[Even Split] Applying per-timepoint cap: {actual_cap} samples (min={min_samples}, requested={samples_per_pt_cap})")

                    # Cap samples per time point
                    for tp in time_grouped_samples:
                        if len(time_grouped_samples[tp]) > actual_cap:
                            # Randomly sample to cap
                            idx_tp = torch.randperm(len(time_grouped_samples[tp]), generator=g)[:actual_cap]
                            time_grouped_samples[tp] = [time_grouped_samples[tp][i] for i in idx_tp]

                # Show LBC class distribution after cap (if LBC)
                if discrete_labels and learning_by_confusion:
                    capped_labels_list = []
                    for samples_list in time_grouped_samples.values():
                        capped_labels_list.extend([sample[1].item() for sample in samples_list])
                    class_0_count = sum(1 for l in capped_labels_list if l == 0)
                    class_1_count = sum(1 for l in capped_labels_list if l == 1)
                    print(f"[Even Split] LBC class distribution after cap: class 0={class_0_count}, class 1={class_1_count}")

            # Now split into train/val by taking from each time point
            train_samples = []
            train_times = []
            val_samples = []
            val_times = []

            for tp in unique_time_points:
                tp_samples = time_grouped_samples[tp]
                n_tp = len(tp_samples)

                # Shuffle samples within this time point
                idx_tp = torch.randperm(n_tp, generator=g)
                tp_samples_shuffled = [tp_samples[i] for i in idx_tp]

                # Split: 70% train, 30% val
                split_train = int(0.7 * n_tp)

                train_samples.extend(tp_samples_shuffled[:split_train])
                train_times.extend([tp] * split_train)
                val_samples.extend(tp_samples_shuffled[split_train:])
                val_times.extend([tp] * (n_tp - split_train))

            print(f"[Even Split] After per-time-point 70/30 split: {len(train_samples)} train, {len(val_samples)} val samples")
            
            # NOTE: cf.data_fraction is deliberately NOT applied on the even_split path.
            # The `main` branch carries a version that subsets by taking the FIRST
            # n time points (unique_time_points[:n]), which truncates the run in time
            # rather than thinning it: for these datasets the tuning parameter is swept
            # with acquisition time, so dropping the tail deletes one whole phase and
            # the transition itself. It also rebuilt train/val_samples without rebuilding
            # train/val_times, desynchronising the time arrays that the weighted-loss
            # path reads. Use samples_per_pt_cap to reduce dataset size instead: it thins
            # every time point evenly and keeps the transition intact.
            # data_fraction still applies on the random-split path below.
        else:
            # Original random split logic
            n_subset = int(cf.data_fraction * len(samples))
            idx = torch.randperm(len(samples), generator=g)[:n_subset]

            split = int(0.7 * n_subset)
            train_samples = [samples[i] for i in idx[:split]]
            val_samples   = [samples[i] for i in idx[split:]]
            train_times   = [sample_times[i] for i in idx[:split]]
            val_times     = [sample_times[i] for i in idx[split:]]
    
    # Create train dataset first (computes normalization from train labels)
    train_dataset = PhaseDataset(train_samples, processor=processor, times=train_times, discrete_labels=discrete_labels,
                                normalize=normalize_labels, label_subset_count=label_subset_count, param_cutoffs=cf.param_cutoffs)

    # Extract normalization parameters from train dataset to apply to val dataset
    # This ensures consistent normalization and follows ML best practice
    if normalize_labels and not discrete_labels:
        y_min = train_dataset.y_min
        y_max = train_dataset.y_max
    else:
        y_min = None
        y_max = None

    # Create val dataset using train normalization parameters
    val_dataset = PhaseDataset(val_samples, processor=processor, times=val_times, discrete_labels=discrete_labels,
                            normalize=normalize_labels, label_subset_count=label_subset_count,
                            param_cutoffs=cf.param_cutoffs, y_min=y_min, y_max=y_max)

    return train_dataset, val_dataset

           

class PhaseDataset(Dataset):
    """
    Main dataset class for phase discovery/classification. 
    """
    def __init__(self, samples, processor, times, discrete_labels=False, normalize=True,
                 label_subset_count=None, param_cutoffs={"delta" : None, "omega" : None},
                 y_min=None, y_max=None):
        self.discrete_labels = discrete_labels
        self.normalize = normalize
        self.param_cutoffs = param_cutoffs

        self.processor = processor # TODO: is it a good pattern to include the full processor? perhaps not...

        if isinstance(self.processor, XYDataProcessor):
            self.cutoff_delta = param_cutoffs["delta"]
        elif isinstance(self.processor, IsingDataProcessor):
            self.cutoff_delta = param_cutoffs["delta"]
            self.cutoff_omega = param_cutoffs["omega"]

        data, labels = zip(*samples)
        labels = torch.stack(labels).to(torch.float32).to(DEVICE)
        data = torch.stack(data)

        self.label_param = processor.label_param
        self.times= torch.as_tensor(times, device=DEVICE)
        self.unique_times = torch.unique(self.times)


        if discrete_labels:
            self.output_dim = 2 
            self._process_snapshot_info( self.times, label_subset_count, normalize) # TODO: should make this more readable/elegant
            # this sets unique labels for snapshot averages, using the times
            self.labels = F.one_hot(labels.long(), num_classes=self.output_dim).float()  # One-hot encode
        else: 
            self.output_dim = labels.shape[1] if len(labels.shape) > 1 else 1
            self.labels = self._process_snapshot_info( labels, label_subset_count, normalize) 

        self.data = data

    def _process_snapshot_info(self, labels, label_subset_count, normalize):
        """
        Processes the snapshot information to set up for phase indication. This includes:
        - Optionally selecting a subset of labels for debugging.
        - Creating sorted indices of labels based on time.
        - Normalizing labels if specified.
        """
        # labels = labels.clone()

        # pick a subset of the labels, to speed up debugging
        # pick every k​th label to get ≈label_subset_count points
        # look at every stepth label; still might not be exactly label_subset_count points, 
        # so additional slicing to obtain exactly label_subset_count unique labels
        # mask data whose label ∈ picked
        if label_subset_count is not None:
            raise NotImplementedError("Deprecated, use samples_per_pt_cap instead.")

        # The following part is required for the phase indicator method.

        # For each true value of the tuning parameter, we want to average predictions ('snapshot average').
        # We also want to show the phase indication for ascending values of the tuning parameter.
        # However, during training these true labels should be shuffled.
        # This means we need to store a list of indices (for data & labels) that corresponds
        # to a sorted ordering of labels.

        # To complicate things more, for ParisData, e.g. Ising the tunings are (δ(t), ω(t)) but
        # we want to sort by ascending t as opposed to (δ, ω).
        # Thus, we loop over the labels, (δ, ω) tuples, find the corresponding time value,
        # and use this to get the index list.

        # set unique unnormalized labels
        if self.label_param in ["delta", "omega", "deltaomega"]:
            if self.processor.label_dict is not None:
                self.dict_times = np.array(list(self.processor.label_dict.keys()))
            else:
                raise ValueError("label_dict is None, cannot access keys.")

            dict_labels = np.array(list(self.processor.label_dict.values()))  # shape (N,output_dim)

            self.unique_labels_unnormalized = torch.as_tensor(dict_labels, device=DEVICE, dtype=torch.float32) # shape (N, output_dim)
            self.unique_times = torch.as_tensor(self.dict_times, device=DEVICE, dtype=torch.float32) # shape (N,)
        else:
            # Single-parameter tuning: the label IS the tuning parameter, so the sweep
            # points are simply its distinct values. This covers 't' for the Rydberg
            # datasets and 'beta' / 'g' / 'Jz' for the simulated ones, none of which
            # need the time -> parameter table that delta and omega go through above.
            self.unique_labels_unnormalized = torch.unique(self.times) # shape (N, output_dim)

        print(f"times shape: {len(self.times)}, labels shape: {labels.shape}")
        # set sorted labels
        # self.sorted_label_indices = np.argsort(times)
        self.sorted_label_indices = torch.argsort(self.times)
        self.unique_labels_unnormalized = torch.as_tensor(self.unique_labels_unnormalized, device=DEVICE)
        self.unique_labels = self.unique_labels_unnormalized.clone() # type: ignore
        if normalize:
            self.y_min = self.unique_labels.min(dim=0).values # shape (output_dim,)
            self.y_max = self.unique_labels.max(dim=0).values  # shape (output_dim,)
            labels = normalize01(labels, self.y_min, self.y_max) # observe that the (δ, ω) tuples in y_min and y_max do not occur in unique_labels.
            self.unique_labels = normalize01(self.unique_labels_unnormalized, self.y_min, self.y_max)

        # finally, the sorted_labels are sorted by time and normalized by the max (δ, ω) values.
        # the labels are just normalized and unsorted.
        self.unique_labels = torch.as_tensor(self.unique_labels, device=DEVICE).squeeze() 
        self.sorted_labels = torch.as_tensor(labels[self.sorted_label_indices], device=DEVICE).squeeze()  # shape (no_unique_labels, output_dim); these are the labels sorted by time
        
        self.sorted_label_indices = torch.as_tensor(self.sorted_label_indices, device=DEVICE)
        self.unique_labels_unnormalized = self.unique_labels_unnormalized.cpu().numpy()
        # self.times = times

        return labels


    def snapshot_average(self, val, output=True):
        """
        Computes snapshot average; was called MVUL("Mean Value per Unique Label") before.

        From this one can compute a phase indicator: the derivative of any value (output or latent variable)
        w.r.t. true labels for continuous predictions.

        output variable=True means we should denormalize since we are considering the network output
            (e.g. delta or time); False means we shoulwe're considering one of the branch activations.
        """
        if isinstance(val, np.ndarray):
            val = torch.as_tensor(val, device=DEVICE, dtype=torch.float32)

        if self.discrete_labels: 
            means, stds = [], []
            for label in self.unique_times:
                subset = val[self.sorted_label_indices][self.sorted_labels == label]
                means.append(subset.mean(dim=0))
                stds.append(subset.std(dim=0))
            means, stds = torch.stack(means), torch.stack(stds)
            return means.cpu().numpy(), stds.cpu().numpy()
        else:
            if output and self.normalize:
                # make sure y_min/y_max are torch tensors on same device
                y_min = torch.as_tensor(self.y_min, device=val.device, dtype=val.dtype)
                y_max = torch.as_tensor(self.y_max, device=val.device, dtype=val.dtype)
                val = denormalize01(val, y_min, y_max)

            if self.processor.label_param == "deltaomega":
                means, stds = [], []
                for label in self.unique_labels:
                    subset = val[self.sorted_label_indices][(self.sorted_labels == label).all(dim=1)]
                    means.append(subset.mean(dim=0))
                    stds.append(subset.std(dim=0))
                means, stds = torch.stack(means), torch.stack(stds)
                return means.cpu().numpy(), stds.cpu().numpy()
            else:
                means, stds = [], []
                for label in self.unique_labels:
                    subset = val[self.sorted_label_indices][self.sorted_labels == label]
                    means.append(subset.mean())
                    stds.append(subset.std())
                means, stds = torch.stack(means), torch.stack(stds)
                return means.cpu().numpy(), stds.cpu().numpy()
            # return avg_sorted_val.cpu().numpy()
        
    def get_phase_indicator(self, MVUL, output=True):

        """"
        Phase indicator: the value of the tuning parameter where the derivative of MVUL is maximal.
            for delta and omega, we need to restrict the domain to prevent numerical divergence

        Args:
            MVUL (np.ndarray): MVUL values. Shape (no_unique_labels, output_dim) or (no_unique_labels,).
            output (bool): Whether to denormalize the output. Default is True.
        """
        assert not self.discrete_labels, "Phase indicator only defined for continuous labels."
        if self.processor.label_param == "deltaomega":
            deltas = self.unique_labels_unnormalized[:,0]
            omegas = self.unique_labels_unnormalized[:,1]

            delta_cut = deltas[ :self.cutoff_delta ]
            omega_cut = omegas[ self.cutoff_omega: ]

            if output:
                delta_pred_cut = MVUL[:,0][:self.cutoff_delta] # type: ignore
                omega_pred_cut = MVUL[:,1][self.cutoff_omega:] # type: ignore
            else: # for branches, MVUL is 1dimensional, but still want two derivs
                delta_pred_cut = MVUL[:self.cutoff_delta] # type: ignore
                omega_pred_cut = MVUL[self.cutoff_omega:] # type: ignore                

            # new method
            deriv_delta_pred = np.gradient( delta_pred_cut ) / np.gradient(deltas, self.dict_times)[:self.cutoff_delta]
            deriv_omega_pred = np.gradient( omega_pred_cut ) / np.gradient(omegas, self.dict_times)[self.cutoff_omega:]
            peak_deriv_delta_delta = np.argmax( np.abs(deriv_delta_pred) ).squeeze()
            peak_deriv_omega_omega = np.argmax( np.abs(deriv_omega_pred) ).squeeze()

            # old method, 'direct_deriv'

            delta_phase_dict = {"tuning_true": delta_cut,
                        "pred": delta_pred_cut,
                        "deriv": deriv_delta_pred,
                        "peak":delta_cut[peak_deriv_delta_delta]}
            omega_phase_dict = {"tuning_true": omega_cut,
                        "pred": omega_pred_cut,
                        "deriv": deriv_omega_pred,
                        "peak":omega_cut[peak_deriv_omega_omega]}

            return delta_phase_dict, omega_phase_dict

        elif self.processor.label_param == 'delta':
            deltas = self.unique_labels_unnormalized.squeeze()
            delta_cut = deltas[ :self.cutoff_delta ]
            delta_pred_cut = MVUL[:self.cutoff_delta].squeeze()  # type: ignore

            # new method
            # (d hat{delta}/ dt) / (d delta/ dt )
            deriv_delta_pred = np.gradient( delta_pred_cut )  /  np.gradient(deltas, self.dict_times)[:self.cutoff_delta] 
            peak_deriv_delta_delta = np.argmax( np.abs(deriv_delta_pred) ).squeeze()
            
            return  {"tuning_true": delta_cut,
                        "pred": delta_pred_cut,
                        "deriv": deriv_delta_pred,
                        "peak":delta_cut[peak_deriv_delta_delta]}
        else: 
            gradient = np.gradient(MVUL)
            peak_deriv_index = np.argmax( np.abs(gradient) ).squeeze()
             
            return {"tuning_true": self.unique_times,
                                            "pred": MVUL,
                                            "deriv": gradient,
                                            "peak": self.unique_labels_unnormalized[peak_deriv_index]}

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset so each item also carries its integer index.

    The weighted-loss path needs the per-sample importance weight that goes with
    each sample, but the training DataLoader shuffles, so a batch's samples are
    not the contiguous slice their position would suggest. Returning the index
    lets us gather the matching weights regardless of shuffling.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i
