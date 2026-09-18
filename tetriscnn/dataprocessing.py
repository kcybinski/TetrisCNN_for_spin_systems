import numpy as np
import torch
import os
import re
import pandas as pd
from pathlib import Path


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a dataset a config asks for is not present on disk.

    Carries the resolved path it looked for, so the message tells the user what to
    put where rather than surfacing a bare ``FileNotFoundError`` from deep inside a
    loader.
    """


#: Directory names searched for the snapshot tree, in order of preference.
#: ``datasets`` is the layout of the public release; ``data`` is the pre-release
#: one, still accepted so working trees that predate the rename keep running.
DATA_ROOT_NAMES = ("datasets", "data")




def get_data_root():
    """Return the directory holding the snapshot datasets.

    Searches upwards from the current working directory for ``datasets/`` and then
    ``data/``, so a notebook run from a subfolder still resolves the same tree as a
    script run from the repository root. Setting ``TETRISCNN_DATA_ROOT`` overrides
    the search, which is how a dataset tree kept outside the repository (on a
    scratch disk, say) is pointed at.
    """
    override = os.environ.get("TETRISCNN_DATA_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise DatasetNotAvailableError(
                f"TETRISCNN_DATA_ROOT is set to '{root}', which is not a directory."
            )
        return root

    curr = Path.cwd().resolve()
    for parent in [curr, *curr.parents]:
        for name in DATA_ROOT_NAMES:
            if (parent / name).is_dir():
                return parent / name
    raise DatasetNotAvailableError(
        "Could not locate a dataset tree. Expected a 'datasets/' (or legacy 'data/') "
        "directory at or above the current working directory, or the environment "
        "variable TETRISCNN_DATA_ROOT pointing at one. See the 'Data and trained runs' section of "
        "the README for the expected layout and where to obtain each dataset."
    )


def require_dataset_dir(relative, what):
    """Resolve ``relative`` inside the dataset tree, or explain what is missing.

    Args:
        relative: path relative to the dataset root, e.g. ``"simulated/1D_TFIM/FM_PM"``.
        what: human-readable name of the dataset, used in the error message.
    """
    path = get_data_root() / relative
    if not path.is_dir():
        raise DatasetNotAvailableError(
            f"{what} not found at '{path}'. Expected the dataset tree to contain "
            f"'{relative}'. See the 'Data and trained runs' section of the README for the expected "
            f"layout and where to obtain this dataset."
        )
    return path


#: Legal values for ``pairing_mode`` (see SnapshotProcessor._build_pairing_indices).
PAIRING_MODES = ("index", "repair_fixed", "repair_resample", "cross_time")


class SnapshotProcessor():
    """Base class for data stored as a pool of snapshots per value of the tuning parameter.

    This is the shape of the Rydberg data and of most experimental or simulated sweeps:
    for every value t of the tuning parameter, a set of snapshots, possibly measured in
    several bases. The base class turns that into the flat ``samples`` /
    ``sample_times`` lists create_datasets() splits itself (see register_dataset in
    tetriscnn/datasets.py), and handles everything that does not depend on the file
    format: stacking bases into channels (and how their snapshots are paired),
    mapping stored 0/1 values to ±1, optional relabelling through ``label_dict``,
    filtering by ``filter_t_values``, and learning-by-confusion relabelling.

    A subclass implements the file format only:

    * ``_load_basis_files(file_list)`` returns ``{t: array}`` for one basis, the
      array holding one snapshot per row (any shape that ``_to_image`` accepts);
    * ``_to_image(snapshot)`` (optional) turns one row into a ``(1, H, W)`` array.
      The default reshapes to ``grid_size``.

    Labels: for regression the label is ``t`` itself (or ``label_dict[t]``). For
    classification with learning by confusion, snapshots are labelled 0/1 on either
    side of a threshold between two neighbouring values of t. Plain classification
    needs to know the phase at each t, which a sweep does not carry by itself: a
    subclass that knows it implements ``_phase_label(t)`` (returning 0 or 1).

    ``data_format='-11'`` (the default) maps stored 0/1 values to ±1; ``'01'`` leaves
    the loaded values untouched.
    """
    def __init__(self,  grid_size, label_param, file_dict = {"X": None, "Z": None}, data_format='-11', label_dict=None,
                 discrete_labels=False, filter_t_values=None,
                 learning_by_confusion=False, partition_index=None,
                 pairing_mode="index", pairing_seed=None, pairing_subset_seed=0):
        assert data_format in ['01', '-11'], "data_format must be '01' or '-11'"
        if pairing_mode not in PAIRING_MODES:
            raise ValueError(f"pairing_mode must be one of {PAIRING_MODES}, got {pairing_mode!r}")

        self.pairing_mode = pairing_mode
        self.pairing_seed = pairing_seed
        self.pairing_subset_seed = pairing_subset_seed

        self.grid_size = grid_size
        self.samples = []
        self.sample_times = []  # Track original time values for even-split grouping
        self.label_dict = label_dict
        self.label_param = label_param
        self.data_format = data_format
        self.filter_t_values = filter_t_values
        
        # file_dict contains as keys either "X", "Z", or both.
        # we loop through them and define a data_dict, containing the same keys, but with the data.
        data_dict = {}
        for basis, file_list in file_dict.items():
            data_dict[basis] = self._load_basis_files(file_list)
            if self.data_format == '-11':
                data_dict[basis] = {t: (data * 2) - 1 for t, data in data_dict[basis].items()}

        # some models have different values of t. 
        # find common times across all bases; also works for just one basis.
        time_sets = [set(d.keys()) for d in data_dict.values()]
        common_keys = sorted(set.intersection(*time_sets))

        # Loop through all (common) values of t. For each value of t, get the number of samples
        # for each basis. We pick the minimum, min_length. Then, we loop up to min_length, and
        # pick from both bases a sample for each index. We concatenate them and append the
        # sample. In append sample, we also change the label if necessary.
        #
        # WHICH snapshot of basis b is paired with which snapshot of basis b' is set by
        # _build_pairing_indices() below. The default ("index") is the historical convention:
        # take the first min_length rows of each basis in file order. Because the bases are
        # measured in *independent experimental realizations*, that convention is arbitrary --
        # see the pairing_mode docstring for the alternatives used by the pairing-robustness
        # ablation.
        basis_names = list(data_dict.keys())
        no_bases = len(basis_names)
        source_times = self._build_source_times(common_keys, no_bases)

        for t_index, t_val in enumerate(common_keys):
            # Channel b reads its snapshots from source_times[t_index][b]. That is t_val itself
            # for every mode except "cross_time", where the non-anchor bases deliberately read a
            # different acquisition time.
            basis_data_arrays = [
                data_dict[basis][source_times[t_index][basis_index]]
                for basis_index, basis in enumerate(basis_names)
            ]
            row_indices = self._build_pairing_indices(
                [data_array.shape[0] for data_array in basis_data_arrays], t_index
            )
            for k in range( len(row_indices[0]) ):
                channels = []
                for basis_index in range( no_bases ):
                    row = row_indices[ basis_index ][ k ]
                    channels.append( self._to_image(basis_data_arrays[ basis_index ][ row ]) )

                combined = np.concatenate(channels, axis=0)
                self._append_sample(combined, t_val)

        labels = torch.stack([lbl for _, lbl in self.samples]).cpu().numpy()
        unique_labels = np.unique(labels, axis=0)
        self.unique_labels = unique_labels

        if discrete_labels:
            if learning_by_confusion:
                assert partition_index is not None, "Must provide partition_index when learning_by_confusion=True"
                self.no_partitions = len(unique_labels) - 1  # total partitions between label values
                if partition_index < 0 or partition_index >= self.no_partitions:
                    raise ValueError(f"partition_index must be in [0, {self.no_partitions-1}]")

                # threshold-based partition 
                t_threshold = 0.5 * (unique_labels[partition_index] + unique_labels[partition_index + 1])
                low_labels = unique_labels[unique_labels < t_threshold]
                high_labels = unique_labels[unique_labels >= t_threshold]

                print(f"\n [LbC] CREATING DATASET: partition {partition_index+1}/{self.no_partitions}, threshold={t_threshold:.4f}, total {len(self.samples)} samples.")
            else:
                self.samples = [(data, torch.tensor(float(self._phase_label(t_val)), dtype=torch.float32))
                                for (data, _), t_val in zip(self.samples, self.sample_times)]
                return

            new_samples = []
            new_sample_times = []  # Preserve original times through LBC filtering
            for (data, lbl), t_val in zip(self.samples, self.sample_times):
                if lbl.item() in low_labels:
                    new_samples.append((data, torch.tensor(0, dtype=torch.float32)))
                    new_sample_times.append(t_val)
                elif lbl.item() in high_labels:
                    new_samples.append((data, torch.tensor(1, dtype=torch.float32)))
                    new_sample_times.append(t_val)

            self.samples = new_samples
            self.sample_times = new_sample_times


    # ------------------------------------------------------------------
    # Snapshot pairing across measurement bases
    # ------------------------------------------------------------------
    # The X and Z snapshots of the XY dataset come from two *independent*
    # experimental runs, so no snapshot of one basis physically corresponds to any
    # particular snapshot of the other. Whatever rule we use to stack them into a
    # two-channel image is therefore a free convention, and the true joint
    # distribution any such rule samples from is exactly the product
    # p(S^X) p(S^Z). Re-pairing can only change the finite-sample realisation, not
    # the distribution -- it injects variance, never bias.
    #
    # These helpers make that convention explicit and swappable so the
    # pairing-robustness ablation can measure how large that variance actually is.
    # Single-basis datasets (Ising, Paris_XY_X, Paris_XY_Z) have nothing to pair and
    # always fall back to "index".

    def _pairing_rng(self, *stream_ids):
        """Independent RNG stream, reproducibly derived from the given integer ids.

        Passing the ids as a seed *sequence* (rather than, say, seeding once and
        drawing in loop order) keeps each (time point, basis) stream independent of
        the order in which the others are consumed, so adding a time point never
        shifts the draws of the others.
        """
        return np.random.default_rng([int(s) for s in stream_ids])

    def _build_source_times(self, common_keys, no_bases):
        """Pick, per target time point and basis, which acquisition time to read from.

        Everything except "cross_time" reads basis b at the same time as the label,
        so this is the identity. For "cross_time" -- the deliberate null control --
        basis 0 stays anchored to the true time while every other basis is read at a
        *different* time, drawn as a derangement (no fixed points, so no channel is
        ever accidentally correct). If the network's branch selection and transition
        estimate survive even this, it reads only per-channel marginals and the
        pairing carries no information at all.
        """
        n_times = len(common_keys)
        source = [[t_val] * no_bases for t_val in common_keys]

        # As in _build_pairing_indices, a missing seed means "do not randomise", so an
        # unseeded cross_time run degrades to the plain index pairing rather than failing.
        if (self.pairing_mode != "cross_time" or no_bases < 2 or n_times < 2
                or self.pairing_seed is None):
            return source

        for basis_index in range(1, no_bases):
            rng = self._pairing_rng(self.pairing_seed, 9999, basis_index)
            # Rejection-sample a derangement. For n >= 2 the derangement fraction is
            # ~1/e, so this terminates almost immediately; the cap only guards n == 2
            # style edge cases from an unlucky stream.
            for _ in range(1000):
                perm = rng.permutation(n_times)
                if not np.any(perm == np.arange(n_times)):
                    break
            else:
                perm = np.roll(np.arange(n_times), 1)  # guaranteed fixed-point-free
            for t_index in range(n_times):
                source[t_index][basis_index] = common_keys[perm[t_index]]

        return source

    def _build_pairing_indices(self, basis_lengths, t_index):
        """Return, per basis, the row indices to read at one time point.

        All returned arrays have the same length m = min(basis_lengths), so the
        number of paired samples per time point -- and hence the total dataset size
        -- is identical across every mode that reads matched times.

        Modes:
          "index"           the historical convention: the first m rows of each basis
                            in file (acquisition) order. Bit-for-bit the pre-ablation
                            behaviour.
          "repair_fixed"    the m surviving snapshots of each basis are fixed by
                            pairing_subset_seed and held constant across pairing
                            seeds; only the assignment of non-anchor bases onto basis
                            0 is permuted. Isolates pairing variance from the
                            variance of *which* snapshots survive truncation.
          "repair_resample" both the surviving subset and the assignment are redrawn
                            per pairing seed. The end-to-end ensemble; note that
                            min-truncation discards ~24% of the raw XY snapshots, so
                            this genuinely resamples the data as well as the pairing.
          "cross_time"      indices as in "index"; the mispairing lives entirely in
                            _build_source_times.
        """
        m = min(basis_lengths)
        no_bases = len(basis_lengths)

        # Nothing to pair with a single basis, and no seed means no randomisation.
        if self.pairing_mode in ("index", "cross_time") or no_bases < 2 or self.pairing_seed is None:
            return [np.arange(m) for _ in basis_lengths]

        if self.pairing_mode == "repair_fixed":
            indices = []
            for basis_index, n_rows in enumerate(basis_lengths):
                # Subset depends only on pairing_subset_seed -> identical for every
                # pairing seed. Sorted so the anchor keeps acquisition order.
                subset = np.sort(self._pairing_rng(
                    self.pairing_subset_seed, t_index, basis_index
                ).choice(n_rows, size=m, replace=False))
                if basis_index > 0:
                    # Only the map moves: same snapshots, different partner.
                    subset = subset[self._pairing_rng(
                        self.pairing_seed, t_index, basis_index
                    ).permutation(m)]
                indices.append(subset)
            return indices

        # "repair_resample": choice() without replacement returns the picks in random
        # order, so a single draw varies both the surviving subset and the assignment.
        return [
            self._pairing_rng(self.pairing_seed, t_index, basis_index).choice(
                n_rows, size=m, replace=False
            )
            for basis_index, n_rows in enumerate(basis_lengths)
        ]

    def _load_basis_files(self, file_list):
        """Read one basis: return ``{t: array}``, one snapshot per row of each array."""
        raise NotImplementedError(f"{type(self).__name__} must implement _load_basis_files().")

    def _to_image(self, snapshot):
        """Turn one stored snapshot into a ``(1, H, W)`` array."""
        return np.asarray(snapshot).reshape(1, *self.grid_size)

    def _phase_label(self, t_val):
        """The phase (0 or 1) at tuning value t_val, for plain classification."""
        raise NotImplementedError(
            f"{type(self).__name__} has no phase labels of its own: implement _phase_label(t), "
            f"or use learning by confusion (task 'partition' or 'lbc' with a partition_index).")

    def _append_sample(self, sample, t_val):
        """"
        From
        """
        label = t_val
        # Skip if not in filter list (if provided)
        if (self.filter_t_values is not None) and (label not in self.filter_t_values):
            return
        
        # Transform to other label 
        if (self.label_param != "t") and self.label_dict is not None:
            label = self.label_dict.get(t_val, None) # returns None if t_val does not exist
            if label is None:
                # this should never happen, because we loop over common_keys!
                raise ValueError(f"t={t_val} not in label_dict, cannot assign label.")
            
        self.samples.append((
            torch.tensor(sample, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        ))
        self.sample_times.append(t_val)  # Store original time value



        
class ParisDataProcessor(SnapshotProcessor):
    """The Rydberg-array (Paris) data format, shared by the Ising and XY datasets.

    One ``t=<time>_ns.dat`` text file per acquisition time, each row a flattened 0/1
    snapshot.
    """

    def _load_basis_files(self, file_list):
        data_by_time = {}
        for filepath in file_list:
            filename = os.path.basename(filepath)

            if filename.startswith('t=') and filename.endswith('.dat'):
                match = re.match(r"t=([\d\.]+)_ns\.dat", filename)
                if match:
                    data_by_time[float(match.group(1))] = np.loadtxt(filepath)

        return data_by_time

    def _to_image(self, snapshot):
        # The raw per-snapshot flat array is unflattened to grid_size and then rotated
        # 90 deg CCW to match the physical atom layout. For the rectangular XY array this
        # is what makes the reshape correct (see XYDataProcessor: grid_size is (7, 6), the
        # true row-major layout, giving a (6, 7) image after rotation). For the square
        # Ising array the rotation only reorients.
        return np.rot90(super()._to_image(snapshot), axes=(1, 2))


class IsingDataProcessor(ParisDataProcessor):
    """
    Processor for 2d, quantum Ising model data.
    - (8, 8) grid size
    - Supports labels as 't', 'delta', or 'omega', or 'deltaomega'.
    """
    def __init__(self, directory=None, data_format='-11', label_param=None,
                 discrete_labels=False, filter_t_values=None,
                 learning_by_confusion=False, partition_index=None):
        if directory:
            if not re.search(r'8x8', directory):
                raise ValueError("Directory must contain '8x8' in its path.")
        else:
            directory = require_dataset_dir(
                'experimental/Ising/Ising_Z_8x8c_0_defects_0_holes',
                'the 8x8 Ising Rydberg snapshots',
            )

        self.delta_dict = dict(zip( np.loadtxt(directory / 'time_in_ns.dat'), np.loadtxt(directory / 'delta_in_MHz.txt') ))
        self.omega_dict = dict(zip( np.loadtxt(directory / 'time_in_ns.dat'), np.loadtxt(directory / 'omega_in_MHz.txt') ))

        grid_size = (8, 8) 
        file_dict = {"Z":[os.path.join(directory, f) for f in os.listdir(directory)]}

        if discrete_labels:
            assert label_param == "t", "label_param must be 't'"
            label_dict = None
        else:
            assert label_param in ['t','delta','omega', 'deltaomega'], "label_param must be 't', 'delta', 'omega' or 'deltaomega'"

            if label_param == 'delta':  
                label_dict = {k: (self.delta_dict[k],) for k in self.delta_dict} # make it a tuple
            elif label_param == 'omega':
                label_dict = {k: (self.omega_dict[k],) for k in self.omega_dict} # make it a tuple
            elif label_param == 'deltaomega':
                label_dict = {k: (self.delta_dict[k], self.omega_dict[k]) for k in self.delta_dict} # (delta, omega) tuple
            else:
                label_dict = None
        
            
        super().__init__( grid_size, label_param, file_dict, data_format, label_dict, discrete_labels, filter_t_values, 
                         learning_by_confusion, partition_index )


    
class XYDataProcessor(ParisDataProcessor):
    """
    Processor for 2d, quantum XY model data.
    - (6, 7) grid size
    - only uses 't' label param
    """
    def __init__(self, basis, directory=None, data_format='-11', label_param=None,
                 discrete_labels=False, filter_t_values=None,
                 learning_by_confusion=False, partition_index=None,
                 pairing_mode="index", pairing_seed=None, pairing_subset_seed=0):
        assert basis in ['X', 'Z', 'XZ'], "basis must be 'X', 'Z', or 'XZ'"

        if directory:
            if not re.search(r'6x7', directory):
                raise ValueError("Directory must contain '6x7' in its path.")
        dir1 = require_dataset_dir(
            'experimental/XY/6x7_atoms_X_ferro_video_15MHz_tau0p3_17032022_0_defects_0_holes',
            'the 6x7 XY Rydberg snapshots (X basis)',
        )
        dir2 = require_dataset_dir(
            'experimental/XY/6x7_atoms_Z_ferro_video_15MHz_tau0p3_16032022_7_0_defects_0_holes',
            'the 6x7 XY Rydberg snapshots (Z basis)',
        )


        # (7, 6) is the true row-major layout of the raw flat array; the base loader
        # rotates 90 deg CCW so the stored image is (6, 7). Using (6, 7) here (the old
        # value) scrambled the pixels because the flat data is not laid out that way.
        grid_size = (7, 6)
        if basis in ['X', 'Z']:
            if directory is None:
                directory = dir1 if basis == 'X' else dir2
            print("directory: ", directory )
            file_dict = {basis: [os.path.join(directory, f) for f in os.listdir(directory)]}
        elif basis == 'XZ':
            file_dict = {"X":[os.path.join(dir1, f) for f in os.listdir(dir1)], 
                         "Z":[os.path.join(dir2, f) for f in os.listdir(dir2)]}
            
        self.delta_dict_x = dict(zip(np.loadtxt(dir1 / 'time_in_ns.dat'), np.loadtxt(dir1 / 'lightshift_in_MHz.txt')))
        self.delta_dict_z = dict(zip(np.loadtxt(dir2 / 'time_in_ns.dat'), np.loadtxt(dir2 / 'lightshift_in_MHz.txt')))
        common_keys = set(self.delta_dict_x) & set(self.delta_dict_z)
        self.delta_dict_combined = { k : ( 0.5 * (self.delta_dict_x[k] + self.delta_dict_z[k]), ) for k in sorted(common_keys) }

        if discrete_labels:
            # assert label_param is None, "For classification, label_param must be None."
            assert label_param == "t", "label_param must be 't'"
            label_dict = None
        else:
            assert label_param in ['t', 'delta'], "label_param must be 't' or 'delta'"
            if label_param == 'delta':
                if basis in ['X', 'Z']:
                    if basis == 'X':
                        label_dict = {k: (self.delta_dict_x[k],) for k in self.delta_dict_x}
                    else:
                        label_dict = {k: (self.delta_dict_z[k],) for k in self.delta_dict_z}
                elif basis == 'XZ':
                    label_dict = self.delta_dict_combined
            elif label_param == 't':
                if basis == 'XZ':
                    label_dict = { k : k for k in sorted(common_keys) } 
                else:
                    label_dict = None

            
        super().__init__( grid_size, label_param, file_dict, data_format, label_dict, discrete_labels, filter_t_values,
                         learning_by_confusion, partition_index,
                         pairing_mode=pairing_mode, pairing_seed=pairing_seed,
                         pairing_subset_seed=pairing_subset_seed )



def partition_threshold(unique_values, partition_index):
    """Place one learning-by-confusion threshold in the tuning parameter.

    The distinct tuning values are assumed sorted. Threshold ``i`` sits midway
    between values ``i`` and ``i + 1``, so a sweep with ``n`` distinct values admits
    ``n - 1`` partitions.

    Returns:
        tuple: (threshold, number of available partitions)
    """
    no_partitions = len(unique_values) - 1
    if no_partitions < 1:
        raise ValueError(
            "Learning by confusion needs at least two distinct values of the tuning "
            f"parameter; got {len(unique_values)}."
        )
    if partition_index < 0 or partition_index >= no_partitions:
        raise ValueError(f"partition_index must be in [0, {no_partitions - 1}]")
    threshold = 0.5 * (unique_values[partition_index] + unique_values[partition_index + 1])
    return threshold, no_partitions


def relabel_by_threshold(samples, tuning_values, threshold):
    """Relabel snapshots 0 below and 1 above a threshold in the tuning parameter.

    The tuning values are carried through unchanged, so the snapshot-average and
    phase-indicator machinery still knows which sweep point each snapshot came from
    after the binary relabelling.
    """
    new_samples, new_tuning = [], []
    for (data, _), tuning in zip(samples, tuning_values):
        cls = 0 if float(tuning) < threshold else 1
        new_samples.append((data, torch.tensor(cls, dtype=torch.float32)))
        new_tuning.append(tuning)
    return new_samples, new_tuning


class ILGTDataProcessor():
    """
    Processor for 2d, ILGT dataset.
    - (16, 16) grid size, two channels (the two link orientations)
    - supports both discrete and continuous labels.
    - for continuous labels, label_param must be 'beta'.

    The dataset ships its own train/test split, so this processor exposes
    ``train_samples``/``val_samples`` (and the matching ``train_times``/``val_times``)
    rather than the single ``samples`` list the Rydberg processors build.
    """
    def __init__(self, discrete_labels=True, label_param=None, filter_label_values=None,
                 directory=None, learning_by_confusion=False, partition_index=None):
        if not discrete_labels:
            assert label_param == 'beta', "For continuous labels in ILGT, label_param must be 'beta'."
        self.grid_size = (16, 16)
        self.label_param = label_param
        # No time -> tuning-parameter table: the label IS the tuning parameter here,
        # unlike the Rydberg datasets where delta and omega are functions of the
        # acquisition time.
        self.label_dict = None

        # The classification file stores only the two phase classes, the regression
        # file stores the inverse temperature beta. Learning by confusion needs a tuning
        # parameter to place its threshold in, so it reads the regression file and
        # derives the binary labels itself.
        needs_tuning_parameter = (not discrete_labels) or learning_by_confusion
        task_str = "regression" if needs_tuning_parameter else "classification"
        if directory is None:
            directory = require_dataset_dir(
                f"simulated/ILGT_{task_str}",
                f"the simulated ILGT {task_str} dataset",
            )
        directory = Path(directory)

        def load_split(split):
            data = torch.tensor(
                np.load(f"{directory}/ilgt_{split}_configs.npy"),
            ).float().permute(0, 3, 1, 2)
            labels = torch.tensor(
                np.load(f"{directory}/ilgt_{split}_labels.npy"),
                dtype=torch.float32
            )

            # Filter by label values if specified; only keep samples whose label is in filter_label_values
            # Useful when using our data with RSMI-NE
            if filter_label_values is not None:
                mask = torch.tensor([lbl.item() in filter_label_values for lbl in labels])
                data = data[mask]
                labels = labels[mask]
            perm = torch.randperm(len(data))
            samples = list(zip(data[perm], labels[perm]))
            times = [float(lbl) for lbl in labels[perm]]
            return samples, times

        self.train_samples, self.train_times = load_split("training")
        self.val_samples, self.val_times = load_split("test")

        self.unique_labels = torch.unique(
            torch.stack([lbl for _, lbl in self.train_samples + self.val_samples])
        ).cpu().numpy()

        if learning_by_confusion:
            assert partition_index is not None, "Must provide partition_index when learning_by_confusion=True"
            unique_tuning = np.unique(np.array(self.train_times + self.val_times))
            threshold, self.no_partitions = partition_threshold(unique_tuning, partition_index)
            self.train_samples, self.train_times = relabel_by_threshold(
                self.train_samples, self.train_times, threshold)
            self.val_samples, self.val_times = relabel_by_threshold(
                self.val_samples, self.val_times, threshold)
            print(f"\n [LbC] CREATING DATASET: partition {partition_index+1}/{self.no_partitions}, "
                  f"threshold={threshold:.4f}, total {len(self.train_samples) + len(self.val_samples)} samples.")


class TFIMDataProcessor():
    """
    Processor for 1d, TFIM (Transverse Field Ising Model) dataset.
    - The base size to be used here is (150, 1), compatibility with datasets of different sizes can also be added.
    - datset itself is enabled for both continuous or discreet labels.
    - For now, I only implement single basis loading.

    Snapshots are stored as ``(1, N, 1)`` tensors, i.e. a one-column image, so the
    same 2D convolutional branches used for the Rydberg lattices apply unchanged: a
    ``(k, 1)`` kernel reads a k-site chain segment. Kernel shapes wider than one
    column simply see nothing, so ``smallkernels`` is the sensible choice here.

    Like ILGT, this dataset ships its own train/test split and therefore exposes
    ``train_samples``/``val_samples`` plus ``train_times``/``val_times``.
    """
    def __init__(self, basis="Z", phase_path="FM_PM", discrete_labels=True, label_param=None,
                 directory=None, learning_by_confusion=False, partition_index=None):
        if not discrete_labels:
            assert label_param == 'g', "For continuous labels in TFIM, label_param must be 'g'."
        assert len(basis) == 1, "Loading of multiple bases is not supported for now. Please just one of 'Z', 'X', 'Y'."

        self.grid_size = (150, 1)
        self.label_param = label_param
        # As for ILGT, the transverse field g is the tuning parameter and the label at
        # once, so there is no separate time -> parameter table.
        self.label_dict = None

        if directory is None:
            directory = require_dataset_dir(
                f"simulated/1D_TFIM/{phase_path}",
                f"the simulated 1D TFIM dataset ({phase_path})",
            )
        directory = Path(directory)

        self.g_dict_dict = {}

        def load_split(directory, split, basis="z"):
            split_dir = directory.joinpath(f"{split}_set").joinpath(f"snapshots_{basis.lower()}")
            if not split_dir.is_dir():
                raise DatasetNotAvailableError(
                    f"The 1D TFIM {split} split in the {basis.upper()} basis was not found at "
                    f"'{split_dir}'. Check that the dataset was downloaded in full and that "
                    f"the requested basis exists for it."
                )
            parameters = self._extract_data_from_folder(split_dir, parse_cl=discrete_labels)

            self.g_dict_dict[split] = dict(zip(parameters['g'], parameters['g']))

            n_reals = parameters['data'][1].shape[1]
            N = parameters['data'][1].shape[0]

            # g is the tuning parameter of the sweep; each csv holds n_reals snapshots
            # at one value of it.
            tuning = np.repeat(parameters['g'], n_reals)

            if discrete_labels:
                # The phase label is recorded per file, in the `class=` field of its name.
                label_values = np.repeat(parameters['class'], n_reals)
            else:
                label_values = tuning

            data_points = np.concatenate(
                [data_pt.to_numpy() for data_pt in parameters["data"]], axis=1
            ).T.reshape(-1, N, 1)

            # (N, 1) -> (1, N, 1): the leading axis is the channel the conv branches read.
            samples = [
                (
                    torch.tensor(snapshot, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(label, dtype=torch.float32),
                )
                for snapshot, label in zip(data_points, label_values)
            ]
            return samples, [float(t) for t in tuning]

        self.train_samples, self.train_times = load_split(directory, "training", basis=basis)
        self.val_samples, self.val_times = load_split(directory, "test", basis=basis)

        self.unique_labels = np.unique(
            np.array([float(lbl) for _, lbl in self.train_samples + self.val_samples])
        )

        if learning_by_confusion:
            assert partition_index is not None, "Must provide partition_index when learning_by_confusion=True"
            unique_tuning = np.unique(np.array(self.train_times + self.val_times))
            threshold, self.no_partitions = partition_threshold(unique_tuning, partition_index)
            self.train_samples, self.train_times = relabel_by_threshold(
                self.train_samples, self.train_times, threshold)
            self.val_samples, self.val_times = relabel_by_threshold(
                self.val_samples, self.val_times, threshold)
            print(f"\n [LbC] CREATING DATASET: partition {partition_index+1}/{self.no_partitions}, "
                  f"threshold={threshold:.4f}, total {len(self.train_samples) + len(self.val_samples)} samples.")

    def _extract_data_from_folder(self,
        path_to_folder,
        verbose=False,
        parse_cl=True,
        multiple=None,
        phys_model="TFIM",
    ):
        p = path_to_folder
        print(f"Folder : ./{p}")

        assert p.is_dir(), f"Folder {p} does not exist!"
        files = [
            el.name
            for el in p.rglob("*")
            if (re.search(".csv", el.name) is not None)
        ]
        if "TFIM" in phys_model:
            parameters = {"N": [], "J": [], "g": [], "data": [], "class": []}
        else:
            raise ValueError("This model is not supported yet!")
        files.sort()
        for f in files:
            if multiple is None:
                N = int(
                    re.search(
                        r"\d+\.*\d*", re.search(rf"N=\d+\.*\d*", f).group() # type: ignore
                    ).group() # type: ignore
                )
                if parse_cl:
                    cl = int(
                        re.search(
                            r"\d+\.*\d*", re.search(rf"class=\d+\.*\d*", f).group() # type: ignore
                        ).group() # type: ignore
                    )
                df = pd.read_csv(p.joinpath(f"{f}"))
                parameters["N"].append(N)
                if parse_cl:
                    parameters["class"].append(cl)
                parameters["data"].append(df)
                if "TFIM" in phys_model:
                    J = float(
                        re.search(
                            r"[-]*\d+\.*\d*",
                            re.search(rf"J=[-]*\d+\.*\d*", f).group(), # type: ignore
                        ).group() # type: ignore
                    )
                    g = float(
                        re.search(
                            r"[-]*\d+\.\d+", re.search(rf"g=[-]*\d+\.\d+", f).group() # type: ignore
                        ).group() # type: ignore
                    )
                    parameters["J"].append(J)
                    parameters["g"].append(g)
                    if verbose:
                        print(f"N = {N} | J = {J} | g = {g:.2f} | Filename: {f}")
                    del N, J, f, df, g
                else:
                    raise ValueError("This model is not supported yet!")
            else:
                N = int(
                    re.search(
                        r"\d+\.*\d*", re.search(rf"N=\d+\.*\d*", f).group() # type: ignore
                    ).group() # type: ignore
                )
                if N == multiple:
                    if parse_cl:
                        cl = int(
                            re.search(
                                r"\d+\.*\d*",
                                re.search(rf"class=\d+\.*\d*", f).group(), # type: ignore
                            ).group() # type: ignore
                        )
                    df = pd.read_csv(p.joinpath(f"{f}"))
                    parameters["N"].append(N)
                    if parse_cl:
                        parameters["class"].append(cl)
                    parameters["data"].append(df)
                    if "TFIM" in phys_model:
                        J = float(
                            re.search(
                                r"[-]*\d+\.*\d*",
                                re.search(rf"J=[-]*\d+\.*\d*", f).group(), # type: ignore
                            ).group() # type: ignore
                        )
                        g = float(
                            re.search(
                                r"[-]*\d+\.\d+",
                                re.search(rf"g=[-]*\d+\.\d+", f).group(), # type: ignore
                            ).group() # type: ignore
                        )
                        parameters["J"].append(J)
                        parameters["g"].append(g)
                        if verbose:
                            print(
                                f"N = {N} | J = {J} | g = {g:.2f} | Filename: {f}"
                            )
                        del N, J, f, df, g
                    else:
                        raise ValueError("This model is not supported yet!")
                else:
                    continue
        if "TFIM" in phys_model:
            parameters["J"] = np.array(parameters["J"]) # type: ignore
            parameters["g"] = np.array(parameters["g"]) # type: ignore
            order = np.argsort(parameters["g"])
            parameters["g"] = parameters["g"][order] # type: ignore
            parameters["J"] = parameters["J"][order] # type: ignore
        else:
            raise ValueError("This model is not supported yet!")
        parameters["N"] = np.array(parameters["N"])[order] # type: ignore
        if parse_cl:
            parameters["class"] = np.array(parameters["class"])[order] # type: ignore
        parameters["data"] = [parameters["data"][i] for i in order]
        return parameters

        

class XXZDataProcessor():
    """
    Processor for 1d, XXZ dataset.
    - (300, 1) grid size
    - supports only continuous labels.

    The anisotropy Jz is both the label and the tuning parameter of the sweep.
    Snapshots are stored as ``(1, 300, 1)`` tensors, the same one-column image
    convention as the 1D TFIM dataset.

    This dataset is not distributed with the repository; see the README for how to
    obtain it. Constructing the processor without it raises
    ``DatasetNotAvailableError`` rather than failing deeper in the loader.
    """
    def __init__(self, discrete_labels=False, label_param="Jz", directory=None,
                 learning_by_confusion=False, partition_index=None):
        # Discrete labels reach this processor only through learning by confusion, which
        # derives them by thresholding Jz below. Plain classification would need a phase
        # label the dataset does not carry, exactly as for the Rydberg processors.
        if discrete_labels and not learning_by_confusion:
            raise NotImplementedError(
                "The XXZ dataset carries no phase labels, so task='classification' is not "
                "available for it. Use task='regression' (label_param='Jz'), or "
                "task='lbc' / task='partition', which derive binary labels by "
                "thresholding the anisotropy."
            )

        self.grid_size = (300, 1)
        self.label_param = label_param
        self.label_dict = None

        if directory is None:
            directory = require_dataset_dir('simulated/XXZ_L', 'the simulated 1D XXZ dataset')
        directory = Path(directory)

        # Snapshot files are named Sz_...Jz<value>....txt; the anisotropy is parsed
        # out of the filename, as there is no separate parameter table.
        self.samples = []
        self.sample_times = []

        for filepath in sorted(directory.iterdir()):
            filename = filepath.name
            if not (filename.startswith('Sz') and filename.endswith('.txt')):
                continue
            match = re.search(r'Jz([-+]?[0-9]*\.?[0-9]+)', filename)
            if not match:
                continue
            label = float(match.group(1))
            data = np.loadtxt(filepath)
            reshaped = data.reshape(self.grid_size[0], self.grid_size[1])
            self.samples.append((
                torch.tensor(reshaped, dtype=torch.float32).unsqueeze(0),
                torch.tensor(label, dtype=torch.float32)
            ))
            self.sample_times.append(label)

        if not self.samples:
            raise DatasetNotAvailableError(
                f"No XXZ snapshot files found in '{directory}'. Expected files named "
                f"'Sz...Jz<value>...txt'."
            )

        labels = torch.stack([lbl for _, lbl in self.samples]).cpu().numpy()
        unique_labels = np.unique(labels)
        self.unique_labels = unique_labels

        if learning_by_confusion:
            assert partition_index is not None, "Must provide partition_index when learning_by_confusion=True"
            threshold, self.no_partitions = partition_threshold(unique_labels, partition_index)
            self.samples, self.sample_times = relabel_by_threshold(
                self.samples, self.sample_times, threshold)
            print(f"\n [LbC] CREATING DATASET: partition {partition_index+1}/{self.no_partitions}, "
                  f"threshold={threshold:.4f}, total {len(self.samples)} samples.")
