# Datasets

Everything TetrisCNN trains on lives under this directory. Two kinds of data are
kept apart, because they differ in origin, in licence, and in what they are good for:

- `experimental/` holds raw snapshots from Rydberg-array quantum simulators. These
  are the data the manuscript analyses.
- `simulated/` holds numerically generated configurations. TetrisCNN was first
  developed as a proof of concept on these, and they remain the fastest way to see
  the method work end to end.

The code finds this directory by searching upwards from the working directory for a
folder named `datasets/` (or, for older working trees, `data/`). Set
`TETRISCNN_DATA_ROOT` to point somewhere else, for instance at a copy kept on a
scratch disk:

```bash
export TETRISCNN_DATA_ROOT=/scratch/tetriscnn/datasets
```

A dataset a config asks for but that is not on disk raises `DatasetNotAvailableError`
naming the path it looked in, rather than failing deeper inside a loader.

## Layout

```
datasets/
├── experimental/
│   ├── Ising/
│   │   └── Ising_Z_8x8c_0_defects_0_holes/     8x8 transverse-field Ising, Z basis
│   └── XY/
│       ├── 6x7_atoms_X_ferro_video_15MHz_tau0p3_17032022_0_defects_0_holes/
│       └── 6x7_atoms_Z_ferro_video_15MHz_tau0p3_16032022_7_0_defects_0_holes/
└── simulated/
    ├── 1D_TFIM/
    │   ├── FM_PM/          ferromagnetic-to-paramagnetic sweep
    │   └── AFM_PM/         antiferromagnetic-to-paramagnetic sweep
    ├── ILGT_classification/
    └── ILGT_regression/
```

The directory names are load-bearing: `tetriscnn/dataprocessing.py` refers to them
verbatim, so renaming one breaks the default path of the processor that reads it.

## The experimental snapshots

Both experimental datasets are projective measurements of a Rydberg tweezer array,
acquired at a sequence of times along an approximately adiabatic sweep of the
Hamiltonian. They are not necessarily ground states: they carry measurement noise, finite-size and edge effects, and the non-equilibrium character of a finite-duration ramp. The number of snapshots varies by up to an order of magnitude across the sweep, which is
what motivates the even-split and sample-capping options described in the main README.

**Ising, 8x8, Z basis** (`Ising_Z_8x8c_0_defects_0_holes`). 28 sweep steps, 64 atoms.
The transverse field and detuning are ramped together; `time_in_ns.dat` indexes the
sweep and `omega_in_MHz.txt` / `delta_in_MHz.txt` give the Hamiltonian parameters at
each step. From Scholl *et al.*, *Quantum simulation of 2D antiferromagnets with
hundreds of Rydberg atoms*, Nature **595**, 233 (2021).

**XY, 6x7, X and Z bases** (`6x7_atoms_{X,Z}_ferro_...`). A dipolar XY model with a
staggered longitudinal field. The two bases come from two independent experimental
runs, so a snapshot in one has no physical partner in the other; how they are stacked
into a two-channel image is a free convention, controlled by `pairing_mode` (see the
main README). `lightshift_in_MHz.txt` gives the staggered field at each step of
`time_in_ns.dat`. From Chen *et al.*, *Continuous symmetry breaking in a
two-dimensional Rydberg array*, Nature **616**, 691 (2023).

Each directory holds one `t=<time>_ns.dat` file per sweep step, plus its `readme.txt`
describing the storage convention, and the atom position tables. A `.dat` file is a
plain text table: one row per experimental repetition, one column per atom.

**Provenance.** These snapshots are the property of the experimental teams that
produced them and are redistributed here with their permission, for the purpose of
reproducing the analysis in the accompanying manuscript. Cite the two papers above,
not this repository, when using the data itself.

## The simulated datasets

**1D TFIM** (`1D_TFIM/`). Snapshots of a 150-site transverse-field Ising chain across
a sweep of the transverse field `g`, in two parameter regimes: a ferromagnetic and an
antiferromagnetic approach to the paramagnet. Each sweep is split into
`training_set`, `val_set` and `test_set`, and within each into `snapshots_x`,
`snapshots_y` and `snapshots_z` by measurement basis. One CSV per value of `g`, named
`N=<sites>_J=<coupling>_class=<phase>_g=<field>_<basis>.csv`; columns are
repetitions, rows are sites. The phase label in the filename is what the
classification task predicts, while `g` stays the sweep coordinate.

Chain snapshots are loaded as one-column `(1, N, 1)` images, so the ordinary 2D
convolutional branches apply to them unchanged. Use `cf.kernel_set = "chainkernels"`,
which carries only vertically extended patterns: the 2D sets contain horizontal
patterns that do not fit a single column.

**2D ILGT** (`ILGT_classification/`, `ILGT_regression/`). Configurations of a 16x16
Ising lattice gauge theory, stored as `(N, 16, 16, 2)` arrays of spins; the trailing
axis is the two link orientations, which the loader moves to the channel axis. Spins
are stored as `int8`, cast to float on load. The regression set labels each
configuration with its inverse temperature `beta` and is the one to use for learning
by confusion, which needs a tuning parameter to place a threshold in; the
classification set carries only the two phase labels.

Both simulated datasets were introduced in Cybinski, Enouen, Georges and Dawid,
*Speak so a physicist can understand you! TetrisCNN for detecting phase transitions
and order parameters*, Machine Learning and the Physical Sciences Workshop at
NeurIPS 2024.

## Integrity

`MANIFEST.sha256` lists every file with its SHA-256 digest. Verify a copy with:

```bash
python scripts/dataset_manifest.py --check
```

Regenerate it after deliberately changing the tree with `--write`.
