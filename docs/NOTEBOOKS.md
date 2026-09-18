# The figure notebooks

Every figure in the manuscript that comes from a computation is produced by one
notebook under `notebooks/`, named after the figure it makes. They read runs that were
trained earlier and recorded under `Plots_data/` (main-text runs directly, appendix
runs in its `App_*_data/` subfolders), rather than retraining, so each executes in
minutes and reproduces exactly the published figure.

Each notebook opens with the question its figure answers, what it reads, and how to
read its output. Taken in order they amount to a guided tour of the method, so a
reader coming from the paper can start at Figure 4 and work outwards.

## Running them

```bash
conda activate tetriscnn
jupyter lab notebooks/          # then open any Figure*.ipynb and run all cells
```

or headless:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/FiguresApp_F_lambdamax.ipynb
```

Figures are written to `Plots/` (PDFs directly, SVGs under `Plots/SVG/`), which is
gitignored and created on first run: the last lines of each notebook's imports cell move
to the repository root (so `Plots_data/` and `Plots/` resolve whether the notebook was
launched from the root or from `notebooks/`) and create the output folders. Only
`FiguresApp_A.ipynb` reads raw snapshots, so it needs `datasets/` present; the rest work
from the recorded runs alone.

Styling is shared and deliberately rigid, so that panels match across figures:
`tetriscnn/paper.mplstyle` carries the manuscript's fonts and sizes, and
`tetriscnn/colors.py` the palette. Both ship inside the installed package, imported as
`from tetriscnn import PAPER_MPLSTYLE` and `from tetriscnn.colors import COLORS`, rather
than depending on the repo root. Changing either changes every figure at once.

Because `paper.mplstyle` sets `text.usetex`, running the notebooks requires a LaTeX
installation with the Latin Modern fonts; see the Installation section of the README. A
missing TeX binary usually surfaces as a confusing complaint about a font file such as
`cmss8.tfm` rather than about LaTeX itself. If TeX is installed but a notebook launched
from a desktop shortcut cannot find it, `from tetriscnn import texpath;
texpath.ensure_tex_on_path()` at the top of the notebook adds the standard install
locations to the `PATH`.

## Figure to notebook

Figure numbers follow the manuscript. Appendix figures are also listed with their
appendix section.

| Manuscript figure | Output file | Notebook |
| --- | --- | --- |
| Fig. 2, experimental datasets | `Fig2.pdf` | `Figure2.ipynb` (base panel only, see below) |
| Fig. 4, interpretable detection | `Fig4.pdf` | `Figure4.ipynb` |
| Fig. 5, mapping branch activations to correlators (XY) | `Fig5.pdf` | `Figure5.ipynb` |
| Fig. 6, decision boundaries | `Fig6_C4.pdf` | `Figure6.ipynb`, with `RUN_TAG = "C4"` |
| Fig. 7 (App. A 1), snapshot counts | `Fig_A1.pdf` | `FiguresApp_A.ipynb` |
| Fig. 8 (App. A 2), XY staggered detuning of the two bases | `Fig_A2.pdf` | `FiguresApp_A.ipynb` |
| Fig. 10 (App. D 3), learning-rate annealing and the activation floor | `Fig_D1.pdf` | `FiguresApp_D.ipynb` |
| Fig. 11 (App. E 1), learning by confusion | `Fig_E1.pdf` | `FiguresApp_E.ipynb` |
| Fig. 12 (App. E 2), prediction-divergence method | `Fig_E2.pdf` | `FiguresApp_E.ipynb` |
| Fig. 13 (App. E 3), PCA/UMAP embeddings | `Fig_E_clustering.pdf` | `FiguresApp_E_clustering.ipynb` |
| Fig. 15 (App. F 1), λmax robustness, rotationally invariant | `Fig_lambdamax_equivariant.pdf` | `FiguresApp_F_lambdamax.ipynb` |
| Fig. 16 (App. F 1/F 3), λmax robustness, unconstrained | `Fig_lambdamax_nonequivariant.pdf` | `FiguresApp_F_lambdamax.ipynb` |
| Fig. 17 (App. F 2), searching for the relevant scale | `Fig_scale_search.pdf` | `FiguresApp_F_scale_search.ipynb` |
| Fig. 18 (App. F 3), invariant vs unconstrained error rates | `model_error_percent_val_equivariance.pdf` | `FiguresApp_F_error_rates_equivariance.ipynb` |
| Fig. 19 (App. F 5), per-time error rates, X vs Z vs XZ | `model_error_rates_val_half.pdf` | `FiguresApp_F_error_rates.ipynb` |
| Fig. 20 (App. F 6), task dependence | `Fig_F1.pdf` | `FiguresApp_F.ipynb` |
| Fig. 21 (App. F 7), complete mapping to correlators | `Fig5_full.pdf` | `Figure5.ipynb` |
| Fig. 22 (App. H 1), regularization path, one network | `optpath_seed46.pdf` | `FiguresApp_H_regularization_path.ipynb` |
| Fig. 23 (App. H 1), regularization path, averaged over initializations | `optpath_modelseed_avg.pdf` | `FiguresApp_H_regularization_path.ipynb` |
| Fig. 24 (App. H 2), plane vs quadratic boundary error | `Fig_xy_plane_vs_square_time.pdf` | `Figure6.ipynb` |
| Fig. 25 (App. I), symbolic-regression parity | `Fig_G1_2col.pdf` | `FiguresApp_I.ipynb`, with `VARIANT = "plain"` |

Notebook names follow the manuscript: `Figure*.ipynb` for main-text figures (some also
produce the appendix versions of those figures) and `FiguresApp_<appendix>*.ipynb` for
appendix figures. One output name predates the final appendix order: `Fig_G1_2col.pdf`
is the symbolic-regression figure of App. I (Fig. 25), kept under the file name the
manuscript includes.

One notebook is a tutorial rather than a figure source. `BYODataset_Tutorial.ipynb`
shows how to train TetrisCNN on data of your own: it writes
a small synthetic dataset to a temporary directory, reads it with a processor built on
`SnapshotProcessor`, registers it with `register_dataset`, trains, and checks that the surviving branches
measure the correlators the data was built to contain. It needs neither `datasets/` nor
`Plots_data/`.

`Figures5_6_combined_pipeline.ipynb` produces no figure of its own. It is kept because it is the
source of the Ising activation-to-correlator fit and of the decision-boundary accuracy
table quoted in the text.

### Three caveats worth knowing before re-running

**Figure 2 is composited by hand.** The notebook produces the base panel (`Fig2.pdf`)
and the nine representative-snapshot images it needs (`Plots/SVG/Fig2_snapshots/`).
The version the manuscript includes has those snapshots placed into panels (a),(c) and
its colour space prepared for print, both done outside the notebook. Re-running
reproduces the panel and the snapshot images, not the final composite.

**Figure 6's output carries a tag.** `Figure6.ipynb` writes `Fig6_C4.pdf` under
its default `RUN_TAG = "C4"`, and the manuscript's `Fig6.pdf` is that file renamed on
export. A plain `Fig6.pdf` in `Plots/` is an older run with `RUN_TAG = "plain"`,
not the published figure.

**Four figures have no notebook here.** The introduction schematic (Fig. 1), the
convolution example (Fig. 3) and the architecture diagram (Fig. 9) were drawn by hand.
The distance-learning comparison (Fig. 14, App. E 4) was generated with code adapted
from the repository accompanying Malyshev *et al.*, as its caption states.

## Where the quoted numbers come from

Numbers stated in the text are printed by a cell in one of these notebooks. The ones a
reader is most likely to want to check:

| Quantity | Printed by |
| --- | --- |
| Minimum snapshot counts, 297 (Ising) and 371 (XY) | `FiguresApp_A.ipynb` |
| XY branch fit $R^2$ values, 1.00 / 0.97 / 0.98 | `Figure5.ipynb` |
| XY sample count, 14471 | `Figure5.ipynb` |
| Ising activation fit, Eq. (12): $z[\blacksquare] = -4.626\,C^Z[\blacksquare] + 0.954$ | `Figures5_6_combined_pipeline.ipynb` |
| Ising decision threshold, $z[\blacksquare] = -0.837$, i.e. $C^Z[\blacksquare] = 0.387$ | `Figure6.ipynb` |
| XY decision boundaries, Eqs. (13)–(14) and App. H 2, with their accuracies (99.84% in activation space, 98.30% written in correlators, and the linear approximations) | `Figures5_6_combined_pipeline.ipynb` |
| Basis-comparison accuracies (99.14%, 93.54%, 91.19%, 70.28%, 93.07%) | `FiguresApp_F_error_rates.ipynb` |
| Clustering silhouette scores and transition intervals (Tab. IV) | `FiguresApp_E_clustering.ipynb` |
| Transition estimates from LBC and the prediction-divergence method | `FiguresApp_E.ipynb` |

Three sets of numbers come from elsewhere. The symbolic-regression table (Tab. VII,
App. I) is assembled by the fitting runs under `Plots_data/App_I_data/`, not by the
plotting notebook. The X/Z re-pairing robustness table (Tab. V, App. F 4) comes from
`scripts/`. The subpattern counts (Tab. II, App. B 2) come from a direct enumeration,
stated as such in the text.

## Where the recorded runs come from

`Plots_data/README.md` documents every run folder: which configuration produced it and
which figure consumes it. A run folder holds the `config.json` it was trained with, its
`metrics.json`, and the two network checkpoints, which is enough to re-derive any
quantity in the figures without retraining. To reproduce a run from scratch instead,
copy its `config.json` values into `setup_experiment()` and run `main.py`.
