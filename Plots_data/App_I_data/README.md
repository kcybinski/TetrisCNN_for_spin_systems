# App_I_data — trained models + PySR fits behind Fig. 25 (App. I)

`FiguresApp_I.ipynb` reads only the `SR_raw_{para,ferro,antiferro}_sr_plot_data.pkl`
files here (SR-prediction vs NN-prediction parity on the three OOD spin-config
families). `model_sr_raw.pkl`, `sr_config.json` and `equations/hof_*` are kept for
provenance. The augmentation `.pdf/.png/.html` plots PySR writes alongside are
regenerable and git-ignored.

Symbolic regression is App. I of the manuscript; its parity figure, Fig. 25, is written
from these files as `Fig_G1_2col.pdf` (that output name predates the final appendix order).

The notebook has a `VARIANT` switch (`"plain"` default, or `"C4"`). Each variant
reads one classifier fit (top row) and one Omega-regression fit (bottom row):

| folder | arch | task | Fig. 20 col | SR | notes |
|---|---|---|---|---|---|
| `models/Ising_CLF/SR_clf_sr=single_logit_topk=1` | plain | partition (classifier) | 1 | `top_k=1`, `single_logit` | bottleneck → 1 branch; SR generalises OOD (R²_syn ≈ 1) |
| `models/Ising_PDM_Omega/SR_default` | plain | regression on Ω | 3 | `top_k=3` (default) | branches [1,2,4] of 10; SR fits in-dist (R²≈0.91) but fails OOD |
| `models/Ising_CLF_equivariant/SR_clf_sr=single_logit_topk=1` | C4 | partition (classifier) | 1 | `top_k=1`, `single_logit` | bottleneck → 1 branch; SR generalises OOD (R²_syn ≈ 1) |
| `models/Ising_PDM_Omega_equivariant/SR_default` | C4 | regression on Ω | 3 | `top_k=3` (default) | see λmax note below |

## Provenance

The **plain** models are SR fits of the checked-in App. F 6 models
(`App_F_data/models/Ising_{CLF,PDM_Omega}`), which were retrained on the fixed
codebase by `scripts/appF_runs.py` (λ ramp `[10, -3, 1, 1]`, `batch_size=64`,
`even_split`, seed 42). So Fig. 25's rows are the SR companions of Fig. 20's
columns 1 and 3 — the same front-ends.

The **C4** models mirror that exact recipe with `equivariant=True`,
`equivariant_group="C4"` (kernel set rebuilt to the 5 canonical C4 orbit
representatives), **except** the C4 Ω-regression run, which uses a **lowered
`λmax = -1`** (`penalty_params [10, -3, -1, 1]`). At the matched `λmax = 1` the
C4 Ω bottleneck collapses to a single branch (orbit averaging folds the
orientation partners together, so `smallkernels` only has 5 branches to start);
`λmax = -1` is the point where it keeps a genuine multi-component bottleneck
(`z_k ≈ [-0.040, 0.037, 0, -0.008, ~0]`, i.e. branches [0, 1] near-equal plus a
weak [3]), val R² 0.80. A short `λmax` sweep (0, -0.5, -1, -1.5, -2) confirmed the
accuracy is flat across it (0.80–0.81) — only the representation changes.

All PySR fits: `niterations=5000`, `gen_samples=2500`, `num_flips=4`, default
operator basis (see each `sr_config.json`).

## Hall-of-Fame leaderboards (Tab. VII, App. I)

`models/Ising_CLF/.../equations/` was regenerated after the fact: that run was
copied here without its `equations/` folder, and the manuscript's Tab. VII
needs the per-equation `r2_val`, `accuracy_val` and OOD `r2_*`/`rho_*` columns. It
was recreated **without refitting PySR** — `sr_toolbox.py` Scenario A against an SR
subfolder, with `fit_sr=False, remake_sr_plot=True`, so `model_sr_raw.pkl` is
reloaded and only the metrics are recomputed. The NN weights for the two plain
runs come from `App_F_data/models/{Ising_CLF,Ising_PDM_Omega}/`, since their
folders here hold no `net*.pt`; the two equivariant runs carry their own.

**Reproducibility.** The OOD probe configurations used to be redrawn on every
call, which left the OOD columns (and hence the Borda `optimized_picked` row)
irreproducible: an independent re-score moved `r2_para` by up to 0.7 and flipped
the Ω Borda pick from complexity 19 to 25. `SRConfig.gen_seed` (default 2137) now
seeds a single `np.random.Generator` threaded through every draw in
`generate_all_generalization_configs`, and all runs here were regenerated with it;
`gen_seed: null` restores the old fresh-draw behaviour. A re-score from a separate
directory now reproduces both `equations/*.csv` and `SR_raw_*_sr_plot_data.pkl`
byte-for-byte.

`plots.py` also stopped writing `SR_raw_*_sr_plot_data.pkl` at some point, so the
copies here had gone stale relative to the code that reads them
(`FiguresApp_I.ipynb`). That dump is restored in `_plot_single_output`, now in
`tetriscnn/symbolic_regression.py`. Note that
those pickles now hold the **validation** split and the **Borda-selected**
equation; the older checked-in copies held the training split and PySR's own pick.
