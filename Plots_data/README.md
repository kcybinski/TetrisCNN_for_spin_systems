# Plots_data — trained models behind the manuscript figures

This directory ships as a GitHub release asset, the same way `datasets/` does (see
the top-level README's Installation section): it is held out of the git tree
(`.gitignore` keeps only this file, `MANIFEST.sha256`, and `App_I_data/README.md`
tracked) and unpacked from a release zip so `Plots_data/` sits at the repository
root. Verify a copy with `python scripts/plots_data_manifest.py --check`, and
regenerate the manifest after an intentional change with `--write`.

Two files directly under `Plots_data/` are not model folders: `results_LBC_Lambdamax_
Ising_summary.pkl` and `results_LBC_Lambdamax_XY_XZ_summary_new.pkl` are the pickled
$\lambda_{\max}$-sweep LBC summaries `Figure4.ipynb` reads as input (produced by
`scripts/lbc_lambdamax_summary.py`). They live here, not under `Plots/`, because they
are not regenerable figure output.

## Main-text figures: single-run and swept models

Each subfolder directly under `Plots_data/` (not nested under an `App_*_data/`
below) is one trained TetrisCNN run, copied out of the (gitignored, not released)
`logs/` tree so the `notebooks/Figure*.ipynb` notebooks run from a fresh clone.
Every model folder holds exactly `net1.pt`, `net2.pt`, `config.json`,
`metrics.json`; load it with the `_load_config_from_folder` /
`_load_model_and_datasets` helpers at the top of each figure notebook. Spin
snapshots are reconstructed from `data/` via `create_datasets(cf)`, so the
experimental `data/` tree must be present.

Model-folder name:
`<dataset>_<arch>_<kernelset>_lam<lambda_max>_seed<seed>_p<partition>`
(`arch` = `C4` for the C4-equivariant branches, `plain` for the standard ones).
All are the `partition` task.

## Single-run models

| Folder | Original `logs/logs_after_bugfix/...` path | Used by |
|---|---|---|
| `Ising_C4_smallkernels_lam3_seed42_p2` | `..._Ising_partition_t_smallkernels_C4_[-3,...,5]/lambdamax_3/seed_42/partition_2` | Figure4, Figure6, Figures5_6_combined_pipeline |
| `XY_XZ_C4_smallkernels_lam3_seed46_p2` | `..._XY_XZ_partition_t_smallkernels_C4_[-3,...,5]/lambdamax_3/seed_46/partition_2` | Figure4, Figure5, Figure6, Figures5_6_combined_pipeline |
| `XY_XZ_plain_smallkernels_lam3_seed46_p2` | `..._XY_XZ_partition_t_smallkernels_[-3,...,5]/lambdamax_3/seed_46/partition_2` | Figure6 / Figures5_6_combined_pipeline (non-equivariant comparison) |
| `Ising_plain_smallkernels_lam3_seed46_p2` | `..._Ising_partition_t_smallkernels_[-3,...,5]/lambdamax_3/seed_46/partition_2` | Figure6 (non-equivariant comparison) |
| `XY_X_C4_smallkernels_lam3_seed46_p2` | `..._XY_X_partition_t_smallkernels_C4_[-3,...,5]/lambdamax_3/seed_46/partition_2` | FiguresApp_F_error_rates panel (b), unmatched (also the seed-46 basis-ablation reference) |
| `XY_Z_C4_smallkernels_lam3_seed46_p4` | `..._XY_Z_partition_t_smallkernels_C4_[-3,...,5]/lambdamax_3/seed_46/partition_4` | FiguresApp_F_error_rates panel (b), unmatched (Z on its native 21-time grid, partition_4 / 587.5ns) |

### 5-seed folders for mean ± std error-rate figures (FiguresApp_F_error_rates\*)

Both `FiguresApp_F_error_rates.ipynb` and `FiguresApp_F_error_rates_equivariance.ipynb`
plot per-time-point error rate as **mean ± std over 5 seeds** (42–46), not a single
seed's point estimate, so every model they use needs all 5 seed folders present, not
just the one canonical seed the interpretability figures use. These follow the same
naming convention with the seed varying and are otherwise identical single-run
copies:

- `Ising_C4_smallkernels_lam3_seed{42,43,44,45,46}_p2`  — panel (a)
- `XY_XZ_C4_smallkernels_lam3_seed{42,43,44,45,46}_p2`  — panel (b) unmatched XZ (also the non-equivariant-comparison arm of `FiguresApp_F_error_rates_equivariance.ipynb`)
- `XY_XZ_plain_smallkernels_lam3_seed{42,43,44,45,46}_p2` (equivariance notebook only)
- `XY_X_C4_smallkernels_lam3_seed{42,43,44,45,46}_p2`  — panel (b) unmatched X (native 13-time grid, partition_2)
- `XY_Z_C4_smallkernels_lam3_seed{42,43,44,45,46}_p4`  — panel (b) unmatched Z (native 21-time grid, partition_4 / 587.5ns)
- `XY_X_C4_smallkernels_lam3_seed{42,43,44,45,46}_p2_matched`  — panel (c)
- `XY_Z_C4_smallkernels_lam3_seed{42,43,44,45,46}_p2_matched`  — panel (c)
- `XY_XZ_C4_smallkernels_lam3_seed{42,43,44,45,46}_p2_matched` — panel (c), `XZ_SNAPSHOTS="equal"`
- `XY_XZ_C4_smallkernels_lam3_seed{42,43,44,45,46}_p2_matched_xzhalf` — panel (c), `XZ_SNAPSHOTS="half"`

`FiguresApp_F_error_rates.ipynb` draws the three panels of Fig. 19: (a) the main-text
Ising model, (b) the main-text XY model (XZ) on its native acquisition grid, and (c) the
X, Z and XZ networks retrained on the snapshot-count-matched setup. The unmatched X-only
and Z-only folders feed a diagnostic version of panel (b) that the notebook keeps
commented out; the manuscript does not show it. The `_p2` (unmatched X/XZ) and `_p4`
(unmatched Z) seeds 42–45 were copied from each basis's own `lambdamax_3` sweep; seed 46
of those was already present.

Only the seed 42/46 members of these sets were already copied for other figures
(interpretability figures are seed-representative, not seed-averaged); the rest
(43–46, or 42–45) were added solely for the two error-rate notebooks.

### Snapshot-count-matched X/Z/XZ (Fig. 19(c), `model_error_rates_val_half.pdf`)

The un-matched X/Z/XZ rows above compare the three bases on their own native
acquisition grids: Z has 21 time points (partition_4, 587.5ns threshold) against
X/XZ's 13 (partition_2, 625ns), and each basis carries a different number of
snapshots per time point. `scripts/matched_basis_survey.py` retrains all three on
the *same* 13 time points, the *same* per-timepoint snapshot cap
(`min(n_X(t), n_Z(t))`, XZ's own pairing count), and the *same* partition_2/625ns
threshold, so the three are directly comparable. See
App. F 5 of the manuscript for the result (XZ beats Z alone once matched, the
opposite of the un-matched ordering).

| Folder | Original `logs/logs_after_bugfix/...` path | Used by |
|---|---|---|
| `XY_X_C4_smallkernels_lam3_seed{42-46}_p2_matched` | `..._XY_X_partition_t_smallkernels_C4_matched/lambdamax_3/seed_{42-46}/partition_2` | FiguresApp_F_error_rates |
| `XY_Z_C4_smallkernels_lam3_seed{42-46}_p2_matched` | `..._XY_Z_partition_t_smallkernels_C4_matched/lambdamax_3/seed_{42-46}/partition_2` | FiguresApp_F_error_rates |
| `XY_XZ_C4_smallkernels_lam3_seed{42-46}_p2_matched` | `..._XY_XZ_partition_t_smallkernels_C4_matched/lambdamax_3/seed_{42-46}/partition_2` | FiguresApp_F_error_rates (`XZ_SNAPSHOTS="equal"`) |
| `XY_XZ_C4_smallkernels_lam3_seed{42-46}_p2_matched_xzhalf` | `..._XY_XZ_partition_t_smallkernels_C4_matched_xzhalf/lambdamax_3/seed_{42-46}/partition_2` | FiguresApp_F_error_rates (`XZ_SNAPSHOTS="half"`, the notebook default and the published Fig. 19(c)) |

The `_matched_xzhalf` XZ folders (`scripts/matched_basis_survey_xzhalf.py`) are the
same matched setup with the XZ per-timepoint cap **halved**
(`min(n_X(t), n_Z(t)) // 2` paired examples). Because an XZ example bundles one
X-basis and one Z-basis snapshot, this gives XZ the *same raw-snapshot budget* as
the (unchanged) X-alone and Z-alone `_matched` arms rather than 2x. Only XZ was
retrained; `FiguresApp_F_error_rates.ipynb`'s `XZ_SNAPSHOTS` switch selects between
the two, and X/Z read the same `_matched` folders either way. With the halved cap, XZ
still beats Z alone on an equal raw-snapshot budget (93.07% against 91.19%, App. F 5).

The canonical C4 runs (Ising seed 42, XY seed 46, both lambda_max = 3) are the
models every interpretability result in the paper is read off; the same folder
is reused everywhere those models appear rather than re-copied. Seed choice is
not load-bearing (the sparsity story is seed-stable); `lambda_max = 3` is.

**Note on the manuscript's Tab. III.** The table lists "RNG seed 42" for both
classifiers, but the XY results quoted in the paper (Figs. 5, 6, 19(b), 22, 24,
Eqs. (13)–(14) and App. H) come from the seed-46 run above; only the Ising model is
seed 42. Pointing the notebooks at `XY_XZ_C4_smallkernels_lam3_seed42_p2` instead
selects the same three branches and the same leading correlators, with small
numerical differences: validation accuracy 93.49% (paper 93.54%), Fig. 5 fit
R² 1.00 / 0.96 / 0.97 (paper 1.00 / 0.97 / 0.98), Eq. (14) becomes
0.65 X² + 0.16 X + Y − 0.78 Z + 0.35 = 0 at 98.23% (paper 0.68 X² + 0.21 X + Y −
0.81 Z + 0.35 = 0 at 98.30%), the quadratic fit in activation space reaches 99.72%
(paper 99.84%), and the two-correlator plane of App. H 2 reaches 97.1% (paper
96.53%). The regularization path (Figs. 22–23) keeps the same four dominant terms.

## lambda-max sweeps (`sweeps/`)

Each folder holds the sweep-level `metrics_per_lambda.json` plus one
representative seed's `config.json` (kernel set + equivariance flag). No model
weights — FiguresApp_F_lambdamax/FiguresApp_F_scale_search only plot the per-lambda metric curves.

| Folder | Used by |
|---|---|
| `sweeps/Ising_C4_smallkernels_partition`, `sweeps/XY_XZ_C4_smallkernels_partition` | FiguresApp_F_lambdamax (equivariant panel) |
| `sweeps/Ising_plain_smallkernels_partition`, `sweeps/XY_XZ_plain_smallkernels_partition` | FiguresApp_F_lambdamax (non-equivariant panel) |
| `sweeps/Ising_plain_bigkernels_partition`, `sweeps/XY_XZ_plain_bigkernels_partition` | FiguresApp_F_scale_search (scale search) |

## Appendix figures: `App_*_data/`

The appendix notebooks each read one of these subfolders rather than the main-text
model folders above. They keep the appendix letter in their name for the same
reason the main-text folders keep their own naming convention: each was copied out
of a different `logs/` run tree, at a different time, for a different notebook.

| Folder | Read by | Contents |
|---|---|---|
| `App_A_data/` | `FiguresApp_A.ipynb` | `{Ising,XY_XZ,XY_Z}.pickle`, precomputed per-time-point snapshot counts; `_old_regression/` is an earlier cut, kept for comparison. |
| `App_D_data/` | `FiguresApp_D.ipynb` | `models/XY_XZ_CLF` and `models/XY_XZ_CLF_no_lr_schedule`, the two runs (`ReduceLROnPlateau` vs. not) behind the noise-floor figure. |
| `App_E_data/` | `FiguresApp_E.ipynb`, `FiguresApp_E_clustering.ipynb` | `models/` (Ising/XY PDM and CLF checkpoints for the predictive-diagnostic panel), `lbc_metrics/{Ising,XY}.pkl` (pre-aggregated learning-by-confusion curves), `clustering/{Ising,XY}/` (the clustering-figure runs). |
| `App_F_data/` | `FiguresApp_F.ipynb` | `models/Ising_{CLF,PDM_Delta,PDM_Omega}`, the task-dependence comparison retrained by `scripts/appF_runs.py` after the reshape/L1-penalty fixes (see that script's docstring). |
| `App_I_data/` | `FiguresApp_I.ipynb` | PySR fits (plain and C4-equivariant) behind the symbolic-regression parity figure; see `App_I_data/README.md` for full provenance, which is the one README in this tree kept tracked in git rather than shipped only in the release zip. |
