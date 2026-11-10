# Analyses of the final saved predictions

These scripts organize the manuscript analyses without fitting or changing a prediction model. Install `pip install '.[analysis]'`. The Figure 3 bootstrap retains the original 32-cancer cohort check. The R scripts require `Rcpp` and `survival`; the plotting packages are listed in `figures/README.md`. `provenance.tsv` lists the original implementations.

Use a separate working directory. The scripts write derived tables to `Source_Data/` and `Analysis/`; rerunning replaces their generated tables. Patient-level predictions and tables remain local and are not bundled here.

## Survival, bootstrap and ROC

Export the 25 public Stage C outputs (`TRAINED/seed_<seed>/fold<fold>/C/`) into a new `WORK/Inputs` directory:

```bash
omnit2i-export-survival --master master_folds.csv --training-root TRAINED --output-dir WORK/Inputs
```

The command reuses saved validation predictions to recover natural-availability and complete-case risk calibration; it does not fit a model. It writes genuine gzip OOF files with boolean event fields for the R readers, the master cohort, sorted `complete_case_patients.tsv`, and full-model-only `performance_per_seed.tsv` (no ablation rows).

The resulting inputs are:

| Working-directory path | Content |
| --- | --- |
| `Inputs/master_folds.csv` | Final five-modality master cohort, with `patient,cancer_type,time,event,fold,has_rna,has_cnv,has_rppa,has_methylation,has_he` |
| `Inputs/oof/seed_<seed>/fold<fold>/survival_oof_predictions.tsv.gz` | Exported Stage C test predictions, including calibrated natural-availability and complete-case risks |
| `Inputs/performance_per_seed.tsv` | Exported natural-availability and complete-case pooled/within-cancer C-indices per seed |

The five seeds are 101, 202, 303, 404 and 505; folds are 0–4. Use follow-up times in days and the original boolean event/availability fields.

```bash
omnit2i-analyze-ipcw --master-folds WORK/Inputs/master_folds.csv --oof-root WORK/Inputs/oof --output-dir WORK/Analysis/auc
Rscript analysis/bootstrap_survival.R WORK
Rscript analysis/km_natural.R WORK
Rscript analysis/km_complete.R WORK
python analysis/km_statistics.py WORK
Rscript analysis/roc_survival.R WORK
Rscript analysis/cancer_auc.R WORK
Rscript analysis/bootstrap_cancer_auc.R WORK
```

Pass an absolute `WORK` path. IPCW uses the corresponding outer-training censoring distribution, with fixed 1-, 3- and 5-year horizons and five-year Uno truncation. Fold metrics use test-patient-count weighting within each seed. C-index/AUC confidence intervals retain the original 1,000 within-cancer patient-bootstrap draws, sharing sampled patients across all five seeds. The cancer-specific AUC calculation estimates censoring within cancer and training fold; undefined estimates remain missing.

KM grouping remains native R: within-cancer/seed risk percentiles, five-seed patient means, then the within-cancer OOF median. `km_statistics.py` reproduces the final table's lifelines Efron Cox fit and approximate PH test with the KM time transform. This last step is required for the final `Fig3c/d_statistics.tsv` fields. The separate `*_proportional_hazards.tsv` files written by R contain native `cox.zph` diagnostics; they are distinct from the final table's explicitly named approximate PH field.

## Clinical adjustment

```bash
omnit2i-analyze-clinical --master master_folds.csv --clinical tcga_survival_supplement.tsv \
  --predictions-root TRAINED --output-dir WORK/Analysis/clinical
```

Use the original TCGA clinical supplement columns and positive OS times in days. `TRAINED` contains `seed_<seed>/fold<fold>/C/` from `omnit2i-train-fold`; direct Stage C directories and historical `survival_oof_predictions.tsv.gz` archives are also accepted. Every test patient, fold and outcome is checked against the master cohort, and saved validation risk standardization is retained before averaging five seed scores.

The command runs the original five fold-specific cancer-stratified Cox adjustments for age, sex and grouped pathological stage, followed by all three original sensitivity specifications: age spline, cancer-by-stage strata, and an age effect changing after 365.25 days. It retains complete clinical cases, rank-transform PH diagnostics, five-fold Bonferroni P-value merging, warnings, and coverage counts. These post-hoc estimates condition on saved OOF scores; the command does not refit OmniT2I. Install the `analysis` extra and choose a new output directory.

## Measured subsets and predicted modalities

Aggregate the 25 evaluation directories (`seed_<seed>/fold<fold>`, each containing the four archives written by `omnit2i-evaluate-subsets`):

```bash
omnit2i-aggregate-subsets --input-root EVALUATIONS --output-dir WORK/Inputs/subsets
```

This produces the measured, fill-all and selective summaries below. Within each seed, C-indices pool the five OOF folds and weight within-cancer concordance by comparable pairs. Reconstruction correlations retain the original test-patient-count weighting across folds. Means and sample SDs summarize the five seeds. The specialist `paired_per_seed.tsv` is a separate model-comparison input.

Put the saved summaries under `Inputs/subsets/`:

- `paired_per_seed.tsv`: specialist versus OmniT2I comparisons over 31 measured subsets and five seeds. The final nine families are selected by excluding the three `adapted_*` models.
- `selective/survival_per_seed.tsv` and `selective/survival_deltas_summary.tsv`: 180 source/target routes, four conditions, five seeds.
- `summary/feature_per_seed.tsv`: 75 single-target reconstruction routes over five seeds.

```bash
python analysis/prepare_subset_sources.py WORK
python analysis/prepare_availability.py --master WORK/Inputs/master_folds.csv --output-dir WORK/Source_Data/Figure2
python analysis/measured_gains.py --per-seed measured_31_subsets_per_seed.tsv --output-dir WORK/Analysis/measured
```

The first command generates Figure 4, Figure 5, Extended Data 2 and Extended Data 3 tables and workbooks. The measured reference for a virtual route is the union of its source and selected target modalities; it is never replaced by the all-five measured condition. Target summaries average routes within each seed before summarizing seeds. Figure 2 uses aggregate availability counts from the master cohort.

For the four prespecified RNA-start paired confidence intervals:

```bash
python analysis/bootstrap_measured_gains.py --patients WORK/Inputs/complete_case_patients.tsv \
  --per-seed WORK/Inputs/subsets/measured/omnit2i_all_31_subsets_per_seed.tsv \
  --predictions-root EVALUATIONS \
  --output-dir WORK/Analysis/measured
```

`omnit2i-export-survival` writes the patients table, and `omnit2i-aggregate-subsets` writes the per-seed measured summary. `EVALUATIONS` is the raw output root passed to that aggregation command, not its summary directory. The patients table is sorted by patient and contains `patient,cancer_type,time,event,fold`. The prediction root contains `seed_<seed>/fold<fold>/predictions.tsv.gz`, with the same patient fields plus `seed,source_subset,risk_source_only`. This manuscript-specific bootstrap requires the original 5,605 complete cases and 1,546 events. The command retains 2,000 cancer-stratified paired draws, seed 20261019, all 75 measured transition summaries, and the original comparable-pair tie rule. `omnit2i.analysis.paired_bootstrap` also exposes the original shared-patient kernels used by paired model comparisons. Intervals condition on saved predictions and do not include model refitting uncertainty.

## Figure 6 inputs

Generate the ablation tables from the 25 public training outputs per variant:

```bash
omnit2i-aggregate-ablations --master WORK/Inputs/master_folds.csv \
  --reference-training TRAINED --training-root ABLATIONS \
  --variants no_pretraining no_c2 no_c3 random_layout no_risk_consistency mean_fusion no_modality_dropout simplified \
  --reference-evaluation EVALUATION_SUMMARY --evaluation-root ABLATION_EVALUATION_SUMMARIES \
  --output-dir WORK/Inputs/ablations
```

`TRAINED` contains `seed_<seed>/fold<fold>/C/`; `ABLATIONS` contains `VARIANT/seed_<seed>/fold<fold>/C/`. The evaluation roots contain the corresponding `omnit2i-aggregate-subsets` outputs, with one subdirectory per variant under `ABLATION_EVALUATION_SUMMARIES`. The optional evaluation arguments must be supplied together to produce the frozen measured/selective pairs. The same command also accepts `mlp_matched` and `mlp_wide`.

Training summaries pool the five OOF folds per seed. Natural-availability risks use the saved validation calibration; complete-case risks use the original outcome-free standardization on complete-case validation patients, recovered from `predictions.csv`. Patients, clinical outcomes and folds are checked against the supplied master. Raw merged and within-fold pair-weighted C-indices remain diagnostics. Differences are full minus ablation, with sample SD over the five seeds. No model is fitted by this command.

The command writes the following inputs accepted by `prepare_ablation_sources.R`:

| `Inputs/ablations/` filename | Saved analysis |
| --- | --- |
| `encoder_paired_survival_per_seed.tsv` | Encoder/input ablations, Stage C |
| `component_paired_survival_per_seed.tsv` | Component ablations, Stage C |
| `simplified_paired_survival_per_seed.tsv` | Simplified fusion, Stage C |
| `paired_measured_summary.tsv` | Component ablations, frozen measured-subset summary |
| `paired_measured_per_seed.tsv` | Component ablations, frozen measured subsets per seed |

After the RNA-start bootstrap, run `Rscript analysis/prepare_ablation_sources.R WORK`. The figure renderer selects the final four encoder/input ablations and uses the original mean ± seed-SD dropout ribbon. The final seven-cohort external panel reads the submitted `Figure6_Source_Data.xlsx` workbook; external data processing and frozen-checkpoint inference are separate from this saved-results plotting step. Then run `Rscript figures/render.R WORK` as described in the figure README.
