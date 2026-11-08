# Raw tables and the shared TCGA cohort

Install `pip install '.[data]'`. Supply the Pan-Cancer Atlas gene-expression matrix, gene-level GISTIC2 CNV matrix, RPPA matrix, 450K beta matrix, TCGA clinical survival supplement, and matching Xena phenotype table. Matrices are gzip TSV files with sample barcodes in their headers. RNA is the supplied batch-adjusted expression scale, not raw sequencing counts. Patient data are not bundled.

The commands preserve the original primary-tumour and positive-OS-time filtering and the original sequence of master-cohort extensions:

```bash
omnit2i-build-rna-cohort --expression rna.tsv.gz --survival survival.tsv --phenotype phenotype.tsv --output cohort/rna_folds.csv
omnit2i-build-cnv-cohort --cnv cnv.tsv.gz --survival survival.tsv --phenotype phenotype.tsv --output cohort/cnv_folds.csv
omnit2i-build-rna-cnv-master --rna-folds cohort/rna_folds.csv --cnv-folds cohort/cnv_folds.csv --output cohort/rna_cnv_master.csv
omnit2i-build-protein-master --rppa rppa.tsv.gz --survival survival.tsv --phenotype phenotype.tsv --base-master cohort/rna_cnv_master.csv --output cohort/rna_cnv_protein_master.csv
omnit2i-build-methylation-cohort --matrix methylation450.tsv.gz --survival survival.tsv --phenotype phenotype.tsv --output cohort/methylation_primary_os.tsv
omnit2i-build-methylation-master --base-master cohort/rna_cnv_protein_master.csv --methylation-samples cohort/methylation_primary_os.tsv --output cohort/four_omics_master.csv
```

RNA outer folds use the established five-fold stratification and seed 123. RNA/CNV merging preserves RNA folds; subsequent extensions preserve all existing assignments and balance newly added patients with the original seeds. These commands create outer folds (`fold` 0–4), not the core/validation partitions required by the training commands. Retain the same shared inner partition for all modalities and stages; the existing manuscript partitions are the reference for reproducing a trained run. RPPA is named `rppa` in cohort columns and `protein` in model inputs.

CNV requires agreement between barcode and phenotype primary-tumour annotations. RPPA prefers the canonical sample over columns with replicate suffixes. Methylation deduplicates repeated technical column names before selecting unique primary-tumour patients; promoter aggregation handles their measurements separately. The clinical tables must retain their original column names, including `sample`, `_PATIENT`, `cancer type abbreviation`, `OS`, `OS.time`, `sample_type_id`, `sample_type`, and `_primary_disease` where used.

Use `cohort/methylation_primary_os.tsv` as the cohort for the raw promoter aggregation commands in the main README. They produce the sample annotation accepted by `omnit2i-build-methylation-master` as well.

## H&E metadata and alignment

```bash
omnit2i-query-he --expression rna.tsv.gz --survival survival.tsv --outdir he_metadata
omnit2i-build-he-master --four-omics-master cohort/four_omics_master.csv \
  --he-embeddings pooled/multislide_patient_mean_embeddings.tsv \
  --he-clinical he_metadata/tcga_open_svs_rna_os_matched_all_slide_files.tsv \
  --output-dir cohort/five_modal
```

The metadata command queries GDC open TCGA slide metadata and writes manifests; it does not download slides. It reuses its cached metadata unless `--force-refresh` is supplied. Representative-slide selection prioritizes tumour/primary annotations and diagnostic slides before file size. For the final multislide workflow, include additional diagnostic slides using the patch and embedding commands in the main README; patient pooling weights slides equally.

The final alignment command reads `case_id`, `n_patches`, and 1,536 `he_emb_*` columns. It checks patient uniqueness and clinical agreement, excludes nonpositive survival times, and preserves the four-omics master folds. It writes the five-modality master CSV, the H&E cohort view, aligned embeddings, feature names and a preparation summary. Original source files are listed in `omnit2i/preprocessing/cohort_provenance.tsv`.

## Shared training partitions

```bash
omnit2i-split-cohort --master cohort/five_modal/rna_cnv_rppa_methylation_he_master_folds.csv --seed 101 --fold 0 --output-dir prepared/seed_101/fold0
```

Use the final master CSV produced by cohort assembly. The command writes `cohort.csv` plus an observed-patient cohort CSV for each modality. All inherit the same 80/20 core/validation split of outer-training patients, using seed `base_seed + fold * 1000`; the outer test fold stays separate. Use these observed-patient CSVs with the molecular preprocessing commands. Their `images.npz` and `features.npz` outputs contain the input images and measured feature targets, respectively.

After preprocessing, place each modality's outputs under `PREPARED/rna/`, `cnv/`, `protein/` and `methylation/`, with frozen patient H&E embeddings at `PREPARED/he.npz`. Then run:

```bash
omnit2i-prepare-targets --cohort PREPARED/cohort.csv --rna PREPARED/rna/features.npz --cnv PREPARED/cnv/features.npz --protein PREPARED/protein/features.npz --methylation PREPARED/methylation/features.npz --he PREPARED/he.npz --output PREPARED/targets.npz
omnit2i-train-fold --prepared-dir PREPARED --output-dir trained/seed_101/fold0 --seed 101 --fold 0 --device cuda
```

The training entry runs the existing four Stage A trainers, Stage B and Stage C in order, stopping on failure. Stage A uses `base_seed + fold * 100`; Stage B and C use `base_seed + fold * 1000`. Prepare and invoke each of seeds 101, 202, 303, 404 and 505 with folds 0–4 separately. Outputs must be new directories. Individual stage commands remain available for running stages separately.
