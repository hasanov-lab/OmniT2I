# OmniT2I

A unified table-to-image framework for survival prediction and cross-modal inference with incomplete multimodal cancer data.

OmniT2I converts molecular profiles into table-to-image representations and integrates them with histopathology embeddings. It supports survival prediction from available modalities and inference of withheld modalities.

## Installation

Python 3.10 or later is required. Install a PyTorch build appropriate for your hardware, then install this package:

```bash
git clone https://github.com/stevenli11/OmniT2I.git
cd OmniT2I
pip install .
```

For the dependency versions used in the release checks, use `pip install -r requirements.txt` from the checkout instead. `pyproject.toml` declares the package dependencies; `requirements.txt` selects the tested versions and installs the package too.

## Model components

The package provides model components, tests, and a synthetic example:

- Spatial CNN encoders for RNA, CNV, methylation, and RPPA/protein table-to-image inputs.
- An adapter for frozen H&E embeddings, shared-projection fusion with availability masks, and Cox risk heads.
- A target-conditioned translator with 20 directed source-to-target routes and target-specific feature decoders. The target is always excluded from its source mask.
- A masked-image reconstruction decoder and the existing pooled-CNN and spatial/global encoder alternatives.

The modality order is **RNA, CNV, protein, methylation, H&E**. The spatial encoders consume floating-point tensors shaped `[patients, 3, height, width]`: RNA/CNV/methylation use 45 × 45 images and protein uses 17 × 17 images. The H&E adapter consumes `[patients, 1536]` frozen embeddings. Encoders produce 96-dimensional representations; the H&E feature extractor itself is not included.

`SharedProjectionFusion` accepts a list of embeddings plus a `[patients, modalities]` availability mask. It supports mean or learned gated fusion over observed modalities; each patient must have at least one observed modality. Missing embedding slots should contain finite placeholders such as zeros. Risk outputs are relative Cox scores, not survival probabilities.

`OmicsHEEndToEndCox` groups the four image encoders, H&E adapter, and fusion head. Its `forward_selected` method uses per-modality row indices, encoding only observed inputs. Each selected batch must contain at least one observation of each modality. For inference with an entirely absent modality, use the encoder components and `SharedProjectionFusion` directly with zero placeholders and the availability mask. The synthetic example selects spatial encoders and mean fusion explicitly; the backbone's existing defaults remain pooled CNNs and gated fusion.

`TargetConditionedCrossModalTranslator` consumes `[patients, 5, latent_dim]` embeddings and a binary source mask. Each row needs an observed source other than the requested target. Output feature counts are supplied in modality order or by modality name. When training the translator with a survival backbone, pass its shared projected embeddings, as shown in the example.

## Synthetic example and tests

```bash
python examples/model_components.py
pip install '.[test,training]'
python -m pytest -q
```

The example runs forward and backward passes on synthetic inputs with randomly initialized models. It demonstrates the component interfaces; its objective and predictions have no scientific or clinical interpretation.

## RNA preprocessing

```bash
omnit2i-prepare-rna --expression expression.tsv.gz --cohort cohort.csv --output-dir outputs/rna
```

The expression TSV has gene IDs in its first column and sample IDs in the remaining column headers. Supply expression on the intended upstream scale; the command does not log-transform raw counts. Gene and sample IDs must be unique, and the selected and invariant genes must have finite values for every requested sample. Missing required genes raise an error.

The cohort CSV contains `patient,sample,cancer_type,time,event,partition`. Each row is one unique patient and sample; `partition` is `core`, `validation`, or `test`, `time` is positive follow-up time in a consistent unit, and `event` is 1 for an observed event and 0 for censoring. Supply the same patient partitions to all modalities. The command preserves those partitions and does not generate a new split.

This package follows the manuscript pipeline preceding the full core-only preprocessing rerun, with RNA placement corrected to snake order. Invariant-gene selection, Top2000 selection, scaling, within-cancer scaling, and correlation-based hierarchical layout use outer-training patients (`core` plus `validation`). Corrected Breslow Cox-score weights use core patients and outcomes only. Outer-test patients enter none of these fitted quantities. All three channels use the same snake layout: globally standardized expression, within-cancer standardized expression, and globally standardized expression multiplied by the Cox-score weight. Unseen cancer types use the outer-training global statistics for the second channel. The output directory must be new; it receives `preprocessing.npz`, `images.npz`, and `cohort.csv`.

Saved parameters can be applied to another cohort without refitting:

```python
import numpy as np
from omnit2i.preprocessing.rna import transform_rna

with np.load("outputs/rna/preprocessing.npz", allow_pickle=False) as saved:
    artifact = dict(saved)
# expression: gene x sample DataFrame; cohort: sample and cancer_type columns.
images = transform_rna(expression, cohort, artifact)
```

## Other molecular preprocessing

```bash
omnit2i-prepare-omics --modality cnv --matrix cnv.tsv.gz --cohort cnv_cohort.csv --output-dir outputs/cnv
omnit2i-prepare-omics --modality protein --matrix rppa.tsv.gz --cohort protein_cohort.csv --output-dir outputs/protein
omnit2i-prepare-omics --modality methylation --matrix promoter_gene_beta.tsv.gz --chromosomes chromosomes.tsv --cohort methylation_cohort.csv --output-dir outputs/methylation
```

Matrices use feature rows and sample columns, like RNA. Each modality's cohort contains its observed patients only, retaining their partitions from the shared master cohort. All three partitions must be present. CNV uses the supplied gene-level scale and selects Top2000 by outer-training variance, breaking ties by feature ID. Protein retains RPPA features with at most 20% outer-training missingness and variance greater than 1e-8 after median imputation; its canvas is 17 × 17. Methylation takes **already aggregated promoter-gene beta values**, converts them to M values with epsilon 1e-5, excludes chrY, filters at 1% outer-training missingness and variance greater than 1e-8, then selects Top2000. Its chromosome TSV requires `feature_id` and `chrom` columns, with annotations such as `chr1` and `chrY`. CNV and methylation use 45 × 45 canvases; `--top-n` can reduce their feature count.

Protein and methylation use outer-training medians for missing values. CNV requires complete selected measurements; missing selected features are errors for every modality. Scaling, cancer references, and layout use core plus validation patients; corrected Cox-score weights use core only. The three snake channels and output filenames match RNA. The final CNV channels use standardized values, so the signed clipping bounds used by older channel alternatives are unnecessary here. Saved artifacts can be reapplied with `transform_omics` from `omnit2i.preprocessing.omics`; it reads the modality from the artifact.

## Molecular pretraining and survival training (Stage A)

```bash
pip install '.[training]'  # Included when installing requirements.txt.
omnit2i-train-rna --images outputs/rna/images.npz --output-dir outputs/rna_training --seed 101 --device cuda
omnit2i-train-molecular --modality cnv --images outputs/cnv/images.npz --output-dir outputs/cnv_training --seed 101 --device cuda
omnit2i-train-molecular --modality protein --images outputs/protein/images.npz --output-dir outputs/protein_training --seed 101 --device cuda
omnit2i-train-molecular --modality methylation --images outputs/methylation/images.npz --output-dir outputs/methylation_training --seed 101 --device cuda
```

These commands train independently learned spatial encoders with masked reconstruction on core patients, then fit each encoder and Cox head on the entire core risk set. Protein uses a 17 × 17 grid; the other three use 45 × 45. `omnit2i-train-rna` remains an alias with RNA as its default modality. Training handles tied follow-up times with Breslow denominators. Validation C-index selects the checkpoint; validation risk mean and standard deviation define the saved risk scaling. Test patients contribute to neither fitting nor checkpoint selection. Masking affects measured pixels only and preserves zero padding.

Defaults are 80 pretraining epochs, up to 180 Cox epochs, patience 25, mask fraction 0.15, and AdamW learning rates 0.001 with weight decay 0.0001. The commands default to CPU unless `--device` is supplied. `--seed` is the final training seed for the supplied partition; prepare each fold separately and pass its intended seed explicitly. In manuscript Stage A runs this is `base_seed + fold * 100`; the supplied partitions still come from the shared master split, whose seed uses `base_seed + fold * 1000`. CUDA runs need enough memory for a full core forward pass: the Cox risk set is not split into independent minibatch losses.

Outputs are `model.pt`, patient-level `predictions.csv`, `embeddings.npz`, training histories, and `options.json`. The checkpoint contains the selected model state and validation risk scaling. Retain its corresponding `preprocessing.npz` to transform new samples. Risk scores are relative Cox scores, not absolute survival probabilities.

A short CPU run using synthetic expression and patient metadata exercises both commands:

```bash
python examples/rna_workflow.py --output-dir outputs/synthetic_rna
```

The synthetic example selects 60 genes for speed. It checks the software workflow and does not reproduce experimental performance.

## Five-modality survival training

`omnit2i-train-fusion` trains the spatial molecular encoders, H&E adapter, shared projection, availability-aware fusion, and survival heads together. It uses one Cox loss over the core risk set, observed-only auxiliary Cox losses, and paired projection alignment. BatchNorm running statistics remain fixed. Validation C-index selects the checkpoint.

Supply the master cohort CSV described above and one NPZ per modality:

| Archive | Required arrays | Input shape |
| --- | --- | --- |
| RNA, CNV, methylation | `patient`, `partition`, `images` | `[observed patients, 3, 45, 45]` |
| Protein | `patient`, `partition`, `images` | `[observed patients, 3, 17, 17]` |
| H&E | `patient`, `embeddings` | `[observed patients, 1536]` |

Each archive contains only genuinely observed patients for that modality. Archive rows may have any order: the loader joins by patient ID, rejects duplicate or unknown patients and inconsistent partitions, and derives availability from membership. Every cohort patient needs at least one observed modality. The current backbone also requires every modality to be represented somewhere in each of the core and validation partitions.

Molecular images must use the manuscript preprocessing: feature selection, imputation, scaling, and layouts fitted on outer-training patients; corrected C3 weights fitted on the shared core patients only. H&E inputs are frozen patient-level embeddings before fold standardization; this command fits their mean and standard deviation on observed outer-training patients and saves those parameters. H&E feature extraction and the CNV/protein/methylation raw-data commands are still being organized.

```bash
omnit2i-train-fusion --cohort cohort.csv \
  --rna rna/images.npz --cnv cnv/images.npz \
  --protein protein/images.npz --methylation methylation/images.npz --he he.npz \
  --rna-checkpoint rna/model.pt --cnv-checkpoint cnv/model.pt \
  --protein-checkpoint protein/model.pt --methylation-checkpoint methylation/model.pt \
  --output-dir outputs/fusion --seed 101 --device cuda
```

The four molecular checkpoints must contain the pretrained spatial encoder under `encoder.*` in `model_state` or `state_dict`. All Stage A commands above write this format. Checkpoints also record their modality, grid size, and feature count. Defaults follow the extracted trainer: gated fusion, modality dropout 0.30, auxiliary weight 0.25, alignment weight 1.0, encoder learning rate 0.0001, and fusion learning rate 0.001. `--fusion-mode mean` selects mean fusion. `--seed` is the final seed for this run.

Outputs include the selected `model.pt`, patient-level predictions, fusion weights, history, and options. The checkpoint stores model constructor arguments, outer-training H&E scaling, and validation risk calibration. Output directories must be new. A short CPU example prepares synthetic molecular matrices, trains all four Stage A encoders, then runs Stage B and C:

```bash
python examples/multimodal_workflow.py --output-dir outputs/synthetic_multimodal
```

This example checks preprocessing, molecular pretraining, patient alignment, fusion training, joint training, frozen subset evaluation, and serialization. It uses synthetic measurements and shortened training schedules solely for the software check.

## Cross-modal joint training

`omnit2i-train-joint` starts from the fusion checkpoint. It first freezes the survival backbone and warms up the translator and feature decoders, then updates those components, the fusion heads, the final molecular convolution blocks, and the H&E adapter. Early molecular layers and BatchNorm running statistics remain fixed. The target modality is excluded from translator inputs; its observed projected embedding is detached supervision. Source dropout retains at least one genuinely observed source.

Reuse the same cohort and input archives as fusion training. The command reuses the checkpoint's outer-training H&E scaling. Supply a target NPZ with these arrays:

| Array | Contents |
| --- | --- |
| `patient` | All master patient IDs, unique; rows are aligned by ID |
| `modalities` | `rna`, `cnv`, `protein`, `methylation`, `he`, in that order |
| `availability` | Boolean `[patients, 5]`, matching the input archives |
| `target_scale` | Five strings describing the prepared feature scales |
| `<modality>_target` | Float `[patients, features]` in the measured target feature space |
| `<modality>_feature_ids` | Unique feature IDs in the corresponding column order |

Targets are prepared molecular profiles or H&E embeddings, rather than image pixels. Their upstream filtering, imputation, and scaling retain the manuscript's outer-training references. Observed target rows must be finite; missing rows may contain NaNs. Optional `cancer_type`, `partition`, and `outer_test` arrays are checked against the cohort when present. The existing cancer-specific mean subtraction is fitted on observed core targets, with the core global mean used for unseen cancers. Validation targets are transformed with those saved means; test targets do not enter optimization or checkpoint selection. Both core and validation need patients with at least two observed modalities.

```bash
omnit2i-train-joint --cohort cohort.csv \
  --rna rna/images.npz --cnv cnv/images.npz \
  --protein protein/images.npz --methylation methylation/images.npz --he he.npz \
  --source-checkpoint outputs/fusion/model.pt --targets targets.npz \
  --output-dir outputs/joint --seed 101 --fold 0 --device cuda
```

The defaults preserve 40 warm-up epochs, up to 80 joint epochs, source dropout 0.35, and the existing latent, feature, and risk-consistency losses. Validation checks run every five epochs. Joint checkpoints are selected when cross-modal validation loss improves and survival C-index remains within 0.005 of the frozen survival model. If none qualify, the selected model retains the best warm-up translator and frozen survival backbone. `--seed` is the base seed; `--fold` contributes the production offset of `fold * 1000` and does not create or change partitions. For fold `k`, use the corresponding fusion checkpoint trained with the final seed `base_seed + k * 1000`.

Outputs include `model.pt`, `frozen_fallback.pt` (the source survival model plus the best warm-up translator), `target_baselines.npz`, predictions, fusion weights, history, and options. Both checkpoints store constructor arguments and risk calibration. Feature decoder outputs are residuals: add the saved cancer-specific mean, or global mean for an unseen cancer, to recover the prepared target scale. The multimodal synthetic example above exercises this command after fusion training.

## Frozen subset evaluation

```bash
omnit2i-evaluate-subsets --cohort cohort.csv \
  --rna rna/images.npz --cnv cnv/images.npz \
  --protein protein/images.npz --methylation methylation/images.npz --he he.npz \
  --checkpoint outputs/joint/model.pt --targets targets.npz \
  --target-baselines outputs/joint/target_baselines.npz \
  --output-dir outputs/evaluation --device cuda
```

This command retains the manuscript's complete-case protocol: 30 nonempty proper source subsets, each evaluated under `source_only`, `cancer_mean`, `virtual`, and `measured` conditions on the same complete-case test patients. Each virtual target is independently predicted from the original measured source subset. `measured` uses all five measured modalities. H&E reconstruction predicts an embedding, rather than an image.

The model remains frozen. Latent cancer means use observed core patients; feature reconstruction adds the saved Stage C cancer means to decoder residuals. Each subset/condition uses its own affine risk calibration fitted on complete-case validation patients. At least two such validation patients and one complete-case test patient are required. Test outcomes enter neither predictions nor calibration. Use the corresponding training cohort, modality preprocessing, target archive, and saved feature means for the selected checkpoint. The command also accepts `frozen_fallback.pt` with the same target means.

Outputs are `predictions.tsv.gz` (30 rows per test patient, with raw and calibrated risks for all four conditions) and `feature_metrics.tsv.gz` (75 subset/target summaries). Reconstruction summaries preserve the original profile correlations, within-cancer correlations, and error improvement relative to core cancer means. Undefined correlations, including constant profiles or cancers with fewer than two test patients, remain NaN. Feature IDs, target scales, patient IDs, and availability are checked against the supplied inputs and saved means.

## Release status

Upstream raw-data preparation (including methylation probe aggregation and H&E embedding extraction) and the remaining analysis/figure scripts are being prepared in stages. Molecular preprocessing, Stage A/B/C training, and frozen 30-subset evaluation are available for prepared matrices and embeddings. This release does not yet reproduce the manuscript experiments end to end. Patient data, research outputs, and pretrained weights are not included; weights are planned for release after publication.

## Development

Principal developer: **Peng Li**

Hasanov Lab, The Ohio State University

## License

Copyright 2026 The Ohio State University.

OmniT2I is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). See the license for permitted uses and conditions. Commercial use requires separate permission from the rights holder.
