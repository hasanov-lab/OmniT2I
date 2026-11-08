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
pip install '.[test]'
python -m pytest -q
```

The example runs forward and backward passes on synthetic inputs with randomly initialized models. It demonstrates the component interfaces; its objective and predictions have no scientific or clinical interpretation.

## RNA preprocessing

```bash
omnit2i-prepare-rna --expression expression.tsv.gz --cohort cohort.csv --output-dir outputs/rna
```

The expression TSV has gene IDs in its first column and sample IDs in the remaining column headers. Supply expression on the intended upstream scale; the command does not log-transform raw counts. Gene and sample IDs must be unique, and the selected and invariant genes must have finite values for every requested sample. Missing required genes raise an error.

The cohort CSV contains `patient,sample,cancer_type,time,event,partition`. Each row is one unique patient and sample; `partition` is `core`, `validation`, or `test`, `time` is positive follow-up time in a consistent unit, and `event` is 1 for an observed event and 0 for censoring. Supply the same patient partitions to all modalities. The command preserves those partitions and does not generate a new split.

Invariant-gene selection, Top2000 selection, scaling, within-cancer scaling, Breslow Cox-score weights, and correlation-based hierarchical layout are fitted on core patients only. All three channels use the snake layout: globally standardized expression, within-cancer standardized expression, and globally standardized expression multiplied by the Cox-score weight. Unseen cancer types use the core global statistics for the second channel. The output directory must be new; it receives `preprocessing.npz`, `images.npz`, and `cohort.csv`.

Saved parameters can be applied to another cohort without refitting:

```python
import numpy as np
from omnit2i.preprocessing.rna import transform_rna

with np.load("outputs/rna/preprocessing.npz", allow_pickle=False) as saved:
    artifact = dict(saved)
# expression: gene x sample DataFrame; cohort: sample and cancer_type columns.
images = transform_rna(expression, cohort, artifact)
```

The `omnit2i.preprocessing` modules also provide the extracted CNV scaling, protein filtering/imputation, and promoter-gene methylation beta-to-M conversion, filtering, and standardization functions. Their raw-data preparation commands are still being organized. The methylation functions operate on promoter-gene profiles, rather than raw probe-level arrays.

## Release status

Multimodal data preparation, training workflows, and evaluation scripts are being prepared in stages. This release does not yet reproduce the manuscript experiments end to end. Patient data, research outputs, and pretrained weights are not included; weights are planned for release after publication.

## Development

Principal developer: **Peng Li**

Hasanov Lab, The Ohio State University

## License

Copyright 2026 The Ohio State University.

OmniT2I is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). See the license for permitted uses and conditions. Commercial use requires separate permission from the rights holder.
