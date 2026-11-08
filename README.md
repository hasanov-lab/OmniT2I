# OmniT2I

A unified table-to-image framework for survival prediction and cross-modal inference with incomplete multimodal cancer data.

OmniT2I combines RNA, copy number, protein/RPPA, DNA methylation and H&E representations. It predicts survival from available modalities and infers withheld molecular profiles or H&E embeddings.

## Installation

Python 3.10 or later is required. Install a PyTorch build appropriate for your hardware, then run:

```bash
git clone https://github.com/stevenli11/OmniT2I.git
cd OmniT2I
pip install -r requirements.txt
```

For the package with its core dependencies only, use `pip install .`. Optional H&E preprocessing requires `pip install '.[he]'`.

## Quick start

Run a forward and backward pass on synthetic inputs:

```bash
python examples/model_components.py
```

For a short synthetic preprocessing and training workflow:

```bash
python examples/multimodal_workflow.py --output-dir outputs/synthetic_multimodal
```

These examples demonstrate the interfaces with randomly initialized models and synthetic data.

## Workflows

| Task | Entry points |
| --- | --- |
| Cohort preparation | [Raw-table and cohort commands](docs/cohorts.md) |
| Molecular preprocessing | `omnit2i-prepare-rna`, `omnit2i-prepare-omics` |
| Raw methylation processing | `omnit2i-map-methylation`, `omnit2i-aggregate-methylation` |
| Frozen H&E extraction and pooling | `omnit2i-sample-he`, `omnit2i-embed-he`, `omnit2i-pool-he` |
| Molecular, fusion and cross-modal training | `omnit2i-train-molecular`, `omnit2i-train-fusion`, `omnit2i-train-joint` |
| Frozen subset evaluation | `omnit2i-evaluate-subsets` |
| Survival statistics and figure source tables | [Analysis scripts](analysis/README.md) |
| Figures 2–6 and Extended Data Figures 1–3 | [R plotting scripts](figures/README.md) |

See [workflow examples and input formats](docs/workflows.md) for command examples. Each command also provides `--help`.

The implementation follows the manuscript version preceding the full core-only preprocessing rerun, with RNA placement corrected to snake order. Feature selection, scaling and layouts use outer-training patients; corrected C3 Cox-score weights use core patients only.

## Data

Download the public inputs from their original repositories:

| Data | Source |
| --- | --- |
| TCGA molecular matrices | [UCSC Xena, TCGA Pan-Cancer Atlas](https://xenabrowser.net/datapages/?hub=https://pancanatlas.xenahubs.net:443) |
| TCGA overall survival | [TCGA Pan-Cancer Clinical Data Resource](https://doi.org/10.1016/j.cell.2018.02.052) |
| TCGA H&E slides and case metadata | [GDC Data Portal](https://portal.gdc.cancer.gov/) |
| CPTAC expression and clinical records | [LinkedOmics](https://www.linkedomics.org/data_download/), [cBioPortal](https://github.com/cBioPortal/datahub/tree/master/public), and GDC |
| SCAN-B expression and clinical records | [GEO GSE96058](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058) |

Cohort-specific releases and adopted inputs are listed in the manuscript's Supplementary Table 20. Downloaded patient data are stored locally rather than duplicated in this repository.

## Pretrained models

Trained OmniT2I weights will be released after publication. Training code and synthetic examples can be used before the weights are released. Frozen H&E extraction uses [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath), subject to the provider's access terms.

## Tests

```bash
pip install '.[test,data,training,analysis]'
python -m pytest -q
```

## Development and license

Principal developer: **Peng Li**, Hasanov Lab, The Ohio State University.

Copyright 2026 The Ohio State University. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires separate permission from the rights holder.
