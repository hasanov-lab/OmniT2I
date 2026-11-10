# OmniT2I

A unified table-to-image framework for survival prediction and cross-modal inference with incomplete multimodal cancer data.

OmniT2I combines RNA, copy number, protein/RPPA, DNA methylation and H&E representations. It predicts survival from available modalities and infers withheld molecular profiles or H&E embeddings.

## Installation

Use Python 3.11 for the pinned environment below (Python 3.13 is incompatible with its PyTorch 2.4.1 pin). Install a PyTorch build appropriate for your hardware, then run:

```bash
git clone https://github.com/stevenli11/OmniT2I.git
cd OmniT2I
pip install -r requirements.txt
```

For the package with its core dependencies only, use `pip install .`. Optional H&E preprocessing requires `pip install '.[he]'`.

## Example with synthetic data

The examples generate synthetic data locally; no patient data or pretrained weights are needed. Run the model example:

```bash
python examples/model_components.py
```

For a short synthetic preprocessing and training workflow:

```bash
python examples/multimodal_workflow.py --output-dir outputs/synthetic_multimodal
```

These examples demonstrate the interfaces with randomly initialized models and synthetic data.

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

## Development and license

Principal developer: **Peng Li**, Hasanov Lab, The Ohio State University.

Copyright 2026 The Ohio State University. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires separate permission from the rights holder.
