import numpy as np
import pandas as pd
import pytest

from omnit2i.preprocessing.build_rna_cnv_master import build_master_folds
from omnit2i.preprocessing.build_protein_master import extend_master_folds as add_protein
from omnit2i.preprocessing.build_methylation_master import extend_master_folds as add_methylation
from omnit2i.preprocessing.cohort_samples import build_primary_os_cohort, select_primary_rppa_samples
from omnit2i.preprocessing.he_metadata import choose_one_slide_per_case


def cohort(n=10):
    patients = [f"TCGA-AA-{i:04d}" for i in range(n)]
    return pd.DataFrame({"patient": patients, "sample": [p + "-01" for p in patients],
                         "cancer_type": "X", "event": [bool(i % 2) for i in range(n)],
                         "time": np.arange(n) + 1., "fold": np.arange(n) % 5})


def test_master_extensions_keep_folds_and_validate_clinical_overlap():
    rna = cohort()
    cnv = cohort(12)
    master, _ = build_master_folds(rna, cnv)
    assert master.set_index("patient").loc[rna.patient, "fold"].tolist() == rna.fold.tolist()
    protein = cohort(14).rename(columns={"sample": "rppa_sample"})
    p, _ = add_protein(master, protein)
    meth = cohort(16)
    m, view, _ = add_methylation(p, meth)
    assert len(m) == 16 and len(view) == 16
    assert m.set_index("patient").loc[master.patient, "fold"].tolist() == master.fold.tolist()
    assert not m.loc[m.patient.isin(cohort(16).patient[-2:]), ["has_rna", "has_cnv", "has_rppa"]].any().any()
    changed = cnv.copy()
    changed.loc[0, "event"] = not changed.loc[0, "event"]
    with pytest.raises(ValueError, match="metadata disagree"):
        build_master_folds(rna, changed)
    with pytest.raises(ValueError, match="duplicate"):
        build_master_folds(rna, pd.concat([cnv, cnv.iloc[:1]]))


def test_primary_os_and_technical_replicate_selection():
    samples = ["TCGA-AA-0001-01", "TCGA-AA-0002-01", "TCGA-AA-0003-11"]
    survival = pd.DataFrame({"sample": samples, "_PATIENT": [s[:12] for s in samples],
        "cancer type abbreviation": "X", "OS": [1, 0, 1], "OS.time": [10, 0, 50]})
    phenotype = pd.DataFrame({"sample": samples, "sample_type_id": ["01", "01", "11"],
        "sample_type": ["Primary Tumor", "Primary Tumor", "Solid Tissue Normal"], "_primary_disease": "X"})
    selected = build_primary_os_cohort(samples, survival, phenotype)
    assert selected["sample"].tolist() == samples[:1]
    rppa = select_primary_rppa_samples([samples[0] + ".1", samples[0], samples[2]])
    assert rppa.rppa_sample.tolist() == samples[:1]
    with pytest.raises(ValueError, match="multiple 01"):
        build_primary_os_cohort([samples[0], samples[0]], survival, phenotype)


def test_diagnostic_slide_priority_precedes_file_size():
    slides = pd.DataFrame({"case_id": ["p", "p"], "tissue_type": "Tumor",
        "sample_type": "Primary Tumor", "experimental_strategy": ["Tissue Slide", "Diagnostic Slide"],
        "file_name": ["p-TS.svs", "p-DX.svs"], "file_size": [1, 100]})
    assert choose_one_slide_per_case(slides).file_name.tolist() == ["p-DX.svs"]
