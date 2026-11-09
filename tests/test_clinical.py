import json

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("lifelines")

from omnit2i.analysis.clinical import SEEDS, bonferroni_merge, grouped_stage, load_data
from omnit2i.analysis.clinical_sensitivity import split_at_one_year


def test_clinical_stage_grouping_and_dependent_fold_p_values():
    assert [grouped_stage(value) for value in ["Stage IIA", "stage ivb", "Stage III", "Stage IS"]] == ["II", "IV", "III", "I"]
    assert pd.isna(grouped_stage("Stage X")) and pd.isna(grouped_stage(np.nan))
    assert bonferroni_merge([.01, .2, .3, .4, .5]) == .05
    assert bonferroni_merge([.9] * 5) == 1
    for values in ([.01] * 4, [.01] * 4 + [np.nan], [.01] * 4 + [-.1]):
        with pytest.raises(ValueError):
            bonferroni_merge(values)


def test_one_year_age_interaction_preserves_events_and_followup():
    patients = pd.DataFrame(dict(patient=["early", "boundary", "late", "censored"],
                                 time=[100., 365.25, 700., 800.], event=[1, 1, 1, 0], age10=[5., 6., 7., 8.]))
    result = split_at_one_year(patients)
    assert len(result) == 6 and result.event.sum() == 3
    assert result.loc[result.start.gt(0), "age10_after_year"].tolist() == [7., 8.]
    assert result.loc[result.start.eq(0), "age10_after_year"].eq(0).all()
    assert result.loc[result.patient.eq("late"), "event"].tolist() == [0, 1]
    assert split_at_one_year(patients.assign(event=patients.event.astype(bool))).event.sum() == 3


def clinical_inputs(tmp_path):
    master = pd.DataFrame(dict(patient=[f"p{i}" for i in range(25)], fold=np.arange(25) % 5,
                               cancer_type="A", time=np.arange(25) + 10., event=np.arange(25) % 2))
    master_path, clinical_path = tmp_path / "master.csv", tmp_path / "clinical.tsv"
    master.to_csv(master_path, index=False)
    clinical = pd.DataFrame({"_PATIENT": master.patient, "age_at_initial_pathologic_diagnosis": 50.,
                             "gender": "FEMALE", "ajcc_pathologic_tumor_stage": "Stage IIA"})
    clinical.loc[0, "ajcc_pathologic_tumor_stage"] = "Stage X"
    pd.concat([clinical, clinical.iloc[[1]]]).to_csv(clinical_path, sep="\t", index=False)
    root = tmp_path / "predictions"
    for seed in SEEDS:
        for fold in range(5):
            folder = root / f"seed_{seed}/fold{fold}"
            raw = np.arange(25) * .1 + seed * .001
            calibration = {"mean": .1, "std": 2.}
            risk = (raw - .1) / 2.
            if fold % 2 == 0:
                folder = folder / "C"
                folder.mkdir(parents=True)
                frame = master.assign(partition=np.where(master.fold.eq(fold), "test", "core"),
                                      risk_raw=raw, risk=risk, seed=seed)
                frame.to_csv(folder / "predictions.csv", index=False)
                torch.save(dict(seed=seed, fold=fold, risk_calibration=calibration), folder / "model.pt")
            else:
                folder.mkdir(parents=True)
                frame = master.assign(seed=seed, risk_score_raw=raw, risk_score_calibrated=risk)
                frame.loc[frame.fold.eq(fold)].to_csv(folder / "survival_oof_predictions.tsv.gz", sep="\t", index=False)
                (folder / "fold_complete.json").write_text(json.dumps(dict(seed=seed, fold=fold, n_test=5,
                    n_inner_core=15, n_inner_validation=5, survival_risk_calibration={"natural": calibration})))
    return master_path, clinical_path, root


def test_clinical_loading_retains_oof_calibration_and_identical_patient_outcomes(tmp_path):
    master, clinical, root = clinical_inputs(tmp_path)
    result, coverage = load_data(master, clinical, root)
    assert len(result) == 24 and coverage["n_stage_unusable"] == 1
    assert coverage["redundant_consistent_clinical_rows"] == 1
    assert result.stage.eq("II").all() and result.age10.eq(5.).all()
    expected = (np.arange(25) * .1 + np.mean(SEEDS) * .001 - .1) / 2.
    np.testing.assert_allclose(result.risk, expected[1:])
    path = root / "seed_505/fold4/C/predictions.csv"
    frame = pd.read_csv(path)
    frame["seed"] = 101
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        load_data(master, clinical, root)
    frame["seed"] = 505
    frame.loc[frame.partition.eq("test"), "time"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        load_data(master, clinical, root)
    frame.loc[frame.partition.eq("test"), "time"] -= 1
    frame.loc[frame.partition.eq("test"), "risk"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        load_data(master, clinical, root)
