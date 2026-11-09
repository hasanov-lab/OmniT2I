"""Post-hoc, fold-wise clinical adjustment of existing OmniT2I OOF scores.

Preserves the final manuscript clinical specifications and saved OOF risks.
"""

import argparse
import json
import platform
import re
import warnings
from pathlib import Path

import lifelines
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


SEEDS = (101, 202, 303, 404, 505)
FOLDS = tuple(range(5))


def grouped_stage(value):
    # Same grouping as the existing grouped-stage run; do not infer unknown stages.
    match = re.fullmatch(r"STAGE (IV|III|II|I)(?:[ABC]|S)?", str(value).strip().upper())
    return match.group(1) if match else np.nan


def bonferroni_merge(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Require five valid fold P values; never drop a failed fold")
    return min(1.0, len(values) * float(values.min()))


def load_data(master_path, clinical_path, predictions_root, output_dir=None):
    master = pd.read_csv(master_path)
    assert not master.patient.duplicated().any()
    if not master.event.isin([0, 1]).all() or not np.isfinite(master.time).all() or not master.time.gt(0).all():
        raise ValueError("Clinical adjustment requires positive finite OS days and binary events")
    columns = ["_PATIENT", "age_at_initial_pathologic_diagnosis", "gender", "ajcc_pathologic_tumor_stage"]
    clinical = pd.read_csv(clinical_path, sep="\t", usecols=columns, low_memory=False)
    clinical = clinical.loc[clinical._PATIENT.isin(master.patient), columns]
    conflicts = clinical.groupby("_PATIENT")[columns[1:]].nunique(dropna=False).gt(1)
    assert not conflicts.any().any(), "Conflicting patient-level clinical profiles"
    redundant = int(clinical._PATIENT.duplicated().sum())
    clinical = clinical.drop_duplicates("_PATIENT").rename(columns={
        "_PATIENT": "patient", "age_at_initial_pathologic_diagnosis": "age",
        "gender": "sex_raw", "ajcc_pathologic_tumor_stage": "stage_raw",
    })
    clinical["age"] = pd.to_numeric(clinical.age, errors="coerce")
    clinical["sex"] = clinical.sex_raw.map({"FEMALE": 0.0, "MALE": 1.0})
    clinical["stage"] = clinical.stage_raw.map(grouped_stage)
    frames, audits = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            folder = predictions_root / f"seed_{seed}" / f"fold{fold}"
            if (folder / "C").is_dir():
                folder = folder / "C"
            if (folder / "predictions.csv").exists():
                f = pd.read_csv(folder / "predictions.csv")
                f = f.loc[f.partition.eq("test")].copy()
                f = f.rename(columns={"risk_raw": "risk_score_raw", "risk": "risk_score_calibrated"})
                if "seed" in f:
                    assert f.seed.eq(seed).all()
                f["seed"] = seed
                checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
                assert checkpoint["seed"] == seed and checkpoint["fold"] == fold
                cal = checkpoint["risk_calibration"]
                del checkpoint
            else:
                f = pd.read_csv(folder / "survival_oof_predictions.tsv.gz", sep="\t")
                meta = json.loads((folder / "fold_complete.json").read_text())
                assert meta["seed"] == seed and meta["fold"] == fold
                assert meta["n_test"] == len(f)
                assert meta["n_inner_core"] + meta["n_inner_validation"] + len(f) == len(master)
                cal = meta["survival_risk_calibration"]["natural"]
            assert not f.patient.duplicated().any()
            actual = f.set_index("patient").sort_index()
            expected = master.loc[master.fold.eq(fold)].set_index("patient").sort_index()
            assert actual.index.equals(expected.index)
            for column in ("fold", "cancer_type", "time", "event"):
                assert np.array_equal(actual[column], expected[column]), column
            assert f.seed.eq(seed).all()
            assert np.isfinite([cal["mean"], cal["std"]]).all() and cal["std"] > 0
            reconstructed = (f.risk_score_raw - cal["mean"]) / cal["std"]
            # Archived risks were serialized from float32 calculations.
            assert np.allclose(reconstructed, f.risk_score_calibrated, rtol=0, atol=1e-6)
            assert np.isfinite(f.risk_score_calibrated).all()
            frames.append(f[["patient", "seed", "risk_score_calibrated"]])
            audits.append({"seed": seed, "fold": fold, "n_test": len(f),
                           "patient_fold_outcome_match": True,
                           "calibration_max_abs_error": float(np.abs(reconstructed - f.risk_score_calibrated).max())})
    scores = pd.concat(frames, ignore_index=True)
    assert not scores.duplicated(["patient", "seed"]).any()
    assert scores.groupby("patient").size().eq(len(SEEDS)).all()
    risk = scores.groupby("patient").risk_score_calibrated.mean().rename("risk")
    data = master.merge(clinical, on="patient", how="left", validate="1:1")
    data = data.merge(risk, on="patient", how="left", validate="1:1")
    assert data.risk.notna().all() and (data.time > 0).all()
    data["eligible"] = data[["age", "sex", "stage"]].notna().all(axis=1)
    data["age10"] = data.age / 10
    data["eligible_event"] = data.event.astype(int) * data.eligible.astype(int)
    counts = data.groupby("cancer_type").agg(
        n_original=("patient", "size"), events_original=("event", "sum"),
        n_clinical_complete=("eligible", "sum"), events_clinical_complete=("eligible_event", "sum"),
    )
    if output_dir is not None:
        counts.to_csv(output_dir / "cohort_by_cancer.csv")
        pd.DataFrame(audits).to_csv(output_dir / "prediction_audit.csv", index=False)
    missing = {"n_original": len(data), "n_clinical_complete": int(data.eligible.sum()),
               "n_excluded": int((~data.eligible).sum()), "n_age_missing": int(data.age.isna().sum()),
               "n_sex_missing": int(data.sex.isna().sum()), "n_stage_unusable": int(data.stage.isna().sum()),
               "redundant_consistent_clinical_rows": redundant,
               "unusable_stage_labels": data.loc[data.stage.isna(), "stage_raw"].fillna("<missing>").value_counts().to_dict()}
    return data.loc[data.eligible].copy(), missing


def analyze(data, coverage, output_dir):
    rows, all_coefficients, all_ph, warning_rows = [], [], [], []
    for fold in FOLDS:
        d = data.loc[data.fold.eq(fold), ["time", "event", "risk", "age10", "sex", "stage", "cancer_type"]].copy()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            cph = CoxPHFitter(penalizer=0.0)
            cph.fit(d, "time", "event", strata=["cancer_type"],
                    formula="risk + age10 + sex + C(stage)")
            coefficients = cph.summary.reset_index().assign(fold=fold)
            all_coefficients.append(coefficients)
            ph = proportional_hazard_test(cph, d, time_transform="rank").summary
            all_ph.append(ph.reset_index().rename(columns={"index": "covariate"}).assign(fold=fold))
        warning_rows.extend({"fold": fold, "category": w.category.__name__, "message": str(w.message)} for w in captured)
        r = cph.summary.loc["risk"]
        row = {"fold": fold, "patients": len(d), "events": int(d.event.sum()),
               "cancer_types": d.cancer_type.nunique(), "risk_log_hr": float(r["coef"]),
               "risk_hr": float(r["exp(coef)"]), "ci95_lower": float(r["exp(coef) lower 95%"]),
               "ci95_upper": float(r["exp(coef) upper 95%"]), "nominal_wald_p": float(r["p"]),
               "risk_ph_rank_p": float(ph.loc["risk", "p"]), "fit_warnings": len(captured)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "fold_adjusted_cox.csv", index=False)
    pd.concat(all_coefficients, ignore_index=True).to_csv(output_dir / "all_cox_coefficients.csv", index=False)
    pd.concat(all_ph, ignore_index=True).to_csv(output_dir / "proportional_hazards_checks.csv", index=False)
    pd.DataFrame(warning_rows, columns=["fold", "category", "message"]).to_csv(output_dir / "fit_warnings.csv", index=False)
    summary = {"analysis": "post_hoc_internal_clinical_adjustment",
               "deep_model_retrained": False, "manuscript_modified": False,
               "clinical_coverage": coverage, "events_analyzed": int(data.event.sum()),
               "cancer_types_analyzed": data.cancer_type.nunique(),
               "risk_definition": "Arithmetic mean of five seed-specific inner-validation-standardized OOF scores; one row per patient",
               "risk_hr_unit": "One unit of the mean standardized score, not necessarily one SD of the ensemble",
               "cox_formula": "risk + age/10 + sex + categorical_stage; stratified by cancer_type",
               "folds_hr_above_one": int(result.risk_hr.gt(1).sum()),
               "nominal_bonferroni_merged_p": bonferroni_merge(result.nominal_wald_p),
               "folds_risk_ph_p_below_005": int(result.risk_ph_rank_p.lt(.05).sum()),
               "fit_warning_count": len(warning_rows),
               "inference_boundary": "P values and CIs are conditional working-model estimates. Five-fold Bonferroni merging allows dependent tests but cannot repair invalid fold tests, model misspecification, or development-stage model selection. Historical pilot records show that development progression consulted outer-test/OOF results; its optimism is not quantified here. Not confirmatory external evidence or proof beyond unmeasured covariates.",
               "software": {"python": platform.python_version(), "lifelines": lifelines.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert result.fold.tolist() == list(FOLDS)
    assert result.patients.sum() == len(data) and result.events.sum() == data.event.sum()
    assert np.isfinite(result[["risk_hr", "ci95_lower", "ci95_upper", "nominal_wald_p"]]).all().all()
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True,
                        help="25 Stage C directories: seed_<seed>/fold<fold>")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data, coverage = load_data(args.master, args.clinical, args.predictions_root, args.output_dir)
    analyze(data, coverage, args.output_dir)
    from .clinical_sensitivity import analyze as sensitivity
    sensitivity(data, args.output_dir)


if __name__ == "__main__":
    main()
