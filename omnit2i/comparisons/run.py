"""Train the manuscript's nine complete-case specialists on shared patient roles."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from omnit2i.data import validate_cohort
from omnit2i.models import MODALITIES
from omnit2i.training.losses import fit_risk_calibration, apply_risk_calibration
from . import estimators, published
from .prepare import RNA_SCALE

AVAILABILITY_COLUMNS = ["has_rna", "has_cnv", "has_rppa", "has_methylation", "has_he"]
MODELS = ("deepsurv", "enet_cox", "lasso_cox", "xgboost_cox", "rsf", "survival_gbm",
          "extra_survival_trees", "coxboost", "survival_svm")
ALL_MODELS = MODELS + published.MODELS

def modality_subsets() -> tuple[tuple[str, ...], ...]:
    """Return all 31 non-empty subsets in size-then-lexicographic order."""
    return tuple(
        subset
        for size in range(1, len(MODALITIES) + 1)
        for subset in combinations(MODALITIES, size)
    )



def subset_slug(subset: tuple[str, ...]) -> str:
    return "__".join(subset)



def _subset_rows(
    targets: list[np.ndarray],
    availability: np.ndarray,
    rows: np.ndarray,
    modality_indices: tuple[int, ...],
) -> np.ndarray:
    if not availability[np.asarray(rows)][:, modality_indices].all():
        raise ValueError("Complete-case subset contains an unavailable input")
    blocks = [np.asarray(targets[index][rows], dtype=np.float32) for index in modality_indices]
    output = np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)
    if output.ndim != 2 or not np.isfinite(output).all():
        raise ValueError("Subset feature matrix is invalid")
    return output



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--subset-index", type=int, choices=range(31), required=True)
    parser.add_argument("--seed", type=int, choices=(101, 202, 303, 404, 505), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)

    parser.add_argument("--coxnet-n-alphas", type=int, default=30)
    parser.add_argument("--coxnet-alpha-min-ratio", type=float, default=0.01)
    parser.add_argument("--coxnet-max-iter", type=int, default=100000)
    parser.add_argument("--coxnet-tol", type=float, default=1e-6)
    parser.add_argument("--coxboost-n-estimators", type=int, default=250)
    parser.add_argument("--coxboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--svm-alpha", type=float, default=1.0)
    parser.add_argument("--svm-max-iter", type=int, default=80)
    parser.add_argument("--extra-n-estimators", type=int, default=150)
    parser.add_argument("--extra-min-samples-split", type=int, default=20)
    parser.add_argument("--extra-min-samples-leaf", type=int, default=15)
    parser.add_argument("--extra-max-features", default="sqrt")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--gbm-n-estimators", type=int, default=200)
    parser.add_argument("--gbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbm-max-depth", type=int, default=3)
    parser.add_argument("--gbm-subsample", type=float, default=0.8)
    parser.add_argument("--gbm-max-features", default="sqrt")
    parser.add_argument("--xgb-n-estimators", type=int, default=1000)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-min-child-weight", type=float, default=5.0)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.5)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=30)

    parser.add_argument("--rsf-n-estimators", type=int, default=150)
    parser.add_argument("--rsf-min-samples-split", type=int, default=20)
    parser.add_argument("--rsf-min-samples-leaf", type=int, default=15)
    parser.add_argument("--rsf-max-features", default="sqrt")
    parser.add_argument("--rsf-n-jobs", type=int, default=12)
    parser.add_argument("--deepsurv-hidden-dims", default="512,128")
    parser.add_argument("--deepsurv-dropout", type=float, default=0.30)
    parser.add_argument("--deepsurv-lr", type=float, default=1e-3)
    parser.add_argument("--deepsurv-weight-decay", type=float, default=1e-4)
    parser.add_argument("--deepsurv-epochs", type=int, default=220)
    parser.add_argument("--deepsurv-patience", type=int, default=30)
    parser.add_argument("--deepsurv-min-delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--senmo-l1", type=float, default=1e-7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def read_targets(path, cohort, fold):
    with np.load(path, allow_pickle=False) as saved:
        patients = saved["patient"].astype(str)
        if len(np.unique(patients)) != len(patients) or len(patients) != len(cohort):
            raise ValueError("Target patients must uniquely cover the shared cohort")
        order = pd.Index(patients).get_indexer(cohort.patient)
        if (order < 0).any() or not np.array_equal(saved["modalities"].astype(str), MODALITIES):
            raise ValueError("Target patients or modality order disagree")
        if not np.array_equal(saved["patient_fold"][order], cohort.fold):
            raise ValueError("Target patient folds disagree")
        if not np.array_equal(saved["outer_test"][order], cohort.fold.eq(fold)):
            raise ValueError("Target outer fold differs")
        availability = cohort[AVAILABILITY_COLUMNS].to_numpy(bool)
        if not np.array_equal(saved["availability"][order], availability):
            raise ValueError("Target availability disagrees with the shared cohort")
        if saved["target_scale"].astype(str)[0] != RNA_SCALE:
            raise ValueError("Specialists require conventional RNA without invariant correction")
        targets, features = [], []
        for index, name in enumerate(MODALITIES):
            values = np.asarray(saved[f"{name}_target"], dtype=np.float32)
            ids = saved[f"{name}_feature_ids"].astype(str)
            if values.shape != (len(cohort), len(ids)) or not len(ids) or len(np.unique(ids)) != len(ids):
                raise ValueError(f"Invalid {name} target shape or feature IDs")
            values = values[order]
            if not np.isfinite(values[availability[:, index]]).all():
                raise ValueError(f"Nonfinite observed {name} target")
            targets.append(values)
            features.append(ids)
    return targets, features


def fit(model, core, core_clinical, validation, validation_clinical, test, args, seed):
    if model == "deepsurv":
        return estimators.fit_deepsurv(core, core_clinical, validation, validation_clinical, test, args, seed)
    if model == "rsf":
        return estimators.fit_rsf(core, core_clinical, validation, validation_clinical, test, args, seed)
    name = f"zero_fill_{model}"
    if model in {"enet_cox", "lasso_cox"}:
        return estimators._fit_coxnet(name, core, core_clinical, validation, validation_clinical, test, args)
    if model == "xgboost_cox":
        return estimators._fit_xgboost(core, core_clinical, validation, validation_clinical, test, args, seed)
    if model not in MODELS:
        raise ValueError(f"Unknown specialist: {model}")
    return estimators._fit_sksurv_model(name, core, core_clinical, validation, validation_clinical, test, args, seed)


def train(args):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    cohort = validate_cohort(pd.read_csv(args.cohort, dtype={"patient": str}))
    if not cohort.fold.isin(range(5)).all() or not np.array_equal(cohort.partition.eq("test"), cohort.fold.eq(args.fold)):
        raise ValueError("Shared test partition disagrees with the requested fold")
    for field, value in (("seed", args.seed), ("outer_fold", args.fold)):
        if field in cohort and not cohort[field].eq(value).all():
            raise ValueError(f"Shared cohort {field} differs")
    if not cohort[AVAILABILITY_COLUMNS].isin([True, False]).all().all():
        raise ValueError("Modality availability must be boolean")
    targets, _ = read_targets(args.targets, cohort, args.fold)
    available = cohort[AVAILABILITY_COLUMNS].to_numpy(bool)
    complete = available.all(axis=1)
    rows = [np.flatnonzero(complete & cohort.partition.eq(role).to_numpy()) for role in ("core", "validation", "test")]
    if any(not len(part) for part in rows):
        raise ValueError("All three complete-case partitions must be nonempty")
    subset = modality_subsets()[args.subset_index]
    indices = tuple(MODALITIES.index(name) for name in subset)
    input_features = sum(targets[i].shape[1] for i in indices)
    if args.model in published.MODELS:
        result = published.fit(args.model, [targets[i] for i in indices], available[:, indices],
                               cohort, rows, args, args.seed + args.fold * 1000)
    else:
        core, validation, test = [_subset_rows(targets, available, part, indices) for part in rows]
        result = fit(args.model, core, cohort.iloc[rows[0]], validation, cohort.iloc[rows[1]], test, args,
                     args.seed + args.fold * 1000)
    calibration = fit_risk_calibration(result.validation_risk, np.ones(len(rows[1]), dtype=bool))
    predictions = cohort.iloc[rows[2]][["patient", "cancer_type", "time", "event", "fold"]].copy()
    predictions["seed"], predictions["baseline"] = args.seed, args.model
    predictions["source_subset"], predictions["observed_count"] = "+".join(subset), len(subset)
    predictions["risk_score_raw"] = result.test_risk.astype(np.float32) if args.model in published.MODELS else result.test_risk
    predictions["risk_score_calibrated"] = apply_risk_calibration(result.test_risk, calibration)
    predictions["risk_score_complete_case_calibrated"] = predictions.risk_score_calibrated
    predictions["complete_case"] = True
    tuning = dict(seed=args.seed, fold=args.fold, baseline=args.model, component="+".join(subset),
                  subset_index=args.subset_index, n_inner_core=len(rows[0]), n_inner_validation=len(rows[1]),
                  n_input_features=input_features, availability_indicators=0, **result.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output_dir / "predictions.tsv.gz", sep="\t", index=False)
    pd.DataFrame([tuning]).to_csv(args.output_dir / "tuning.tsv", sep="\t", index=False)
    (args.output_dir / "options.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    marker = dict(status="complete", seed=args.seed, fold=args.fold, model=args.model,
                  source_subset="+".join(subset), subset_index=args.subset_index, observed_count=len(subset),
                  n_complete_cases=int(complete.sum()), n_outer_test=len(rows[2]),
                  n_inner_core=len(rows[0]), n_inner_validation=len(rows[1]), availability_indicators=0,
                  risk_calibration=calibration)
    (args.output_dir / "complete.json").write_text(json.dumps(marker, indent=2) + "\n")


def main():
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
