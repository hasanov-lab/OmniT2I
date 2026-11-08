#!/usr/bin/env python3
"""Compute censoring-robust metrics for final OmniT2I OOF predictions.

For every seed and outer fold, the censoring distribution is estimated from
the corresponding outer-training patients only. Uno's C-index is truncated at
five years. Cumulative/dynamic AUC is evaluated at fixed 1-, 3-, and 5-year
horizons. No model is trained or updated by this script.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sksurv
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv


SEEDS = (101, 202, 303, 404, 505)
FOLDS = (0, 1, 2, 3, 4)
YEAR_DAYS = 365.25
EVALUATION_TIMES = np.asarray([1.0, 3.0, 5.0]) * YEAR_DAYS
UNO_TAU = 5.0 * YEAR_DAYS
AVAILABILITY_COLUMNS = (
    "has_rna",
    "has_cnv",
    "has_rppa",
    "has_methylation",
    "has_he",
)


def survival_array(frame: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(
        event=frame["event"].astype(bool).to_numpy(),
        time=frame["time"].astype(float).to_numpy(),
    )


def validate_test_rows(
    test: pd.DataFrame,
    expected: pd.DataFrame,
    seed: int,
    fold: int,
    scenario: str,
) -> None:
    if test["patient"].duplicated().any():
        raise ValueError(f"Duplicate OOF patient: seed={seed}, fold={fold}")
    if set(test["patient"]) != set(expected["patient"]):
        raise ValueError(
            f"OOF membership mismatch: seed={seed}, fold={fold}, {scenario}"
        )
    check = test[["patient", "time", "event"]].merge(
        expected[["patient", "time", "event"]],
        on="patient",
        suffixes=("_oof", "_master"),
        validate="one_to_one",
    )
    if not np.allclose(check["time_oof"], check["time_master"], atol=0.0):
        raise ValueError(f"Time mismatch: seed={seed}, fold={fold}, {scenario}")
    if not np.array_equal(
        check["event_oof"].astype(bool), check["event_master"].astype(bool)
    ):
        raise ValueError(f"Event mismatch: seed={seed}, fold={fold}, {scenario}")


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    risk_column: str,
) -> dict[str, float | int]:
    y_train = survival_array(train)
    y_test = survival_array(test)
    risk = test[risk_column].astype(float).to_numpy()
    if not np.isfinite(risk).all():
        raise ValueError(f"Non-finite risks in {risk_column}")

    censoring = CensoringDistributionEstimator().fit(y_train)
    censoring_survival = censoring.predict_proba(EVALUATION_TIMES)
    if not np.isfinite(censoring_survival).all() or np.any(
        censoring_survival <= 0
    ):
        raise ValueError("Insufficient outer-training censoring support")

    uno = concordance_index_ipcw(
        y_train,
        y_test,
        risk,
        tau=UNO_TAU,
    )
    auc, mean_auc = cumulative_dynamic_auc(
        y_train,
        y_test,
        risk,
        EVALUATION_TIMES,
    )

    result: dict[str, float | int] = {
        "n_train": int(len(train)),
        "train_events": int(train["event"].astype(bool).sum()),
        "n_test": int(len(test)),
        "test_events": int(test["event"].astype(bool).sum()),
        "uno_c_tau_5y": float(uno[0]),
        "uno_concordant": int(uno[1]),
        "uno_discordant": int(uno[2]),
        "uno_tied_risk": int(uno[3]),
        "uno_tied_time": int(uno[4]),
        "auc_1y": float(auc[0]),
        "auc_3y": float(auc[1]),
        "auc_5y": float(auc[2]),
        "mean_auc_1_3_5y": float(mean_auc),
        "train_max_followup_days": float(train["time"].max()),
        "test_max_followup_days": float(test["time"].max()),
        "censoring_survival_1y": float(censoring_survival[0]),
        "censoring_survival_3y": float(censoring_survival[1]),
        "censoring_survival_5y": float(censoring_survival[2]),
    }
    for years, time in zip((1, 3, 5), EVALUATION_TIMES):
        event = test["event"].astype(bool).to_numpy()
        observed_time = test["time"].astype(float).to_numpy()
        result[f"cases_by_{years}y"] = int((event & (observed_time <= time)).sum())
        result[f"controls_at_{years}y"] = int((observed_time > time).sum())
        result[f"censored_by_{years}y"] = int(
            ((~event) & (observed_time <= time)).sum()
        )
    return result


def weighted_seed_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "uno_c_tau_5y",
        "auc_1y",
        "auc_3y",
        "auc_5y",
        "mean_auc_1_3_5y",
    )
    rows = []
    for (scenario, seed), frame in fold_metrics.groupby(
        ["scenario", "seed"], sort=True
    ):
        weights = frame["n_test"].astype(float).to_numpy()
        row: dict[str, float | int | str] = {
            "scenario": scenario,
            "seed": int(seed),
            "outer_folds": int(len(frame)),
            "n_patients": int(frame["n_test"].sum()),
            "n_events": int(frame["test_events"].sum()),
        }
        for metric in metric_columns:
            row[metric] = float(np.average(frame[metric], weights=weights))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "seed"])


def across_seed_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "uno_c_tau_5y",
        "auc_1y",
        "auc_3y",
        "auc_5y",
        "mean_auc_1_3_5y",
    )
    rows = []
    for scenario, frame in per_seed.groupby("scenario", sort=True):
        row: dict[str, float | int | str] = {
            "scenario": scenario,
            "n_seeds": int(len(frame)),
            "n_patients": int(frame["n_patients"].iloc[0]),
            "n_events": int(frame["n_events"].iloc[0]),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-folds", type=Path, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.master_folds)
    complete_mask = master[list(AVAILABILITY_COLUMNS)].astype(bool).all(axis=1)
    scenarios = {
        "natural_availability": {
            "master_mask": np.ones(len(master), dtype=bool),
            "risk_column": "risk_score_calibrated",
        },
        "complete_case": {
            "master_mask": complete_mask.to_numpy(),
            "risk_column": "risk_score_complete_case_calibrated",
        },
    }

    rows = []
    oof_files = []
    for seed in SEEDS:
        for fold in FOLDS:
            oof_path = (
                args.oof_root
                / f"seed_{seed}"
                / f"fold{fold}"
                / "survival_oof_predictions.tsv.gz"
            )
            oof = pd.read_csv(oof_path, sep="\t")
            oof_files.append(str(oof_path))
            for scenario, definition in scenarios.items():
                scenario_mask = definition["master_mask"]
                train = master.loc[
                    scenario_mask & master["fold"].ne(fold)
                ].copy()
                expected_test = master.loc[
                    scenario_mask & master["fold"].eq(fold)
                ].copy()
                test = oof.copy()
                if scenario == "complete_case":
                    test = test.loc[test["complete_case"].astype(bool)].copy()
                validate_test_rows(test, expected_test, seed, fold, scenario)
                metrics = evaluate_fold(
                    train,
                    test,
                    str(definition["risk_column"]),
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "fold": fold,
                        "risk_column": definition["risk_column"],
                        "uno_tau_days": UNO_TAU,
                        "auc_times_days": "365.25,1095.75,1826.25",
                        **metrics,
                    }
                )

    fold_metrics = pd.DataFrame(rows).sort_values(["scenario", "seed", "fold"])
    per_seed = weighted_seed_summary(fold_metrics)
    summary = across_seed_summary(per_seed)

    fold_metrics.to_csv(
        args.output_dir / "fold_metrics.tsv", sep="\t", index=False
    )
    per_seed.to_csv(args.output_dir / "per_seed.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)

    metadata = {
        "status": "complete",
        "analysis": "final OmniT2I OOF censoring-robust sensitivity",
        "model_training_performed": False,
        "scenarios": list(scenarios),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "fold_evaluations": int(len(fold_metrics)),
        "uno_tau_days": UNO_TAU,
        "uno_tau_years": 5.0,
        "auc_times_days": EVALUATION_TIMES.tolist(),
        "ipcw_reference": "corresponding outer-training patients only",
        "fold_aggregation_within_seed": "outer-test-patient-count weighted mean",
        "seed_aggregation": "mean and sample standard deviation across five seeds",
        "uncertainty_boundary": (
            "Seed SD quantifies training-seed variability and is not a patient-"
            "sampling confidence interval."
        ),
        "master_folds": str(args.master_folds),
        "oof_files": oof_files,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_survival": sksurv.__version__,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
