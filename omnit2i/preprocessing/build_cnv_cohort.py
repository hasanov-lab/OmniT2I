"""Build CNV-specific five-fold OS cohort from TCGA primary tumors only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .cohort_samples import read_matrix_header


def assign_folds(cohort: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    cancer = cohort["cancer_type"].astype(str)
    event = cohort["event"].astype(int).astype(str)
    joint = cancer + "_e" + event
    joint_counts = joint.value_counts()

    # If either outcome group within a cancer is too small for K folds,
    # stratify that entire cancer by cancer type only. This preserves cancer
    # balance without creating an undersized fallback stratum.
    rare_cancers = set(cancer[joint.map(joint_counts) < n_splits])
    strata = joint.where(~cancer.isin(rare_cancers), cancer + "_all_events")
    if strata.value_counts().min() < n_splits:
        raise ValueError("A CNV primary-tumor stratum has fewer samples than folds")

    fold_values = pd.Series(-1, index=cohort.index, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_indices) in enumerate(splitter.split(cohort, strata)):
        fold_values.iloc[test_indices] = fold
    if (fold_values < 0).any():
        raise RuntimeError("Some CNV primary-tumor samples were not assigned a fold")
    result = cohort.copy()
    result["fold"] = fold_values.to_numpy()
    return result


def build_cohort(
    cnv_samples: list[str],
    survival: pd.DataFrame,
    phenotype: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    primary_samples = [sample for sample in cnv_samples if sample[13:15] == "01"]
    cnv = pd.DataFrame({"sample": primary_samples})

    survival_columns = [
        "sample",
        "_PATIENT",
        "cancer type abbreviation",
        "OS",
        "OS.time",
    ]
    missing_survival = set(survival_columns) - set(survival.columns)
    if missing_survival:
        raise ValueError(f"Survival table is missing columns: {sorted(missing_survival)}")
    phenotype_columns = ["sample", "sample_type_id", "sample_type"]
    missing_phenotype = set(phenotype_columns) - set(phenotype.columns)
    if missing_phenotype:
        raise ValueError(f"Phenotype table is missing columns: {sorted(missing_phenotype)}")

    cohort = cnv.merge(
        survival[survival_columns], on="sample", how="left", validate="one_to_one"
    ).merge(
        phenotype[phenotype_columns], on="sample", how="left", validate="one_to_one"
    )
    cohort["event"] = pd.to_numeric(cohort["OS"], errors="coerce")
    cohort["time"] = pd.to_numeric(cohort["OS.time"], errors="coerce")
    cohort = cohort.loc[
        cohort["event"].isin([0, 1]) & cohort["time"].gt(0)
    ].copy()
    cohort = cohort.rename(
        columns={"_PATIENT": "patient", "cancer type abbreviation": "cancer_type"}
    )
    cohort["patient"] = cohort["patient"].astype(str)
    cohort["cancer_type"] = cohort["cancer_type"].astype(str)
    cohort["sample_type_id"] = cohort["sample_type_id"].astype(str).str.zfill(2)
    if not cohort["sample"].str.slice(13, 15).eq("01").all():
        raise ValueError("Non-primary CNV sample entered the primary cohort")
    discordant_sample_type = ~cohort["sample_type_id"].eq("01")
    n_discordant_sample_type = int(discordant_sample_type.sum())
    # A sample is primary only when its barcode and phenotype annotation agree.
    # Exclude discordant records rather than choosing one source silently.
    cohort = cohort.loc[~discordant_sample_type].copy()
    if not cohort["patient"].eq(cohort["sample"].str[:12]).all():
        raise ValueError("Survival patient ID disagrees with CNV sample barcode")
    if cohort["patient"].duplicated().any():
        raise ValueError("CNV primary OS cohort contains duplicate patients")

    cohort["event"] = cohort["event"].astype(bool)
    cohort["time_years"] = cohort["time"] / 365.25
    cohort = cohort[
        [
            "sample",
            "patient",
            "cancer_type",
            "sample_type",
            "sample_type_id",
            "event",
            "time",
            "time_years",
        ]
    ].sort_values(["cancer_type", "sample"]).reset_index(drop=True)
    cohort = assign_folds(cohort, n_splits, seed)

    summary: dict[str, int | str] = {
        "cohort_definition": "TCGA CNV sample type 01 (Primary Tumor) with valid OS",
        "n_cnv_samples_total": len(cnv_samples),
        "n_cnv_primary_samples": len(primary_samples),
        "n_valid_os_before_phenotype_concordance": len(cohort)
        + n_discordant_sample_type,
        "n_barcode_phenotype_sample_type_discordant": n_discordant_sample_type,
        "n_valid_os_samples": len(cohort),
        "n_unique_patients": int(cohort["patient"].nunique()),
        "n_cancer_types": int(cohort["cancer_type"].nunique()),
        "n_events": int(cohort["event"].sum()),
        "n_outer_folds": n_splits,
        "seed": seed,
    }
    return cohort, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnv", type=Path, required=True)
    parser.add_argument("--survival", type=Path, required=True)
    parser.add_argument("--phenotype", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, cnv_samples = read_matrix_header(args.cnv)
    survival = pd.read_csv(args.survival, sep="\t", dtype=str)
    phenotype = pd.read_csv(args.phenotype, sep="\t", dtype=str)
    cohort, summary = build_cohort(
        cnv_samples, survival, phenotype, args.n_splits, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    summary_path = args.output.with_name("cnv_primary_os_cohort_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
