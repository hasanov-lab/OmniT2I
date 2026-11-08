"""Extend the RNA/CNV patient-level master folds with primary-tumor RPPA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_rna_cnv_master import _assign_cnv_only_folds
from .cohort_samples import read_rppa_header, select_primary_rppa_samples


def build_rppa_cohort(
    samples: list[str],
    survival: pd.DataFrame,
    phenotype: pd.DataFrame,
) -> pd.DataFrame:
    selected = select_primary_rppa_samples(samples)
    survival_columns = [
        "sample", "_PATIENT", "cancer type abbreviation", "OS", "OS.time"
    ]
    phenotype_columns = ["sample", "sample_type_id", "sample_type"]
    survival_view = survival[survival_columns].rename(
        columns={"sample": "rppa_sample"}
    )
    phenotype_view = phenotype[phenotype_columns].rename(
        columns={"sample": "rppa_sample"}
    )
    cohort = selected.merge(
        survival_view,
        on="rppa_sample",
        how="left",
        validate="one_to_one",
    ).merge(
        phenotype_view,
        on="rppa_sample",
        how="left",
        validate="one_to_one",
    )
    cohort["event"] = pd.to_numeric(cohort["OS"], errors="coerce")
    cohort["time"] = pd.to_numeric(cohort["OS.time"], errors="coerce")
    cohort["sample_type_id"] = cohort["sample_type_id"].astype(str).str.zfill(2)
    valid = (
        cohort["event"].isin([0, 1])
        & cohort["time"].gt(0)
        & cohort["sample_type_id"].eq("01")
        & cohort["_PATIENT"].eq(cohort["patient"])
    )
    cohort = cohort.loc[valid].rename(
        columns={
            "cancer type abbreviation": "cancer_type",
        }
    )
    cohort["event"] = cohort["event"].astype(bool)
    cohort["time"] = cohort["time"].astype(float)
    cohort["cancer_type"] = cohort["cancer_type"].astype(str)
    if cohort["patient"].duplicated().any():
        raise ValueError("RPPA valid-OS cohort contains duplicate patients")
    return cohort[
        [
            "rppa_sample", "patient", "cancer_type", "event", "time",
            "sample_type", "sample_type_id",
        ]
    ].reset_index(drop=True)


def extend_master_folds(
    master: pd.DataFrame,
    rppa: pd.DataFrame,
    seed: int = 20260725,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = master.copy()
    required = {
        "patient", "rna_sample", "cnv_sample", "has_rna", "has_cnv",
        "cancer_type", "event", "time", "fold",
    }
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"Base master is missing columns: {sorted(missing)}")
    if base["patient"].duplicated().any() or rppa["patient"].duplicated().any():
        raise ValueError("Master/RPPA patient IDs must be unique")

    overlap = base.merge(
        rppa[["patient", "cancer_type", "event", "time"]],
        on="patient",
        suffixes=("_master", "_rppa"),
        validate="one_to_one",
    )
    disagreements = {
        "cancer_type": int(
            (overlap["cancer_type_master"] != overlap["cancer_type_rppa"]).sum()
        ),
        "event": int((overlap["event_master"].astype(bool) != overlap["event_rppa"]).sum()),
        "time": int(
            (~np.isclose(overlap["time_master"], overlap["time_rppa"], atol=1e-8)).sum()
        ),
    }
    if any(disagreements.values()):
        raise ValueError(f"Master/RPPA overlap metadata disagree: {disagreements}")

    base = base.drop(columns=["availability", "time_years"], errors="ignore")
    base = base.merge(
        rppa[["patient", "rppa_sample"]],
        on="patient",
        how="left",
        validate="one_to_one",
    )
    base["has_rppa"] = base["rppa_sample"].notna()

    base_patients = set(base["patient"])
    rppa_only = rppa.loc[~rppa["patient"].isin(base_patients)].copy()
    if len(rppa_only):
        rppa_only["fold"] = _assign_cnv_only_folds(
            rppa_only[["patient", "cancer_type", "event"]],
            base[["patient", "cancer_type", "event", "fold"]],
            n_splits=5,
            seed=seed,
        ).to_numpy()
        rppa_only["rna_sample"] = np.nan
        rppa_only["cnv_sample"] = np.nan
        rppa_only["has_rna"] = False
        rppa_only["has_cnv"] = False
        rppa_only["has_rppa"] = True
        base = pd.concat(
            [
                base,
                rppa_only[
                    [
                        "patient", "rna_sample", "cnv_sample", "rppa_sample",
                        "has_rna", "has_cnv", "has_rppa", "cancer_type",
                        "event", "time", "fold",
                    ]
                ],
            ],
            ignore_index=True,
        )

    def availability(row: pd.Series) -> str:
        present = [
            name
            for name, column in [
                ("rna", "has_rna"), ("cnv", "has_cnv"), ("rppa", "has_rppa")
            ]
            if bool(row[column])
        ]
        return "+".join(present)

    base["event"] = base["event"].astype(bool)
    base["fold"] = base["fold"].astype(int)
    base["time"] = base["time"].astype(float)
    base["time_years"] = base["time"] / 365.25
    base["availability"] = base.apply(availability, axis=1)
    columns = [
        "patient", "rna_sample", "cnv_sample", "rppa_sample",
        "has_rna", "has_cnv", "has_rppa", "availability",
        "cancer_type", "event", "time", "time_years", "fold",
    ]
    base = base[columns].sort_values(
        ["fold", "cancer_type", "patient"], kind="stable"
    ).reset_index(drop=True)
    if base["patient"].duplicated().any():
        raise RuntimeError("Extended master contains duplicate patients")

    summary: dict[str, object] = {
        "definition": "Union of valid-OS TCGA 01 RNA, CNV, and RPPA cohorts",
        "fold_policy": (
            "Preserve every RNA/CNV master fold; assign only new RPPA-only "
            "patients by cancer/event-balanced deterministic allocation"
        ),
        "seed_for_rppa_only_assignment": seed,
        "n_patients_union": int(len(base)),
        "n_rna": int(base["has_rna"].sum()),
        "n_cnv": int(base["has_cnv"].sum()),
        "n_rppa": int(base["has_rppa"].sum()),
        "n_rppa_overlap_existing_master": int((base["has_rppa"] & (base["has_rna"] | base["has_cnv"])).sum()),
        "n_rppa_only": int((base["has_rppa"] & ~base["has_rna"] & ~base["has_cnv"]).sum()),
        "n_cancer_types": int(base["cancer_type"].nunique()),
        "n_events": int(base["event"].sum()),
        "fold_sizes": {
            str(k): int(v) for k, v in base["fold"].value_counts().sort_index().items()
        },
        "availability_counts": {
            str(k): int(v) for k, v in base["availability"].value_counts().items()
        },
        "overlap_metadata_disagreements": disagreements,
    }
    return base, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rppa", type=Path, required=True)
    parser.add_argument("--survival", type=Path, required=True)
    parser.add_argument("--phenotype", type=Path, required=True)
    parser.add_argument("--base-master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, samples = read_rppa_header(args.rppa)
    survival = pd.read_csv(args.survival, sep="\t", dtype=str)
    phenotype = pd.read_csv(args.phenotype, sep="\t", dtype=str)
    master = pd.read_csv(args.base_master)
    cohort = build_rppa_cohort(samples, survival, phenotype)
    extended, summary = extend_master_folds(master, cohort, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extended.to_csv(args.output, index=False)
    rppa_view = (
        extended.loc[extended["has_rppa"]]
        .rename(columns={"rppa_sample": "sample"})
        [["sample", "patient", "cancer_type", "event", "time", "time_years", "fold"]]
    )
    rppa_view.to_csv(args.output.with_name("rppa_master_folds.csv"), index=False)
    args.output.with_name("rna_cnv_rppa_master_folds_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
