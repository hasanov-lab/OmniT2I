"""Add primary-tumour methylation to the existing RNA/CNV/Protein master folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_rna_cnv_master import _assign_cnv_only_folds


def extend_master_folds(
    master: pd.DataFrame,
    methylation: pd.DataFrame,
    seed: int = 20260727,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    base = master.copy()
    methylation = methylation.copy()
    required_master = {
        "patient",
        "rna_sample",
        "cnv_sample",
        "rppa_sample",
        "has_rna",
        "has_cnv",
        "has_rppa",
        "cancer_type",
        "event",
        "time",
        "fold",
    }
    required_methylation = {
        "sample",
        "patient",
        "cancer_type",
        "event",
        "time",
    }
    missing_master = required_master - set(base.columns)
    missing_methylation = required_methylation - set(methylation.columns)
    if missing_master:
        raise ValueError(f"Base master missing columns: {sorted(missing_master)}")
    if missing_methylation:
        raise ValueError(
            f"Methylation cohort missing columns: {sorted(missing_methylation)}"
        )
    if base["patient"].duplicated().any() or methylation["patient"].duplicated().any():
        raise ValueError("Master and methylation patient IDs must be unique")
    if not methylation["sample"].astype(str).str.slice(13, 15).eq("01").all():
        raise ValueError("Methylation cohort contains non-primary samples")
    if not methylation["patient"].astype(str).eq(
        methylation["sample"].astype(str).str[:12]
    ).all():
        raise ValueError("Methylation patient IDs disagree with sample barcodes")

    overlap = base.merge(
        methylation[["patient", "cancer_type", "event", "time"]],
        on="patient",
        suffixes=("_master", "_methylation"),
        validate="one_to_one",
    )
    disagreements = {
        "cancer_type": int(
            (
                overlap["cancer_type_master"].astype(str)
                != overlap["cancer_type_methylation"].astype(str)
            ).sum()
        ),
        "event": int(
            (
                overlap["event_master"].astype(bool)
                != overlap["event_methylation"].astype(bool)
            ).sum()
        ),
        "time": int(
            (
                ~np.isclose(
                    overlap["time_master"].astype(float),
                    overlap["time_methylation"].astype(float),
                    atol=1e-8,
                )
            ).sum()
        ),
    }
    if any(disagreements.values()):
        raise ValueError(
            f"Master/methylation overlap metadata disagree: {disagreements}"
        )

    base = base.drop(columns=["availability", "time_years"], errors="ignore")
    methylation = methylation.rename(columns={"sample": "methylation_sample"})
    base = base.merge(
        methylation[["patient", "methylation_sample"]],
        on="patient",
        how="left",
        validate="one_to_one",
    )
    base["has_methylation"] = base["methylation_sample"].notna()

    existing = set(base["patient"])
    methylation_only = methylation.loc[
        ~methylation["patient"].isin(existing)
    ].copy()
    if len(methylation_only):
        methylation_only["fold"] = _assign_cnv_only_folds(
            methylation_only[["patient", "cancer_type", "event"]],
            base[["patient", "cancer_type", "event", "fold"]],
            n_splits=5,
            seed=seed,
        ).to_numpy()
        for sample_column, flag_column in [
            ("rna_sample", "has_rna"),
            ("cnv_sample", "has_cnv"),
            ("rppa_sample", "has_rppa"),
        ]:
            methylation_only[sample_column] = np.nan
            methylation_only[flag_column] = False
        methylation_only["has_methylation"] = True
        base = pd.concat(
            [
                base,
                methylation_only[
                    [
                        "patient",
                        "rna_sample",
                        "cnv_sample",
                        "rppa_sample",
                        "methylation_sample",
                        "has_rna",
                        "has_cnv",
                        "has_rppa",
                        "has_methylation",
                        "cancer_type",
                        "event",
                        "time",
                        "fold",
                    ]
                ],
            ],
            ignore_index=True,
        )

    modality_columns = [
        ("rna", "has_rna"),
        ("cnv", "has_cnv"),
        ("rppa", "has_rppa"),
        ("methylation", "has_methylation"),
    ]
    base["availability"] = base.apply(
        lambda row: "+".join(
            name for name, column in modality_columns if bool(row[column])
        ),
        axis=1,
    )
    base["event"] = base["event"].astype(bool)
    base["time"] = base["time"].astype(float)
    base["fold"] = base["fold"].astype(int)
    base["time_years"] = base["time"] / 365.25
    columns = [
        "patient",
        "rna_sample",
        "cnv_sample",
        "rppa_sample",
        "methylation_sample",
        "has_rna",
        "has_cnv",
        "has_rppa",
        "has_methylation",
        "availability",
        "cancer_type",
        "event",
        "time",
        "time_years",
        "fold",
    ]
    extended = base[columns].sort_values(
        ["fold", "cancer_type", "patient"], kind="stable"
    ).reset_index(drop=True)
    if extended["patient"].duplicated().any():
        raise RuntimeError("Extended master contains duplicate patients")

    methylation_view = (
        extended.loc[extended["has_methylation"]]
        .rename(columns={"methylation_sample": "sample"})
        [
            [
                "sample",
                "patient",
                "cancer_type",
                "event",
                "time",
                "time_years",
                "fold",
            ]
        ]
        .reset_index(drop=True)
    )
    summary: dict[str, object] = {
        "definition": (
            "Union of valid-OS TCGA 01 RNA, CNV, RPPA, and promoter-gene "
            "methylation cohorts"
        ),
        "fold_policy": (
            "Preserve every RNA/CNV/RPPA master fold; assign only genuinely new "
            "methylation-only patients by deterministic cancer/event balancing"
        ),
        "seed_for_methylation_only_assignment": seed,
        "n_patients_union": int(len(extended)),
        "n_methylation": int(extended["has_methylation"].sum()),
        "n_methylation_overlap_existing_master": int(
            (
                extended["has_methylation"]
                & (extended["has_rna"] | extended["has_cnv"] | extended["has_rppa"])
            ).sum()
        ),
        "n_methylation_only": int(
            (
                extended["has_methylation"]
                & ~extended["has_rna"]
                & ~extended["has_cnv"]
                & ~extended["has_rppa"]
            ).sum()
        ),
        "n_cancer_types_methylation": int(
            methylation_view["cancer_type"].nunique()
        ),
        "n_events_methylation": int(methylation_view["event"].sum()),
        "fold_sizes_methylation": {
            str(key): int(value)
            for key, value in methylation_view["fold"]
            .value_counts()
            .sort_index()
            .items()
        },
        "overlap_metadata_disagreements": disagreements,
    }
    return extended, methylation_view, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-master", type=Path, required=True)
    parser.add_argument("--methylation-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    master = pd.read_csv(args.base_master)
    methylation = pd.read_csv(args.methylation_samples, sep="\t")
    extended, methylation_view, summary = extend_master_folds(
        master, methylation, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extended.to_csv(args.output, index=False)
    methylation_view.to_csv(
        args.output.with_name("methylation_master_folds.csv"), index=False
    )
    args.output.with_name(
        "rna_cnv_rppa_methylation_master_folds_summary.json"
    ).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
