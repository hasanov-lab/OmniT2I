"""Build leakage-safe patient-level master folds for RNA/CNV fusion.

RNA patients retain their established outer-fold assignment. CNV-only patients
are assigned deterministically while balancing cancer type, event, and fold
size. A patient therefore has one fold regardless of which modalities exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"sample", "patient", "cancer_type", "event", "time", "fold"}


def _validate_cohort(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{name} cohort is missing columns: {sorted(missing)}")
    out = frame.copy()
    out["patient"] = out["patient"].astype(str)
    out["sample"] = out["sample"].astype(str)
    out["cancer_type"] = out["cancer_type"].astype(str)
    out["event"] = out["event"].astype(bool)
    out["time"] = pd.to_numeric(out["time"], errors="raise")
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(int)
    if out["patient"].duplicated().any():
        raise ValueError(f"{name} cohort contains duplicate patients")
    if not out["patient"].eq(out["sample"].str[:12]).all():
        raise ValueError(f"{name} patient IDs disagree with sample barcodes")
    if not out["sample"].str.slice(13, 15).eq("01").all():
        raise ValueError(f"{name} cohort contains non-primary samples")
    if not out["fold"].between(0, 4).all():
        raise ValueError(f"{name} fold IDs must be in [0, 4]")
    if not np.isfinite(out["time"]).all() or not out["time"].gt(0).all():
        raise ValueError(f"{name} contains invalid survival times")
    return out


def _assign_cnv_only_folds(
    cnv_only: pd.DataFrame,
    fixed_rna: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.Series:
    """Greedily balance each cancer-event stratum without changing RNA folds."""
    rng = np.random.default_rng(seed)
    fold_ids = list(range(n_splits))
    total = fixed_rna["fold"].value_counts().reindex(fold_ids, fill_value=0).to_dict()
    event_counts = (
        fixed_rna.groupby(["fold", "event"]).size().to_dict()
    )
    cancer_counts = (
        fixed_rna.groupby(["fold", "cancer_type"]).size().to_dict()
    )
    joint_counts = (
        fixed_rna.groupby(["fold", "cancer_type", "event"]).size().to_dict()
    )

    work = cnv_only.copy()
    work["_random"] = rng.random(len(work))
    stratum_size = work.groupby(["cancer_type", "event"])["patient"].transform("size")
    # Rare strata first makes balancing deterministic and avoids leaving rare
    # cancers to a fold chosen only by overall size.
    work["_stratum_size"] = stratum_size
    work = work.sort_values(
        ["_stratum_size", "cancer_type", "event", "_random", "patient"],
        kind="stable",
    )

    assigned: dict[int, int] = {}
    for idx, row in work.iterrows():
        cancer = str(row["cancer_type"])
        event = bool(row["event"])
        tie = rng.permutation(n_splits).tolist()
        tie_rank = {fold: rank for rank, fold in enumerate(tie)}
        fold = min(
            fold_ids,
            key=lambda f: (
                joint_counts.get((f, cancer, event), 0),
                cancer_counts.get((f, cancer), 0),
                event_counts.get((f, event), 0),
                total[f],
                tie_rank[f],
            ),
        )
        assigned[idx] = fold
        total[fold] += 1
        event_counts[(fold, event)] = event_counts.get((fold, event), 0) + 1
        cancer_counts[(fold, cancer)] = cancer_counts.get((fold, cancer), 0) + 1
        joint_counts[(fold, cancer, event)] = (
            joint_counts.get((fold, cancer, event), 0) + 1
        )
    return pd.Series(assigned, name="fold", dtype=int).reindex(cnv_only.index)


def build_master_folds(
    rna: pd.DataFrame,
    cnv: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 20260721,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rna = _validate_cohort(rna, "RNA")
    cnv = _validate_cohort(cnv, "CNV")
    if sorted(rna["fold"].unique()) != list(range(n_splits)):
        raise ValueError("RNA cohort does not contain every requested fold")

    overlap = rna.merge(cnv, on="patient", suffixes=("_rna", "_cnv"), validate="one_to_one")
    disagreements = {
        "sample": int((overlap["sample_rna"] != overlap["sample_cnv"]).sum()),
        "cancer_type": int(
            (overlap["cancer_type_rna"] != overlap["cancer_type_cnv"]).sum()
        ),
        "event": int((overlap["event_rna"] != overlap["event_cnv"]).sum()),
        "time": int(
            (~np.isclose(overlap["time_rna"], overlap["time_cnv"], rtol=0, atol=1e-8)).sum()
        ),
    }
    if any(disagreements.values()):
        raise ValueError(f"RNA/CNV overlap metadata disagree: {disagreements}")

    rna_core = rna[["patient", "sample", "cancer_type", "event", "time", "fold"]].copy()
    rna_core = rna_core.rename(columns={"sample": "rna_sample"})
    rna_core["has_rna"] = True

    cnv_core = cnv[["patient", "sample", "cancer_type", "event", "time"]].copy()
    cnv_core = cnv_core.rename(
        columns={
            "sample": "cnv_sample",
            "cancer_type": "cnv_cancer_type",
            "event": "cnv_event",
            "time": "cnv_time",
        }
    )
    master = rna_core.merge(cnv_core, on="patient", how="outer", validate="one_to_one")
    master["has_rna"] = master["has_rna"].eq(True)
    master["has_cnv"] = master["cnv_sample"].notna()

    cnv_only_mask = ~master["has_rna"] & master["has_cnv"]
    master.loc[cnv_only_mask, "cancer_type"] = master.loc[
        cnv_only_mask, "cnv_cancer_type"
    ]
    master.loc[cnv_only_mask, "event"] = master.loc[cnv_only_mask, "cnv_event"]
    master.loc[cnv_only_mask, "time"] = master.loc[cnv_only_mask, "cnv_time"]
    master["event"] = master["event"].astype(bool)
    master["time"] = master["time"].astype(float)

    cnv_only = master.loc[cnv_only_mask, ["patient", "cancer_type", "event"]].copy()
    master.loc[cnv_only_mask, "fold"] = _assign_cnv_only_folds(
        cnv_only, rna_core, n_splits, seed
    )
    master["fold"] = master["fold"].astype(int)
    master["time_years"] = master["time"] / 365.25
    master["availability"] = np.select(
        [master["has_rna"] & master["has_cnv"], master["has_rna"], master["has_cnv"]],
        ["rna+cnv", "rna_only", "cnv_only"],
        default="none",
    )
    if (master["availability"] == "none").any():
        raise RuntimeError("Master cohort contains a patient with no modality")
    observed_rna_folds = master.loc[master["has_rna"], ["patient", "fold"]].set_index(
        "patient"
    )["fold"]
    expected_rna_folds = rna_core.set_index("patient")["fold"]
    if observed_rna_folds.to_dict() != expected_rna_folds.to_dict():
        raise RuntimeError("RNA fold assignments changed while building master folds")

    keep = [
        "patient", "rna_sample", "cnv_sample", "has_rna", "has_cnv",
        "availability", "cancer_type", "event", "time", "time_years", "fold",
    ]
    master = master[keep].sort_values(["fold", "cancer_type", "patient"]).reset_index(drop=True)

    fold_availability = (
        master.groupby(["fold", "availability"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(n_splits), fill_value=0)
    )
    summary: dict[str, object] = {
        "definition": "Union of valid-OS TCGA 01 Primary Tumor RNA and CNV cohorts",
        "fold_policy": "Preserve established RNA folds; deterministically assign CNV-only patients",
        "seed_for_cnv_only_assignment": seed,
        "n_folds": n_splits,
        "n_patients_union": int(len(master)),
        "n_rna": int(master["has_rna"].sum()),
        "n_cnv": int(master["has_cnv"].sum()),
        "n_rna_cnv": int((master["availability"] == "rna+cnv").sum()),
        "n_rna_only": int((master["availability"] == "rna_only").sum()),
        "n_cnv_only": int((master["availability"] == "cnv_only").sum()),
        "n_cancer_types": int(master["cancer_type"].nunique()),
        "n_events": int(master["event"].sum()),
        "overlap_metadata_disagreements": disagreements,
        "fold_sizes": {
            str(k): int(v) for k, v in master["fold"].value_counts().sort_index().items()
        },
        "fold_availability": {
            str(fold): {str(k): int(v) for k, v in row.items()}
            for fold, row in fold_availability.iterrows()
        },
    }
    return master, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-folds", type=Path, required=True)
    parser.add_argument("--cnv-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rna = pd.read_csv(args.rna_folds)
    cnv = pd.read_csv(args.cnv_folds)
    master, summary = build_master_folds(rna, cnv, args.n_splits, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)
    common_columns = ["patient", "cancer_type", "event", "time", "time_years", "fold"]
    rna_view = master.loc[master["has_rna"], ["rna_sample", *common_columns]].rename(
        columns={"rna_sample": "sample"}
    )
    cnv_view = master.loc[master["has_cnv"], ["cnv_sample", *common_columns]].rename(
        columns={"cnv_sample": "sample"}
    )
    rna_view.to_csv(args.output.with_name("rna_master_folds.csv"), index=False)
    cnv_view.to_csv(args.output.with_name("cnv_master_folds.csv"), index=False)
    summary_path = args.output.with_name("rna_cnv_master_folds_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
