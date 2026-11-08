"""Prepare fold-aligned frozen GigaPath embeddings for omics + H&E fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-omics-master", type=Path, required=True)
    parser.add_argument("--he-embeddings", type=Path, required=True)
    parser.add_argument("--he-clinical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _one_clinical_row_per_case(path: Path) -> pd.DataFrame:
    columns = ["case_id", "cancer_type", "OS", "OS.time"]
    clinical = pd.read_csv(path, sep="\t", usecols=columns)
    for column in columns[1:]:
        inconsistent = clinical.groupby("case_id")[column].nunique(dropna=False)
        if (inconsistent > 1).any():
            examples = inconsistent[inconsistent > 1].index[:5].tolist()
            raise ValueError(
                f"Inconsistent {column} values within H&E cases: {examples}"
            )
    return clinical.drop_duplicates("case_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(args.four_omics_master)
    if master["patient"].duplicated().any():
        raise ValueError("Four-omics master has duplicate patients")
    expected_folds = set(range(5))
    if set(master["fold"].astype(int)) != expected_folds:
        raise ValueError("Four-omics master must contain folds 0-4")

    he = pd.read_csv(args.he_embeddings, sep="\t")
    feature_columns = [column for column in he if column.startswith("he_emb_")]
    if len(feature_columns) != 1536:
        raise ValueError(f"Expected 1,536 H&E features, found {len(feature_columns)}")
    if he["case_id"].duplicated().any():
        raise ValueError("H&E embedding table has duplicate cases")
    embedding_values = he[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(embedding_values).all():
        raise ValueError("H&E embedding table contains non-finite values")

    clinical = _one_clinical_row_per_case(args.he_clinical)
    he = he[["case_id", "n_patches"]].merge(
        clinical, on="case_id", how="left", validate="one_to_one"
    )
    if he[["cancer_type", "OS", "OS.time"]].isna().any(axis=None):
        raise ValueError("Some H&E embeddings lack clinical annotations")

    # A valid survival observation requires positive follow-up time. The 93
    # GigaPath cases outside the established master all have OS.time == 0 and
    # are therefore excluded rather than creating a misleading H&E-only union.
    valid = he["OS.time"].astype(float).gt(0)
    valid_cases = set(he.loc[valid, "case_id"].astype(str))
    master["has_he"] = master["patient"].astype(str).isin(valid_cases)
    master["he_case"] = master["patient"].where(master["has_he"], np.nan)
    master["availability"] = (
        master["availability"].astype(str)
        + np.where(master["has_he"], "+he", "")
    )
    if not master["has_he"].any():
        raise ValueError("No valid H&E patients overlap the four-omics master")

    he_lookup = pd.Series(np.arange(len(he), dtype=int), index=he["case_id"])
    he_master = master.loc[
        master["has_he"],
        ["patient", "he_case", "cancer_type", "event", "time", "time_years", "fold"],
    ].copy()
    he_master = he_master.rename(columns={"he_case": "sample"})
    he_master["n_patches"] = (
        he_lookup.loc[he_master["patient"]]
        .map(he["n_patches"])
        .to_numpy(dtype=int)
    )

    matched = he_master.merge(
        he[["case_id", "cancer_type", "OS", "OS.time"]],
        left_on="patient",
        right_on="case_id",
        how="left",
        validate="one_to_one",
        suffixes=("_master", "_he"),
    )
    if not np.array_equal(
        matched["event"].astype(int).to_numpy(),
        matched["OS"].astype(int).to_numpy(),
    ):
        raise ValueError("H&E and master event labels disagree")
    if not np.allclose(
        matched["time"].astype(float), matched["OS.time"].astype(float)
    ):
        raise ValueError("H&E and master survival times disagree")
    if not (
        matched["cancer_type_master"].astype(str)
        == matched["cancer_type_he"].astype(str)
    ).all():
        raise ValueError("H&E and master cancer types disagree")

    embedding_rows = he_lookup.loc[he_master["patient"]].to_numpy(dtype=int)
    aligned_embeddings = embedding_values[embedding_rows]
    if len(aligned_embeddings) != len(he_master):
        raise ValueError("H&E cohort and aligned embedding matrix disagree")

    master_path = args.output_dir / "rna_cnv_rppa_methylation_he_master_folds.csv"
    cohort_path = args.output_dir / "he_master_folds.csv"
    matrix_path = args.output_dir / "he_gigapath_case_embeddings.npy"
    feature_path = args.output_dir / "he_gigapath_features.tsv"
    summary_path = args.output_dir / "he_fusion_preparation_summary.json"
    master.to_csv(master_path, index=False)
    he_master.to_csv(cohort_path, index=False)
    np.save(matrix_path, aligned_embeddings.astype(np.float32))
    pd.DataFrame({"feature_id": feature_columns}).to_csv(
        feature_path, sep="\t", index=False
    )

    four_available = ["has_rna", "has_cnv", "has_rppa", "has_methylation"]
    summary = {
        "source_he_cases": int(len(he)),
        "source_embedding_dim": int(len(feature_columns)),
        "source_cases_with_positive_survival_time": int(valid.sum()),
        "excluded_zero_time_cases": int((~valid).sum()),
        "master_patients": int(len(master)),
        "master_he_patients": int(master["has_he"].sum()),
        "complete_case_all_five": int(
            (master[four_available].all(axis=1) & master["has_he"]).sum()
        ),
        "he_patients_by_fold": {
            str(key): int(value)
            for key, value in he_master["fold"].value_counts().sort_index().items()
        },
        "master_output": str(master_path),
        "he_cohort_output": str(cohort_path),
        "he_embedding_output": str(matrix_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
