#!/usr/bin/env python3
"""Equal-mean original and extra slide embeddings into patient embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-case-embeddings", type=Path, required=True)
    parser.add_argument("--extra-slide-embeddings", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--target-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort)
    patients = cohort["patient"].astype(str)
    baseline = pd.read_csv(args.baseline_case_embeddings, sep="\t")
    extra = pd.read_csv(args.extra_slide_embeddings, sep="\t")
    target = pd.read_csv(args.target_table, sep="\t")
    feature_columns = [column for column in baseline if column.startswith("he_emb_")]
    if len(feature_columns) != 1536:
        raise RuntimeError(f"Expected 1536 baseline features, found {len(feature_columns)}")
    if feature_columns != [
        column for column in extra if column.startswith("he_emb_")
    ]:
        raise RuntimeError("Baseline and extra embedding columns disagree")

    baseline = baseline.loc[baseline["case_id"].astype(str).isin(set(patients))].copy()
    if baseline["case_id"].duplicated().any() or len(baseline) != len(patients):
        raise RuntimeError("Baseline must contain exactly one row per cohort patient")
    baseline["file_id"] = "baseline"
    baseline["file_name"] = "existing_one_slide"
    baseline["source"] = "existing_one_slide"
    extra["source"] = "additional_diagnostic_slide"
    slides = pd.concat(
        [
            baseline[
                ["case_id", "file_id", "file_name", "n_patches", "source"]
                + feature_columns
            ],
            extra[
                ["case_id", "file_id", "file_name", "n_patches", "source"]
                + feature_columns
            ],
        ],
        ignore_index=True,
    )
    if not np.isfinite(slides[feature_columns].to_numpy(dtype=np.float32)).all():
        raise RuntimeError("Non-finite slide embeddings")
    expected_slides = (
        target.loc[target["case_id"].astype(str).isin(set(patients))]
        .groupby("case_id")
        .size()
    )
    observed_slides = slides.groupby("case_id").size()
    if not expected_slides.equals(observed_slides.reindex(expected_slides.index)):
        mismatch = expected_slides.compare(
            observed_slides.reindex(expected_slides.index), keep_shape=False
        )
        raise RuntimeError(f"Slide counts disagree for {len(mismatch)} patients")

    values = slides[feature_columns].to_numpy(dtype=np.float32)
    value_frame = pd.DataFrame(values, columns=feature_columns)
    value_frame.insert(0, "case_id", slides["case_id"].astype(str).to_numpy())
    patient_embeddings = value_frame.groupby("case_id", sort=True).mean()
    patient_embeddings.insert(0, "n_slides", observed_slides[patient_embeddings.index])
    patient_embeddings.insert(
        1,
        "n_patches",
        slides.groupby("case_id")["n_patches"].sum()[patient_embeddings.index],
    )
    patient_embeddings = patient_embeddings.reset_index()
    if set(patient_embeddings["case_id"]) != set(patients):
        raise RuntimeError("Final patient set differs from cohort")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slide_output = args.output_dir / "all_slide_embeddings.tsv.gz"
    patient_output = args.output_dir / "multislide_patient_mean_embeddings.tsv"
    slides.to_csv(slide_output, sep="\t", index=False, compression="gzip")
    patient_embeddings.to_csv(patient_output, sep="\t", index=False)
    summary = {
        "aggregation": "equal mean over slide embeddings per patient",
        "patients": int(len(patient_embeddings)),
        "slides": int(len(slides)),
        "baseline_slides": int((slides["source"] == "existing_one_slide").sum()),
        "extra_diagnostic_slides": int(
            (slides["source"] == "additional_diagnostic_slide").sum()
        ),
        "patients_with_multiple_slides": int(
            (patient_embeddings["n_slides"] > 1).sum()
        ),
        "embedding_dim": len(feature_columns),
        "slide_output": str(slide_output),
        "patient_output": str(patient_output),
    }
    (args.output_dir / "multislide_aggregation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
