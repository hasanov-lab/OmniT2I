"""Prepare the manuscript's shared core/validation/test patient partitions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from omnit2i.data import validate_cohort


def inner_split(clinical, seed):
    cancer = clinical["cancer_type"].astype(str)
    event = clinical["event"].astype(int).astype(str)
    joint_labels = cancer + "_e" + event
    joint_counts = joint_labels.value_counts()
    rare_cancers = set(cancer[joint_labels.map(joint_counts) < 2])
    labels = joint_labels.where(~cancer.isin(rare_cancers), cancer + "_all_events")
    if labels.value_counts().min() < 2:
        labels = event
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(splitter.split(clinical, labels))


def build_shared_cohort(master, fold, seed):
    required = {"patient", "cancer_type", "event", "time", "fold"}
    if required - set(master):
        raise ValueError(f"Master missing columns: {sorted(required - set(master))}")
    cohort = master.copy().reset_index(drop=True)
    if cohort.patient.isna().any() or cohort.patient.duplicated().any():
        raise ValueError("Master patient IDs must be nonmissing and unique")
    if not cohort.fold.isin(range(5)).all() or fold not in cohort.fold.unique():
        raise ValueError("Master folds must be 0-4 and contain the requested test fold")
    if "sample" not in cohort:
        cohort["sample"] = cohort["patient"]
    # Validate outcomes before stratification; test patients never enter the split.
    cohort["partition"] = np.where(cohort.fold.eq(fold), "test", "core")
    if cohort[["cancer_type", "event", "time"]].isna().any(axis=None):
        raise ValueError("Master cancer types and outcomes must be nonmissing")
    if not cohort.event.isin([0, 1]).all() or not np.isfinite(cohort.time).all() or not cohort.time.gt(0).all():
        raise ValueError("Master time must be positive and finite; event must be 0 or 1")
    outer = np.flatnonzero(cohort.fold.ne(fold).to_numpy())
    core, validation = inner_split(cohort.iloc[outer].reset_index(drop=True), seed + fold * 1000)
    cohort.loc[outer[core], "partition"] = "core"
    cohort.loc[outer[validation], "partition"] = "validation"
    cohort["seed"], cohort["outer_fold"] = seed, fold
    return validate_cohort(cohort)


def modality_cohorts(cohort):
    views = {}
    for name, sample, flag in [
        ("rna", "rna_sample", "has_rna"), ("cnv", "cnv_sample", "has_cnv"),
        ("protein", "rppa_sample", "has_rppa"),
        ("methylation", "methylation_sample", "has_methylation"),
        ("he", "he_case", "has_he"),
    ]:
        if sample not in cohort or flag not in cohort:
            raise ValueError(f"Master requires {sample} and {flag}")
        if not cohort[flag].isin([True, False]).all():
            raise ValueError(f"{flag} must contain boolean availability")
        observed = cohort[flag].astype(bool)
        if not np.array_equal(cohort[sample].notna(), observed):
            raise ValueError(f"{sample} disagrees with {flag}")
        view = cohort.loc[observed].copy()
        view["sample"] = view[sample]
        views[name] = validate_cohort(view)
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cohort = build_shared_cohort(pd.read_csv(args.master, dtype={"patient": str}), args.fold, args.seed)
    views = modality_cohorts(cohort)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cohort.to_csv(args.output_dir / "cohort.csv", index=False)
    for name, view in views.items():
        view.to_csv(args.output_dir / f"{name}_cohort.csv", index=False)
    print(f"Prepared shared seed {args.seed}, fold {args.fold}: {args.output_dir}")


if __name__ == "__main__":
    main()
