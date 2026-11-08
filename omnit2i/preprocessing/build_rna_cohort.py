"""Build the established RNA primary-tumour OS outer folds from raw table headers."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from .cohort_samples import read_matrix_header

def endpoint_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    df = cohort[["sample", "patient", "cancer_type", "sample_type", "sample_type_id", "OS", "OS.time"]].copy()
    df = df.rename(columns={"OS": "event", "OS.time": "time"})
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df[df["event"].isin([0, 1]) & df["time"].notna() & (df["time"] > 0)].copy()
    df["event"] = df["event"].astype(bool)
    df["time_years"] = df["time"] / 365.25
    return df.sort_values(["cancer_type", "sample"]).reset_index(drop=True)


def stratification_labels(df: pd.DataFrame, n_splits: int) -> pd.Series:
    labels = df["cancer_type"].astype(str) + "_e" + df["event"].astype(int).astype(str)
    counts = labels.value_counts()
    return labels.where(labels.map(counts) >= n_splits, df["event"].astype(int).astype(str))


def assign_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    labels = stratification_labels(df, n_splits)
    folds = np.full(len(df), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_idx) in enumerate(splitter.split(df, labels)):
        folds[test_idx] = fold
    out = df.copy()
    out["fold"] = folds
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expression", "survival", "phenotype", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    _, samples = read_matrix_header(args.expression)
    survival = pd.read_csv(args.survival, sep="\t", dtype=str)
    phenotype = pd.read_csv(args.phenotype, sep="\t", dtype=str)
    cohort = survival.loc[survival["sample"].isin(samples) & survival["sample"].str.endswith("-01")].rename(
        columns={"_PATIENT": "patient", "cancer type abbreviation": "cancer_type"})
    cohort = cohort.merge(phenotype[["sample", "sample_type_id", "sample_type", "_primary_disease"]], on="sample", how="left")
    folds = assign_folds(endpoint_frame(cohort), 5, 123)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
