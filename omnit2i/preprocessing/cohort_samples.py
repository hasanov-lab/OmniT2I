from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

def read_matrix_header(path: Path) -> tuple[str, list[str]]:
    with gzip.open(path, "rt") as handle:
        fields = handle.readline().rstrip("\r\n").split("\t")
    if len(fields) < 2:
        raise ValueError(f"CNV matrix has fewer than two columns: {path}")
    return fields[0], fields[1:]

def read_rppa_header(path: Path) -> tuple[str, list[str]]:
    with gzip.open(path, "rt") as handle:
        fields = handle.readline().rstrip("\r\n").split("\t")
    if len(fields) < 2:
        raise ValueError(f"RPPA matrix has fewer than two columns: {path}")
    return fields[0], fields[1:]


def select_primary_rppa_samples(sample_ids: Iterable[str]) -> pd.DataFrame:
    """Choose one canonical 01 sample per patient.

    Some Xena columns carry a trailing ``.1`` replicate suffix. Prefer the
    canonical column without that suffix, then use lexical order as a stable
    tie breaker.
    """
    frame = pd.DataFrame({"rppa_sample": list(map(str, sample_ids))})
    frame = frame.loc[frame["rppa_sample"].str.slice(13, 15).eq("01")].copy()
    frame["patient"] = frame["rppa_sample"].str[:12]
    frame["_replicate_suffix"] = frame["rppa_sample"].str.contains(r"\.", regex=True)
    frame = frame.sort_values(
        ["patient", "_replicate_suffix", "rppa_sample"], kind="stable"
    )
    return (
        frame.drop_duplicates("patient", keep="first")
        .drop(columns="_replicate_suffix")
        .reset_index(drop=True)
    )

def build_primary_os_cohort(
    samples: list[str],
    survival: pd.DataFrame,
    phenotype: pd.DataFrame,
) -> pd.DataFrame:
    """Select one valid-OS 01 Primary Tumor sample per patient."""
    sample_frame = pd.DataFrame({"sample": list(map(str, samples))})
    sample_frame["patient"] = sample_frame["sample"].str[:12]
    sample_frame["barcode_sample_type"] = sample_frame["sample"].str[13:15]
    primary = sample_frame.loc[
        sample_frame["barcode_sample_type"].eq("01")
    ].copy()
    if primary["patient"].duplicated().any():
        duplicates = primary.loc[
            primary["patient"].duplicated(False), "patient"
        ].unique()
        raise ValueError(
            f"Mutation matrix has multiple 01 samples for {len(duplicates)} patients"
        )

    survival_view = survival[
        ["sample", "_PATIENT", "cancer type abbreviation", "OS", "OS.time"]
    ].copy()
    phenotype_view = phenotype[
        ["sample", "sample_type_id", "sample_type", "_primary_disease"]
    ].copy()
    cohort = primary.merge(
        survival_view, on="sample", how="left", validate="one_to_one"
    ).merge(
        phenotype_view, on="sample", how="left", validate="one_to_one"
    )
    cohort["event"] = pd.to_numeric(cohort["OS"], errors="coerce")
    cohort["time"] = pd.to_numeric(cohort["OS.time"], errors="coerce")
    cohort["sample_type_id"] = (
        cohort["sample_type_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    )
    valid = (
        cohort["event"].isin([0, 1])
        & cohort["time"].gt(0)
        & cohort["sample_type_id"].eq("01")
        & cohort["_PATIENT"].eq(cohort["patient"])
    )
    cohort = cohort.loc[valid].rename(
        columns={"cancer type abbreviation": "cancer_type"}
    )
    cohort["event"] = cohort["event"].astype(bool)
    cohort["time"] = cohort["time"].astype(float)
    cohort["time_years"] = cohort["time"] / 365.25
    cohort["cancer_type"] = cohort["cancer_type"].astype(str)
    columns = [
        "sample",
        "patient",
        "cancer_type",
        "sample_type",
        "sample_type_id",
        "event",
        "time",
        "time_years",
    ]
    cohort = cohort[columns].reset_index(drop=True)
    if cohort["patient"].duplicated().any():
        raise ValueError("Valid mutation OS cohort contains duplicate patients")
    return cohort
