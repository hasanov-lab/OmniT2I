"""Patient-level input validation for the public workflows."""

import numpy as np
import pandas as pd


def validate_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    required = {"patient", "sample", "cancer_type", "time", "event", "partition"}
    if required - set(cohort.columns):
        raise ValueError(f"Cohort is missing columns: {sorted(required - set(cohort.columns))}")
    cohort = cohort.copy()
    for column in ("patient", "sample", "cancer_type", "partition"):
        if cohort[column].isna().any() or cohort[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Cohort {column} must not contain missing or empty values")
        cohort[column] = cohort[column].astype(str)
    for column in ("patient", "sample"):
        if cohort[column].duplicated().any():
            raise ValueError(f"Cohort must have one unique {column} per row")
    if set(cohort["partition"]) != {"core", "validation", "test"}:
        raise ValueError("partition must contain core, validation, and test, with no other labels")
    for column in ("time", "event"):
        cohort[column] = pd.to_numeric(cohort[column], errors="raise")
        if not np.isfinite(cohort[column].to_numpy(float)).all():
            raise ValueError(f"Cohort {column} must be finite")
    if (cohort["time"] <= 0).any() or not cohort["event"].isin([0, 1]).all():
        raise ValueError("time must be positive and event must be 0 or 1")
    if cohort["partition"].eq("core").sum() < 2:
        raise ValueError("At least two core patients are required")
    return cohort
