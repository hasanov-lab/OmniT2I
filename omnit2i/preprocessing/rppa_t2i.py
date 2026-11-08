"""rppa t2i preprocessing utilities."""

from __future__ import annotations

import numpy as np


def fit_feature_filter(
    x_train: np.ndarray,
    max_missing_fraction: float = 0.20,
    min_variance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(x_train, dtype=np.float32)
    missing_fraction = np.isnan(values).mean(axis=0)
    medians = np.nanmedian(values, axis=0)
    imputed = np.where(np.isnan(values), medians, values)
    variance = np.nanvar(imputed, axis=0)
    keep = (
        (missing_fraction <= max_missing_fraction)
        & np.isfinite(medians)
        & np.isfinite(variance)
        & (variance > min_variance)
    )
    if not keep.any():
        raise ValueError("RPPA feature filter removed every feature")
    return keep, missing_fraction.astype(np.float32), variance.astype(np.float32)


def fit_impute_standardize(
    x_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(x_train, dtype=np.float32)
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    if not np.isfinite(medians).all():
        raise ValueError("At least one retained RPPA feature has no training observations")
    imputed = np.where(np.isnan(values), medians, values).astype(np.float32)
    means = imputed.mean(axis=0).astype(np.float32)
    stds = imputed.std(axis=0, ddof=0).astype(np.float32)
    stds[~np.isfinite(stds) | (stds < 1e-6)] = 1.0
    return medians, means, stds


def transform_impute_standardize(
    x: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    imputed = np.where(np.isnan(values), medians, values)
    return ((imputed - means) / stds).astype(np.float32)
