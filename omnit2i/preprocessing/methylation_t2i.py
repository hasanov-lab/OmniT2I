"""methylation t2i preprocessing utilities."""

from __future__ import annotations

import numpy as np
from .cnv_t2i import snake_pixel_indices


def beta_to_m_value(beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Convert beta values to log2 methylated/unmethylated odds."""
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie between zero and 0.5")
    values = np.asarray(beta, dtype=np.float32)
    finite = np.isfinite(values)
    if ((values[finite] < 0.0) | (values[finite] > 1.0)).any():
        raise ValueError("Finite beta values must lie in [0, 1]")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    output = np.log2(clipped / (1.0 - clipped)).astype(np.float32)
    output[~finite] = np.nan
    return output


def fit_feature_filter(
    x_train_m: np.ndarray,
    eligible: np.ndarray,
    max_missing_fraction: float,
    min_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(x_train_m, dtype=np.float32)
    eligible = np.asarray(eligible, dtype=bool)
    if values.ndim != 2 or eligible.shape != (values.shape[1],):
        raise ValueError("Feature-filter inputs have incompatible shapes")
    missing_fraction = np.isnan(values).mean(axis=0)
    medians = np.nanmedian(values, axis=0)
    imputed = np.where(np.isnan(values), medians, values)
    variance = np.nanvar(imputed, axis=0)
    keep = (
        eligible
        & (missing_fraction <= max_missing_fraction)
        & np.isfinite(medians)
        & np.isfinite(variance)
        & (variance > min_variance)
    )
    return keep, missing_fraction.astype(np.float32), variance.astype(np.float32)


def select_top_variance(
    feature_ids: np.ndarray,
    keep: np.ndarray,
    variance: np.ndarray,
    top_n: int,
) -> np.ndarray:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    candidates = np.flatnonzero(np.asarray(keep, dtype=bool))
    if len(candidates) < top_n:
        raise ValueError(
            f"Only {len(candidates)} methylation genes pass QC; requested {top_n}"
        )
    identifiers = np.asarray(feature_ids, dtype=str)
    # np.lexsort uses the final key as primary: descending variance, then ID.
    order = np.lexsort((identifiers[candidates], -variance[candidates]))
    return candidates[order[:top_n]]


def fit_impute_standardize(
    x_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(x_train, dtype=np.float32)
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    if not np.isfinite(medians).all():
        raise ValueError("A retained methylation gene has no training observations")
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


def render_option2(
    standardized: np.ndarray,
    cancer_centered: np.ndarray,
    cox_scores: np.ndarray,
    order: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    gene = np.asarray(standardized, dtype=np.float32)
    cancer = np.asarray(cancer_centered, dtype=np.float32)
    scores = np.asarray(cox_scores, dtype=np.float32)
    order = np.asarray(order, dtype=np.int64)
    if gene.ndim != 2 or cancer.shape != gene.shape:
        raise ValueError("Methylation channels must have samples x features shape")
    if scores.shape != (gene.shape[1],):
        raise ValueError("Cox scores have incompatible shape")
    if len(order) != gene.shape[1] or set(order.tolist()) != set(range(len(order))):
        raise ValueError("Layout order must permute every selected feature")
    if height * width < len(order):
        raise ValueError("Image canvas cannot contain selected methylation genes")
    pixels = snake_pixel_indices(height, width)[: len(order)]
    channels = [gene, cancer, gene * scores.reshape(1, -1)]
    images = []
    for channel in channels:
        canvas = np.zeros((gene.shape[0], height * width), dtype=np.float32)
        canvas[:, pixels] = channel[:, order]
        images.append(canvas.reshape(gene.shape[0], 1, height, width))
    output = np.concatenate(images, axis=1)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite methylation image values detected")
    return output
