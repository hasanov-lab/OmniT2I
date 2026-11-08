"""cnv t2i preprocessing utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class AsymmetricBounds:
    lower: float
    upper: float
    lower_quantile: float
    upper_quantile: float


def fit_asymmetric_bounds(
    x_train: np.ndarray,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> AsymmetricBounds:
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Quantiles must satisfy 0 < lower < upper < 1")
    finite = np.asarray(x_train, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Training matrix has no finite CNV values")
    lower, upper = np.quantile(finite, [lower_quantile, upper_quantile])
    if lower >= 0 or upper <= 0:
        raise ValueError(
            f"Zero-anchored scaling requires a negative lower and positive upper bound; got {lower}, {upper}"
        )
    return AsymmetricBounds(float(lower), float(upper), lower_quantile, upper_quantile)


def transform_zero_anchored(x: np.ndarray, bounds: AsymmetricBounds) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    negative = values < 0
    positive = values > 0
    out[negative] = values[negative] / abs(bounds.lower)
    out[positive] = values[positive] / bounds.upper
    out = np.clip(out, -1.0, 1.0)
    out[~np.isfinite(values)] = np.nan
    return out.astype(np.float32)


def fit_gene_standardization(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(x_train, axis=0).astype(np.float32)
    stds = np.nanstd(x_train, axis=0, ddof=0).astype(np.float32)
    stds[~np.isfinite(stds) | (stds < 1e-6)] = 1.0
    return means, stds


def transform_gene_standardization(
    x: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - means) / stds).astype(np.float32)


def cox_score_z(x_train: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    from .cox_score_corrected import cox_score_z as corrected_score
    return corrected_score(x_train, time, event)


def corr_hclust_snake_order(x_train: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(np.asarray(x_train, dtype=np.float32), rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    return leaves_list(tree).astype(np.int64)


def snake_pixel_indices(height: int, width: int) -> np.ndarray:
    indices: list[int] = []
    for row in range(height):
        columns = range(width) if row % 2 == 0 else range(width - 1, -1, -1)
        indices.extend(row * width + column for column in columns)
    return np.asarray(indices, dtype=np.int64)


def fit_cancer_standardization(
    x_train_z: np.ndarray,
    cancer_types: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit fold-local within-cancer centering/scaling on standardized CNV."""
    values = np.asarray(x_train_z, dtype=np.float32)
    labels = np.asarray(list(map(str, cancer_types)), dtype=str)
    unique_labels = np.asarray(sorted(np.unique(labels)), dtype=str)
    global_mean = np.nanmean(values, axis=0).astype(np.float32)
    global_std = np.nanstd(values, axis=0, ddof=0).astype(np.float32)
    global_std[~np.isfinite(global_std) | (global_std < 1e-6)] = 1.0
    means, stds = [], []
    for label in unique_labels:
        subset = values[labels == label]
        mean = np.nanmean(subset, axis=0).astype(np.float32)
        std = np.nanstd(subset, axis=0, ddof=0).astype(np.float32)
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        means.append(mean)
        stds.append(std)
    return (
        unique_labels,
        np.vstack(means).astype(np.float32),
        np.vstack(stds).astype(np.float32),
        global_mean,
        global_std,
    )


def apply_cancer_standardization(
    x_z: np.ndarray,
    cancer_types: Iterable[str],
    labels: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> np.ndarray:
    """Apply fold-local within-cancer z-scoring with a global fallback."""
    values = np.asarray(x_z, dtype=np.float32)
    observed = np.asarray(list(map(str, cancer_types)), dtype=str)
    lookup = {str(label): index for index, label in enumerate(labels.astype(str))}
    output = np.empty_like(values, dtype=np.float32)
    for label in np.unique(observed):
        mask = observed == label
        if label in lookup:
            index = lookup[label]
            mean, std = means[index], stds[index]
        else:
            mean, std = global_mean, global_std
        output[mask] = (values[mask] - mean) / std
    return output.astype(np.float32)
