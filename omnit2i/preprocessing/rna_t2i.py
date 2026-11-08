"""rna t2i preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def select_invariant_genes(
    expr: pd.DataFrame,
    train_samples: list[str],
    n_bins: int = 3,
    per_bin: int = 50,
) -> pd.DataFrame:
    train = expr.loc[:, train_samples]
    mean = train.mean(axis=1)
    variance = train.var(axis=1, ddof=0)
    bins = pd.qcut(mean.rank(method="first"), q=n_bins, labels=False)
    rows: list[pd.DataFrame] = []
    for bin_id in range(n_bins):
        mask = np.asarray(bins == bin_id)
        subset = pd.DataFrame(
            {
                "gene": expr.index.to_numpy()[mask],
                "mean": mean.to_numpy()[mask],
                "variance": variance.to_numpy()[mask],
            }
        ).sort_values(["variance", "gene"], kind="stable").head(per_bin)
        subset["expression_bin"] = bin_id
        subset["rank_in_bin"] = np.arange(1, len(subset) + 1)
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)[
        ["expression_bin", "rank_in_bin", "gene", "mean", "variance"]
    ]


def sample_gamma(expr: pd.DataFrame, samples: list[str], genes: list[str]) -> pd.Series:
    present = [gene for gene in genes if gene in expr.index]
    if not present:
        raise ValueError("No invariant genes are present in RNA expression")
    return expr.loc[present, samples].median(axis=0)


def fit_cancer_standardization(
    x_train: np.ndarray, cancer: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = cancer.astype(str).to_numpy()
    names = np.asarray(sorted(np.unique(labels)), dtype=str)
    global_mean = x_train.mean(axis=0).astype(np.float32)
    global_std = x_train.std(axis=0).astype(np.float32)
    global_std[global_std < 1e-6] = 1.0
    means, stds = [], []
    for name in names:
        values = x_train[labels == name]
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        means.append(mean)
        stds.append(std)
    return names, np.stack(means), np.stack(stds), global_mean, global_std


def apply_cancer_standardization(
    x: np.ndarray,
    cancer: pd.Series,
    labels: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> np.ndarray:
    lookup = {str(label): i for i, label in enumerate(labels.astype(str))}
    output = np.empty_like(x, dtype=np.float32)
    observed = cancer.astype(str).to_numpy()
    for name in np.unique(observed):
        mask = observed == name
        if name in lookup:
            idx = lookup[name]
            mean, std = means[idx], stds[idx]
        else:
            mean, std = global_mean, global_std
        output[mask] = (x[mask] - mean) / std
    return output


def cox_score_z(x_train: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    from .cox_score_corrected import cox_score_z as corrected_score
    return corrected_score(x_train, time, event)


def corr_hclust_order(x_train: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(x_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    distance = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    return leaves_list(
        linkage(squareform(distance, checks=False), method="average")
    ).astype(np.int64)


def render_option2(
    x_gene: np.ndarray,
    x_cancer: np.ndarray,
    score_z: np.ndarray,
    order: np.ndarray,
    height: int = 45,
    width: int = 45,
) -> np.ndarray:
    """Place ordered features along the same serpentine path as other molecular inputs."""
    from .cnv_t2i import snake_pixel_indices
    n_pixels = height * width
    if len(order) > n_pixels:
        raise ValueError("RNA feature count exceeds image capacity")
    channels = [x_gene, x_cancer, x_gene * score_z.reshape(1, -1)]
    rendered = []
    for values in channels:
        image = np.zeros((values.shape[0], n_pixels), dtype=np.float32)
        image[:, snake_pixel_indices(height, width)[:len(order)]] = values[:, order]
        rendered.append(image.reshape(values.shape[0], 1, height, width))
    output = np.concatenate(rendered, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite RNA image values detected")
    return output
