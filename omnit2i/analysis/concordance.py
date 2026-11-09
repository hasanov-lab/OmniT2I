"""Original comparable-pair decomposition of pooled and within-cancer C-index."""

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored


def c_index_parts(frame: pd.DataFrame) -> tuple[float, np.ndarray]:
    result = concordance_index_censored(
        frame["event"].astype(bool).to_numpy(),
        frame["time"].astype(float).to_numpy(),
        frame["risk_score"].astype(float).to_numpy(),
    )
    return float(result[0]), np.asarray(result[1:], dtype=np.int64)


def c_index_from_counts(counts: np.ndarray) -> float:
    denominator = int(counts[:3].sum())
    if denominator == 0:
        return float("nan")
    return float((counts[0] + 0.5 * counts[2]) / denominator)


def fold_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    pooled_c, total = c_index_parts(frame)
    within = np.zeros(4, dtype=np.int64)
    for _, cancer_frame in frame.groupby("cancer_type", sort=True):
        try:
            _, counts = c_index_parts(cancer_frame)
        except ValueError:
            continue
        within += counts
    cross = total - within
    within_pairs = int(within[:3].sum())
    cross_pairs = int(cross[:3].sum())
    all_pairs = within_pairs + cross_pairs
    return {
        "pooled_c_index": pooled_c,
        "within_cancer_c_index": c_index_from_counts(within),
        "cross_cancer_c_index": c_index_from_counts(cross),
        "within_cancer_comparable_pairs": within_pairs,
        "cross_cancer_comparable_pairs": cross_pairs,
        "within_cancer_pair_fraction": (
            within_pairs / all_pairs if all_pairs else float("nan")
        ),
    }
