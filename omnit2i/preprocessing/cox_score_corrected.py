"""cox score corrected preprocessing utilities."""

from __future__ import annotations

import numpy as np


def cox_score_z(x_train: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    order = np.argsort(-time.astype(np.float64))
    x_sorted = x_train[order].astype(np.float64, copy=False)
    time_sorted = time[order].astype(np.float64)
    event_sorted = event[order].astype(bool)
    cumulative = np.cumsum(x_sorted, axis=0)
    # Boundaries must follow the descending rows, including every tied risk-set member.
    group_ends = np.r_[np.flatnonzero(np.diff(time_sorted) != 0) + 1, len(time_sorted)]
    score = np.zeros(x_train.shape[1], dtype=np.float64)
    start = 0
    for stop in group_ends:
        event_mask = event_sorted[start:stop]
        n_events = int(event_mask.sum())
        if n_events:
            score += (
                x_sorted[start:stop][event_mask].sum(axis=0)
                - n_events * cumulative[stop - 1] / stop
            )
        start = stop
    scale = score.std()
    score = (score - score.mean()) / (scale if scale > 1e-8 else 1.0)
    return np.clip(score, -3.0, 3.0).astype(np.float32)
