"""Original comparable-pair bootstrap kernels for saved OOF predictions."""
import numpy as np
from numba import njit, prange

def comparable_pairs(time, event, cancer):
    left, right = [], []
    for label in np.unique(cancer):
        ids = np.flatnonzero(cancer == label)
        t, e = time[ids], event[ids]
        eligible = e[:, None] & ((t[:, None] < t[None, :]) |
                                 ((t[:, None] == t[None, :]) & ~e[None, :]))
        i, j = np.nonzero(eligible)
        left.append(ids[i])
        right.append(ids[j])
    return np.concatenate(left), np.concatenate(right)

def pair_scores(risks, left, right):
    # Shared denominators make averaging seed-wise concordances exactly linear.
    delta = risks[:, :, left] - risks[:, :, right]
    return np.ascontiguousarray(((delta > 1e-8) +
        0.5 * (np.abs(delta) <= 1e-8)).mean(axis=1).T)

@njit(parallel=True)
def weighted_c(weights, left, right, scores):
    out = np.empty((len(weights), scores.shape[1]))
    for b in prange(len(weights)):
        denominator = 0.0
        totals = np.zeros(scores.shape[1])
        for k in range(len(left)):
            weight = float(weights[b, left[k]]) * float(weights[b, right[k]])
            denominator += weight
            for m in range(scores.shape[1]):
                totals[m] += weight * scores[k, m]
        for m in range(scores.shape[1]):
            out[b, m] = totals[m] / denominator if denominator else np.nan
    return out
