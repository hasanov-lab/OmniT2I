"""Reconstruction metrics used in the manuscript evaluation."""

import numpy as np


def _correlation_rows(
    truth: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    actual = actual - actual.mean(axis=1, keepdims=True)
    estimate = estimate - estimate.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(
        estimate, axis=1
    )
    return np.divide(
        np.sum(actual * estimate, axis=1),
        denominator,
        out=np.full(len(actual), np.nan),
        where=denominator > 0,
    )


def _correlation_columns(
    truth: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    actual = actual - actual.mean(axis=0, keepdims=True)
    estimate = estimate - estimate.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(actual**2, axis=0) * np.sum(estimate**2, axis=0)
    )
    return np.divide(
        np.sum(actual * estimate, axis=0),
        denominator,
        out=np.full(actual.shape[1], np.nan),
        where=denominator > 0,
    )


def cancer_center(
    truth: np.ndarray,
    prediction: np.ndarray,
    cancer_type: np.ndarray,
    minimum_group_size: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center truth and prediction separately within held-out cancer groups."""
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    labels = np.asarray(cancer_type, dtype=str)
    keep = np.zeros(len(labels), dtype=bool)
    centered_actual = np.zeros_like(actual)
    centered_estimate = np.zeros_like(estimate)
    for label in np.unique(labels):
        rows = labels == label
        if rows.sum() < minimum_group_size:
            continue
        keep |= rows
        centered_actual[rows] = actual[rows] - actual[rows].mean(
            axis=0, keepdims=True
        )
        centered_estimate[rows] = estimate[rows] - estimate[rows].mean(
            axis=0, keepdims=True
        )
    return centered_actual[keep], centered_estimate[keep], keep


def within_cancer_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    cancer_baseline: np.ndarray,
    cancer_type: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Return residual error and cancer-centered correlation summaries."""
    actual = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    baseline = np.asarray(cancer_baseline, dtype=np.float64)
    model_mse_by_feature = np.mean((estimate - actual) ** 2, axis=0)
    baseline_mse_by_feature = np.mean((baseline - actual) ** 2, axis=0)
    residual_r2 = np.divide(
        baseline_mse_by_feature - model_mse_by_feature,
        baseline_mse_by_feature,
        out=np.full(actual.shape[1], np.nan),
        where=baseline_mse_by_feature > 0,
    )

    centered_actual, centered_estimate, keep = cancer_center(
        actual, estimate, cancer_type
    )
    feature_correlation = _correlation_columns(
        centered_actual, centered_estimate
    )
    finite_feature = feature_correlation[np.isfinite(feature_correlation)]
    patient_correlation = _correlation_rows(
        actual - baseline, estimate - baseline
    )
    model_mse = float(np.mean(model_mse_by_feature))
    baseline_mse = float(np.mean(baseline_mse_by_feature))
    metrics: dict[str, float | int] = {
        "n_patients": int(len(actual)),
        "n_patients_in_correlation": int(keep.sum()),
        "n_cancer_types": int(len(np.unique(cancer_type))),
        "model_mse": model_mse,
        "cancer_baseline_mse": baseline_mse,
        "residual_mse_improvement": (
            float(1.0 - model_mse / baseline_mse)
            if baseline_mse > 0
            else float("nan")
        ),
        "within_cancer_patient_pearson_mean": float(
            np.nanmean(patient_correlation)
        ),
        "within_cancer_feature_pearson_mean": (
            float(np.mean(finite_feature))
            if len(finite_feature)
            else float("nan")
        ),
        "within_cancer_feature_pearson_median": (
            float(np.median(finite_feature))
            if len(finite_feature)
            else float("nan")
        ),
        "positive_within_feature_pearson_fraction": (
            float(np.mean(finite_feature > 0))
            if len(finite_feature)
            else float("nan")
        ),
        "positive_residual_r2_fraction": (
            float(np.mean(residual_r2[np.isfinite(residual_r2)] > 0))
            if np.isfinite(residual_r2).any()
            else float("nan")
        ),
        "residual_r2_mean": float(np.nanmean(residual_r2)),
        "residual_r2_median": float(np.nanmedian(residual_r2)),
    }
    return metrics, feature_correlation, residual_r2


def _profile_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    feature = _correlation_columns(truth, prediction)
    patient = _correlation_rows(truth, prediction)
    return {
        "feature_pearson_mean": float(np.nanmean(feature)),
        "feature_pearson_median": float(np.nanmedian(feature)),
        "patient_pearson_mean": float(np.nanmean(patient)),
    }


def _feature_metric_row(
    truth: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    cancers: np.ndarray,
) -> dict[str, float | int]:
    profile = _profile_metrics(truth, prediction)
    within, _, _ = within_cancer_metrics(truth, prediction, baseline, cancers)
    return {
        **profile,
        **within,
        "model_squared_error_sum": float(np.square(prediction - truth).sum()),
        "cancer_baseline_squared_error_sum": float(
            np.square(baseline - truth).sum()
        ),
        "n_feature_values": int(truth.size),
    }
