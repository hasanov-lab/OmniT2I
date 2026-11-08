"""Production losses helpers."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from sksurv.metrics import concordance_index_censored
from omnit2i.models import paired_alignment_mse
from .rna import cox_partial_likelihood


def _cox_if_available(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if int(event[mask].sum().item()) == 0:
        return risk.sum() * 0.0
    return cox_partial_likelihood(risk[mask], time[mask], event[mask])


def _survival_loss(
    result: dict[str, Any],
    time: torch.Tensor,
    event: torch.Tensor,
    availability: torch.Tensor,
    aux_weight: float,
    alignment_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fusion = cox_partial_likelihood(result["risk"], time, event)
    auxiliary = torch.stack(
        [
            _cox_if_available(risk, time, event, availability[:, index])
            for index, risk in enumerate(result["aux_risks"])
        ]
    ).mean()
    alignment = paired_alignment_mse(result["projected"], availability)
    total = fusion + float(aux_weight) * auxiliary + float(alignment_weight) * alignment
    return total, {"fusion_cox": fusion, "aux_cox": auxiliary, "alignment": alignment}


def fit_risk_calibration(
    risk: np.ndarray,
    reference_mask: np.ndarray,
) -> dict[str, float | int]:
    """Fit an outcome-free affine risk calibration on inner validation rows."""
    values = np.asarray(risk, dtype=np.float64)
    reference = np.asarray(reference_mask, dtype=bool)
    if values.shape != reference.shape:
        raise ValueError("risk and reference_mask shapes differ")
    selected = values[reference]
    if len(selected) < 2 or not np.isfinite(selected).all():
        raise ValueError("Risk calibration requires at least two finite references")
    mean = float(selected.mean())
    std = float(selected.std(ddof=0))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("Risk calibration reference has near-zero variance")
    return {"mean": mean, "std": std, "n_reference": int(len(selected))}


def apply_risk_calibration(
    risk: np.ndarray,
    calibration: dict[str, float | int],
) -> np.ndarray:
    return (
        (np.asarray(risk, dtype=np.float64) - float(calibration["mean"]))
        / float(calibration["std"])
    ).astype(np.float32)


def c_index(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(np.asarray(event, dtype=bool), np.asarray(time, dtype=float), np.asarray(risk, dtype=float))[0])
