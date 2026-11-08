"""Missing-modality fusion with one projection shared across modalities."""

from __future__ import annotations

import torch
from torch import nn

from .masked_gated_fusion import MaskedGatedFusion


def paired_alignment_mse(
    projected: list[torch.Tensor],
    availability: torch.Tensor,
) -> torch.Tensor:
    """Sum IntegrAO-style MSE over naturally paired modality projections."""
    if len(projected) < 2:
        raise ValueError("Alignment requires at least two modality projections")
    if availability.ndim != 2 or availability.shape[1] != len(projected):
        raise ValueError("Availability must have shape [patients, modalities]")
    if any(z.ndim != 2 for z in projected):
        raise ValueError("Each projected embedding must have shape [patients, latent]")
    if any(z.shape != projected[0].shape for z in projected[1:]):
        raise ValueError("All projected embeddings must have identical shapes")
    if projected[0].shape[0] != availability.shape[0]:
        raise ValueError("Projection and availability patient counts differ")

    availability = availability.bool()
    zero = sum(z.sum() for z in projected) * 0.0
    pair_losses = []
    for left in range(len(projected)):
        for right in range(left + 1, len(projected)):
            paired = availability[:, left] & availability[:, right]
            if paired.any():
                pair_losses.append(
                    torch.mean((projected[left][paired] - projected[right][paired]) ** 2)
                )
    # Average rather than sum so the alignment scale does not grow merely
    # because a third (or later) modality adds more available pairs. For the
    # established two-modality model this is numerically unchanged.
    return torch.stack(pair_losses).mean() if pair_losses else zero


class SharedProjectionFusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 96,
        latent_dim: int = 96,
        n_modalities: int = 2,
        fusion_mode: str = "mean",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"mean", "gated"}:
            raise ValueError("fusion_mode must be 'mean' or 'gated'")
        self.n_modalities = n_modalities
        self.fusion_mode = fusion_mode
        self.shared_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.gates = (
            nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(n_modalities)])
            if fusion_mode == "gated"
            else None
        )
        # Keep auxiliary heads modality-specific in all configurations so the
        # projection and fusion rules are the only experimental differences.
        self.aux_heads = nn.ModuleList(
            [nn.Linear(latent_dim, 1) for _ in range(n_modalities)]
        )
        self.fused_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    def forward(
        self,
        embeddings: list[torch.Tensor],
        availability: torch.Tensor,
        modality_dropout_probability: float = 0.0,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if len(embeddings) != self.n_modalities:
            raise ValueError("Embedding count does not match configured modalities")
        if availability.ndim != 2 or availability.shape[1] != self.n_modalities:
            raise ValueError("Availability must have shape [patients, modalities]")
        availability = availability.bool()
        if (~availability.any(dim=1)).any():
            raise ValueError("Every patient must have at least one available modality")
        effective = (
            MaskedGatedFusion.modality_dropout(
                availability, modality_dropout_probability
            )
            if self.training
            else availability
        )
        projected = [self.shared_projection(x) for x in embeddings]
        if self.fusion_mode == "mean":
            weights = effective.to(projected[0].dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
        else:
            assert self.gates is not None
            gate_logits = torch.cat(
                [gate(z) for gate, z in zip(self.gates, projected)], dim=1
            )
            gate_logits = gate_logits.masked_fill(
                ~effective, torch.finfo(gate_logits.dtype).min
            )
            weights = torch.softmax(gate_logits, dim=1)
        stacked = torch.stack(projected, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return {
            "risk": self.fused_head(fused).flatten(),
            "weights": weights,
            "fused": fused,
            "projected": projected,
            "aux_risks": [
                head(z).flatten() for head, z in zip(self.aux_heads, projected)
            ],
            "effective_availability": effective,
        }
