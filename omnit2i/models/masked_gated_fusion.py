"""Symmetric missing-modality fusion for modality-specific embeddings."""

from __future__ import annotations

import torch
from torch import nn


class MaskedGatedFusion(nn.Module):
    def __init__(
        self,
        input_dims: list[int],
        latent_dim: int = 96,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                    nn.GELU(),
                )
                for dim in input_dims
            ]
        )
        self.gates = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in input_dims])
        self.aux_heads = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in input_dims])
        self.fused_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),
        )

    @staticmethod
    def modality_dropout(mask: torch.Tensor, probability: float) -> torch.Tensor:
        if probability <= 0:
            return mask
        keep = mask & (torch.rand_like(mask, dtype=torch.float32) >= probability)
        lost_all = mask.any(dim=1) & ~keep.any(dim=1)
        if lost_all.any():
            rows = torch.nonzero(lost_all, as_tuple=False).flatten()
            for row in rows:
                available = torch.nonzero(mask[row], as_tuple=False).flatten()
                choice = available[torch.randint(len(available), (1,), device=mask.device)]
                keep[row, choice] = True
        return keep

    def forward(
        self,
        embeddings: list[torch.Tensor],
        availability: torch.Tensor,
        modality_dropout_probability: float = 0.0,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if len(embeddings) != len(self.projections):
            raise ValueError("Embedding count does not match configured modalities")
        if availability.ndim != 2 or availability.shape[1] != len(embeddings):
            raise ValueError("Availability must have shape [patients, modalities]")
        availability = availability.bool()
        if (~availability.any(dim=1)).any():
            raise ValueError("Every patient must have at least one available modality")
        effective = (
            self.modality_dropout(availability, modality_dropout_probability)
            if self.training
            else availability
        )
        projected = [projection(x) for projection, x in zip(self.projections, embeddings)]
        gate_logits = torch.cat(
            [gate(z) for gate, z in zip(self.gates, projected)], dim=1
        )
        gate_logits = gate_logits.masked_fill(~effective, torch.finfo(gate_logits.dtype).min)
        weights = torch.softmax(gate_logits, dim=1)
        stacked = torch.stack(projected, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return {
            "risk": self.fused_head(fused).flatten(),
            "weights": weights,
            "fused": fused,
            "projected": projected,
            "aux_risks": [head(z).flatten() for head, z in zip(self.aux_heads, projected)],
            "effective_availability": effective,
        }
