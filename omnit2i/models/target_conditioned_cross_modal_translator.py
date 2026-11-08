"""Target-conditioned cross-modal translation for five-modal T2I models.

The translator consumes the differentiable 96-D embeddings produced by the
modality-specific encoders.  Each directed source-to-target route owns a small
mapper, while a target-specific masked gate combines only genuinely available
non-target sources.  A target decoder then maps the predicted target latent to
the corresponding measured feature space.

The module deliberately contains no optimizer or training loop.  It can
therefore be attached either to frozen encoders (the modular fallback) or to
trainable encoders (the end-to-end model) without duplicating architecture
code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .constants import MODALITIES


class LowRankLinear(nn.Module):
    """Bias-free low-rank linear map used as a route's linear skip branch."""

    def __init__(self, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        if min(int(input_dim), int(output_dim), int(rank)) < 1:
            raise ValueError("input_dim, output_dim, and rank must be positive")
        if int(rank) > min(int(input_dim), int(output_dim)):
            raise ValueError("rank cannot exceed input_dim or output_dim")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.down = nn.Linear(self.input_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.output_dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(values))


class ResidualMapper(nn.Module):
    """Lightweight route-specific mapper from one modality to another.

    The low-rank branch preserves a direct linear route analogous to the
    strong embedding-Ridge baseline.  The nonlinear branch learns the
    remaining source-to-target residual.
    """

    def __init__(
        self,
        latent_dim: int = 96,
        hidden_dim: int = 192,
        linear_rank: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.latent_dim = int(latent_dim)
        self.linear_skip = LowRankLinear(
            self.latent_dim, self.latent_dim, int(linear_rank)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.latent_dim:
            raise ValueError("values must have shape [patients, latent_dim]")
        return self.linear_skip(values) + self.residual(values)


class TargetFeatureDecoder(nn.Module):
    """Decode a predicted target latent into one measured feature space."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(int(latent_dim), int(hidden_dim), int(output_dim)) < 1:
            raise ValueError("Decoder dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent must have shape [patients, latent_dim]")
        return self.network(latent)


class TargetConditionedCrossModalTranslator(nn.Module):
    """Predict any one target modality from available non-target modalities.

    Parameters
    ----------
    output_dims:
        Either five feature counts in ``MODALITIES`` order or a mapping with
        exactly the five modality names.  Feature counts refer to the measured
        training targets, such as selected genes, RPPA features, or the H&E
        patient embedding dimensions.

    Notes
    -----
    The target is removed from the effective source mask inside the module,
    even when the caller passes natural availability that marks it observed.
    Thus the cross-modal prediction cannot take an identity shortcut.
    """

    modalities = MODALITIES

    def __init__(
        self,
        output_dims: Mapping[str, int] | Sequence[int],
        latent_dim: int = 96,
        mapper_hidden_dim: int = 192,
        decoder_hidden_dim: int = 384,
        linear_rank: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.n_modalities = len(self.modalities)
        self.latent_dim = int(latent_dim)
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        self.output_dims = self._normalize_output_dims(output_dims)

        self.route_mappers = nn.ModuleDict()
        self.route_gates = nn.ModuleDict()
        for target in self.modalities:
            for source in self.modalities:
                if source == target:
                    continue
                key = self.route_key(source, target)
                self.route_mappers[key] = ResidualMapper(
                    latent_dim=self.latent_dim,
                    hidden_dim=int(mapper_hidden_dim),
                    linear_rank=int(linear_rank),
                    dropout=float(dropout),
                )
                self.route_gates[key] = nn.Sequential(
                    nn.LayerNorm(self.latent_dim),
                    nn.Linear(self.latent_dim, 1),
                )

        self.feature_decoders = nn.ModuleDict(
            {
                target: TargetFeatureDecoder(
                    latent_dim=self.latent_dim,
                    hidden_dim=int(decoder_hidden_dim),
                    output_dim=self.output_dims[target],
                    dropout=float(dropout),
                )
                for target in self.modalities
            }
        )

    @classmethod
    def _normalize_output_dims(
        cls, output_dims: Mapping[str, int] | Sequence[int]
    ) -> dict[str, int]:
        if isinstance(output_dims, Mapping):
            missing = set(cls.modalities) - set(output_dims)
            extra = set(output_dims) - set(cls.modalities)
            if missing or extra:
                raise ValueError(
                    "output_dims mapping must contain exactly: "
                    + ", ".join(cls.modalities)
                )
            normalized = {
                modality: int(output_dims[modality]) for modality in cls.modalities
            }
        else:
            values = tuple(int(value) for value in output_dims)
            if len(values) != len(cls.modalities):
                raise ValueError("output_dims must contain one value per modality")
            normalized = dict(zip(cls.modalities, values, strict=True))
        if any(value < 1 for value in normalized.values()):
            raise ValueError("Every target output dimension must be positive")
        return normalized

    @classmethod
    def resolve_target(cls, target: int | str) -> tuple[int, str]:
        if isinstance(target, bool):
            raise ValueError("target must be a modality name or integer index")
        if isinstance(target, str):
            if target not in cls.modalities:
                raise ValueError(f"Unknown target modality: {target}")
            return cls.modalities.index(target), target
        if isinstance(target, int) and 0 <= target < len(cls.modalities):
            return int(target), cls.modalities[int(target)]
        raise ValueError("target must be a valid modality name or integer index")

    @staticmethod
    def route_key(source: str, target: str) -> str:
        return f"{source}_to_{target}"

    def _validate_inputs(
        self,
        embeddings: torch.Tensor,
        source_mask: torch.Tensor,
        target_index: int,
    ) -> torch.Tensor:
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("embeddings must be a torch.Tensor")
        if not isinstance(source_mask, torch.Tensor):
            raise TypeError("source_mask must be a torch.Tensor")
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape [patients, modalities, latent_dim]"
            )
        if embeddings.shape[0] == 0:
            raise ValueError("embeddings must contain at least one patient")
        if embeddings.shape[1:] != (self.n_modalities, self.latent_dim):
            raise ValueError("Unexpected modality count or latent dimension")
        if not torch.is_floating_point(embeddings):
            raise ValueError("embeddings must be floating point")
        if source_mask.shape != embeddings.shape[:2]:
            raise ValueError("source_mask shape does not match embeddings")
        if source_mask.is_complex():
            raise ValueError("source_mask must contain only binary 0/1 values")

        if source_mask.dtype == torch.bool:
            available = source_mask.to(device=embeddings.device).clone()
        else:
            mask_on_device = source_mask.to(device=embeddings.device)
            if torch.is_floating_point(mask_on_device) and not torch.isfinite(
                mask_on_device
            ).all():
                raise ValueError("source_mask contains non-finite values")
            binary = (mask_on_device == 0) | (mask_on_device == 1)
            if not binary.all():
                raise ValueError("source_mask must contain only binary 0/1 values")
            available = mask_on_device.bool()

        # Target exclusion is unconditional: callers may pass natural
        # availability, but the translation branch never sees the target.
        available[:, target_index] = False
        if (~available.any(dim=1)).any():
            raise ValueError("Every row needs at least one available non-target source")

        finite_by_modality = torch.isfinite(embeddings).all(dim=-1)
        if (~finite_by_modality & available).any():
            raise ValueError("An available source embedding contains non-finite values")
        return available

    def predict_latent(
        self,
        embeddings: torch.Tensor,
        source_mask: torch.Tensor,
        target: int | str,
    ) -> dict[str, torch.Tensor]:
        """Return target latent, masked route weights, and route candidates."""
        target_index, target_name = self.resolve_target(target)
        available = self._validate_inputs(embeddings, source_mask, target_index)

        candidates: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        negative_infinity = torch.finfo(embeddings.dtype).min
        for source_index, source_name in enumerate(self.modalities):
            if source_index == target_index:
                candidates.append(
                    embeddings.new_zeros((embeddings.shape[0], self.latent_dim))
                )
                logits.append(
                    embeddings.new_full((embeddings.shape[0], 1), negative_infinity)
                )
                continue

            # Missing slots may contain NaNs in exported arrays.  Replace them
            # before running a mapper; available non-finite values were rejected
            # above, so this cannot hide corrupt observed data.
            source_is_available = available[:, source_index].unsqueeze(-1)
            safe_source = torch.where(
                source_is_available,
                embeddings[:, source_index],
                torch.zeros_like(embeddings[:, source_index]),
            )
            key = self.route_key(source_name, target_name)
            candidate = self.route_mappers[key](safe_source)
            candidates.append(candidate)
            logits.append(self.route_gates[key](candidate))

        route_candidates = torch.stack(candidates, dim=1)
        gate_logits = torch.cat(logits, dim=1)
        gate_logits = gate_logits.masked_fill(~available, negative_infinity)
        weights = torch.softmax(gate_logits, dim=1)
        latent_prediction = (weights.unsqueeze(-1) * route_candidates).sum(dim=1)
        return {
            "latent_prediction": latent_prediction,
            "weights": weights,
            "route_candidates": route_candidates,
            "effective_source_mask": available,
        }

    def decode_features(
        self, latent_prediction: torch.Tensor, target: int | str
    ) -> torch.Tensor:
        """Decode a predicted target latent into measured target features."""
        if not isinstance(latent_prediction, torch.Tensor):
            raise TypeError("latent_prediction must be a torch.Tensor")
        _, target_name = self.resolve_target(target)
        return self.feature_decoders[target_name](latent_prediction)

    def forward(
        self,
        embeddings: torch.Tensor,
        source_mask: torch.Tensor,
        target: int | str,
    ) -> dict[str, torch.Tensor]:
        translated = self.predict_latent(embeddings, source_mask, target)
        translated["feature_prediction"] = self.decode_features(
            translated["latent_prediction"], target
        )
        return translated
