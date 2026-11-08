"""Five-modality survival backbone and cross-modal model container."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .constants import MODALITIES
from .encoders import (
    CNVImageEncoder, CNNSpatialProjectorEncoder, CNNSpatialGlobalEncoder,
    HEEmbeddingAdapter,
)
from .shared_projection_fusion import SharedProjectionFusion
from .target_conditioned_cross_modal_translator import TargetConditionedCrossModalTranslator


class OmicsHEEndToEndCox(nn.Module):
    """Four T2I CNNs and one GigaPath adapter feeding shared fusion."""

    def __init__(
        self,
        he_input_dim: int = 1536,
        latent_dim: int = 96,
        dropout: float = 0.25,
        rna_encoder_architecture: str = "current_cnn",
        cnv_encoder_architecture: str = "current_cnn",
        protein_encoder_architecture: str = "current_cnn",
        methylation_encoder_architecture: str = "current_cnn",
        fusion_mode: str = "gated",
    ) -> None:
        super().__init__()
        if rna_encoder_architecture == "current_cnn":
            rna_encoder: nn.Module = CNVImageEncoder(3)
        elif rna_encoder_architecture == "masked_cnn_spatial_projector":
            rna_encoder = CNNSpatialProjectorEncoder()
        elif rna_encoder_architecture == "masked_cnn_spatial_global":
            rna_encoder = CNNSpatialGlobalEncoder()
        else:
            raise ValueError(
                f"Unsupported RNA encoder architecture: {rna_encoder_architecture}"
            )
        if cnv_encoder_architecture == "current_cnn":
            cnv_encoder: nn.Module = CNVImageEncoder(3)
        elif cnv_encoder_architecture == "masked_cnn_spatial_projector":
            cnv_encoder = CNNSpatialProjectorEncoder(grid_height=45, grid_width=45)
        else:
            raise ValueError(
                f"Unsupported CNV encoder architecture: {cnv_encoder_architecture}"
            )
        if protein_encoder_architecture == "current_cnn":
            protein_encoder: nn.Module = CNVImageEncoder(3)
        elif protein_encoder_architecture == "masked_cnn_spatial_projector":
            protein_encoder = CNNSpatialProjectorEncoder(
                grid_height=17,
                grid_width=17,
            )
        else:
            raise ValueError(
                "Unsupported protein encoder architecture: "
                f"{protein_encoder_architecture}"
            )
        if methylation_encoder_architecture == "current_cnn":
            methylation_encoder: nn.Module = CNVImageEncoder(3)
        elif methylation_encoder_architecture == "masked_cnn_spatial_projector":
            methylation_encoder = CNNSpatialProjectorEncoder()
        else:
            raise ValueError(
                "Unsupported methylation encoder architecture: "
                f"{methylation_encoder_architecture}"
            )
        self.rna_encoder_architecture = rna_encoder_architecture
        self.cnv_encoder_architecture = cnv_encoder_architecture
        self.protein_encoder_architecture = protein_encoder_architecture
        self.methylation_encoder_architecture = methylation_encoder_architecture
        self.encoders = nn.ModuleList(
            [
                rna_encoder,
                cnv_encoder,
                protein_encoder,
                methylation_encoder,
                HEEmbeddingAdapter(
                    input_dim=he_input_dim,
                    output_dim=96,
                    dropout=dropout,
                ),
            ]
        )
        self.fusion = SharedProjectionFusion(
            input_dim=96,
            latent_dim=latent_dim,
            n_modalities=len(MODALITIES),
            fusion_mode=fusion_mode,
            dropout=dropout,
        )

    def freeze_batchnorm_statistics(self) -> None:
        for encoder in self.encoders:
            for module in encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    @staticmethod
    def _encode_chunks(
        encoder: nn.Module,
        inputs: torch.Tensor,
        chunk_size: int,
        use_gradient_checkpointing: bool,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, len(inputs), chunk_size):
            block = inputs[start : start + chunk_size]
            outputs.append(
                checkpoint(encoder, block, use_reentrant=False)
                if use_gradient_checkpointing
                else encoder(block)
            )
        return torch.cat(outputs, dim=0)

    def forward_selected(
        self,
        modality_inputs: list[torch.Tensor],
        local_indices: list[torch.Tensor],
        availability: torch.Tensor,
        chunk_size: int,
        modality_dropout_probability: float = 0.0,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if len(modality_inputs) != len(MODALITIES):
            raise ValueError("Expected four omics inputs and one H&E input")
        n_patients = availability.shape[0]
        embeddings = []
        for modality, encoder in enumerate(self.encoders):
            present = availability[:, modality].bool()
            positions = torch.nonzero(present, as_tuple=False).flatten()
            if positions.numel() == 0:
                raise ValueError(
                    f"Selected batch has no {MODALITIES[modality]} patients"
                )
            input_rows = local_indices[modality][present]
            encoded = self._encode_chunks(
                encoder,
                modality_inputs[modality][input_rows],
                chunk_size,
                use_gradient_checkpointing=self.training,
            )
            empty = torch.zeros(
                (n_patients, encoded.shape[1]),
                device=encoded.device,
                dtype=encoded.dtype,
            )
            embeddings.append(empty.index_copy(0, positions, encoded))
        return self.fusion(
            embeddings,
            availability,
            modality_dropout_probability=modality_dropout_probability,
        )

    def encode_all(
        self,
        modality_inputs: list[torch.Tensor],
        master_positions: list[torch.Tensor],
        n_master: int,
        chunk_size: int,
    ) -> list[torch.Tensor]:
        embeddings = []
        for encoder, inputs, positions in zip(
            self.encoders, modality_inputs, master_positions
        ):
            encoded = self._encode_chunks(
                encoder,
                inputs,
                chunk_size,
                use_gradient_checkpointing=False,
            )
            empty = torch.zeros(
                (n_master, encoded.shape[1]),
                device=encoded.device,
                dtype=encoded.dtype,
            )
            embeddings.append(empty.index_copy(0, positions, encoded))
        return embeddings


class FiveModalCrossModalMultitaskModel(nn.Module):
    """Survival backbone plus target-conditioned measured-feature translator."""

    def __init__(
        self,
        survival: OmicsHEEndToEndCox,
        translator: TargetConditionedCrossModalTranslator,
    ) -> None:
        super().__init__()
        self.survival = survival
        self.translator = translator
