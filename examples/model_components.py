"""Run survival and cross-modal components on synthetic inputs."""

import torch

from omnit2i.models import (
    FiveModalCrossModalMultitaskModel,
    OmicsHEEndToEndCox,
    TargetConditionedCrossModalTranslator,
)


def main():
    torch.manual_seed(101)
    survival = OmicsHEEndToEndCox(
        rna_encoder_architecture="masked_cnn_spatial_projector",
        cnv_encoder_architecture="masked_cnn_spatial_projector",
        protein_encoder_architecture="masked_cnn_spatial_projector",
        methylation_encoder_architecture="masked_cnn_spatial_projector",
        fusion_mode="mean",
    )
    # Feature dimensions below are synthetic; set these from prepared targets.
    translator = TargetConditionedCrossModalTranslator((20, 20, 10, 20, 1536))
    model = FiveModalCrossModalMultitaskModel(survival, translator)
    model.train()
    model.survival.freeze_batchnorm_statistics()

    n = 3
    inputs = [
        torch.randn(n, 3, 45, 45),  # RNA
        torch.randn(n, 3, 45, 45),  # CNV
        torch.randn(n, 3, 17, 17),  # RPPA/protein
        torch.randn(n, 3, 45, 45),  # Methylation
        torch.randn(n, 1536),      # Frozen H&E embeddings
    ]
    availability = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.bool
    )
    output = model.survival.forward_selected(
        inputs, [torch.arange(n) for _ in inputs], availability, chunk_size=2
    )
    translated = model.translator(
        torch.stack(output["projected"], dim=1), availability, target="protein"
    )
    # A smoke-test objective, not the survival or reconstruction training loss.
    loss = output["risk"].square().mean() + translated["feature_prediction"].square().mean()
    loss.backward()
    print("Survival risk shape:", tuple(output["risk"].shape))
    print("Predicted protein feature shape:", tuple(translated["feature_prediction"].shape))
    print("Backward pass completed on synthetic data.")


if __name__ == "__main__":
    main()
