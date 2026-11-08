import io
from itertools import product

import pytest
import torch

from omnit2i.models import (
    FiveModalCrossModalMultitaskModel,
    OmicsHEEndToEndCox,
    SharedProjectionFusion,
    TargetConditionedCrossModalTranslator,
    paired_alignment_mse,
)


def make_survival(fusion_mode):
    return OmicsHEEndToEndCox(
        dropout=0.0,
        rna_encoder_architecture="masked_cnn_spatial_projector",
        cnv_encoder_architecture="masked_cnn_spatial_projector",
        protein_encoder_architecture="masked_cnn_spatial_projector",
        methylation_encoder_architecture="masked_cnn_spatial_projector",
        fusion_mode=fusion_mode,
    )


@pytest.mark.parametrize("fusion_mode", ["mean", "gated"])
def test_joint_gradients_and_checkpoint_round_trip(fusion_mode):
    torch.manual_seed(7)
    model = FiveModalCrossModalMultitaskModel(
        make_survival(fusion_mode),
        TargetConditionedCrossModalTranslator((11, 7, 5, 9, 13), dropout=0.0),
    )
    model.train()
    model.survival.freeze_batchnorm_statistics()
    n = 4
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 0, 1, 0]],
        dtype=torch.bool,
    )
    inputs = [torch.randn(n, 3, grid, grid) for grid in (45, 45, 17, 45)]
    inputs.append(torch.randn(n, 1536))
    for values, present in zip(inputs, mask.T):
        values[~present] = torch.nan
        values.requires_grad_()
    indices = [torch.arange(n) for _ in inputs]
    output = model.survival.forward_selected(inputs, indices, mask, chunk_size=2)
    translated = model.translator(torch.stack(output["projected"], dim=1), mask, "he")
    assert torch.isfinite(output["risk"]).all()
    assert torch.isfinite(translated["feature_prediction"]).all()
    assert torch.count_nonzero(output["weights"][~mask]) == 0
    assert torch.count_nonzero(translated["weights"][:, 4]) == 0
    if fusion_mode == "mean":
        assert model.survival.fusion.gates is None
        torch.testing.assert_close(output["weights"], mask.float() / mask.sum(1, keepdim=True))
    loss = output["risk"].square().mean()
    loss = loss + translated["feature_prediction"].square().mean()
    loss = loss + paired_alignment_mse(output["projected"], mask)
    loss.backward()
    for values, present in zip(inputs, mask.T):
        assert values.grad[present].abs().sum() > 0
        assert torch.count_nonzero(values.grad[~present]) == 0
    for encoder in model.survival.encoders:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    assert model.survival.fusion.shared_projection[0].weight.grad.abs().sum() > 0
    assert model.translator.feature_decoders["he"].network[0].weight.grad.abs().sum() > 0
    if fusion_mode == "gated":
        assert model.survival.fusion.gates[0].weight.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        before = model.survival.forward_selected(inputs, indices, mask, chunk_size=2)
        features_before = model.translator(torch.stack(before["projected"], 1), mask, "he")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = FiveModalCrossModalMultitaskModel(
            make_survival(fusion_mode),
            TargetConditionedCrossModalTranslator((11, 7, 5, 9, 13), dropout=0.0),
        ).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        after = restored.survival.forward_selected(inputs, indices, mask, chunk_size=3)
        features_after = restored.translator(torch.stack(after["projected"], 1), mask, "he")
        torch.testing.assert_close(before["risk"], after["risk"])
        torch.testing.assert_close(features_before["feature_prediction"], features_after["feature_prediction"])


@pytest.mark.parametrize("fusion_mode", ["mean", "gated"])
def test_fusion_accepts_all_31_nonempty_availability_patterns(fusion_mode):
    mask = torch.tensor(list(product([False, True], repeat=5))[1:])
    fusion = SharedProjectionFusion(n_modalities=5, fusion_mode=fusion_mode).eval()
    output = fusion([torch.randn(31, 96) for _ in range(5)], mask)
    assert torch.isfinite(output["risk"]).all()
    torch.testing.assert_close(output["weights"].sum(1), torch.ones(31))
    assert torch.count_nonzero(output["weights"][~mask]) == 0
    with pytest.raises(ValueError, match="at least one available"):
        fusion([torch.randn(1, 96) for _ in range(5)], torch.zeros(1, 5, dtype=torch.bool))


def test_selected_batch_requires_observations_for_each_modality():
    model = make_survival("mean").eval()
    inputs = [torch.randn(2, 3, grid, grid) for grid in (45, 45, 17, 45)]
    inputs.append(torch.randn(2, 1536))
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, 0] = False
    with pytest.raises(ValueError, match="no rna patients"):
        model.forward_selected(inputs, [torch.arange(2) for _ in inputs], mask, chunk_size=2)
