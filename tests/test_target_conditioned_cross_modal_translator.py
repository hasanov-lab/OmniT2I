from __future__ import annotations

import pytest
import torch

from omnit2i.models.target_conditioned_cross_modal_translator import (
    LowRankLinear,
    TargetConditionedCrossModalTranslator,
)


OUTPUT_DIMS = (11, 7, 5, 9, 13)


def make_model() -> TargetConditionedCrossModalTranslator:
    return TargetConditionedCrossModalTranslator(
        OUTPUT_DIMS,
        latent_dim=8,
        mapper_hidden_dim=12,
        decoder_hidden_dim=16,
        linear_rank=3,
        dropout=0.0,
    )


def test_builds_all_twenty_routes_with_low_rank_skips() -> None:
    model = make_model()
    assert len(model.route_mappers) == 20
    assert len(model.route_gates) == 20
    for mapper in model.route_mappers.values():
        assert isinstance(mapper.linear_skip, LowRankLinear)
        assert mapper.linear_skip.down.out_features == 3
        assert mapper.linear_skip.up.in_features == 3


def test_target_and_missing_sources_receive_zero_weight() -> None:
    torch.manual_seed(4)
    model = make_model()
    embeddings = torch.randn(3, 5, 8)
    source_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    output = model(embeddings, source_mask, target="protein")
    assert output["latent_prediction"].shape == (3, 8)
    assert output["feature_prediction"].shape == (3, OUTPUT_DIMS[2])
    assert output["route_candidates"].shape == (3, 5, 8)
    assert output["effective_source_mask"].shape == (3, 5)
    assert torch.allclose(output["weights"].sum(dim=1), torch.ones(3))
    assert torch.count_nonzero(output["weights"][:, 2]) == 0
    assert torch.count_nonzero(output["weights"][0, 3:]) == 0
    assert torch.count_nonzero(output["weights"][1, [1, 2, 4]]) == 0


def test_target_embedding_cannot_change_cross_modal_prediction() -> None:
    torch.manual_seed(8)
    model = make_model().eval()
    embeddings = torch.randn(2, 5, 8)
    source_mask = torch.ones((2, 5), dtype=torch.bool)
    original = model(embeddings, source_mask, target="rna")
    changed = embeddings.clone()
    changed[:, 0] = 1_000.0 * torch.randn_like(changed[:, 0])
    perturbed = model(changed, source_mask, target="rna")
    assert torch.allclose(
        original["latent_prediction"], perturbed["latent_prediction"]
    )
    assert torch.allclose(
        original["feature_prediction"], perturbed["feature_prediction"]
    )


def test_masked_source_payload_cannot_change_cross_modal_prediction() -> None:
    torch.manual_seed(9)
    model = make_model().eval()
    embeddings = torch.randn(2, 5, 8)
    source_mask = torch.ones((2, 5), dtype=torch.bool)
    source_mask[:, 3] = False
    original = model(embeddings, source_mask, target="protein")
    changed = embeddings.clone()
    changed[:, 3] = torch.nan
    perturbed = model(changed, source_mask, target="protein")
    assert torch.allclose(
        original["latent_prediction"], perturbed["latent_prediction"]
    )
    assert torch.allclose(
        original["feature_prediction"], perturbed["feature_prediction"]
    )


def test_single_source_has_unit_weight_and_only_it_receives_gradient() -> None:
    torch.manual_seed(12)
    model = make_model()
    embeddings = torch.randn(2, 5, 8, requires_grad=True)
    source_mask = torch.zeros((2, 5), dtype=torch.bool)
    source_mask[:, 1] = True
    output = model(embeddings, source_mask, target=4)
    assert torch.allclose(output["weights"][:, 1], torch.ones(2))
    assert torch.count_nonzero(output["weights"][:, [0, 2, 3, 4]]) == 0
    output["feature_prediction"].sum().backward()
    assert embeddings.grad is not None
    assert torch.count_nonzero(embeddings.grad[:, 1]) > 0
    assert torch.count_nonzero(embeddings.grad[:, [0, 2, 3, 4]]) == 0


def test_each_target_supports_latent_only_and_feature_prediction() -> None:
    model = make_model()
    embeddings = torch.randn(2, 5, 8)
    availability = torch.ones((2, 5), dtype=torch.bool)
    for target, output_dim in enumerate(OUTPUT_DIMS):
        latent = model.predict_latent(embeddings, availability, target)
        assert latent["latent_prediction"].shape == (2, 8)
        decoded = model.decode_features(
            latent["latent_prediction"], model.modalities[target]
        )
        assert decoded.shape == (2, output_dim)


def test_multiple_sources_backpropagate_through_mapper_gate_and_decoder() -> None:
    torch.manual_seed(14)
    model = make_model()
    embeddings = torch.randn(3, 5, 8, requires_grad=True)
    source_mask = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )
    output = model(embeddings, source_mask, target="he")
    output["feature_prediction"].square().mean().backward()

    assert embeddings.grad is not None
    assert torch.count_nonzero(embeddings.grad[:, :3]) > 0
    assert torch.count_nonzero(embeddings.grad[:, 3:]) == 0
    mapper = model.route_mappers[model.route_key("rna", "he")]
    gate = model.route_gates[model.route_key("rna", "he")]
    decoder = model.feature_decoders["he"]
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in mapper.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in gate.parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in decoder.parameters()
    )


def test_named_output_dimensions_follow_modality_names() -> None:
    named_dims = dict(zip(make_model().modalities, OUTPUT_DIMS, strict=True))
    model = TargetConditionedCrossModalTranslator(
        named_dims,
        latent_dim=8,
        mapper_hidden_dim=12,
        decoder_hidden_dim=16,
        linear_rank=3,
        dropout=0.0,
    )
    embeddings = torch.randn(1, 5, 8)
    availability = torch.ones((1, 5), dtype=torch.bool)
    for modality, expected_dim in named_dims.items():
        assert model(embeddings, availability, modality)[
            "feature_prediction"
        ].shape == (1, expected_dim)

    with pytest.raises(ValueError, match="exactly"):
        TargetConditionedCrossModalTranslator(
            {**named_dims, "unexpected": 3}, latent_dim=8
        )


def test_missing_nan_slots_are_safe_but_available_nan_is_rejected() -> None:
    model = make_model().eval()
    embeddings = torch.randn(2, 5, 8)
    availability = torch.ones((2, 5), dtype=torch.bool)
    availability[:, 3] = False
    embeddings[:, 3] = torch.nan
    output = model(embeddings, availability, target="protein")
    assert torch.isfinite(output["latent_prediction"]).all()
    assert torch.isfinite(output["feature_prediction"]).all()

    availability[:, 3] = True
    with pytest.raises(ValueError, match="available source embedding"):
        model(embeddings, availability, target="protein")


@pytest.mark.parametrize(
    ("source_mask", "message"),
    [
        (torch.tensor([[0, 0, 1, 0, 0]]), "non-target source"),
        (torch.tensor([[1.0, 0.5, 0.0, 0.0, 0.0]]), "binary 0/1"),
    ],
)
def test_invalid_source_masks_are_rejected(
    source_mask: torch.Tensor, message: str
) -> None:
    model = make_model()
    embeddings = torch.randn(1, 5, 8)
    with pytest.raises(ValueError, match=message):
        model(embeddings, source_mask, target="protein")
