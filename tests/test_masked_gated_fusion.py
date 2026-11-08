import torch

from omnit2i.models.masked_gated_fusion import MaskedGatedFusion


def test_masked_weights_sum_to_one_and_missing_modalities_get_zero():
    model = MaskedGatedFusion([4, 4], latent_dim=3, dropout=0.0).eval()
    x = [torch.randn(3, 4), torch.randn(3, 4)]
    mask = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
    output = model(x, mask)
    weights = output["weights"]
    assert torch.allclose(weights.sum(dim=1), torch.ones(3))
    assert weights[1, 1].item() == 0.0
    assert weights[2, 0].item() == 0.0
    assert weights[1, 0].item() == 1.0
    assert weights[2, 1].item() == 1.0


def test_modality_dropout_never_removes_every_available_modality():
    torch.manual_seed(5)
    mask = torch.tensor([[1, 1], [1, 0], [0, 1]] * 20, dtype=torch.bool)
    dropped = MaskedGatedFusion.modality_dropout(mask, 0.99)
    assert dropped.any(dim=1).all()
    assert (~dropped | mask).all()
