import torch

from omnit2i.models import SharedProjectionFusion, paired_alignment_mse


def test_shared_projection_mean_is_missing_aware_equal_average():
    model = SharedProjectionFusion(
        input_dim=4, latent_dim=3, n_modalities=2, fusion_mode="mean", dropout=0.0
    )
    model.eval()
    same = torch.randn(3, 4)
    availability = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
    output = model([same, same], availability)
    expected = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    assert torch.equal(output["weights"], expected)
    assert torch.allclose(output["projected"][0], output["projected"][1])


def test_paired_alignment_uses_only_naturally_shared_patients():
    rna = torch.tensor([[1.0, 2.0], [4.0, 5.0], [8.0, 9.0]])
    cnv = torch.tensor([[1.0, 2.0], [100.0, 100.0], [-20.0, -20.0]])
    availability = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool)
    loss = paired_alignment_mse([rna, cnv], availability)
    assert torch.equal(loss, torch.tensor(0.0))


def test_paired_alignment_backpropagates_to_both_modalities():
    rna = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    cnv = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    availability = torch.ones((2, 2), dtype=torch.bool)
    loss = paired_alignment_mse([rna, cnv], availability)
    loss.backward()
    assert rna.grad is not None
    assert cnv.grad is not None
    assert torch.count_nonzero(rna.grad) > 0
    assert torch.count_nonzero(cnv.grad) > 0
