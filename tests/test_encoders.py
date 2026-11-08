import pytest
import torch

from omnit2i.models import (
    CNNSpatialGlobalEncoder,
    CNNSpatialProjectorEncoder,
    HEEmbeddingAdapter,
    MaskedRNAImageAutoencoder,
)


@pytest.mark.parametrize("grid", [17, 45])
def test_spatial_encoder_preserves_positions_and_backpropagates(grid):
    encoder = CNNSpatialProjectorEncoder(grid_height=grid, grid_width=grid)
    image = torch.randn(2, 3, grid, grid, requires_grad=True)
    assert encoder.projected_feature_map(image).shape == (2, 16, grid // 4, grid // 4)
    embedding = encoder(image)
    assert embedding.shape == (2, 96)
    embedding.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert encoder.trunk.network[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(2, 45, 45), (2, 1, 45, 45), (2, 3, 17, 17)])
def test_spatial_encoder_rejects_incorrect_images(shape):
    with pytest.raises(ValueError, match="shape"):
        CNNSpatialProjectorEncoder()(torch.randn(shape))


def test_he_adapter_and_masked_decoder_backpropagate():
    adapter = HEEmbeddingAdapter(dropout=0.0)
    embeddings = torch.randn(2, 1536, requires_grad=True)
    adapted = adapter(embeddings)
    assert adapted.shape == (2, 96)
    adapted.square().mean().backward()
    assert embeddings.grad.abs().sum() > 0

    autoencoder = MaskedRNAImageAutoencoder(CNNSpatialProjectorEncoder(), 20)
    reconstructed = autoencoder(torch.randn(2, 3, 45, 45))
    assert reconstructed.shape == (2, 3, 20)
    reconstructed.square().mean().backward()
    assert autoencoder.encoder.trunk.network[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="positive"):
        MaskedRNAImageAutoencoder(CNNSpatialProjectorEncoder(), 0)


def test_spatial_global_encoder_loads_spatial_state_without_changing_output():
    spatial = CNNSpatialProjectorEncoder().eval()
    combined = CNNSpatialGlobalEncoder().eval()
    combined.load_spatial_state_dict(spatial.state_dict())
    image = torch.randn(2, 3, 45, 45)
    torch.testing.assert_close(combined(image), spatial(image), rtol=0, atol=0)
    assert not combined.global_image_feature.weight.requires_grad
    with pytest.raises(ValueError, match="Spatial RNA state differs"):
        combined.load_spatial_state_dict({})
