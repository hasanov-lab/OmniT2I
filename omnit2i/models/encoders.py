"""Table-to-image CNN encoders and frozen-embedding H&E adapter."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


N_CHANNELS = 3
GRID_HEIGHT = 45
GRID_WIDTH = 45
EMBEDDING_DIMENSION = 96
CNN_FEATURE_CHANNELS = 96
CNN_FEATURE_HEIGHT = 11
CNN_FEATURE_WIDTH = 11
SPATIAL_PROJECTION_CHANNELS = 16
N_MEASURED_FEATURES = 2000


class CNVImageEncoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image).flatten(1)


class CNVT2ICoxModel(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.encoder = CNVImageEncoder(in_channels)
        self.head = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(image)).flatten()


def _validate_image(
    image: torch.Tensor,
    grid_height: int = GRID_HEIGHT,
    grid_width: int = GRID_WIDTH,
) -> None:
    if image.ndim != 4:
        raise ValueError(
            "RNA image must have shape [patients, channels, height, width]"
        )
    expected = (N_CHANNELS, int(grid_height), int(grid_width))
    if tuple(image.shape[1:]) != expected:
        raise ValueError(
            f"RNA image shape must end in {expected}, got {tuple(image.shape)}"
        )


class LockedCNNSpatialTrunk(nn.Module):
    """Locked CNN convolutional layers without the global-average pool."""

    def __init__(
        self,
        grid_height: int = GRID_HEIGHT,
        grid_width: int = GRID_WIDTH,
    ) -> None:
        super().__init__()
        if grid_height < 4 or grid_width < 4:
            raise ValueError("Spatial CNN grids must be at least 4 x 4")
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.feature_height = self.grid_height // 4
        self.feature_width = self.grid_width // 4
        self.network = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, CNN_FEATURE_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(CNN_FEATURE_CHANNELS),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _validate_image(image, self.grid_height, self.grid_width)
        feature_map = self.network(image)
        expected = (
            CNN_FEATURE_CHANNELS,
            self.feature_height,
            self.feature_width,
        )
        if tuple(feature_map.shape[1:]) != expected:
            raise RuntimeError(
                f"Locked CNN produced {tuple(feature_map.shape[1:])}, expected {expected}"
            )
        return feature_map


class CNNSpatialProjectorEncoder(nn.Module):
    """CNN image encoder with an explicitly position-preserving readout."""

    def __init__(
        self,
        grid_height: int = GRID_HEIGHT,
        grid_width: int = GRID_WIDTH,
    ) -> None:
        super().__init__()
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.trunk = LockedCNNSpatialTrunk(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
        )
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(
                CNN_FEATURE_CHANNELS,
                SPATIAL_PROJECTION_CHANNELS,
                kernel_size=1,
            ),
            nn.GELU(),
        )
        self.output = nn.Linear(
            SPATIAL_PROJECTION_CHANNELS
            * self.trunk.feature_height
            * self.trunk.feature_width,
            EMBEDDING_DIMENSION,
        )

    def feature_map(self, image: torch.Tensor) -> torch.Tensor:
        """Return the locked pre-pooling CNN feature map for audit tests."""
        return self.trunk(image)

    def projected_feature_map(self, image: torch.Tensor) -> torch.Tensor:
        return self.spatial_projection(self.feature_map(image))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        projected = self.projected_feature_map(image)
        return self.output(projected.flatten(start_dim=1))


class CNNSpatialGlobalEncoder(CNNSpatialProjectorEncoder):
    """Position-preserving CNN with one frozen whole-image scalar feature."""

    def __init__(self) -> None:
        super().__init__()
        self.global_image_feature = nn.Conv2d(
            in_channels=N_CHANNELS,
            out_channels=1,
            kernel_size=(GRID_HEIGHT, GRID_WIDTH),
            bias=False,
        )
        self.global_direction = nn.Parameter(torch.zeros(EMBEDDING_DIMENSION))
        self.global_image_feature.requires_grad_(False)

    def load_spatial_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> None:
        """Load an exact ``CNNSpatialProjectorEncoder`` state dictionary."""

        module_prefixes = {
            "trunk": self.trunk,
            "spatial_projection": self.spatial_projection,
            "output": self.output,
        }
        expected = {
            f"{prefix}.{key}"
            for prefix, module in module_prefixes.items()
            for key in module.state_dict()
        }
        observed = set(state_dict)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                "Spatial RNA state differs: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for prefix, module in module_prefixes.items():
            selected = {
                key.removeprefix(f"{prefix}."): value
                for key, value in state_dict.items()
                if key.startswith(f"{prefix}.")
            }
            module.load_state_dict(selected, strict=True)

    def load_global_image_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        scale: float,
    ) -> None:
        """Load and scale the fold-local whole-image convolution."""

        if set(state_dict) != {"weight"}:
            raise ValueError("Global image state must contain only 'weight'")
        scaled = {"weight": state_dict["weight"] * float(scale)}
        self.global_image_feature.load_state_dict(scaled, strict=True)
        self.global_image_feature.requires_grad_(False)

    def global_scalar(self, image: torch.Tensor) -> torch.Tensor:
        return self.global_image_feature(image).flatten(start_dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        spatial_embedding = super().forward(image)
        return spatial_embedding + self.global_scalar(image) * self.global_direction


class HEEmbeddingAdapter(nn.Module):
    """Trainable adapter from frozen 1,536-D GigaPath vectors to 96-D."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 384,
        output_dim: int = 96,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class HEOnlyCox(nn.Module):
    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.adapter = HEEmbeddingAdapter(dropout=dropout)
        self.risk_head = nn.Sequential(
            nn.Linear(96, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.risk_head(self.adapter(values)).flatten()


class MaskedRNAImageAutoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_measured_features: int = N_MEASURED_FEATURES,
    ) -> None:
        super().__init__()
        if n_measured_features < 1:
            raise ValueError("n_measured_features must be positive")
        self.encoder = encoder
        self.n_measured_features = int(n_measured_features)
        self.decoder = nn.Linear(
            EMBEDDING_DIMENSION, N_CHANNELS * self.n_measured_features
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image)).reshape(
            image.shape[0], N_CHANNELS, self.n_measured_features
        )
