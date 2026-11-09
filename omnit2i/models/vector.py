"""Original vector controls on the same rendered three-channel features."""

import torch
from torch import nn

VARIANTS = ("mlp_matched", "mlp_wide")


class VectorEncoder(nn.Module):
    def __init__(self, variant, grid_height=45, grid_width=45):
        super().__init__()
        if variant not in VARIANTS or (grid_height, grid_width) not in ((45, 45), (17, 17)):
            raise ValueError("Unsupported vector variant or molecular feature grid")
        self.grid_height, self.grid_width = grid_height, grid_width
        self.n_features = 2000 if grid_height == 45 else 217
        matched_width = 40 if self.n_features == 2000 else 106
        self.width = matched_width if variant == "mlp_matched" else max(matched_width, 96)
        self.input_block = nn.Sequential(
            nn.Linear(3 * self.n_features, self.width), nn.BatchNorm1d(self.width), nn.ReLU())
        self.output = nn.Linear(self.width, 96)

    def forward(self, image):
        if tuple(image.shape[1:]) != (3, self.grid_height, self.grid_width):
            raise ValueError("Unexpected molecular input shape")
        # The shared renderer fills ordered features before padding; no spatial operation follows.
        values = image.flatten(2)[:, :, :self.n_features].transpose(1, 2).flatten(1)
        return self.output(self.input_block(values))

