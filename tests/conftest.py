from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from omnit2i.models import MODALITIES, CNNSpatialProjectorEncoder, FiveModalCrossModalMultitaskModel, OmicsHEEndToEndCox, TargetConditionedCrossModalTranslator


@pytest.fixture
def multimodal_files(tmp_path):
    rng = np.random.default_rng(81)
    cohort = pd.DataFrame({
        "patient": [f"p{i}" for i in range(16)], "sample": [f"s{i}" for i in range(16)],
        "cancer_type": ["A", "B"] * 8, "time": np.arange(1, 17), "event": [1] * 16,
        "partition": ["core"] * 8 + ["validation"] * 4 + ["test"] * 4,
    })
    cohort.to_csv(tmp_path / "cohort.csv", index=False)
    availability = np.ones((16, 5), dtype=bool)
    availability[np.arange(16), np.arange(16) % 5] = False
    paths, checkpoints = [], []
    for index, name in enumerate(MODALITIES):
        rows = rng.permutation(np.flatnonzero(availability[:, index]))
        grid = 17 if name == "protein" else 45
        shape = (1536,) if name == "he" else (3, grid, grid)
        values = rng.normal(size=(len(rows), *shape)).astype(np.float32)
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, patient=cohort.iloc[rows]["patient"].to_numpy(str), partition=cohort.iloc[rows]["partition"].to_numpy(str), **{"embeddings" if name == "he" else "images": values})
        paths.append(path)
        if name != "he":
            torch.manual_seed(10 + index)
            encoder = CNNSpatialProjectorEncoder(grid_height=grid, grid_width=grid)
            checkpoint = tmp_path / f"{name}.pt"
            key = "model_state" if name == "rna" else "state_dict"
            torch.save({key: {f"encoder.{key}": value for key, value in encoder.state_dict().items()}}, checkpoint)
            checkpoints.append(checkpoint)
    return tmp_path / "cohort.csv", paths, checkpoints, availability


@pytest.fixture
def evaluation_inputs():
    torch.manual_seed(174)
    cohort = pd.DataFrame({
        "patient": [f"p{i}" for i in range(12)], "cancer_type": ["A"] * 12,
        "time": np.arange(1, 13), "event": 1,
        "partition": ["core"] * 6 + ["validation"] * 3 + ["test"] * 3,
    })
    model = FiveModalCrossModalMultitaskModel(OmicsHEEndToEndCox(latent_dim=8), TargetConditionedCrossModalTranslator(output_dims={name: 12 for name in MODALITIES}, latent_dim=8, linear_rank=4))
    projected = torch.randn(12, 5, 8)
    targets = [np.random.default_rng(i).normal(size=(12, 12)).astype(np.float32) for i in range(5)]
    baselines = {name: {"target_scale": "synthetic", "global_mean": np.zeros(12, np.float32), "cancer_labels": np.array(["A"]), "cancer_means": np.zeros((1, 12), np.float32)} for name in MODALITIES}
    return model, projected, cohort, np.ones((12, 5), bool), targets, baselines
