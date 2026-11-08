"""Exercise five-modality training with synthetic inputs and random encoder checkpoints."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from omnit2i.models import MODALITIES, CNNSpatialProjectorEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(81)
    cohort = pd.DataFrame({
        "patient": [f"synthetic_p{i}" for i in range(20)],
        "sample": [f"synthetic_s{i}" for i in range(20)],
        "cancer_type": ["A", "B"] * 10, "time": np.arange(1, 21), "event": [1] * 20,
        "partition": ["core"] * 10 + ["validation"] * 5 + ["test"] * 5,
    })
    cohort_path = args.output_dir / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    availability = np.ones((20, 5), dtype=bool)
    availability[np.arange(20), np.arange(20) % 5] = False
    paths, checkpoints = [], []
    for index, name in enumerate(MODALITIES):
        rows = rng.permutation(np.flatnonzero(availability[:, index]))
        grid = 17 if name == "protein" else 45
        shape = (1536,) if name == "he" else (3, grid, grid)
        values = rng.normal(size=(len(rows), *shape)).astype(np.float32)
        path = args.output_dir / f"{name}.npz"
        np.savez_compressed(path, patient=cohort.iloc[rows]["patient"].to_numpy(str), partition=cohort.iloc[rows]["partition"].to_numpy(str), **{"embeddings" if name == "he" else "images": values})
        paths += [f"--{name}", str(path)]
        if name != "he":
            torch.manual_seed(10 + index)
            encoder = CNNSpatialProjectorEncoder(grid_height=grid, grid_width=grid)
            checkpoint = args.output_dir / f"{name}.pt"
            torch.save({"model_state": {f"encoder.{key}": value for key, value in encoder.state_dict().items()}}, checkpoint)
            checkpoints += [f"--{name}-checkpoint", str(checkpoint)]
    subprocess.run([
        sys.executable, "-m", "omnit2i.training.fusion", "--cohort", str(cohort_path),
        *paths, *checkpoints, "--output-dir", str(args.output_dir / "fusion"), "--epochs", "2",
    ], check=True)
    print("Synthetic fusion workflow completed; random initial checkpoints are not pretrained models.")


if __name__ == "__main__":
    main()
