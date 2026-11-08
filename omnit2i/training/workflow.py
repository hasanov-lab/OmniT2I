"""Run the existing Stage A/B/C commands for one prepared seed/fold."""

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from omnit2i.models import MODALITIES


def train_fold(prepared, output, seed, fold, device):
    cohort = prepared / "cohort.csv"
    paths = [prepared / name / "images.npz" for name in MODALITIES[:4]] + [prepared / "he.npz"]
    for path in [cohort, prepared / "targets.npz", *paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    table = pd.read_csv(cohort)
    for field, value in (("seed", seed), ("outer_fold", fold)):
        if field in table and not table[field].eq(value).all():
            raise ValueError(f"Prepared cohort {field} disagrees with this invocation")
    output.mkdir(parents=True, exist_ok=False)
    common = ["--device", device]
    checkpoints = []
    for name, path in zip(MODALITIES[:4], paths[:4]):
        destination = output / "A" / name
        subprocess.run([
            sys.executable, "-m", "omnit2i.training.rna", "--modality", name,
            "--images", str(path), "--seed", str(seed + fold * 100),
            "--output-dir", str(destination), *common,
        ], check=True)
        checkpoints += [f"--{name}-checkpoint", str(destination / "model.pt")]
    data = ["--cohort", str(cohort)]
    for name, path in zip(MODALITIES, paths):
        data += [f"--{name}", str(path)]
    subprocess.run([
        sys.executable, "-m", "omnit2i.training.fusion", *data, *checkpoints,
        "--seed", str(seed + fold * 1000), "--output-dir", str(output / "B"), *common,
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "omnit2i.training.crossmodal", *data,
        "--source-checkpoint", str(output / "B/model.pt"),
        "--targets", str(prepared / "targets.npz"), "--seed", str(seed), "--fold", str(fold),
        "--output-dir", str(output / "C"), *common,
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_fold(args.prepared_dir, args.output_dir, args.seed, args.fold, args.device)


if __name__ == "__main__":
    main()
