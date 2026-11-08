"""Run preprocessing and a short RNA training pass with synthetic patient data."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(817)
    samples = [f"synthetic_sample_{i}" for i in range(30)]
    expression = pd.DataFrame(
        rng.normal(size=(240, 30)).astype(np.float32),
        index=[f"gene_{i}" for i in range(240)], columns=samples,
    )
    cohort = pd.DataFrame({
        "patient": [f"synthetic_patient_{i}" for i in range(30)], "sample": samples,
        "cancer_type": ["A", "B", "C"] * 10, "time": np.arange(1, 31), "event": [1] * 30,
        "partition": ["core"] * 20 + ["validation"] * 5 + ["test"] * 5,
    })
    expression.to_csv(args.output_dir / "expression.tsv.gz", sep="\t")
    cohort.to_csv(args.output_dir / "cohort.csv", index=False)
    subprocess.run([
        sys.executable, "-m", "omnit2i.preprocessing.rna",
        "--expression", str(args.output_dir / "expression.tsv.gz"),
        "--cohort", str(args.output_dir / "cohort.csv"),
        "--output-dir", str(args.output_dir / "prepared"), "--top-n", "60", "--invariant-per-bin", "5",
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "omnit2i.training.rna",
        "--images", str(args.output_dir / "prepared/images.npz"),
        "--output-dir", str(args.output_dir / "trained"), "--pretrain-epochs", "1", "--epochs", "3",
    ], check=True)
    print("Synthetic workflow completed; these predictions have no scientific interpretation.")


if __name__ == "__main__":
    main()
