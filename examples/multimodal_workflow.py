"""Exercise molecular preprocessing, Stage A, fusion, and joint training on synthetic inputs."""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from omnit2i.models import MODALITIES


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
    availability[np.arange(0, 20, 2), np.arange(0, 20, 2) % 5] = False
    paths, checkpoints = [], []
    for index, name in enumerate(MODALITIES):
        rows = rng.permutation(np.flatnonzero(availability[:, index]))
        observed = cohort.iloc[rows]
        if name == "he":
            path = args.output_dir / "he.npz"
            np.savez_compressed(path, patient=observed["patient"].to_numpy(str), embeddings=rng.normal(size=(len(rows), 1536)).astype(np.float32))
        else:
            matrix = pd.DataFrame(rng.normal(size=(80, len(rows))).astype(np.float32), index=[f"gene{i}" for i in range(80)], columns=observed["sample"])
            if name == "methylation":
                matrix = 1 / (1 + np.exp(-matrix))
            matrix_path = args.output_dir / f"{name}.tsv.gz"
            matrix.to_csv(matrix_path, sep="\t")
            observed_path = args.output_dir / f"{name}_cohort.csv"
            observed.to_csv(observed_path, index=False)
            prepared = args.output_dir / f"{name}_prepared"
            if name == "rna":
                prepare = ["omnit2i.preprocessing.rna", "--expression", str(matrix_path), "--invariant-per-bin", "2"]
            else:
                prepare = ["omnit2i.preprocessing.omics", "--modality", name, "--matrix", str(matrix_path)]
                if name == "methylation":
                    chrom_path = args.output_dir / "chromosomes.tsv"
                    pd.DataFrame({"feature_id": matrix.index, "chrom": "chr1"}).to_csv(chrom_path, sep="\t", index=False)
                    prepare += ["--chromosomes", str(chrom_path)]
            subprocess.run([sys.executable, "-m", *prepare, "--cohort", str(observed_path), "--output-dir", str(prepared), "--top-n", "60"], check=True)
            path = prepared / "images.npz"
            trained = args.output_dir / f"{name}_trained"
            subprocess.run([sys.executable, "-m", "omnit2i.training.rna", "--modality", name, "--images", str(path), "--output-dir", str(trained), "--pretrain-epochs", "1", "--epochs", "2"], check=True)
            checkpoints += [f"--{name}-checkpoint", str(trained / "model.pt")]
        paths += [f"--{name}", str(path)]
    subprocess.run([
        sys.executable, "-m", "omnit2i.training.fusion", "--cohort", str(cohort_path),
        *paths, *checkpoints, "--output-dir", str(args.output_dir / "fusion"), "--epochs", "2",
    ], check=True)
    targets = {
        "patient": cohort["patient"].to_numpy(str), "cancer_type": cohort["cancer_type"].to_numpy(str),
        "partition": cohort["partition"].to_numpy(str), "availability": availability,
        "modalities": np.array(MODALITIES), "target_scale": np.array(["synthetic_z"] * 5),
    }
    for index, name in enumerate(MODALITIES):
        values = rng.normal(size=(len(cohort), 12 + index)).astype(np.float32)
        values[~availability[:, index]] = np.nan
        targets[f"{name}_target"] = values
        targets[f"{name}_feature_ids"] = np.array([f"feature_{i}" for i in range(values.shape[1])])
    target_path = args.output_dir / "targets.npz"
    np.savez_compressed(target_path, **targets)
    subprocess.run([
        sys.executable, "-m", "omnit2i.training.crossmodal", "--cohort", str(cohort_path),
        *paths, "--source-checkpoint", str(args.output_dir / "fusion" / "model.pt"),
        "--targets", str(target_path), "--output-dir", str(args.output_dir / "joint"),
        "--warmup-epochs", "1", "--joint-epochs", "1", "--validation-frequency", "1",
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "omnit2i.evaluation", "--cohort", str(cohort_path),
        *paths, "--checkpoint", str(args.output_dir / "joint/model.pt"),
        "--targets", str(target_path), "--target-baselines", str(args.output_dir / "joint/target_baselines.npz"),
        "--output-dir", str(args.output_dir / "evaluation"),
    ], check=True)
    print("Synthetic preprocessing, Stage A/B/C, and frozen evaluation completed; these models have no scientific interpretation.")


if __name__ == "__main__":
    main()
