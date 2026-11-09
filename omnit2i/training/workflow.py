"""Run the existing Stage A/B/C commands for one prepared seed/fold."""

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

from omnit2i.models import MODALITIES


def train_fold(prepared, output, seed, fold, device, *, variant=None, reference=None):
    if variant is not None:
        from .ablations import VARIANTS
        if variant not in VARIANTS:
            raise ValueError(f"Unknown ablation: {variant}")
    cohort = prepared / "cohort.csv"
    paths = [prepared / name / "images.npz" for name in MODALITIES[:4]] + [prepared / "he.npz"]
    for path in [cohort, prepared / "targets.npz", *paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    table = pd.read_csv(cohort)
    for field, value in (("seed", seed), ("outer_fold", fold)):
        if field in table and not table[field].eq(value).all():
            raise ValueError(f"Prepared cohort {field} disagrees with this invocation")
    reuse_a = variant in {"no_risk_consistency", "mean_fusion", "no_modality_dropout", "simplified"}
    reuse_b = variant in {"no_risk_consistency", "simplified"}
    if reuse_a and reference is None:
        raise ValueError("This ablation requires its reference training directory")
    reused = [reference / "B/model.pt"] if reuse_b else ([reference / "A" / name / "model.pt" for name in MODALITIES[:4]] if reuse_a else [])
    for path in reused:
        if not path.is_file():
            raise FileNotFoundError(path)
    if variant == "simplified":
        import torch
        source = torch.load(reference / "B/model.pt", map_location="cpu", weights_only=True)
        if source["model_kwargs"]["fusion_mode"] != "mean":
            raise ValueError("Joint simplification requires the mean-fusion Stage B")
    output.mkdir(parents=True, exist_ok=False)
    if variant in {"no_c2", "no_c3", "random_layout"}:
        from .ablations import prepare_inputs
        prepared = prepare_inputs(prepared, output / "inputs", variant, fold)
        paths = [prepared / name / "images.npz" for name in MODALITIES[:4]] + [prepared / "he.npz"]
    common = ["--device", device]
    checkpoints = []
    for name, path in zip(MODALITIES[:4], paths[:4]):
        destination = (reference if reuse_a else output) / "A" / name
        extra = ["--pretrain-epochs", "0"] if variant == "no_pretraining" else (["--input-ablation", variant] if variant in {"no_c2", "no_c3"} else [])
        if variant in {"mlp_matched", "mlp_wide"}:
            extra += ["--encoder-architecture", variant]
        if not reuse_a:
            subprocess.run([
                sys.executable, "-m", "omnit2i.training.rna", "--modality", name,
                "--images", str(path), "--seed", str(seed + fold * 100),
                "--output-dir", str(destination), *common, *extra,
            ], check=True)
        checkpoints += [f"--{name}-checkpoint", str(destination / "model.pt")]
    data = ["--cohort", str(cohort)]
    for name, path in zip(MODALITIES, paths):
        data += [f"--{name}", str(path)]
    b_extra = ["--fusion-mode", "mean"] if variant == "mean_fusion" else (["--modality-dropout", "0"] if variant == "no_modality_dropout" else [])
    if variant in {"mlp_matched", "mlp_wide"}:
        b_extra += ["--encoder-architecture", variant]
    if not reuse_b:
        subprocess.run([
            sys.executable, "-m", "omnit2i.training.fusion", *data, *checkpoints,
            "--seed", str(seed + fold * 1000), "--output-dir", str(output / "B"), *common, *b_extra,
        ], check=True)
    c_extra = ["--risk-consistency-weight", "0"] if variant in {"no_risk_consistency", "simplified"} else (["--modality-dropout", "0"] if variant == "no_modality_dropout" else [])
    subprocess.run([
        sys.executable, "-m", "omnit2i.training.crossmodal", *data,
        "--source-checkpoint", str((reference if reuse_b else output) / "B/model.pt"),
        "--targets", str(prepared / "targets.npz"), "--seed", str(seed), "--fold", str(fold),
        "--output-dir", str(output / "C"), *common, *c_extra,
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
