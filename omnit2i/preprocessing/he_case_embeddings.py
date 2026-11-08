#!/usr/bin/env python
"""Extract case-level mean Prov-GigaPath embeddings from pre-sampled H&E patches."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset





class PatchDataset(Dataset):
    def __init__(self, table: pd.DataFrame, image_transform) -> None:
        self.table = table.reset_index(drop=True)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        image = Image.open(row.patch_path).convert("RGB")
        return self.image_transform(image), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-table", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-patch-tissue-fraction", type=float, default=0.40)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--prefix", default="full_he_gigapath_64patch_mean")
    return parser.parse_args()


def build_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def collate_batch(batch):
    images, indices = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(indices, dtype=torch.long)


def load_gigapath_tile_encoder(device: torch.device):
    import timm

    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model.eval()
    model.to(device)
    return model


def normalize_output(output):
    if isinstance(output, dict):
        for key in ("features", "embedding", "embeddings", "x"):
            if key in output:
                return output[key]
        raise ValueError(f"Unsupported model output dict keys: {list(output)}")
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def embedding_columns(width: int) -> list[str]:
    return [f"he_emb_{i:04d}" for i in range(width)]


def main() -> None:
    from tqdm import tqdm

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.patch_table, sep="\t")
    table = table[table["status"].eq("ok") & table["patch_path"].astype(str).ne("")].copy()
    if args.min_patch_tissue_fraction is not None:
        table = table[table["patch_tissue_fraction"] >= args.min_patch_tissue_fraction].copy()
    table["patch_index"] = pd.to_numeric(table["patch_index"], errors="coerce")
    table = table.sort_values(["case_id", "patch_index"]).reset_index(drop=True)
    if args.limit_cases is not None:
        keep_cases = table["case_id"].drop_duplicates().head(args.limit_cases)
        table = table[table["case_id"].isin(keep_cases)].copy()
    if table.empty:
        raise ValueError("No patches left after filtering.")

    missing = [path for path in table["patch_path"].head(50) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing patch files, first examples: {missing[:3]}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available inside this job.")
    model = load_gigapath_tile_encoder(device)
    loader = DataLoader(
        PatchDataset(table, build_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="GigaPath case mean"):
            images = images.to(device, non_blocking=True)
            features = normalize_output(model(images)).detach().cpu().numpy().astype(np.float32)
            batch_table = table.iloc[indices.numpy()]
            for case_id, vector in zip(batch_table["case_id"].astype(str), features):
                if case_id not in sums:
                    sums[case_id] = np.zeros_like(vector, dtype=np.float64)
                sums[case_id] += vector.astype(np.float64)
                counts[case_id] += 1

    rows = []
    emb_dim = len(next(iter(sums.values())))
    emb_cols = embedding_columns(emb_dim)
    for case_id in sorted(sums):
        mean = (sums[case_id] / counts[case_id]).astype(np.float32)
        row = {"case_id": case_id, "n_patches": int(counts[case_id])}
        row.update({col: float(val) for col, val in zip(emb_cols, mean)})
        rows.append(row)

    out_path = args.outdir / f"{args.prefix}_case_mean_embeddings.tsv"
    manifest_path = args.outdir / f"{args.prefix}_run_manifest.json"
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    manifest = {
        "model": "prov-gigapath/prov-gigapath",
        "level": "tile encoder mean-pooled to case",
        "patch_table": str(args.patch_table),
        "min_patch_tissue_fraction": args.min_patch_tissue_fraction,
        "n_input_patches": int(len(table)),
        "n_cases": int(len(rows)),
        "embedding_dim": int(emb_dim),
        "device": str(device),
        "output": str(out_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
