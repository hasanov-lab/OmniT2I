#!/usr/bin/env python3
"""Extract frozen Prov-GigaPath embeddings and mean-pool within each extra slide."""

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
    def __init__(self, table: pd.DataFrame, transform) -> None:
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        image = Image.open(self.table.iloc[index]["patch_path"]).convert("RGB")
        return self.transform(image), index


def build_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )


def normalize_output(output):
    if isinstance(output, dict):
        for key in ("features", "embedding", "embeddings", "x"):
            if key in output:
                return output[key]
        raise ValueError(f"Unsupported model output keys: {list(output)}")
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def main() -> None:
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-patch-tissue-fraction", type=float, default=0.40)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    table = pd.read_csv(args.patch_table, sep="\t")
    table = table.loc[
        table["status"].eq("ok")
        & table["patch_path"].astype(str).ne("")
        & pd.to_numeric(table["patch_tissue_fraction"], errors="coerce").ge(
            args.min_patch_tissue_fraction
        )
    ].copy()
    table = table.sort_values(["case_id", "file_id", "patch_index"]).reset_index(
        drop=True
    )
    if table.empty:
        raise RuntimeError("No patches remain after filtering")
    missing = [path for path in table["patch_path"].head(100) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing patch files: {missing[:3]}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    import timm

    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model.eval().to(device)
    loader = DataLoader(
        PatchDataset(table, build_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)
    metadata: dict[str, tuple[str, str]] = {}
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="GigaPath extra slides"):
            images = images.to(device, non_blocking=True)
            vectors = (
                normalize_output(model(images))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            batch = table.iloc[indices.numpy()]
            for (_, row), vector in zip(batch.iterrows(), vectors):
                file_id = str(row["file_id"])
                if file_id not in sums:
                    sums[file_id] = np.zeros_like(vector, dtype=np.float64)
                    metadata[file_id] = (str(row["case_id"]), str(row["file_name"]))
                sums[file_id] += vector.astype(np.float64)
                counts[file_id] += 1

    dimension = len(next(iter(sums.values())))
    columns = [f"he_emb_{index:04d}" for index in range(dimension)]
    rows = []
    for file_id in sorted(sums):
        case_id, file_name = metadata[file_id]
        vector = (sums[file_id] / counts[file_id]).astype(np.float32)
        row: dict[str, object] = {
            "case_id": case_id,
            "file_id": file_id,
            "file_name": file_name,
            "n_patches": counts[file_id],
        }
        row.update(dict(zip(columns, vector.tolist())))
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "extra_slide_mean_embeddings.tsv"
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)
    manifest = {
        "model": "prov-gigapath/prov-gigapath",
        "level": "tile encoder mean-pooled within slide",
        "patch_table": str(args.patch_table),
        "min_patch_tissue_fraction": args.min_patch_tissue_fraction,
        "input_patches": int(len(table)),
        "slides": int(len(rows)),
        "patients": int(len({row["case_id"] for row in rows})),
        "embedding_dim": dimension,
        "output": str(output),
    }
    (args.output_dir / "extra_slide_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
