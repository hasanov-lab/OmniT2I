#!/usr/bin/env python3
"""Sample patch bags from additional TCGA H&E slides without filename collisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-table", type=Path, required=True)
    parser.add_argument("--svs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument("--part-id", required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patches-per-slide", type=int, default=64)
    parser.add_argument("--mask-size", type=int, default=1600)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.45)
    parser.add_argument("--max-attempts-factor", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_slide_seed(seed: int, file_id: str) -> int:
    digest = hashlib.blake2b(file_id.encode("utf-8"), digest_size=8).digest()
    return (seed + int.from_bytes(digest, "little")) % (2**32)


def tissue_mask_from_rgb(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    hsv_arr = np.asarray(hsv)
    rgb_arr = np.asarray(rgb)
    return (
        (hsv_arr[:, :, 1] > 20)
        & (hsv_arr[:, :, 2] < 245)
        & (hsv_arr[:, :, 2] > 25)
        & (rgb_arr.std(axis=2) > 4)
    )


def region_fraction(
    mask: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> float:
    height, width = mask.shape
    x0, x1 = max(0, min(width, x0)), max(0, min(width, x1))
    y0, y1 = max(0, min(height, y0)), max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def patch_stats(image: Image.Image) -> tuple[float, float, float]:
    mask = tissue_mask_from_rgb(image)
    stat = ImageStat.Stat(image.convert("RGB"))
    brightness = sum(stat.mean) / len(stat.mean)
    dynamic_range = sum(high - low for low, high in stat.extrema) / len(stat.extrema)
    return float(mask.mean()), float(brightness), float(dynamic_range)


def sample_locations(
    mask: np.ndarray,
    slide_width: int,
    slide_height: int,
    patch_size: int,
    n_patches: int,
    min_tissue_fraction: float,
    max_attempts: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, float]]:
    mask_height, mask_width = mask.shape
    tissue_y, tissue_x = np.where(mask)
    if len(tissue_x) == 0:
        return []
    scale_x, scale_y = mask_width / slide_width, mask_height / slide_height
    patch_width_mask = max(1, int(round(patch_size * scale_x)))
    patch_height_mask = max(1, int(round(patch_size * scale_y)))
    selected: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for _ in range(max_attempts):
        if len(selected) >= n_patches:
            break
        index = int(rng.integers(0, len(tissue_x)))
        x = int(round(tissue_x[index] / scale_x - patch_size / 2))
        y = int(round(tissue_y[index] / scale_y - patch_size / 2))
        x = max(0, min(slide_width - patch_size, x))
        y = max(0, min(slide_height - patch_size, y))
        key = (x // patch_size, y // patch_size)
        if key in seen:
            continue
        mask_x, mask_y = int(round(x * scale_x)), int(round(y * scale_y))
        fraction = region_fraction(
            mask,
            mask_x,
            mask_y,
            mask_x + patch_width_mask,
            mask_y + patch_height_mask,
        )
        if fraction < min_tissue_fraction:
            continue
        seen.add(key)
        selected.append((x, y, fraction))
    return selected


def locate_svs(root: Path, file_id: str, file_name: str) -> Path:
    candidates = list(root.glob(f"*/{file_id}/{file_name}"))
    if len(candidates) != 1:
        candidates = list(root.rglob(file_name))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one SVS for {file_id}/{file_name}, found {len(candidates)}"
        )
    return candidates[0]


def main() -> None:
    from tiffslide import TiffSlide

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(args.target_table, sep="\t")
    target = target.loc[target["already_present"].astype(int).eq(0)].copy()
    target = target.sort_values(["case_id", "file_id"]).reset_index(drop=True)
    target = target.iloc[args.start_index : args.end_index].copy()

    patch_rows: list[dict[str, object]] = []
    slide_rows: list[dict[str, object]] = []
    for index, row in target.reset_index(drop=True).iterrows():
        case_id, file_id = str(row["case_id"]), str(row["file_id"])
        file_name = str(row["file_name"])
        svs_path: Path | None = None
        n_saved, status, error = 0, "ok", ""
        try:
            svs_path = locate_svs(args.svs_root, file_id, file_name)
            slide_dir = args.outdir / case_id / file_id
            slide_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(stable_slide_seed(args.seed, file_id))
            with TiffSlide(str(svs_path)) as slide:
                slide_width, slide_height = slide.dimensions
                thumb = slide.get_thumbnail((args.mask_size, args.mask_size)).convert(
                    "RGB"
                )
                mask = tissue_mask_from_rgb(thumb)
                locations = sample_locations(
                    mask,
                    slide_width,
                    slide_height,
                    args.patch_size,
                    args.patches_per_slide,
                    args.min_tissue_fraction,
                    args.patches_per_slide * args.max_attempts_factor,
                    rng,
                )
                tissue_y, tissue_x = np.where(mask)
                scale_x, scale_y = (
                    mask.shape[1] / slide_width,
                    mask.shape[0] / slide_height,
                )
                while len(locations) < args.patches_per_slide and len(tissue_x) > 0:
                    sampled = int(rng.integers(0, len(tissue_x)))
                    x = int(round(tissue_x[sampled] / scale_x - args.patch_size / 2))
                    y = int(round(tissue_y[sampled] / scale_y - args.patch_size / 2))
                    x = max(0, min(slide_width - args.patch_size, x))
                    y = max(0, min(slide_height - args.patch_size, y))
                    locations.append((x, y, 0.0))
                for patch_index, (x, y, coarse_fraction) in enumerate(
                    locations[: args.patches_per_slide]
                ):
                    patch_path = (
                        slide_dir
                        / f"{file_id}_{patch_index:03d}_x{x}_y{y}"
                        f"_ps{args.patch_size}_im{args.image_size}.jpg"
                    )
                    if args.force or not patch_path.exists():
                        patch = (
                            slide.read_region(
                                (x, y), 0, (args.patch_size, args.patch_size)
                            )
                            .convert("RGB")
                            .resize(
                                (args.image_size, args.image_size),
                                Image.Resampling.BILINEAR,
                            )
                        )
                        patch.save(patch_path, quality=90)
                    else:
                        patch = Image.open(patch_path).convert("RGB")
                    tissue_fraction, brightness, dynamic_range = patch_stats(patch)
                    patch_rows.append(
                        {
                            "case_id": case_id,
                            "file_id": file_id,
                            "file_name": file_name,
                            "cancer_type": row["cancer_type"],
                            "OS": row["OS"],
                            "OS.time": row["OS.time"],
                            "svs_path": str(svs_path),
                            "patch_path": str(patch_path),
                            "patch_index": patch_index,
                            "x": x,
                            "y": y,
                            "patch_size": args.patch_size,
                            "image_size": args.image_size,
                            "coarse_tissue_fraction": coarse_fraction,
                            "patch_tissue_fraction": tissue_fraction,
                            "mean_brightness": brightness,
                            "dynamic_range": dynamic_range,
                            "status": "ok",
                            "error": "",
                        }
                    )
                    n_saved += 1
            if n_saved < args.patches_per_slide:
                status = "partial"
        except Exception as exc:  # noqa: BLE001
            status, error = "failed", repr(exc)
        slide_rows.append(
            {
                "case_id": case_id,
                "file_id": file_id,
                "file_name": file_name,
                "cancer_type": row["cancer_type"],
                "OS": row["OS"],
                "OS.time": row["OS.time"],
                "svs_path": str(svs_path) if svs_path else "",
                "n_patches": n_saved,
                "status": status,
                "error": error,
            }
        )
        if (index + 1) % 5 == 0 or index + 1 == len(target):
            print(f"processed {index + 1}/{len(target)} slides", flush=True)

    prefix = f"extra_he_64patch_part{args.part_id}"
    patch_path = args.outdir / f"{prefix}_patch_table.tsv"
    slide_path = args.outdir / f"{prefix}_slide_table.tsv"
    pd.DataFrame(patch_rows).to_csv(patch_path, sep="\t", index=False)
    pd.DataFrame(slide_rows).to_csv(slide_path, sep="\t", index=False)
    report = {
        "part_id": args.part_id,
        "slides_requested": int(len(target)),
        "slides_ok_or_partial": int(
            sum(
                row["status"] in {"ok", "partial"} and row["n_patches"] > 0
                for row in slide_rows
            )
        ),
        "slides_failed": int(sum(row["status"] == "failed" for row in slide_rows)),
        "patch_rows": int(len(patch_rows)),
        "patch_table": str(patch_path),
        "slide_table": str(slide_path),
        "failed_examples": [
            row for row in slide_rows if row["status"] == "failed"
        ][:10],
    }
    (args.outdir / f"{prefix}_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if report["slides_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
