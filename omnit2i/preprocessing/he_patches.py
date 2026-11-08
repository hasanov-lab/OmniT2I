#!/usr/bin/env python
"""Build H&E patch bags per TCGA case for MIL experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageStat





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--svs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="full_he_64patch_bag")
    parser.add_argument("--report-name", default="14_full_he_64patch_bags.md")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patches-per-case", type=int, default=64)
    parser.add_argument("--mask-size", type=int, default=1600)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.45)
    parser.add_argument("--max-attempts-factor", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def tissue_mask_from_rgb(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    hsv_arr = np.asarray(hsv)
    rgb_arr = np.asarray(rgb)
    return (hsv_arr[:, :, 1] > 20) & (hsv_arr[:, :, 2] < 245) & (hsv_arr[:, :, 2] > 25) & (rgb_arr.std(axis=2) > 4)


def region_fraction(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    h, w = mask.shape
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def patch_stats(image: Image.Image) -> tuple[float, float, float]:
    mask = tissue_mask_from_rgb(image)
    stat = ImageStat.Stat(image.convert("RGB"))
    brightness = sum(stat.mean) / len(stat.mean)
    dynamic_range = sum(hi - lo for lo, hi in stat.extrema) / len(stat.extrema)
    return float(mask.mean()), float(brightness), float(dynamic_range)


def find_svs_paths(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.rglob("*.svs")}


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
    mask_h, mask_w = mask.shape
    tissue_y, tissue_x = np.where(mask)
    if len(tissue_x) == 0:
        return []
    scale_x = mask_w / slide_width
    scale_y = mask_h / slide_height
    patch_w_mask = max(1, int(round(patch_size * scale_x)))
    patch_h_mask = max(1, int(round(patch_size * scale_y)))
    selected: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for _ in range(max_attempts):
        if len(selected) >= n_patches:
            break
        idx = int(rng.integers(0, len(tissue_x)))
        mx, my = int(tissue_x[idx]), int(tissue_y[idx])
        x = int(round(mx / scale_x - patch_size / 2))
        y = int(round(my / scale_y - patch_size / 2))
        x = max(0, min(slide_width - patch_size, x))
        y = max(0, min(slide_height - patch_size, y))
        key = (x // patch_size, y // patch_size)
        if key in seen:
            continue
        mx0 = int(round(x * scale_x))
        my0 = int(round(y * scale_y))
        frac = region_fraction(mask, mx0, my0, mx0 + patch_w_mask, my0 + patch_h_mask)
        if frac < min_tissue_fraction:
            continue
        seen.add(key)
        selected.append((x, y, frac))
    return selected


def main() -> None:
    from tiffslide import TiffSlide

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table, sep="\t")
    df = df.iloc[args.start_index : args.end_index].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()
    svs_by_name = find_svs_paths(args.svs_root)
    rng = np.random.default_rng(args.seed)
    rows = []
    case_rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        case_id = str(row["case_id"])
        file_name = str(row["file_name"] if "file_name" in row else row["filename"])
        svs_path = svs_by_name.get(file_name)
        status = "ok"
        error = ""
        n_saved = 0
        try:
            if svs_path is None:
                raise FileNotFoundError(file_name)
            case_dir = args.outdir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            with TiffSlide(str(svs_path)) as slide:
                slide_width, slide_height = slide.dimensions
                thumb = slide.get_thumbnail((args.mask_size, args.mask_size)).convert("RGB")
                mask = tissue_mask_from_rgb(thumb)
                locs = sample_locations(
                    mask,
                    slide_width,
                    slide_height,
                    args.patch_size,
                    args.patches_per_case,
                    args.min_tissue_fraction,
                    args.patches_per_case * args.max_attempts_factor,
                    rng,
                )
                tissue_y, tissue_x = np.where(mask)
                scale_x = mask.shape[1] / slide_width
                scale_y = mask.shape[0] / slide_height
                while len(locs) < args.patches_per_case and len(tissue_x) > 0:
                    idx = int(rng.integers(0, len(tissue_x)))
                    x = int(round(tissue_x[idx] / scale_x - args.patch_size / 2))
                    y = int(round(tissue_y[idx] / scale_y - args.patch_size / 2))
                    x = max(0, min(slide_width - args.patch_size, x))
                    y = max(0, min(slide_height - args.patch_size, y))
                    locs.append((x, y, 0.0))
                for patch_index, (x, y, coarse_frac) in enumerate(locs[: args.patches_per_case]):
                    patch_path = case_dir / f"{case_id}_{patch_index:03d}_x{x}_y{y}_ps{args.patch_size}_im{args.image_size}.jpg"
                    if args.force or not patch_path.exists():
                        patch = slide.read_region((x, y), 0, (args.patch_size, args.patch_size)).convert("RGB")
                        patch = patch.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
                        patch.save(patch_path, quality=90)
                    else:
                        patch = Image.open(patch_path).convert("RGB")
                    frac, bright, dyn = patch_stats(patch)
                    rows.append(
                        {
                            "case_id": case_id,
                            "cancer_type": row.get("cancer_type", ""),
                            "OS": row.get("OS", ""),
                            "OS.time": row.get("OS.time", ""),
                            "file_name": file_name,
                            "svs_path": str(svs_path),
                            "patch_path": str(patch_path),
                            "patch_index": patch_index,
                            "x": x,
                            "y": y,
                            "patch_size": args.patch_size,
                            "image_size": args.image_size,
                            "coarse_tissue_fraction": coarse_frac,
                            "patch_tissue_fraction": frac,
                            "mean_brightness": bright,
                            "dynamic_range": dyn,
                            "status": "ok",
                            "error": "",
                        }
                    )
                    n_saved += 1
            if n_saved < args.patches_per_case:
                status = "partial"
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = repr(exc)
        case_rows.append(
            {
                "case_id": case_id,
                "cancer_type": row.get("cancer_type", ""),
                "OS": row.get("OS", ""),
                "OS.time": row.get("OS.time", ""),
                "file_name": file_name,
                "svs_path": str(svs_path) if svs_path is not None else "",
                "n_patches": n_saved,
                "status": status,
                "error": error,
            }
        )
        if (i + 1) % 25 == 0 or i + 1 == len(df):
            print(f"processed {i + 1}/{len(df)} cases", flush=True)
    patch_table = args.outdir / f"{args.prefix}_patch_table.tsv"
    case_table = args.outdir / f"{args.prefix}_case_table.tsv"
    pd.DataFrame(rows).to_csv(patch_table, sep="\t", index=False)
    pd.DataFrame(case_rows).to_csv(case_table, sep="\t", index=False)
    ok_cases = sum(1 for row in case_rows if row["status"] in {"ok", "partial"} and row["n_patches"] > 0)
    failed_cases = [row for row in case_rows if row["status"] == "failed"]
    report_path = args.report_dir / args.report_name
    report = {
        "n_cases_requested": int(len(df)),
        "ok_or_partial_cases": int(ok_cases),
        "failed_cases": int(len(failed_cases)),
        "patches_per_case": args.patches_per_case,
        "patch_size": args.patch_size,
        "image_size": args.image_size,
        "patch_table": str(patch_table),
        "case_table": str(case_table),
        "failed_examples": failed_cases[:10],
    }
    report_path.write_text("# 14 Full TCGA H&E 64-Patch Bags\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
