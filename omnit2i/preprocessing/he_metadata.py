#!/usr/bin/env python
"""Build metadata-only TCGA H&E/RNA/OS matching tables and download manifests.

This script does not download SVS images. It queries GDC metadata for open TCGA
SVS slide images, matches them to the existing pan-cancer RNA+OS cohort, and
chooses one representative tumor slide per case for staged downloads.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--survival", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--limits", default="100,500,1000,2000,5000")
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def gdc_filters() -> dict[str, Any]:
    return {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.program.name", "value": "TCGA"}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ],
    }


def first_nested(hit: dict[str, Any]) -> dict[str, Any]:
    case = (hit.get("cases") or [{}])[0]
    sample = (case.get("samples") or [{}])[0]
    project = case.get("project") or {}
    return {
        "file_id": hit.get("file_id") or hit.get("id"),
        "id": hit.get("id") or hit.get("file_id"),
        "file_name": hit.get("file_name"),
        "md5sum": hit.get("md5sum"),
        "file_size": hit.get("file_size"),
        "state": hit.get("state"),
        "access": hit.get("access"),
        "data_type": hit.get("data_type"),
        "data_format": hit.get("data_format"),
        "experimental_strategy": hit.get("experimental_strategy"),
        "case_uuid": case.get("case_id"),
        "case_id": case.get("submitter_id"),
        "project_id": project.get("project_id"),
        "sample_type": sample.get("sample_type"),
        "tissue_type": sample.get("tissue_type"),
    }


def fetch_gdc_slide_metadata(out_path: Path, page_size: int, sleep_s: float) -> pd.DataFrame:
    url = "https://api.gdc.cancer.gov/files"
    fields = ",".join(
        [
            "file_id",
            "file_name",
            "md5sum",
            "file_size",
            "state",
            "access",
            "data_type",
            "data_format",
            "experimental_strategy",
            "cases.case_id",
            "cases.submitter_id",
            "cases.project.project_id",
            "cases.samples.sample_type",
            "cases.samples.tissue_type",
        ]
    )
    records = []
    offset = 0
    total = None
    while total is None or offset < total:
        payload = {
            "filters": gdc_filters(),
            "fields": fields,
            "format": "JSON",
            "size": page_size,
            "from": offset,
            "sort": "cases.submitter_id:asc,file_size:asc",
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()["data"]
        total = int(data["pagination"]["total"])
        hits = data["hits"]
        records.extend(first_nested(hit) for hit in hits)
        print(f"Fetched {len(records):,}/{total:,} GDC slide files")
        offset += page_size
        time.sleep(sleep_s)

    df = pd.DataFrame(records)
    df["file_size"] = pd.to_numeric(df["file_size"], errors="coerce").astype("Int64")
    df.to_csv(out_path, sep="\t", index=False)
    return df


def read_rna_os_cohort(expr_path: Path, survival_path: Path) -> pd.DataFrame:

    with gzip.open(expr_path, "rt") as handle:
        rna_samples = set(handle.readline().rstrip("\n").split("\t")[1:])

    survival = pd.read_csv(survival_path, sep="\t", dtype=str)
    survival = survival.rename(columns={"_PATIENT": "case_id", "cancer type abbreviation": "cancer_type"})
    survival["OS"] = pd.to_numeric(survival["OS"], errors="coerce")
    survival["OS.time"] = pd.to_numeric(survival["OS.time"], errors="coerce")
    cohort = survival[
        survival["sample"].isin(rna_samples)
        & survival["sample"].str.endswith("-01", na=False)
        & survival["OS"].isin([0, 1])
        & survival["OS.time"].notna()
    ].copy()
    cohort = cohort.drop_duplicates("case_id", keep="first")
    return cohort[["case_id", "sample", "cancer_type", "OS", "OS.time"]].sort_values("case_id")


def slide_rank(row: pd.Series) -> tuple:
    strategy = str(row.get("experimental_strategy") or "")
    file_name = str(row.get("file_name") or "")
    sample_type = str(row.get("sample_type") or "")
    tissue_type = str(row.get("tissue_type") or "")

    tumor_rank = 0 if tissue_type == "Tumor" else 1
    primary_rank = 0 if sample_type == "Primary Tumor" else 1
    strategy_rank = 0 if strategy == "Diagnostic Slide" else 1 if strategy == "Tissue Slide" else 2
    filename_rank = 0 if "-DX" in file_name else 1 if "-TS" in file_name else 2
    size = int(row["file_size"]) if pd.notna(row["file_size"]) else 10**18
    return (tumor_rank, primary_rank, strategy_rank, filename_rank, size, file_name)


def choose_one_slide_per_case(slides: pd.DataFrame) -> pd.DataFrame:
    ranked = slides.copy()
    ranked["_rank"] = ranked.apply(slide_rank, axis=1)
    idx = ranked.sort_values(["case_id", "_rank"]).groupby("case_id", sort=True).head(1).index
    chosen = ranked.loc[idx].drop(columns=["_rank"]).sort_values("case_id").reset_index(drop=True)
    return chosen


def write_gdc_manifest(df: pd.DataFrame, path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "id": df["file_id"],
            "filename": df["file_name"],
            "md5": df["md5sum"],
            "size": df["file_size"],
            "state": df["state"],
        }
    )
    manifest.to_csv(path, sep="\t", index=False)


def human_tb(bytes_value: float) -> float:
    return bytes_value / 1024**4


def build_download_plans(chosen: pd.DataFrame, limits: list[int], outdir: Path) -> pd.DataFrame:
    rows = []
    ordered = chosen.sort_values(["project_id", "case_id"]).reset_index(drop=True)
    for limit in limits:
        subset = ordered.head(min(limit, len(ordered))).copy()
        manifest_path = outdir / f"tcga_pan_he_one_slide_per_case_first{len(subset)}_manifest.tsv"
        write_gdc_manifest(subset, manifest_path)
        rows.append(
            {
                "plan": f"first_{len(subset)}",
                "n_cases": len(subset),
                "n_files": len(subset),
                "total_size_gib": subset["file_size"].sum() / 1024**3,
                "total_size_tib": human_tb(subset["file_size"].sum()),
                "manifest": str(manifest_path),
            }
        )
    all_manifest = outdir / "tcga_pan_he_one_slide_per_case_all_matched_manifest.tsv"
    write_gdc_manifest(ordered, all_manifest)
    rows.append(
        {
            "plan": "all_matched",
            "n_cases": len(ordered),
            "n_files": len(ordered),
            "total_size_gib": ordered["file_size"].sum() / 1024**3,
            "total_size_tib": human_tb(ordered["file_size"].sum()),
            "manifest": str(all_manifest),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.outdir / "tcga_open_svs_slide_metadata.tsv"
    if metadata_path.exists() and not args.force_refresh:
        slides = pd.read_csv(metadata_path, sep="\t")
    else:
        slides = fetch_gdc_slide_metadata(metadata_path, args.page_size, args.sleep)

    rna_os = read_rna_os_cohort(args.expression, args.survival)
    matched = slides.merge(rna_os, on="case_id", how="inner", suffixes=("", "_rna_os"))
    tumor = matched[
        matched["tissue_type"].eq("Tumor") & matched["sample_type"].isin(["Primary Tumor", "Recurrent Tumor", "Metastatic"])
    ].copy()
    chosen = choose_one_slide_per_case(tumor)

    matched_path = args.outdir / "tcga_open_svs_rna_os_matched_all_slide_files.tsv"
    chosen_path = args.outdir / "tcga_open_svs_rna_os_one_slide_per_case.tsv"
    rna_os_path = args.outdir / "tcga_pancan_rna_os_valid_cases.tsv"
    matched.to_csv(matched_path, sep="\t", index=False)
    chosen.to_csv(chosen_path, sep="\t", index=False)
    rna_os.to_csv(rna_os_path, sep="\t", index=False)

    limits = [int(x) for x in args.limits.split(",") if x]
    plans = build_download_plans(chosen, limits, args.outdir)
    plan_path = args.outdir / "tcga_he_download_size_plans.tsv"
    plans.to_csv(plan_path, sep="\t", index=False)

    by_project = (
        chosen.groupby("project_id", as_index=False)
        .agg(n_cases=("case_id", "nunique"), total_size_gib=("file_size", lambda x: x.sum() / 1024**3))
        .sort_values("n_cases", ascending=False)
    )
    by_project.to_csv(args.outdir / "tcga_he_one_slide_per_case_by_project.tsv", sep="\t", index=False)

    summary = {
        "tcga_open_svs_files": int(len(slides)),
        "tcga_open_svs_cases": int(slides["case_id"].nunique()),
        "rna_os_valid_cases": int(len(rna_os)),
        "rna_os_matched_slide_files": int(len(matched)),
        "rna_os_matched_cases_any_slide": int(matched["case_id"].nunique()),
        "rna_os_matched_tumor_cases": int(tumor["case_id"].nunique()),
        "one_slide_per_case_cases": int(len(chosen)),
        "one_slide_per_case_total_tib": float(human_tb(chosen["file_size"].sum())),
        "outputs": {
            "metadata": str(metadata_path),
            "matched_all_slides": str(matched_path),
            "one_slide_per_case": str(chosen_path),
            "download_plans": str(plan_path),
        },
    }
    summary_path = args.outdir / "tcga_he_rna_os_match_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nDownload plans:")
    print(plans.to_string(index=False))


if __name__ == "__main__":
    main()
