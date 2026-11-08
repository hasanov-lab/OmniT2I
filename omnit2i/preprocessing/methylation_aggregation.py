"""Stream TCGA 450K beta values into a promoter-gene matrix.

Technical duplicate sample columns are merged probe-wise by their available
mean. Only primary-tumor patients with valid overall survival from the audited
cohort are retained. Promoter probes with cross-gene ambiguity are excluded by
the upstream mapping step.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


MISSING_REPLACEMENTS = (
    ("NA", "nan"),
    ("null", "nan"),
    ("NULL", "nan"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-promoter-probes", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=25000)
    return parser.parse_args()


def read_matrix_header(path: Path) -> tuple[str, list[str]]:
    with gzip.open(path, "rt") as handle:
        fields = handle.readline().rstrip("\r\n").split("\t")
    if len(fields) < 2:
        raise ValueError("Methylation matrix header has fewer than two columns")
    return fields[0], fields[1:]


def build_sample_positions(
    matrix_samples: list[str],
    cohort_samples: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(matrix_samples):
        positions[sample].append(index)
    missing = [sample for sample in cohort_samples if sample not in positions]
    if missing:
        raise ValueError(f"{len(missing)} cohort samples are absent from matrix header")
    too_many = {
        sample: indices
        for sample, indices in positions.items()
        if sample in set(cohort_samples) and len(indices) > 2
    }
    if too_many:
        raise ValueError(f"Unexpected >2 technical columns: {too_many}")

    primary = np.array([positions[sample][0] for sample in cohort_samples], dtype=int)
    secondary = np.array(
        [
            positions[sample][1] if len(positions[sample]) == 2 else -1
            for sample in cohort_samples
        ],
        dtype=int,
    )
    duplicate_counts = {
        sample: len(positions[sample])
        for sample in cohort_samples
        if len(positions[sample]) > 1
    }
    return primary, secondary, duplicate_counts


def parse_beta_values(value_text: str, expected_count: int) -> np.ndarray:
    normalized = value_text
    for source, replacement in MISSING_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    # np.fromstring does not reliably preserve empty delimited fields. Pad the
    # row, replace every empty field explicitly, then remove the padding.
    normalized = f"\t{normalized}\t"
    while "\t\t" in normalized:
        normalized = normalized.replace("\t\t", "\tnan\t")
    normalized = normalized[1:-1]
    values = np.fromstring(normalized, dtype=np.float32, sep="\t")
    if len(values) != expected_count:
        raise ValueError(
            f"Parsed {len(values)} beta values; expected {expected_count}"
        )
    return values


def merge_technical_columns(
    values: np.ndarray,
    primary_positions: np.ndarray,
    secondary_positions: np.ndarray,
) -> np.ndarray:
    merged = values[primary_positions].copy()
    duplicate_mask = secondary_positions >= 0
    if not duplicate_mask.any():
        return merged
    first = merged[duplicate_mask]
    second = values[secondary_positions[duplicate_mask]]
    first_finite = np.isfinite(first)
    second_finite = np.isfinite(second)
    both = first_finite & second_finite
    first[both] = (first[both] + second[both]) / 2.0
    first[~first_finite & second_finite] = second[~first_finite & second_finite]
    first[~first_finite & ~second_finite] = np.nan
    merged[duplicate_mask] = first
    return merged


def prepare_gene_mapping(
    mapping_path: Path,
    minimum_promoter_probes: int,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    if minimum_promoter_probes < 1:
        raise ValueError("--minimum-promoter-probes must be at least one")
    mapping = pd.read_csv(mapping_path, sep="\t", dtype={"probe_id": str})
    if mapping["ambiguous_gene_mapping"].astype(bool).any():
        raise ValueError("Main mapping contains ambiguous cross-gene probes")
    if mapping["probe_id"].duplicated().any():
        raise ValueError("Main mapping must assign each probe to exactly one gene")
    gene_profile = (
        mapping.groupby(["gene_id", "gene_name"], as_index=False)
        .agg(n_promoter_probes=("probe_id", "nunique"))
    )
    gene_profile = gene_profile.loc[
        gene_profile["n_promoter_probes"] >= minimum_promoter_probes
    ].sort_values(["gene_id", "gene_name"]).reset_index(drop=True)
    selected_gene_ids = set(gene_profile["gene_id"])
    mapping = mapping.loc[mapping["gene_id"].isin(selected_gene_ids)].copy()
    gene_index = {
        gene_id: index for index, gene_id in enumerate(gene_profile["gene_id"])
    }
    probe_to_gene = {
        probe_id: gene_index[gene_id]
        for probe_id, gene_id in zip(mapping["probe_id"], mapping["gene_id"])
    }
    return mapping, probe_to_gene, gene_profile


def stream_aggregate(
    matrix_path: Path,
    matrix_sample_count: int,
    primary_positions: np.ndarray,
    secondary_positions: np.ndarray,
    probe_to_gene: dict[str, int],
    n_genes: int,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    n_samples = len(primary_positions)
    sums = np.zeros((n_genes, n_samples), dtype=np.float32)
    counts = np.zeros((n_genes, n_samples), dtype=np.uint16)
    n_rows = 0
    n_selected = 0
    selected_seen: set[str] = set()

    with gzip.open(matrix_path, "rt") as handle:
        next(handle)
        for line in handle:
            n_rows += 1
            separator = line.find("\t")
            if separator < 0:
                raise ValueError(f"Matrix row {n_rows + 1} has no tab separator")
            probe_id = line[:separator]
            gene_index = probe_to_gene.get(probe_id)
            if gene_index is None:
                continue
            if probe_id in selected_seen:
                raise ValueError(f"Duplicate selected probe row: {probe_id}")
            selected_seen.add(probe_id)
            values = parse_beta_values(
                line[separator + 1 :].rstrip("\r\n"),
                matrix_sample_count,
            )
            merged = merge_technical_columns(
                values, primary_positions, secondary_positions
            )
            finite = np.isfinite(merged)
            if finite.any():
                sums[gene_index, finite] += merged[finite]
                counts[gene_index, finite] += 1
            n_selected += 1
            if progress_every > 0 and n_selected % progress_every == 0:
                print(
                    f"processed selected probes: {n_selected:,} "
                    f"(matrix rows scanned: {n_rows:,})",
                    flush=True,
                )

    missing_selected = set(probe_to_gene) - selected_seen
    if missing_selected:
        examples = sorted(missing_selected)[:10]
        raise ValueError(
            f"{len(missing_selected)} selected probes absent from matrix; "
            f"examples={examples}"
        )
    beta = sums
    np.divide(beta, counts, out=beta, where=counts > 0)
    beta[counts == 0] = np.nan
    return beta, counts, {
        "n_matrix_probe_rows_scanned": n_rows,
        "n_selected_probe_rows_aggregated": n_selected,
    }


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort, sep="\t")
    if cohort["patient"].duplicated().any():
        raise ValueError("Primary OS cohort contains duplicate patients")
    feature_column, matrix_samples = read_matrix_header(args.matrix)
    primary_positions, secondary_positions, duplicate_counts = (
        build_sample_positions(matrix_samples, cohort["sample"].astype(str).tolist())
    )
    mapping, probe_to_gene, gene_profile = prepare_gene_mapping(
        args.mapping, args.minimum_promoter_probes
    )
    beta, counts, stream_summary = stream_aggregate(
        args.matrix,
        len(matrix_samples),
        primary_positions,
        secondary_positions,
        probe_to_gene,
        len(gene_profile),
        args.progress_every,
    )

    finite = np.isfinite(beta)
    finite_values = beta[finite]
    n_out_of_range = int(
        ((finite_values < 0.0) | (finite_values > 1.0)).sum()
    )
    if n_out_of_range:
        raise ValueError(f"{n_out_of_range} aggregated beta values fall outside [0,1]")

    gene_profile["n_observed_matrix_probes"] = gene_profile[
        "n_promoter_probes"
    ]
    gene_profile["maximum_observed_probes_per_patient"] = counts.max(axis=1)
    gene_profile["minimum_observed_probes_per_patient"] = counts.min(axis=1)
    gene_profile["median_observed_probes_per_patient"] = np.median(
        counts, axis=1
    )
    gene_profile["missing_fraction"] = 1.0 - finite.mean(axis=1)
    sample_qc = cohort.copy()
    sample_qc["missing_gene_fraction"] = 1.0 - finite.mean(axis=0)
    sample_qc["median_observed_promoter_probes_per_gene"] = np.median(
        counts, axis=0
    )

    atomic_save_npy(args.output_dir / "methylation_promoter_gene_beta.npy", beta)
    gene_profile.to_csv(
        args.output_dir / "methylation_promoter_gene_features.tsv",
        sep="\t",
        index=False,
    )
    sample_qc.to_csv(
        args.output_dir / "methylation_promoter_gene_samples.tsv",
        sep="\t",
        index=False,
    )
    summary = {
        "feature_column": feature_column,
        "matrix_orientation": "genes x patients",
        "aggregation": "mean beta across unambiguous promoter probes",
        "technical_duplicate_rule": (
            "probe-wise mean when both observed; available value when one observed"
        ),
        "minimum_promoter_probes_per_gene": args.minimum_promoter_probes,
        "n_matrix_sample_columns": len(matrix_samples),
        "n_primary_os_patients": len(cohort),
        "n_primary_samples_with_duplicate_technical_columns": len(duplicate_counts),
        "n_selected_genes": len(gene_profile),
        "n_selected_mapping_probes": len(mapping),
        "n_finite_gene_patient_values": int(finite.sum()),
        "gene_patient_missing_fraction": float(1.0 - finite.mean()),
        "beta_min": float(finite_values.min()),
        "beta_max": float(finite_values.max()),
        "beta_mean": float(finite_values.mean()),
        "beta_standard_deviation": float(finite_values.std()),
        **stream_summary,
    }
    with (args.output_dir / "methylation_promoter_gene_summary.json").open(
        "w"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
