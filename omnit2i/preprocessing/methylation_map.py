"""Build a rigorous hg19 450K probe-to-promoter-gene mapping.

The main mapping uses protein-coding transcript TSSs from GENCODE v19 and a
strand-aware promoter window spanning 1,500 bp upstream to 500 bp downstream.
Probe coordinates come from the UCSC Xena TCGA Legacy 450K probe map.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ATTRIBUTE_PATTERN = re.compile(r'(\S+)\s+"([^"]+)"')
STANDARD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


@dataclass(frozen=True)
class Promoter:
    chrom: str
    start: int
    end: int
    gene_id: str
    gene_name: str
    strand: str
    transcript_id: str
    tss: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--probe-map", type=Path, required=True)
    parser.add_argument("--upstream-bp", type=int, default=1500)
    parser.add_argument("--downstream-bp", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_attributes(field: str) -> dict[str, str]:
    return dict(ATTRIBUTE_PATTERN.findall(field))


def strip_version(identifier: str) -> str:
    return identifier.split(".", maxsplit=1)[0]


def read_protein_coding_promoters(
    gtf_path: Path,
    upstream_bp: int,
    downstream_bp: int,
) -> list[Promoter]:
    if upstream_bp < 0 or downstream_bp < 0:
        raise ValueError("Promoter offsets must be non-negative")
    promoters: list[Promoter] = []
    opener = gzip.open if gtf_path.suffix == ".gz" else open
    with opener(gtf_path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "transcript":
                continue
            chrom, start_text, end_text, strand = (
                fields[0],
                fields[3],
                fields[4],
                fields[6],
            )
            if chrom not in STANDARD_CHROMS or strand not in {"+", "-"}:
                continue
            attributes = parse_attributes(fields[8])
            if attributes.get("gene_type") != "protein_coding":
                continue
            gene_id = attributes.get("gene_id")
            gene_name = attributes.get("gene_name")
            transcript_id = attributes.get("transcript_id")
            if not gene_id or not gene_name or not transcript_id:
                continue

            # GTF is 1-based inclusive. Store all coordinates as 0-based,
            # half-open intervals; tss is a 0-based genomic point.
            gtf_start = int(start_text)
            gtf_end = int(end_text)
            tss = gtf_start - 1 if strand == "+" else gtf_end - 1
            if strand == "+":
                promoter_start = max(0, tss - upstream_bp)
                promoter_end = tss + downstream_bp + 1
            else:
                promoter_start = max(0, tss - downstream_bp)
                promoter_end = tss + upstream_bp + 1
            promoters.append(
                Promoter(
                    chrom=chrom,
                    start=promoter_start,
                    end=promoter_end,
                    gene_id=strip_version(gene_id),
                    gene_name=gene_name,
                    strand=strand,
                    transcript_id=strip_version(transcript_id),
                    tss=tss,
                )
            )
    return promoters


def deduplicate_promoters(promoters: list[Promoter]) -> list[Promoter]:
    """Collapse duplicate transcript TSSs for the same gene without merging loci."""
    unique: dict[tuple[str, int, int, str, str], Promoter] = {}
    for promoter in promoters:
        key = (
            promoter.chrom,
            promoter.start,
            promoter.end,
            promoter.gene_id,
            promoter.strand,
        )
        if (
            key not in unique
            or promoter.transcript_id < unique[key].transcript_id
        ):
            unique[key] = promoter
    return sorted(
        unique.values(),
        key=lambda value: (value.chrom, value.start, value.end, value.gene_id),
    )


def read_probe_map(path: Path) -> pd.DataFrame:
    probes_raw = pd.read_csv(
        path,
        sep="\t",
        usecols=["#id", "chrom", "chromStart", "chromEnd"],
        dtype={"#id": str, "chrom": str, "chromStart": "Int64", "chromEnd": "Int64"},
    ).rename(columns={"#id": "probe_id"})
    n_raw = len(probes_raw)
    n_missing_chrom = int(probes_raw["chrom"].isna().sum())
    probes = probes_raw
    probes = probes.dropna(
        subset=["probe_id", "chrom", "chromStart", "chromEnd"]
    ).copy()
    probes["chrom_start"] = probes["chromStart"].astype(int)
    probes["chrom_end"] = probes["chromEnd"].astype(int)
    invalid = probes["chrom_end"] <= probes["chrom_start"]
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} probe intervals have non-positive width")
    probes["probe_midpoint"] = (
        probes["chrom_start"] + probes["chrom_end"] - 1
    ) // 2
    result = probes[
        ["probe_id", "chrom", "chrom_start", "chrom_end", "probe_midpoint"]
    ].copy()
    result.attrs["n_probe_map_rows_raw"] = n_raw
    result.attrs["n_missing_chromosome"] = n_missing_chrom
    return result


def map_probes_to_promoters(
    probes: pd.DataFrame,
    promoters: list[Promoter],
) -> pd.DataFrame:
    """Point-overlap probes with promoters using an exact chromosome sweep."""
    promoter_by_chrom: dict[str, list[Promoter]] = defaultdict(list)
    for promoter in promoters:
        promoter_by_chrom[promoter.chrom].append(promoter)
    for chrom in promoter_by_chrom:
        promoter_by_chrom[chrom].sort(key=lambda value: (value.start, value.end))

    records: list[dict[str, object]] = []
    for chrom, chrom_probes in probes.groupby("chrom", sort=False):
        intervals = promoter_by_chrom.get(chrom, [])
        interval_index = 0
        active_heap: list[tuple[int, int]] = []
        active: dict[int, Promoter] = {}
        ordered_probes = chrom_probes.sort_values(
            ["probe_midpoint", "probe_id"], kind="stable"
        )
        for probe in ordered_probes.itertuples(index=False):
            position = int(probe.probe_midpoint)
            while (
                interval_index < len(intervals)
                and intervals[interval_index].start <= position
            ):
                promoter = intervals[interval_index]
                active[interval_index] = promoter
                heapq.heappush(active_heap, (promoter.end, interval_index))
                interval_index += 1
            while active_heap and active_heap[0][0] <= position:
                _, expired_index = heapq.heappop(active_heap)
                active.pop(expired_index, None)

            genes: dict[str, dict[str, object]] = {}
            for promoter in active.values():
                if position >= promoter.end:
                    continue
                gene = genes.setdefault(
                    promoter.gene_id,
                    {
                        "gene_id": promoter.gene_id,
                        "gene_name": promoter.gene_name,
                        "strand": promoter.strand,
                        "transcript_ids": set(),
                        "tss_positions": set(),
                    },
                )
                gene["transcript_ids"].add(promoter.transcript_id)
                gene["tss_positions"].add(promoter.tss)

            n_candidate_genes = len(genes)
            for gene_id in sorted(genes):
                gene = genes[gene_id]
                records.append(
                    {
                        "probe_id": probe.probe_id,
                        "chrom": chrom,
                        "chrom_start": int(probe.chrom_start),
                        "chrom_end": int(probe.chrom_end),
                        "probe_midpoint": position,
                        "gene_id": gene["gene_id"],
                        "gene_name": gene["gene_name"],
                        "strand": gene["strand"],
                        "n_candidate_genes": n_candidate_genes,
                        "ambiguous_gene_mapping": n_candidate_genes > 1,
                        "n_overlapping_transcripts": len(gene["transcript_ids"]),
                        "n_overlapping_tss": len(gene["tss_positions"]),
                        "transcript_ids": ",".join(sorted(gene["transcript_ids"])),
                        "tss_positions": ",".join(
                            str(value) for value in sorted(gene["tss_positions"])
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_mapping(
    probes: pd.DataFrame,
    raw_promoters: list[Promoter],
    promoters: list[Promoter],
    mapping: pd.DataFrame,
    upstream_bp: int,
    downstream_bp: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    mapped_probe_ids = set(mapping["probe_id"]) if len(mapping) else set()
    ambiguous_probe_ids = set(
        mapping.loc[mapping["ambiguous_gene_mapping"], "probe_id"]
    )
    main = mapping.loc[~mapping["ambiguous_gene_mapping"]].copy()
    gene_counts = (
        main.groupby(["gene_id", "gene_name"], as_index=False)
        .agg(n_unique_promoter_probes=("probe_id", "nunique"))
        .sort_values(
            ["n_unique_promoter_probes", "gene_name"], ascending=[False, True]
        )
    )
    standard_probe_count = int(probes["chrom"].isin(STANDARD_CHROMS).sum())
    summary = {
        "coordinate_system": "hg19 / GRCh37",
        "annotation": "GENCODE v19",
        "gene_type": "protein_coding",
        "promoter_definition": (
            f"TSS -{upstream_bp} bp to +{downstream_bp} bp, strand-aware"
        ),
        "probe_position_rule": "0-based midpoint of Xena half-open probe interval",
        "ambiguous_mapping_rule": "exclude probes overlapping promoters of >1 gene",
        "n_probe_map_rows_raw": int(probes.attrs["n_probe_map_rows_raw"]),
        "n_probe_map_rows_with_coordinates": int(len(probes)),
        "n_probe_map_rows_missing_chromosome": int(
            probes.attrs["n_missing_chromosome"]
        ),
        "n_standard_chromosome_probes": standard_probe_count,
        "n_transcript_promoters_raw": int(len(raw_promoters)),
        "n_unique_gene_tss_promoters": int(len(promoters)),
        "n_protein_coding_genes_in_annotation": int(
            len({value.gene_id for value in promoters})
        ),
        "n_promoter_mapped_probes": int(len(mapped_probe_ids)),
        "promoter_probe_fraction_of_standard_chromosome_probes": (
            len(mapped_probe_ids) / standard_probe_count
            if standard_probe_count
            else None
        ),
        "n_ambiguous_cross_gene_probes": int(len(ambiguous_probe_ids)),
        "n_unambiguous_promoter_probes": int(
            len(mapped_probe_ids - ambiguous_probe_ids)
        ),
        "n_genes_with_unambiguous_promoter_probe": int(len(gene_counts)),
        "n_genes_with_at_least_3_unambiguous_promoter_probes": int(
            (gene_counts["n_unique_promoter_probes"] >= 3).sum()
        ),
        "n_genes_with_at_least_5_unambiguous_promoter_probes": int(
            (gene_counts["n_unique_promoter_probes"] >= 5).sum()
        ),
        "promoter_probes_per_gene_quantiles": {
            str(key): float(value)
            for key, value in gene_counts["n_unique_promoter_probes"]
            .quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])
            .items()
        },
    }
    return summary, gene_counts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_promoters = read_protein_coding_promoters(
        args.gtf, args.upstream_bp, args.downstream_bp
    )
    promoters = deduplicate_promoters(raw_promoters)
    probes = read_probe_map(args.probe_map)
    mapping = map_probes_to_promoters(probes, promoters)
    summary, gene_counts = summarize_mapping(
        probes,
        raw_promoters,
        promoters,
        mapping,
        args.upstream_bp,
        args.downstream_bp,
    )

    promoter_table = pd.DataFrame(
        [
            {
                "chrom": value.chrom,
                "promoter_start": value.start,
                "promoter_end": value.end,
                "gene_id": value.gene_id,
                "gene_name": value.gene_name,
                "strand": value.strand,
                "transcript_id_representative": value.transcript_id,
                "tss": value.tss,
            }
            for value in promoters
        ]
    )
    promoter_table.to_csv(
        args.output_dir / "gencode_v19_protein_coding_promoters.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    mapping.to_csv(
        args.output_dir / "methylation450k_probe_promoter_gene_map.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    mapping.loc[~mapping["ambiguous_gene_mapping"]].to_csv(
        args.output_dir / "methylation450k_probe_promoter_gene_map_main.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    gene_counts.to_csv(
        args.output_dir / "methylation450k_promoter_probe_counts_by_gene.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    with (args.output_dir / "methylation450k_promoter_mapping_summary.json").open(
        "w"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
