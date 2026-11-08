from pathlib import Path

import pandas as pd

from omnit2i.preprocessing.methylation_map import (
    Promoter,
    deduplicate_promoters,
    map_probes_to_promoters,
)


def test_deduplicate_promoters_collapses_same_gene_tss() -> None:
    first = Promoter("chr1", 100, 201, "ENSG1", "A", "+", "ENST1", 150)
    duplicate = Promoter("chr1", 100, 201, "ENSG1", "A", "+", "ENST2", 150)
    distinct = Promoter("chr1", 300, 401, "ENSG1", "A", "+", "ENST3", 350)
    observed = deduplicate_promoters([duplicate, distinct, first])
    assert observed == [first, distinct]


def test_map_probes_uses_half_open_intervals_and_marks_ambiguity() -> None:
    promoters = [
        Promoter("chr1", 100, 201, "ENSG1", "A", "+", "ENST1", 150),
        Promoter("chr1", 180, 251, "ENSG2", "B", "-", "ENST2", 220),
    ]
    probes = pd.DataFrame(
        {
            "probe_id": ["left", "overlap", "right_edge", "outside"],
            "chrom": ["chr1"] * 4,
            "chrom_start": [100, 190, 200, 251],
            "chrom_end": [101, 191, 201, 252],
            "probe_midpoint": [100, 190, 200, 251],
        }
    )
    mapped = map_probes_to_promoters(probes, promoters)

    left = mapped.loc[mapped["probe_id"] == "left"]
    assert left["gene_id"].tolist() == ["ENSG1"]
    overlap = mapped.loc[mapped["probe_id"] == "overlap"]
    assert overlap["gene_id"].tolist() == ["ENSG1", "ENSG2"]
    assert overlap["ambiguous_gene_mapping"].all()
    right_edge = mapped.loc[mapped["probe_id"] == "right_edge"]
    assert set(right_edge["gene_id"]) == {"ENSG1", "ENSG2"}
    assert "outside" not in set(mapped["probe_id"])
