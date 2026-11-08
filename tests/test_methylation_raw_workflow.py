import gzip
import json
import sys

import numpy as np
import pandas as pd

from omnit2i.preprocessing import methylation_aggregation, methylation_map


def test_raw_methylation_commands(tmp_path, monkeypatch):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        'chr1\ttest\ttranscript\t2001\t3000\t.\t+\t.\t'
        'gene_id "ENSG1.1"; gene_name "A"; gene_type "protein_coding"; '
        'transcript_id "ENST1.1";\n'
        'chr2\ttest\ttranscript\t1000\t3000\t.\t-\t.\t'
        'gene_id "ENSG2.2"; gene_name "B"; gene_type "protein_coding"; '
        'transcript_id "ENST2.2";\n'
    )
    probe_map = tmp_path / "probes.tsv"
    probe_map.write_text(
        "#id\tchrom\tchromStart\tchromEnd\n"
        "p1\tchr1\t500\t501\np2\tchr1\t2000\t2001\n"
        "p3\tchr1\t2500\t2501\np4\tchr2\t2499\t2500\n"
        "p5\tchr2\t2999\t3000\np6\tchr2\t4499\t4500\n"
    )
    reference = tmp_path / "reference"
    monkeypatch.setattr(sys, "argv", [
        "omnit2i-map-methylation", "--gtf", str(gtf),
        "--probe-map", str(probe_map), "--output-dir", str(reference),
    ])
    methylation_map.main()
    mapping = reference / "methylation450k_probe_promoter_gene_map_main.tsv.gz"
    mapped = pd.read_csv(mapping, sep="\t")
    assert mapped.groupby("gene_id").size().to_dict() == {"ENSG1": 3, "ENSG2": 3}
    cohort = tmp_path / "cohort.tsv"
    cohort.write_text("patient\tsample\npatient_B\tB\npatient_A\tA\n")
    matrix = tmp_path / "beta.tsv.gz"
    with gzip.open(matrix, "wt") as handle:
        handle.write("probe\tA\tB\tA\n")
        for probe in range(1, 7):
            handle.write(f"p{probe}\t0.2\t0.8\t0.6\n")
    output = tmp_path / "aggregated"
    monkeypatch.setattr(sys, "argv", [
        "omnit2i-aggregate-methylation", "--matrix", str(matrix),
        "--mapping", str(mapping), "--cohort", str(cohort),
        "--output-dir", str(output),
    ])
    methylation_aggregation.main()
    np.testing.assert_allclose(
        np.load(output / "methylation_promoter_gene_beta.npy"),
        [[0.8, 0.4], [0.8, 0.4]],
    )
    samples = pd.read_csv(output / "methylation_promoter_gene_samples.tsv", sep="\t")
    assert samples["sample"].tolist() == ["B", "A"]
    summary = json.loads((output / "methylation_promoter_gene_summary.json").read_text())
    assert summary["n_selected_mapping_probes"] == 6
    assert summary["minimum_promoter_probes_per_gene"] == 3
