import gzip

import numpy as np

from omnit2i.preprocessing.methylation_aggregation import (
    build_sample_positions,
    merge_technical_columns,
    parse_beta_values,
    stream_aggregate,
)


def test_sample_positions_preserve_cohort_order_and_duplicates() -> None:
    primary, secondary, duplicates = build_sample_positions(
        ["A", "B", "A", "C"],
        ["C", "A", "B"],
    )
    np.testing.assert_array_equal(primary, [3, 0, 1])
    np.testing.assert_array_equal(secondary, [-1, 2, -1])
    assert duplicates == {"A": 2}


def test_parse_and_merge_technical_columns() -> None:
    values = parse_beta_values("0.2\tNA\t0.6\t0.8", 4)
    merged = merge_technical_columns(
        values,
        np.array([0, 1, 3]),
        np.array([2, 2, -1]),
    )
    np.testing.assert_allclose(merged, [0.4, 0.6, 0.8])


def test_parse_preserves_empty_missing_fields() -> None:
    values = parse_beta_values("\t0.2\t\t0.4\t", 5)
    np.testing.assert_allclose(values[[1, 3]], [0.2, 0.4])
    assert np.isnan(values[[0, 2, 4]]).all()


def test_merge_keeps_nan_if_both_technical_columns_missing() -> None:
    values = np.array([np.nan, np.nan], dtype=np.float32)
    merged = merge_technical_columns(
        values,
        np.array([0]),
        np.array([1]),
    )
    assert np.isnan(merged[0])


def test_stream_aggregate_merges_duplicates_and_averages_probes(tmp_path) -> None:
    matrix = tmp_path / "tiny.tsv.gz"
    with gzip.open(matrix, "wt") as handle:
        handle.write("probe\tA\tB\tA\n")
        handle.write("p1\t0.2\t0.1\t0.6\n")
        handle.write("ignored\t0.9\t0.9\t0.9\n")
        handle.write("p2\t0.4\t0.3\tNA\n")
    primary, secondary, _ = build_sample_positions(
        ["A", "B", "A"], ["A", "B"]
    )
    beta, counts, summary = stream_aggregate(
        matrix,
        matrix_sample_count=3,
        primary_positions=primary,
        secondary_positions=secondary,
        probe_to_gene={"p1": 0, "p2": 0},
        n_genes=1,
        progress_every=0,
    )
    np.testing.assert_allclose(beta, [[0.4, 0.2]])
    np.testing.assert_array_equal(counts, [[2, 2]])
    assert summary == {
        "n_matrix_probe_rows_scanned": 3,
        "n_selected_probe_rows_aggregated": 2,
    }
