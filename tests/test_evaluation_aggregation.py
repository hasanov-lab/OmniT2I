import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")

from omnit2i.analysis.aggregation import aggregate, aggregate_predictions, subset_manifest, SEEDS
from omnit2i.analysis.concordance import fold_metrics
from omnit2i.evaluation import evaluate_routes


def test_saved_evaluations_generate_manuscript_summary_tables(tmp_path, evaluation_inputs):
    fill, features, measured, selective = evaluate_routes(*evaluation_inputs)
    root = tmp_path / "evaluation"
    for seed in SEEDS:
        for fold in range(5):
            folder = root / f"seed_{seed}" / f"fold{fold}"
            folder.mkdir(parents=True)
            for name, frame in (("predictions", fill), ("feature_metrics", features),
                                ("measured_predictions", measured), ("selective_predictions", selective)):
                part = frame.copy()
                part["seed"], part["fold"] = seed, fold
                if "patient" in part:
                    part["patient"] = part.patient + f"_f{fold}"
                    part["time"] += fold * 10
                part.to_csv(folder / f"{name}.tsv.gz", sep="\t", index=False)
    output = tmp_path / "summary"
    aggregate(root, output)
    read = lambda name: pd.read_csv(output / name, sep="\t")
    results = read("measured/omnit2i_all_31_subsets_per_seed.tsv")
    assert len(results) == 155 and results.n_patients.eq(15).all()
    assert len(read("selective/survival_per_seed.tsv")) == 3600
    assert len(read("summary/feature_per_seed.tsv")) == 375
    deltas = read("selective/survival_deltas_summary.tsv")
    assert len(deltas) == 180 and deltas.n_seeds.eq(5).all()
    actual = results.loc[results.seed.eq(101) & results.source_subset.eq("rna")].iloc[0]
    parts = [pd.read_csv(root / f"seed_101/fold{fold}/measured_predictions.tsv.gz", sep="\t") for fold in range(5)]
    frame = pd.concat(parts)
    expected = fold_metrics(frame.loc[frame.source_subset.eq("rna")])
    assert actual.pooled_c_index == pytest.approx(expected["pooled_c_index"], abs=1e-14)
    row = read("summary/feature_per_seed.tsv").iloc[0]
    assert row.n_test == 15 and row.n_folds == 5
    with pytest.raises(FileExistsError):
        aggregate(root, output)


def test_aggregation_rejects_duplicate_and_inconsistent_patients():
    frame = pd.DataFrame(dict(patient=["a", "b", "c"], seed=101, fold=0,
                              source_subset="rna", observed_count=1, cancer_type="A",
                              time=[1., 2., 3.], event=[1, 1, 0], risk_score=[3., 2., 1.]))
    reference = frame.set_index("patient")[["cancer_type", "time", "event", "fold"]]
    args = (["source_subset", "observed_count"], {"measured": "risk_score"}, {("rna", 1)}, reference)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_predictions(pd.concat([frame, frame.iloc[:1]]), *args)
    changed = frame.copy()
    changed.loc[0, "event"] = 0
    with pytest.raises(AssertionError):
        aggregate_predictions(changed, *args)
    changed = frame.copy()
    changed.loc[0, "risk_score"] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        aggregate_predictions(changed, *args)
    measured, routes = subset_manifest()
    assert len(measured) == 31 and len(routes) == 180


def test_concordance_preserves_censoring_ties_and_pair_weighting():
    frame = pd.DataFrame(dict(cancer_type=["A"] * 3 + ["B"] * 2,
                              time=[1., 1., 2., 2., 3.], event=[1, 0, 1, 1, 0],
                              risk_score=[3., 3., 1., 2., 0.]))
    result = fold_metrics(frame)
    assert result["within_cancer_comparable_pairs"] == 3
    assert result["within_cancer_c_index"] == 2.5 / 3
    assert result["pooled_c_index"] == pytest.approx(result["within_cancer_c_index"] * result["within_cancer_pair_fraction"]
                                                    + result["cross_cancer_c_index"] * (1 - result["within_cancer_pair_fraction"]), abs=1e-14)
