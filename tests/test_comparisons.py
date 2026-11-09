import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")
pytest.importorskip("xgboost")

from omnit2i.comparisons import prepare, run
from omnit2i.comparisons import aggregate as aggregation
from omnit2i.models import MODALITIES


def inputs():
    rng = np.random.default_rng(42)
    n = 150
    fold = np.arange(n) % 5
    partition = np.where(fold == 0, "test", np.where(fold == 1, "validation", "core"))
    cohort = pd.DataFrame(dict(patient=[f"P{i}" for i in range(n)], sample=[f"S{i}" for i in range(n)],
                               cancer_type=np.where(np.arange(n) % 2, "A", "B"),
                               fold=fold, partition=partition, seed=101, outer_fold=0,
                               time=rng.exponential(100, n) + 1, event=np.ones(n, dtype=int)))
    availability = np.ones((n, 5), dtype=bool)
    availability[[0, 1, 2], [0, 1, 2]] = False
    cohort[run.AVAILABILITY_COLUMNS] = availability
    source = dict(patient=cohort.patient.to_numpy(str), modalities=np.asarray(MODALITIES),
                  availability=availability, outer_test=fold == 0,
                  target_scale=np.asarray(["image_target"] * 5))
    for name in MODALITIES:
        source[f"{name}_target"] = rng.normal(size=(n, 4)).astype(np.float32)
        source[f"{name}_feature_ids"] = np.asarray([f"{name}{i}" for i in range(4)])
    rna = cohort.loc[cohort.has_rna, ["patient", "sample", "fold"]].copy()
    expression = pd.DataFrame(rng.normal(size=(8, len(rna))).astype(np.float32),
                              index=[f"G{i}" for i in range(8)], columns=rna["sample"])
    return cohort, rna, expression, source


def test_conventional_rna_outer_training_and_other_blocks():
    cohort, rna, expression, source = inputs()
    prepared = prepare.prepare(cohort, rna, expression, source, 0, top_n=4)
    train = expression.loc[:, rna.loc[rna.fold.ne(0), "sample"]]
    ids = train.var(axis=1, ddof=0).sort_values(ascending=False).index[:4]
    np.testing.assert_array_equal(prepared["rna_feature_ids"], ids)
    means = train.loc[ids].to_numpy().mean(axis=1, dtype=np.float64).astype(np.float32)
    stds = train.loc[ids].to_numpy().std(axis=1, dtype=np.float64).astype(np.float32)
    values = ((expression.loc[ids].to_numpy().T - means) / stds).astype(np.float32)
    np.testing.assert_array_equal(prepared["rna_target"][cohort.has_rna], values)
    assert not prepared["rna_target"][~cohort.has_rna].any()
    expression.loc[:, rna.loc[rna.fold.eq(0), "sample"]] += 100
    changed = prepare.prepare(cohort, rna, expression, source, 0, top_n=4)
    for key in ("rna_feature_ids", "rna_train_mean", "rna_train_std", "rna_train_variance"):
        np.testing.assert_array_equal(prepared[key], changed[key])
    for name in MODALITIES[1:]:
        np.testing.assert_array_equal(prepared[f"{name}_target"], source[f"{name}_target"])
    with pytest.raises(ValueError, match="folds disagree"):
        prepare.prepare(cohort, rna.assign(fold=0), expression, source, 0, 4)


def small_args(tmp_path, model="deepsurv"):
    args = run.build_parser().parse_args([
        "--model", model, "--seed", "101", "--fold", "0", "--subset-index", "30",
        "--cohort", str(tmp_path / "cohort.csv"), "--targets", str(tmp_path / "targets.npz"),
        "--output-dir", str(tmp_path / "out"),
    ])
    args.deepsurv_hidden_dims = "8,4"
    args.deepsurv_epochs, args.deepsurv_patience = 3, 2
    args.rsf_n_estimators = args.extra_n_estimators = 8
    args.gbm_n_estimators = args.coxboost_n_estimators = 8
    args.xgb_n_estimators, args.xgb_early_stopping_rounds = 10, 3
    args.n_jobs = args.rsf_n_jobs = 1
    return args


@pytest.mark.parametrize("model", run.MODELS)
def test_nine_original_estimators_fit_on_cpu(tmp_path, model):
    args = small_args(tmp_path, model)
    rng = np.random.default_rng(74)
    values = rng.normal(size=(120, 6)).astype(np.float32)
    clinical = pd.DataFrame(dict(time=np.exp(3 - values[:, 0] + rng.normal(0, .2, 120)), event=1))
    result = run.fit(model, values[:70], clinical.iloc[:70], values[70:100],
                     clinical.iloc[70:100], values[100:], args, 101)
    assert result.validation_risk.shape == (30,)
    assert result.test_risk.shape == (20,)
    assert np.isfinite(result.test_risk).all()
    assert np.std(result.validation_risk) > 1e-8


def test_runner_preserves_shared_roles_complete_cases_and_calibration(tmp_path, monkeypatch):
    cohort, rna, expression, source = inputs()
    targets = prepare.prepare(cohort, rna, expression, source, 0, 4)
    args = small_args(tmp_path)
    cohort.to_csv(args.cohort, index=False)
    np.savez_compressed(args.targets, **targets)
    seen = {}

    def capture(model, core, core_clinical, validation, validation_clinical, test, options, seed):
        seen.update(core=core, validation=validation, test=test, core_ids=core_clinical.patient.tolist(),
                    validation_ids=validation_clinical.patient.tolist(), seed=seed)
        return run.estimators.FitResult(np.arange(len(validation), dtype=float),
                                       np.arange(len(test), dtype=float), {})

    monkeypatch.setattr(run, "fit", capture)
    run.train(args)
    complete = cohort[run.AVAILABILITY_COLUMNS].all(axis=1)
    assert seen["core_ids"] == cohort.loc[complete & cohort.partition.eq("core"), "patient"].tolist()
    assert seen["validation_ids"] == cohort.loc[complete & cohort.partition.eq("validation"), "patient"].tolist()
    assert seen["core"].shape[1] == 20
    assert seen["seed"] == 101
    expected = cohort.loc[complete & cohort.partition.eq("test"), "patient"].tolist()
    predictions = pd.read_csv(args.output_dir / "predictions.tsv.gz", sep="\t")
    assert predictions.patient.tolist() == expected
    n_val = len(seen["validation"])
    calibrated = (np.arange(len(expected)) - np.arange(n_val).mean()) / np.arange(n_val).std()
    np.testing.assert_allclose(predictions.risk_score_calibrated, calibrated, rtol=1e-7)
    marker = json.loads((args.output_dir / "complete.json").read_text())
    assert marker["n_complete_cases"] == int(complete.sum())
    assert marker["availability_indicators"] == 0
    with pytest.raises(FileExistsError):
        run.train(args)
    targets["target_scale"] = np.asarray(["wrong"] * 5)
    np.savez_compressed(args.targets, **targets)
    with pytest.raises(ValueError, match="without invariant correction"):
        run.read_targets(args.targets, cohort, 0)


def predictions_for(expected, seed=101, fold=0, subset=("rna",)):
    frame = expected[["patient", "cancer_type", "time", "event", "fold"]].copy()
    frame["seed"], frame["baseline"] = seed, "deepsurv"
    frame["source_subset"], frame["observed_count"] = "+".join(subset), len(subset)
    frame["risk_score_calibrated"] = np.log(frame.time.to_numpy()) * -1
    frame["complete_case"] = True
    return frame


def test_fold_reader_rejects_mixed_outcomes(tmp_path):
    cohort, *_ = inputs()
    master = cohort.loc[cohort[run.AVAILABILITY_COLUMNS].all(axis=1)]
    expected = master.loc[master.fold.eq(0)]
    folder = tmp_path / "deepsurv/rna/seed_101/fold0"
    folder.mkdir(parents=True)
    predictions = predictions_for(expected)
    predictions.to_csv(folder / "predictions.tsv.gz", sep="\t", index=False)
    pd.DataFrame([dict(baseline="deepsurv", component="rna", availability_indicators=0)]).to_csv(
        folder / "tuning.tsv", sep="\t", index=False)
    marker = dict(status="complete", model="deepsurv", source_subset="rna", subset_index=0,
                  seed=101, fold=0, n_complete_cases=len(master), availability_indicators=0)
    (folder / "complete.json").write_text(json.dumps(marker))
    # Use the same CSV precision at both input boundaries.
    expected = pd.read_csv(__import__("io").StringIO(expected.to_csv(index=False)))
    aggregation._read_fold(tmp_path, "deepsurv", ("rna",), 0, 101, 0, expected, len(master))
    predictions.loc[predictions.index[0], "event"] = 0
    predictions.to_csv(folder / "predictions.tsv.gz", sep="\t", index=False)
    with pytest.raises(ValueError, match="event disagrees"):
        aggregation._read_fold(tmp_path, "deepsurv", ("rna",), 0, 101, 0, expected, len(master))


def test_oof_aggregation_and_paired_tables(tmp_path, monkeypatch):
    cohort, *_ = inputs()
    master = cohort.loc[cohort[run.AVAILABILITY_COLUMNS].all(axis=1)]
    metrics = aggregation.fold_metrics(predictions_for(master).rename(columns={"risk_score_calibrated": "risk_score"}))
    omni = pd.DataFrame([dict(seed=seed, source_subset="+".join(subset), observed_count=len(subset),
                             n_patients=len(master), n_events=len(master), **metrics)
                         for seed in aggregation.SEEDS for subset in run.modality_subsets()])
    calls = []

    def read(root, model, subset, index, seed, fold, expected, n_complete):
        calls.append((index, seed, fold))
        assert expected.fold.eq(fold).all() and n_complete == len(master)
        return predictions_for(expected, seed, fold, subset), pd.DataFrame([dict(model=model)]), {}

    monkeypatch.setattr(aggregation, "_read_fold", read)
    aggregation.aggregate(tmp_path, cohort, tmp_path / "summary", omni, models=("deepsurv",))
    assert len(set(calls)) == 31 * 5 * 5
    per_seed = pd.read_csv(tmp_path / "summary/model_subset_per_seed.tsv", sep="\t")
    assert len(per_seed) == 31 * 5 and per_seed.patients.eq(len(master)).all()
    paired = pd.read_csv(tmp_path / "summary/paired_per_seed.tsv", sep="\t")
    assert paired.delta_pooled_c_index.eq(0).all()
    summary = pd.read_csv(tmp_path / "summary/vs_all_models.tsv", sep="\t")
    assert summary.positive_seeds_pooled_c_index.eq(0).all()
    assert summary.delta_pooled_c_index_sd.eq(0).all()
    assert len(pd.read_csv(tmp_path / "summary/subsetwise_best_descriptive.tsv", sep="\t")) == 62
    with pytest.raises(ValueError, match="patient/event counts"):
        aggregation.comparison_tables(per_seed, omni.assign(n_patients=1))
    with pytest.raises(ValueError, match="Duplicate"):
        aggregation.comparison_tables(pd.concat([per_seed, per_seed.iloc[:1]]), omni)
    with pytest.raises(ValueError, match="Incomplete"):
        aggregation.comparison_tables(per_seed, omni.iloc[1:])
