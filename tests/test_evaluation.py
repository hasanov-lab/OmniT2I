import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("sksurv")

from omnit2i.evaluation import CONDITIONS, evaluate_routes, evaluate_subsets, load_evaluation_targets
from omnit2i.metrics import _feature_metric_row
from omnit2i.models import MODALITIES, FiveModalCrossModalMultitaskModel, OmicsHEEndToEndCox, TargetConditionedCrossModalTranslator


def evaluation_inputs():
    torch.manual_seed(174)
    cohort = pd.DataFrame({
        "patient": [f"p{i}" for i in range(12)], "cancer_type": ["A"] * 12,
        "time": np.arange(1, 13), "event": 1,
        "partition": ["core"] * 6 + ["validation"] * 3 + ["test"] * 3,
    })
    model = FiveModalCrossModalMultitaskModel(OmicsHEEndToEndCox(latent_dim=8), TargetConditionedCrossModalTranslator(output_dims={name: 12 for name in MODALITIES}, latent_dim=8, linear_rank=4))
    projected = torch.randn(12, 5, 8)
    targets = [np.random.default_rng(i).normal(size=(12, 12)).astype(np.float32) for i in range(5)]
    baselines = {name: {"target_scale": "synthetic", "global_mean": np.zeros(12, np.float32), "cancer_labels": np.array(["A"]), "cancer_means": np.zeros((1, 12), np.float32)} for name in MODALITIES}
    return model, projected, cohort, np.ones((12, 5), bool), targets, baselines


def test_30_subsets_independent_translation_and_frozen_parameters(monkeypatch):
    model, projected, cohort, availability, targets, baselines = evaluation_inputs()
    original = model.translator.forward
    calls = []

    def checked(embeddings, source_mask, target):
        torch.testing.assert_close(embeddings, projected[6:], rtol=0, atol=0)
        assert not source_mask[:, target].any()
        calls.append(target)
        return original(embeddings, source_mask, target)

    monkeypatch.setattr(model.translator, "forward", checked)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    predictions, metrics = evaluate_subsets(model, projected, cohort, availability, targets, baselines)
    assert len(predictions) == 90 and len(metrics) == len(calls) == 75
    assert predictions["source_subset"].nunique() == 30
    for condition in CONDITIONS:
        assert np.isfinite(predictions[f"risk_{condition}"]).all()
        assert predictions[f"calibration_n_{condition}"].eq(3).all()
    assert set(predictions["patient"]) == {"p9", "p10", "p11"}
    for _, group in predictions.groupby("patient"):
        assert group["risk_measured"].nunique() == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    changed = cohort.copy()
    changed.loc[9:, "time"] *= 5
    changed.loc[9:, "event"] = 0
    second, second_metrics = evaluate_subsets(model, projected, changed, availability, targets, baselines)
    pd.testing.assert_frame_equal(predictions.filter(regex="risk_|calibration_"), second.filter(regex="risk_|calibration_"))
    pd.testing.assert_frame_equal(metrics, second_metrics)
    availability[-4:] = False
    with pytest.raises(ValueError, match="complete-case"):
        evaluate_subsets(model, projected, cohort, availability, targets, baselines)


def test_reconstruction_metrics_have_expected_perfect_and_baseline_scores():
    truth = np.arange(60, dtype=np.float32).reshape(5, 12) ** 1.1
    baseline = np.zeros_like(truth)
    metrics = _feature_metric_row(truth, truth.copy(), baseline, np.array(["A"] * 5))
    assert metrics["model_mse"] == metrics["model_squared_error_sum"] == 0
    assert metrics["residual_mse_improvement"] == metrics["feature_pearson_mean"] == 1
    with pytest.warns(RuntimeWarning):
        failed = _feature_metric_row(truth, baseline.copy(), baseline, np.array(["A"] * 5))
    assert failed["residual_mse_improvement"] == 0


def test_31_measured_subsets_and_180_selective_routes(monkeypatch):
    from omnit2i.training.crossmodal import fused_risk_from_projected
    from omnit2i.training.losses import fit_risk_calibration, apply_risk_calibration

    model, projected, cohort, availability, targets, baselines = evaluation_inputs()
    original = model.translator.forward

    def checked(values, mask, target):
        torch.testing.assert_close(values, projected[6:], rtol=0, atol=0)
        assert not mask[:, target].any()
        return original(values, mask, target)

    monkeypatch.setattr(model.translator, "forward", checked)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    fill, metrics, measured, routes = evaluate_routes(model, projected, cohort, availability, targets, baselines)
    assert len(metrics) == 75 and len(measured) == 31 * 3 and len(routes) == 180 * 3
    assert not measured.duplicated(["source_subset", "patient"]).any()
    assert not routes.duplicated(["source_subset", "virtual_targets", "patient"]).any()
    reference = measured.set_index(["source_subset", "patient"])
    for row in routes.itertuples():
        source, chosen = set(row.source_subset.split("+")), set(row.virtual_targets.split("+"))
        assert not source & chosen
        assert set(row.measured_subset.split("+")) == source | chosen
        assert row.final_count == len(source | chosen)
        assert row.risk_measured_matched == reference.loc[(row.measured_subset, row.patient), "risk_score"]
    full_fill = routes.loc[routes.final_count.eq(5)].sort_values(["source_subset", "patient"])
    expected = fill.sort_values(["source_subset", "patient"])
    pd.testing.assert_frame_equal(full_fill[fill.columns].reset_index(drop=True), expected.reset_index(drop=True))
    # Hand-compute a partial route: the three unfilled modalities stay masked.
    source_mask = torch.zeros((6, 5), dtype=torch.bool)
    source_mask[:, 0] = True
    values = projected[6:].clone()
    with torch.no_grad():
        values[:, 1] = original(projected[6:], source_mask, 1)["latent_prediction"]
        source_mask[:, 1] = True
        raw = fused_risk_from_projected(model.survival.fusion, values, source_mask)[0].numpy()
    calibration = fit_risk_calibration(raw, np.array([True] * 3 + [False] * 3))
    partial = routes.loc[routes.source_subset.eq("rna") & routes.virtual_targets.eq("cnv")]
    np.testing.assert_array_equal(partial.risk_virtual_raw, raw[3:])
    np.testing.assert_array_equal(partial.risk_virtual, apply_risk_calibration(raw, calibration)[3:])
    assert np.isfinite(routes.filter(regex="^risk_").to_numpy()).all()
    changed = cohort.copy()
    changed.loc[9:, "time"], changed.loc[9:, "event"] = 100, 0
    again = evaluate_routes(model, projected, changed, availability, targets, baselines)
    for first, second in zip((fill, measured, routes), (again[0], again[2], again[3])):
        pd.testing.assert_frame_equal(first.filter(regex="risk_|calibration_"), second.filter(regex="risk_|calibration_"))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_saved_target_alignment_and_feature_order_validation(tmp_path):
    from types import SimpleNamespace

    _, _, cohort, availability, targets, baselines = evaluation_inputs()
    data = SimpleNamespace(cohort=cohort, availability=availability)
    order = np.arange(12)[::-1]
    archive = {"patient": cohort["patient"].to_numpy(str)[order], "modalities": np.array(MODALITIES), "availability": availability[order], "target_scale": np.array(["synthetic"] * 5)}
    fitted = {}
    for i, name in enumerate(MODALITIES):
        features = np.array([f"g{j}" for j in range(12)])
        archive[f"{name}_target"], archive[f"{name}_feature_ids"] = targets[i][order], features
        baselines[name]["feature_ids"] = features
        fitted.update({f"{name}_{key}": value for key, value in baselines[name].items()})
    np.savez(tmp_path / "targets.npz", **archive)
    np.savez(tmp_path / "means.npz", **fitted)
    aligned, restored = load_evaluation_targets(tmp_path / "targets.npz", tmp_path / "means.npz", data, {name: 12 for name in MODALITIES})
    for a, b in zip(targets, aligned):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(restored["rna"]["global_mean"], baselines["rna"]["global_mean"])
    archive["rna_feature_ids"] = archive["rna_feature_ids"][::-1]
    np.savez(tmp_path / "targets.npz", **archive)
    with pytest.raises(ValueError, match="feature IDs"):
        load_evaluation_targets(tmp_path / "targets.npz", tmp_path / "means.npz", data, {name: 12 for name in MODALITIES})
