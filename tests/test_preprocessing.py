import numpy as np
import pandas as pd
import pytest

from omnit2i.data import validate_cohort
from omnit2i.preprocessing.cnv_t2i import (
    apply_cancer_standardization,
    fit_asymmetric_bounds,
    fit_cancer_standardization,
    snake_pixel_indices,
    transform_zero_anchored,
)
from omnit2i.preprocessing.cox_score_corrected import cox_score_z
from omnit2i.preprocessing.methylation_t2i import beta_to_m_value, render_option2
from omnit2i.preprocessing.rna import fit_rna_preprocessing, main, transform_rna
from omnit2i.preprocessing.rna_t2i import render_option2 as render_rna_option2, sample_gamma, load_expression
from omnit2i.preprocessing.rppa_t2i import fit_impute_standardize, transform_impute_standardize


def rna_inputs():
    rng = np.random.default_rng(23)
    samples = [f"sample{i}" for i in range(12)]
    expression = pd.DataFrame(
        rng.normal(size=(30, 12)).astype(np.float32),
        index=[f"gene{i}" for i in range(30)], columns=samples,
    )
    cohort = pd.DataFrame({
        "patient": [f"patient{i}" for i in range(12)], "sample": samples,
        "cancer_type": ["A", "B"] * 6, "time": np.arange(1, 13),
        "event": [1, 0] * 6, "partition": ["core"] * 8 + ["validation"] * 2 + ["test"] * 2,
    })
    return expression, cohort


def test_rna_excludes_outer_test_and_serializes_without_pickle(tmp_path):
    expression, cohort = rna_inputs()
    artifact = fit_rna_preprocessing(expression, cohort, top_n=12, invariant_per_bin=2)
    changed_expression = expression.copy()
    changed_expression.iloc[:, 10:] *= 100
    changed_cohort = cohort.copy()
    changed_cohort.loc[10:, "time"] *= 17
    changed_cohort.loc[10:, "event"] = 1
    changed_cohort.loc[10:, "cancer_type"] = "unseen"
    changed = fit_rna_preprocessing(changed_expression, changed_cohort, top_n=12, invariant_per_bin=2)
    for key in artifact:
        if np.issubdtype(artifact[key].dtype, np.number):
            np.testing.assert_allclose(artifact[key], changed[key], atol=1e-6, rtol=1e-5)
        else:
            np.testing.assert_array_equal(artifact[key], changed[key])
    np.savez_compressed(tmp_path / "artifact.npz", **artifact)
    with np.load(tmp_path / "artifact.npz", allow_pickle=False) as saved:
        restored = dict(saved)
    images = transform_rna(expression, cohort, restored)
    assert images.shape == (12, 3, 45, 45) and np.isfinite(images).all()
    reordered = transform_rna(expression, cohort.iloc[::-1], restored)
    np.testing.assert_array_equal(reordered, images[::-1])


def test_rna_outer_training_references_and_core_only_c3():
    expression, cohort = rna_inputs()
    artifact = fit_rna_preprocessing(expression, cohort, top_n=12, invariant_per_bin=2)
    train = cohort.loc[cohort["partition"].ne("test")]
    samples = train["sample"].tolist()
    gamma = sample_gamma(expression, samples, artifact["invariant_genes"].tolist())
    matrix = expression.loc[artifact["feature_ids"], samples].sub(gamma, axis=1).T
    np.testing.assert_array_equal(artifact["train_means"], matrix.mean(axis=0).to_numpy(np.float32))
    np.testing.assert_array_equal(artifact["train_stds"], matrix.std(axis=0, ddof=0).to_numpy(np.float32))
    standardized = (matrix.to_numpy(np.float32) - artifact["train_means"]) / artifact["train_stds"]
    core = train["partition"].eq("core").to_numpy()
    expected = cox_score_z(standardized[core], train.loc[core, "time"].to_numpy(float), train.loc[core, "event"].to_numpy(bool))
    np.testing.assert_array_equal(artifact["cox_score_z"], expected)
    changed = cohort.copy()
    changed.loc[changed["partition"].eq("validation"), "time"] *= 10
    changed.loc[changed["partition"].eq("validation"), "event"] = 1
    second = fit_rna_preprocessing(expression, changed, top_n=12, invariant_per_bin=2)
    for key in artifact:
        np.testing.assert_array_equal(artifact[key], second[key])


def test_cohort_rejects_patient_overlap_and_bad_survival():
    _, cohort = rna_inputs()
    cohort.loc[11, "patient"] = cohort.loc[0, "patient"]
    with pytest.raises(ValueError, match="unique patient"):
        validate_cohort(cohort)
    _, cohort = rna_inputs()
    cohort.loc[0, "event"] = 2
    with pytest.raises(ValueError, match="event must be"):
        validate_cohort(cohort)


def test_rna_rejects_missing_required_genes_and_nonfinite_values():
    expression, cohort = rna_inputs()
    artifact = fit_rna_preprocessing(expression, cohort, top_n=12, invariant_per_bin=2)
    with pytest.raises(KeyError):
        transform_rna(expression.drop(artifact["feature_ids"][0]), cohort, artifact)
    expression.loc[artifact["invariant_genes"][0], "sample11"] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        transform_rna(expression, cohort, artifact)


def test_cox_score_matches_eventwise_tied_risk_sets():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(41, 17))
    time = rng.integers(1, 12, 41)
    event = rng.integers(0, 2, 41).astype(bool)
    score = sum((x[i] - x[time >= time[i]].mean(axis=0) for i in np.flatnonzero(event)))
    expected = np.clip((score - score.mean()) / score.std(), -3, 3)
    np.testing.assert_allclose(cox_score_z(x, time, event), expected, atol=1e-6)
    order = rng.permutation(len(time))
    np.testing.assert_allclose(cox_score_z(x[order], time[order], event[order]), expected, atol=1e-6)


@pytest.mark.parametrize("renderer", [render_option2, render_rna_option2])
def test_serpentine_channels_and_zero_padding(renderer):
    x = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    order = np.array([4, 0, 3, 1, 2])
    scores = np.array([2, -1, 0, 0.5, 3])
    image = renderer(x, -x, scores, order, 2, 3)
    np.testing.assert_array_equal(snake_pixel_indices(2, 3), [0, 1, 2, 5, 4, 3])
    pixels = np.array([0, 1, 2, 5, 4])
    flat = image.reshape(1, 3, -1)
    np.testing.assert_array_equal(flat[0, 0, pixels], x[0, order])
    np.testing.assert_array_equal(flat[0, 1, pixels], -x[0, order])
    np.testing.assert_array_equal(flat[0, 2, pixels], (x * scores)[0, order])
    np.testing.assert_array_equal(flat[:, :, 3], 0)


def test_cnv_clipping_and_new_cancer_global_fallback():
    train = np.array([[-2, -1], [0, 2], [3, 4]], dtype=np.float32)
    bounds = fit_asymmetric_bounds(train)
    transformed = transform_zero_anchored(np.array([[-100, 0, 100, np.nan]]), bounds)
    np.testing.assert_array_equal(transformed[0, :3], [-1, 0, 1])
    assert np.isnan(transformed[0, 3])
    parameters = fit_cancer_standardization(train, ["A", "A", "B"])
    actual = apply_cancer_standardization(train[:1], ["new"], *parameters)
    np.testing.assert_allclose(actual, (train[:1] - train.mean(0)) / train.std(0))


def test_protein_imputation_and_methylation_scale():
    core = np.array([[1, 2], [3, np.nan], [5, 6]], dtype=np.float32)
    parameters = fit_impute_standardize(core)
    held_out = np.array([[np.nan, 100]], dtype=np.float32)
    transformed = transform_impute_standardize(held_out, *parameters)
    assert transformed[0, 0] == 0 and transformed[0, 1] > 10
    np.testing.assert_allclose(beta_to_m_value(np.array([0.2, 0.5, 0.8])), [-2, 0, 2], atol=1e-6)
    with pytest.raises(ValueError, match="lie in"):
        beta_to_m_value(np.array([1.1]))


def test_rna_command_preserves_patient_order_and_refuses_overwrite(tmp_path, monkeypatch):
    expression, cohort = rna_inputs()
    dirty = pd.concat([expression, expression.iloc[:1] + 2, expression.iloc[:1].rename(index={"gene0": "missing"})])
    dirty.loc["missing", "sample11"] = np.nan
    dirty.to_csv(tmp_path / "expression.tsv.gz", sep="\t")
    cleaned = load_expression(tmp_path / "expression.tsv.gz", cohort["sample"].tolist())
    assert cleaned.index.is_unique and "missing" not in cleaned.index
    np.testing.assert_allclose(cleaned.loc["gene0"], expression.loc["gene0"] + 1, atol=1e-6)
    expected = fit_rna_preprocessing(cleaned, cohort, top_n=12, invariant_per_bin=2)
    cohort.to_csv(tmp_path / "cohort.csv", index=False)
    output = tmp_path / "prepared"
    monkeypatch.setattr("sys.argv", [
        "omnit2i-prepare-rna", "--expression", str(tmp_path / "expression.tsv.gz"),
        "--cohort", str(tmp_path / "cohort.csv"), "--output-dir", str(output),
        "--top-n", "12", "--invariant-per-bin", "2",
    ])
    main()
    with np.load(output / "images.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["patient"], cohort["patient"].to_numpy(str))
        np.testing.assert_array_equal(saved["partition"], cohort["partition"].to_numpy(str))
        np.testing.assert_array_equal(saved["images"], transform_rna(cleaned, cohort, expected))
    with pytest.raises(FileExistsError):
        main()


def test_rna_reader_matches_manuscript_cleanup_on_requested_samples(tmp_path):
    source = pd.DataFrame({"s1": [1., 3., 7., np.nan, 5.], "s2": [2., 6., 8., 9., 10.],
                           "unused": [np.nan] * 5}, index=["dup", "dup", "good", "missing", "keep"])
    path = tmp_path / "expression.tsv.gz"
    source.to_csv(path, sep="\t")
    actual = load_expression(path, ["s2", "s1"])
    expected = source[["s2", "s1"]].astype(np.float32).groupby(level=0).mean().dropna()
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc["dup"].tolist() == [4., 2.]
    with pytest.raises(ValueError):
        load_expression(path, ["absent"])
