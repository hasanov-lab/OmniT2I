import sys

import numpy as np
import pandas as pd
import pytest

from omnit2i.preprocessing.cnv_t2i import cox_score_z, snake_pixel_indices
from omnit2i.preprocessing.omics import fit_omics_preprocessing, main, transform_omics, transform_omics_features


def inputs(modality):
    rng = np.random.default_rng(23)
    samples = [f"sample{i}" for i in range(12)]
    matrix = pd.DataFrame(rng.normal(size=(30, 12)).astype(np.float32), index=[f"gene{i}" for i in range(30)], columns=samples)
    cohort = pd.DataFrame({
        "patient": [f"patient{i}" for i in range(12)], "sample": samples,
        "cancer_type": ["A", "B"] * 6, "time": np.arange(1, 13),
        "event": [1, 0] * 6, "partition": ["core"] * 8 + ["validation"] * 2 + ["test"] * 2,
    })
    chromosomes = pd.Series("chr1", index=matrix.index)
    chromosomes.iloc[-1] = "chrY"
    if modality == "methylation":
        matrix = 1 / (1 + np.exp(-matrix))
    elif modality == "protein":
        matrix.iloc[0, 0] = np.nan
        matrix.iloc[-1, :5] = np.nan
    return matrix, cohort, chromosomes


@pytest.mark.parametrize("modality", ["cnv", "protein", "methylation"])
def test_manuscript_scope_channels_and_serialization(modality, tmp_path):
    matrix, cohort, chromosomes = inputs(modality)
    artifact = fit_omics_preprocessing(matrix, cohort, modality, top_n=12, chromosomes=chromosomes)
    altered = matrix.copy()
    altered.iloc[:, 10:] = 0.99 if modality == "methylation" else 1000
    changed_cohort = cohort.copy()
    changed_cohort.loc[8:, "time"] *= 100
    changed_cohort.loc[8:, "event"] = 1
    changed = fit_omics_preprocessing(altered, changed_cohort, modality, top_n=12, chromosomes=chromosomes)
    for key in artifact:
        np.testing.assert_array_equal(artifact[key], changed[key])
    train = cohort["partition"].ne("test").to_numpy()
    core = cohort.loc[train, "partition"].eq("core").to_numpy()
    raw = matrix.loc[artifact["feature_ids"], cohort.loc[train, "sample"]].T.to_numpy(np.float32)
    if modality == "methylation":
        raw = np.log2(np.clip(raw, 1e-5, 1 - 1e-5) / (1 - np.clip(raw, 1e-5, 1 - 1e-5)))
        assert "gene29" not in artifact["feature_ids"]
    if modality != "cnv":
        raw = np.where(np.isnan(raw), np.nanmedian(raw, axis=0), raw)
    np.testing.assert_allclose(artifact["train_means"], raw.mean(axis=0), atol=2e-7)
    np.testing.assert_allclose(artifact["train_stds"], raw.std(axis=0), atol=2e-7)
    z = (raw - artifact["train_means"]) / artifact["train_stds"]
    expected = cox_score_z(z[core], cohort.loc[train].loc[core, "time"].to_numpy(float), cohort.loc[train].loc[core, "event"].to_numpy(bool))
    np.testing.assert_allclose(artifact["cox_score_z"], expected, atol=1e-6)
    np.savez_compressed(tmp_path / "fit.npz", **artifact)
    with np.load(tmp_path / "fit.npz", allow_pickle=False) as saved:
        restored = dict(saved)
    images = transform_omics(matrix, cohort, restored)
    side = 17 if modality == "protein" else 45
    assert images.shape == (12, 3, side, side) and np.isfinite(images).all()
    pixels = snake_pixel_indices(side, side)[:len(artifact["feature_ids"])]
    order = artifact["layout_order"]
    flat = images[train].reshape(train.sum(), 3, -1)
    np.testing.assert_allclose(flat[:, 0, pixels], z[:, order], atol=1e-6)
    np.testing.assert_allclose(flat[:, 2, pixels], (z * expected)[:, order], atol=2e-6)
    np.testing.assert_array_equal(transform_omics(matrix, cohort.iloc[::-1], restored), images[::-1])
    unseen = cohort.copy()
    unseen["cancer_type"] = "new"
    unseen_images = transform_omics(matrix, unseen, restored)[train].reshape(train.sum(), 3, -1)
    np.testing.assert_allclose(unseen_images[:, 1, pixels], ((z - artifact["global_cancer_mean"]) / artifact["global_cancer_std"])[:, order], atol=1e-6)
    with pytest.raises(ValueError, match="missing selected"):
        transform_omics(matrix.drop(artifact["feature_ids"][0]), cohort, restored)


@pytest.mark.parametrize("modality", ["cnv", "protein", "methylation"])
def test_cli_output_matches_transform(modality, tmp_path, monkeypatch):
    matrix, cohort, chromosomes = inputs(modality)
    matrix.to_csv(tmp_path / "matrix.tsv.gz", sep="\t")
    cohort.to_csv(tmp_path / "cohort.csv", index=False)
    chromosomes.rename_axis("feature_id").rename("chrom").to_csv(tmp_path / "chrom.tsv", sep="\t")
    monkeypatch.setattr(sys, "argv", ["prepare", "--modality", modality, "--matrix", str(tmp_path / "matrix.tsv.gz"), "--cohort", str(tmp_path / "cohort.csv"), "--chromosomes", str(tmp_path / "chrom.tsv"), "--top-n", "12", "--output-dir", str(tmp_path / "output")])
    main()
    with np.load(tmp_path / "output/preprocessing.npz", allow_pickle=False) as saved:
        artifact = dict(saved)
    with np.load(tmp_path / "output/images.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["images"], transform_omics(matrix, cohort, artifact), atol=2e-6)
        np.testing.assert_array_equal(saved["patient"], cohort["patient"])
    with np.load(tmp_path / "output/features.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["features"], transform_omics_features(matrix, cohort, artifact), atol=2e-6)
        np.testing.assert_array_equal(saved["feature_ids"], artifact["feature_ids"])
    with pytest.raises(FileExistsError):
        main()


def test_bad_annotations_missingness_and_capacity():
    matrix, cohort, chromosomes = inputs("methylation")
    with pytest.raises(ValueError, match="chromosome"):
        fit_omics_preprocessing(matrix, cohort, "methylation", top_n=12)
    matrix.iloc[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_omics_preprocessing(matrix, cohort, "methylation", top_n=12, chromosomes=chromosomes)
    matrix, cohort, _ = inputs("cnv")
    matrix.iloc[:, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_omics_preprocessing(matrix, cohort, "cnv", top_n=12)
    matrix = pd.concat([matrix.fillna(0).rename(index=lambda x: f"{i}_{x}") for i in range(10)])
    with pytest.raises(ValueError, match="fit the image"):
        fit_omics_preprocessing(matrix, cohort, "protein")
