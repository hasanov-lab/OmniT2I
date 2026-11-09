import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("sksurv")

from omnit2i.external import map_expression, transform_external, scanb_expression, predict_rna, bootstrap, summarize_members, SEEDS
from omnit2i.models import OmicsHEEndToEndCox
from omnit2i.preprocessing.rna import fit_rna_preprocessing, transform_rna
from omnit2i.preprocessing.cnv_t2i import snake_pixel_indices


def fitted_rna():
    rng = np.random.default_rng(91)
    clinical = pd.DataFrame(dict(patient=[f"p{i}" for i in range(12)], sample=[f"s{i}" for i in range(12)],
                                  cancer_type="A", time=np.arange(12) + 1., event=1,
                                  partition=["core"] * 6 + ["validation"] * 3 + ["test"] * 3))
    expression = pd.DataFrame(rng.normal(size=(24, 12)), index=[f"g{i}" for i in range(24)], columns=clinical["sample"]).astype(np.float32)
    saved = fit_rna_preprocessing(expression, clinical, top_n=16, expression_bins=2, invariant_per_bin=2, height=4, width=4)
    return expression, clinical, saved


def test_external_transform_uses_saved_statistics_fixed_positions_and_partial_invariants():
    expression, clinical, saved = fitted_rna()
    actual, _ = transform_external(expression, saved, clinical.cancer_type)
    np.testing.assert_array_equal(actual, transform_rna(expression, clinical, saved))
    gene = saved["feature_ids"][0]
    partial = expression.drop(index=gene)
    images, audit = transform_external(partial, saved, clinical.cancer_type)
    assert audit["mean_filled_input_genes"] == 1
    layout_position = np.flatnonzero(saved["layout_order"] == 0).item()
    pixel = snake_pixel_indices(4, 4)[layout_position]
    assert (images[:, 0].reshape(12, -1)[:, pixel] == 0).all()
    assert (images[:, 2].reshape(12, -1)[:, pixel] == 0).all()
    assert np.isfinite(images).all()
    with pytest.raises(ValueError, match="invariant"):
        transform_external(expression.drop(index=saved["invariant_genes"]), saved, clinical.cancer_type)
    mapping = pd.DataFrame(dict(saved_gene_symbol=expression.index, source_gene_id=expression.index, match_status="exact"))
    pd.testing.assert_frame_equal(map_expression(expression, mapping, saved).sort_index(),
                                  map_expression(expression, None, saved).sort_index(), check_names=False)
    mapping.loc[mapping.saved_gene_symbol.eq(gene), "match_status"] = "ambiguous"
    assert gene not in map_expression(expression, mapping, saved).index
    mapping["source_gene_id"] = "g0"
    with pytest.raises(ValueError, match="one-to-one"):
        map_expression(expression, mapping, saved)


def test_scanb_excludes_replicates_but_uses_unselected_genes_in_denominator(tmp_path):
    fpkm = pd.DataFrame({"original": [1., 3., 96.], "replicate": [10., 20., 30.]}, index=["g1", "g2", "other"])
    source = tmp_path / "deposited.csv.gz"
    np.log2(fpkm + .1).to_csv(source)
    actual, genes = scanb_expression(source, ["original"], ["g1", "g2"])
    assert genes == 3 and actual.columns.tolist() == ["original"]
    np.testing.assert_allclose(actual.original, np.log2(np.array([1., 3.]) / 100 * 1e6 + 1), rtol=1e-6)


def test_frozen_rna_batch_padding_and_native_fusion_agree():
    torch.manual_seed(101)
    model = OmicsHEEndToEndCox(latent_dim=8).eval()
    images = np.random.default_rng(51).normal(size=(3, 3, 45, 45)).astype(np.float32)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    actual = predict_rna(model, images)
    with torch.no_grad():
        embedding = model.encoders[0](torch.from_numpy(np.concatenate([images, np.repeat(images[-1:], 29, axis=0)])))
        mask = torch.zeros((32, 5), dtype=torch.bool)
        mask[:, 0] = True
        expected = model.fusion([embedding] + [torch.zeros_like(embedding) for _ in range(4)], mask)["risk"][:3].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_external_ensemble_checks_all_members_and_reproduces_patient_bootstrap(tmp_path):
    clinical = pd.DataFrame(dict(patient=[f"p{i}" for i in range(8)], sample=[f"s{i}" for i in range(8)],
                                  cancer_type="A", time=np.arange(8) + 1., event=[1, 1, 1, 0, 1, 1, 0, 0]))
    root = tmp_path / "members"
    for seed in SEEDS:
        for fold in range(5):
            frame = clinical.copy()
            frame["seed"], frame["fold"] = seed, fold
            frame["risk_raw"] = -frame.time
            frame["risk"] = -frame.time + seed * .001 + fold * .01
            folder = root / f"seed_{seed}/fold{fold}"
            folder.mkdir(parents=True)
            frame.to_csv(folder / "predictions.csv", index=False)
    output = tmp_path / "summary"
    summarize_members(root, output)
    averaged = pd.read_csv(output / "patient_predictions.csv")
    np.testing.assert_allclose(averaged.risk, -clinical.time + np.mean(SEEDS) * .001 + .02)
    summary = pd.read_csv(output / "summary.csv").iloc[0]
    assert summary.c_index == summary.ci_low == summary.ci_high == 1.
    assert summary.bootstrap_attempted == 2000 and summary.bootstrap_valid > 1900
    frame = clinical.copy()
    frame["risk"] = np.array([1., 1., 2., 3., 5., 8., 5., 4.])
    first, second = bootstrap(frame, 100), bootstrap(frame, 100)
    assert first == second
    broken = root / "seed_505/fold4/predictions.csv"
    frame = pd.read_csv(broken)
    frame.loc[0, "time"] = 100
    frame.to_csv(broken, index=False)
    with pytest.raises(AssertionError):
        summarize_members(root, tmp_path / "bad")


def test_scanb_series_matrix_retains_originals_and_rejects_discordant_outcomes(tmp_path):
    import csv
    import gzip
    from omnit2i.external import scanb_clinical

    source = tmp_path / "series.txt.gz"
    rows = [
        ["!Sample_title", "original", "originalrepl", "second"],
        ["!Sample_characteristics_ch1", "scan-b external id: X.C01.Y", "scan-b external id: X.C01.Y", "scan-b external id: X.C02.Y"],
        ["!Sample_characteristics_ch1", "overall survival days: 10", "overall survival days: 10", "overall survival days: 20"],
        ["!Sample_characteristics_ch1", "overall survival event: 1", "overall survival event: 1", "overall survival event: 0"],
    ]
    with gzip.open(source, "wt") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)
    frame = scanb_clinical([source])
    assert frame.patient.tolist() == ["C01", "C02"]
    assert frame["sample"].tolist() == ["original", "second"]
    assert frame.time.tolist() == [10, 20] and frame.event.tolist() == [1, 0]
    rows[-1][2] = "overall survival event: 0"
    with gzip.open(source, "wt") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)
    with pytest.raises(ValueError, match="Discordant"):
        scanb_clinical([source])


def test_external_cli_uses_released_checkpoint_calibration(tmp_path, monkeypatch):
    import sys
    from omnit2i.external import main

    expression, clinical, saved = fitted_rna()
    saved["grid_height"], saved["grid_width"] = 45, 45
    np.savez(tmp_path / "preprocessing.npz", **saved)
    expression.to_csv(tmp_path / "expression.tsv", sep="\t")
    clinical = clinical.iloc[:3].drop(columns="partition")
    clinical.to_csv(tmp_path / "clinical.csv", index=False)
    torch.manual_seed(101)
    model = OmicsHEEndToEndCox(latent_dim=8).eval()
    torch.save(dict(survival_model_kwargs={"latent_dim": 8},
                    model_state={f"survival.{key}": value for key, value in model.state_dict().items()},
                    risk_calibration={"mean": 1., "std": 2.}, seed=101, fold=0), tmp_path / "model.pt")
    images, _ = transform_external(expression.loc[:, clinical["sample"]], saved, clinical.cancer_type)
    expected = predict_rna(model, images)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["predict", "--expression", str(tmp_path / "expression.tsv"),
        "--clinical", str(tmp_path / "clinical.csv"), "--preprocessing", str(tmp_path / "preprocessing.npz"),
        "--checkpoint", str(tmp_path / "model.pt"), "--output-dir", str(output)])
    main()
    result = pd.read_csv(output / "predictions.csv")
    np.testing.assert_allclose(result.risk_raw, expected, atol=1e-6)
    np.testing.assert_allclose(result.risk, (expected - 1.) / 2., atol=1e-6)
    assert result.seed.eq(101).all() and result.fold.eq(0).all()
    with pytest.raises(FileExistsError):
        main()
