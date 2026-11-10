import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")

from omnit2i.analysis import ablations
from omnit2i.training.losses import fit_risk_calibration, apply_risk_calibration


def training_tree(root, master, sign):
    for seed in ablations.SEEDS:
        for fold in range(5):
            frame = master.copy()
            frame["sample"] = frame.patient
            frame["partition"] = np.where(frame.fold.eq(fold), "test", np.where(frame.fold.eq((fold + 1) % 5), "validation", "core"))
            raw = sign * frame.time.to_numpy(np.float32)
            cal = fit_risk_calibration(raw, frame.partition.eq("validation"))
            frame["risk_raw"], frame["risk"] = raw, apply_risk_calibration(raw, cal)
            folder = root / f"seed_{seed}/fold{fold}/C"
            folder.mkdir(parents=True)
            frame.to_csv(folder / "predictions.csv", index=False)
            (folder / "options.json").write_text(json.dumps(dict(seed=seed, fold=fold)))


def test_stage_c_summaries_pair_oof_and_recover_complete_validation_calibration(tmp_path):
    n = 100
    master = pd.DataFrame(dict(patient=[f"P{i}" for i in range(n)], cancer_type=np.where(np.arange(n) % 2, "A", "B"),
                               time=np.arange(1, n + 1, dtype=float), event=1, fold=np.arange(n) % 5))
    master[ablations.FLAGS] = True
    master.loc[np.arange(n) % 7 == 0, "has_rna"] = False
    full, other = tmp_path / "full", tmp_path / "other"
    training_tree(full, master, -1)
    training_tree(other, master, 1)
    tables = ablations.training_tables(full, {"no_c2": other}, master)
    per_seed = tables["survival_per_seed"]
    assert len(per_seed) == 20
    assert per_seed.loc[per_seed.scenario.eq("natural_availability"), "n_patients"].eq(100).all()
    assert per_seed.loc[per_seed.scenario.eq("complete_case"), "n_patients"].eq(85).all()
    paired = tables["paired_survival_per_seed"]
    assert paired.loc[paired.metric.eq("raw_merged_pooled_c_index"), "full_minus_ablation"].eq(1).all()
    assert tables["paired_survival_summary"].n_seeds.eq(5).all()
    assert len(tables["encoder_paired_survival_per_seed"]) == 60
    first = ablations.read_training_fold(full, master, 101, 0)
    source = pd.read_csv(full / "seed_101/fold0/C/predictions.csv")
    valid = source.partition.eq("validation") & source.patient.isin(master.loc[master.has_rna, "patient"])
    expected = (source.risk_raw.to_numpy(np.float32) - source.loc[valid, "risk_raw"].mean()) / source.loc[valid, "risk_raw"].std(ddof=0)
    selected = source.partition.eq("test") & source.patient.isin(first.loc[first.complete_case, "patient"])
    np.testing.assert_allclose(first.loc[first.complete_case, "risk_score_complete_case_calibrated"], expected[selected], atol=1e-6)
    with pytest.raises(ValueError, match="time differs"):
        ablations.read_training_fold(full, master.assign(time=master.time + 1), 101, 0)
    source.loc[0, "risk"] += 1
    source.to_csv(full / "seed_101/fold0/C/predictions.csv", index=False)
    with pytest.raises(ValueError, match="standardization"):
        ablations.read_training_fold(full, master, 101, 0)


def frozen_tree(root, score):
    measured, routes = ablations.subset_manifest()
    metrics = ["pooled_c_index", "within_cancer_c_index", "cross_cancer_c_index"]
    frames = {
        "measured/omnit2i_all_31_subsets_per_seed.tsv": pd.DataFrame([
            dict(seed=seed, source_subset=name, observed_count=len(name.split("+")), n_patients=100, n_events=50,
                 **{metric: score for metric in metrics}) for seed in ablations.SEEDS for name in measured]),
        "selective/survival_per_seed.tsv": pd.DataFrame([
            dict(seed=seed, **dict(zip(ablations.ROUTE, route)), condition=condition, n_patients=100, n_events=50,
                 **{metric: score for metric in metrics}) for seed in ablations.SEEDS for route in routes
            for condition in ("source_only", "cancer_mean", "virtual", "measured")]),
        "selective/survival_deltas_per_seed.tsv": pd.DataFrame([
            dict(seed=seed, **dict(zip(ablations.ROUTE, route)),
                 **{f"delta_virtual_minus_source_only_{metric}": score / 10 for metric in metrics})
            for seed in ablations.SEEDS for route in routes]),
    }
    for filename, frame in frames.items():
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)


def test_frozen_pairing_direction_sd_and_missing_routes(tmp_path):
    full, other = tmp_path / "full", tmp_path / "other"
    frozen_tree(full, .8)
    frozen_tree(other, .6)
    tables = ablations.frozen_tables(full, {"no_modality_dropout": other})
    assert len(tables["paired_measured_per_seed"]) == 31 * 5 * 3
    assert len(tables["paired_selective_per_seed"]) == 180 * 5 * 4 * 3
    summary = tables["paired_measured_summary"]
    np.testing.assert_allclose(summary.delta_mean, .2)
    np.testing.assert_allclose(summary.delta_sd, 0)
    assert summary.full_higher_seeds.eq(5).all()
    np.testing.assert_allclose(tables["paired_virtual_gain_per_seed"].full_minus_ablation, .02)
    path = other / "measured/omnit2i_all_31_subsets_per_seed.tsv"
    pd.read_csv(path, sep="\t").iloc[1:].to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        ablations.frozen_tables(full, {"no_modality_dropout": other})


def test_survival_export_connects_stage_c_to_ipcw_and_r_inputs(tmp_path, monkeypatch):
    from omnit2i.analysis import ipcw

    n = 100
    master = pd.DataFrame(dict(patient=[f"P{i}" for i in range(n)], cancer_type=np.where(np.arange(n) % 2, "A", "B"),
                               time=np.repeat(np.arange(n // 5) * 200. + 1, 5), event=1, fold=np.arange(n) % 5))
    master[ablations.FLAGS] = True
    master.loc[np.arange(n) % 7 == 0, "has_rna"] = False
    master.to_csv(tmp_path / "master.csv", index=False)
    trained, output = tmp_path / "trained", tmp_path / "Inputs"
    training_tree(trained, master, -1)
    monkeypatch.setattr("sys.argv", ["export", "--master", str(tmp_path / "master.csv"),
        "--training-root", str(trained), "--output-dir", str(output)])
    ablations.export_main()
    paths = list(output.glob("oof/seed_*/fold*/survival_oof_predictions.tsv.gz"))
    assert len(paths) == 25
    for path in paths:
        assert path.read_bytes()[:2] == b"\x1f\x8b"
        frame = pd.read_csv(path, sep="\t")
        assert frame.event.dtype == frame.complete_case.dtype == bool
        assert frame.event.all()
        assert frame.loc[frame.complete_case, "risk_score_complete_case_calibrated"].notna().all()
    performance = pd.read_csv(output / "performance_per_seed.tsv", sep="\t")
    assert len(performance) == 10 and "variant" not in performance
    assert performance.raw_merged_pooled_c_index.eq(1).all()
    assert performance.within_fold_pooled_c_index_pair_weighted.eq(1).all()
    patients = pd.read_csv(output / "complete_case_patients.tsv", sep="\t")
    expected = master.loc[master.has_rna, ["patient", "cancer_type", "time", "event", "fold"]].sort_values("patient").reset_index(drop=True)
    expected["event"] = expected.event.astype(bool)
    pd.testing.assert_frame_equal(patients, expected)
    with pytest.raises(FileExistsError):
        ablations.export_main()
    monkeypatch.setattr("sys.argv", ["ipcw", "--master-folds", str(output / "master_folds.csv"),
        "--oof-root", str(output / "oof"), "--output-dir", str(tmp_path / "auc")])
    ipcw.main()
    assert (tmp_path / "auc/per_seed.tsv").is_file()
