import subprocess

import numpy as np
import pandas as pd
import pytest

from omnit2i.models import MODALITIES
from omnit2i.preprocessing.feature_targets import prepare_targets
from omnit2i.preprocessing.splits import build_shared_cohort, inner_split, modality_cohorts
from omnit2i.training import workflow
from omnit2i.training.crossmodal import load_target_state
from omnit2i.training.multimodal_data import load_multimodal_data


def test_shared_partition_is_patient_level_and_excludes_outer_test():
    n = 100
    master = pd.DataFrame({
        "patient": [f"p{i}" for i in range(n)], "cancer_type": ["A", "B"] * 50,
        "time": np.arange(n) + 1., "event": [0, 0, 1, 1] * 25, "fold": np.arange(n) % 5,
    })
    for sample, flag in [("rna_sample", "has_rna"), ("cnv_sample", "has_cnv"),
                         ("rppa_sample", "has_rppa"), ("methylation_sample", "has_methylation"),
                         ("he_case", "has_he")]:
        master[sample], master[flag] = master.patient + "_sample", True
    master.loc[::7, "rna_sample"], master.loc[::7, "has_rna"] = np.nan, False
    result = build_shared_cohort(master, fold=2, seed=101)
    assert result.partition.value_counts().to_dict() == {"core": 64, "test": 20, "validation": 16}
    assert result.partition.eq("test").equals(master.fold.eq(2))
    changed = master.copy()
    changed.loc[changed.fold.eq(2), "event"] = 1 - changed.loc[changed.fold.eq(2), "event"]
    pd.testing.assert_series_equal(result.partition, build_shared_cohort(changed, 2, 101).partition)
    assert not result.partition.equals(build_shared_cohort(master, 2, 202).partition)
    for view in modality_cohorts(result).values():
        pd.testing.assert_series_equal(view.set_index("patient").partition, result.set_index("patient").loc[view.patient].partition)
    with pytest.raises(ValueError, match="unique"):
        build_shared_cohort(pd.concat([master, master.iloc[:1]]), 2, 101)


def test_split_retains_rare_cancer_and_singleton_fallbacks():
    # Collapse both event groups of the rare cancer, then use event-only strata
    # when the entire cancer contains only one patient.
    clinical = pd.DataFrame({"cancer_type": ["A"] * 20 + ["B"] * 5,
                             "event": [0, 1] * 10 + [1, 0, 0, 0, 0]})
    for rows in (clinical, clinical.iloc[:21]):
        core, validation = inner_split(rows, 101)
        assert not set(core) & set(validation)
        assert set(core) | set(validation) == set(range(len(rows)))


def feature_archives(cohort_path, paths):
    cohort = pd.read_csv(cohort_path)
    for index, path in enumerate(paths[:4]):
        with np.load(path, allow_pickle=False) as saved:
            patients = saved["patient"]
        rows = cohort.set_index("patient").loc[patients]
        np.savez_compressed(path, patient=patients, features=np.arange(len(patients) * 3).reshape(-1, 3).astype(np.float32),
                            feature_ids=np.array(["f0", "f1", "f2"]),
                            **{field: rows[field].to_numpy(str if field in {"partition", "cancer_type"} else float)
                               for field in ("partition", "cancer_type", "time", "event")})
    return cohort


def test_targets_align_missing_modalities_and_share_he_transform(multimodal_files, tmp_path):
    cohort_path, paths, _, availability = multimodal_files
    data = load_multimodal_data(cohort_path, paths)
    cohort = feature_archives(cohort_path, paths)
    payload = prepare_targets(cohort, paths)
    np.testing.assert_array_equal(payload["availability"], availability)
    for index, name in enumerate(MODALITIES):
        np.testing.assert_array_equal(payload[f"{name}_target"][~availability[:, index]], 0)
        with np.load(paths[index], allow_pickle=False) as saved:
            positions = pd.Index(cohort.patient).get_indexer(saved["patient"])
            expected = data.inputs[4] if name == "he" else saved["features"]
            np.testing.assert_array_equal(payload[f"{name}_target"][positions], expected)
    np.testing.assert_array_equal(payload["he_train_mean"], data.he_standardization["mean"])
    target_path = tmp_path / "targets.npz"
    np.savez_compressed(target_path, **payload)
    state = load_target_state(target_path, data)
    assert len(state.residual_targets) == 5
    changed = cohort.iloc[::-1].reset_index(drop=True)
    reordered = prepare_targets(changed, paths)
    np.testing.assert_array_equal(reordered["rna_target"], payload["rna_target"][::-1])
    with np.load(paths[4], allow_pickle=False) as saved:
        raw = dict(saved)
    test = cohort.set_index("patient").loc[raw["patient"]].partition.eq("test").to_numpy()
    raw["embeddings"][test] *= 1000
    np.savez_compressed(paths[4], **raw)
    altered = prepare_targets(cohort, paths)
    for key in ("he_train_mean", "he_train_std"):
        np.testing.assert_array_equal(altered[key], payload[key])


@pytest.mark.parametrize("bad", ["patient", "partition", "features"])
def test_targets_reject_mismatched_inputs(multimodal_files, bad):
    cohort_path, paths, *_ = multimodal_files
    cohort = feature_archives(cohort_path, paths)
    with np.load(paths[0], allow_pickle=False) as saved:
        payload = dict(saved)
    if bad == "patient":
        payload["patient"][0] = "unknown"
    elif bad == "partition":
        payload["partition"][:] = "test"
    else:
        payload["features"][0, 0] = np.nan
    np.savez_compressed(paths[0], **payload)
    with pytest.raises(ValueError):
        prepare_targets(cohort, paths)


def test_training_sequence_offsets_and_failures(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    pd.DataFrame({"seed": [101], "outer_fold": [2]}).to_csv(prepared / "cohort.csv", index=False)
    for name in MODALITIES[:4]:
        (prepared / name).mkdir()
        (prepared / name / "images.npz").touch()
    for name in ("he.npz", "targets.npz"):
        (prepared / name).touch()
    calls = []
    monkeypatch.setattr(workflow.subprocess, "run", lambda argv, check: calls.append(argv))
    workflow.train_fold(prepared, tmp_path / "trained", 101, 2, "cpu")
    assert len(calls) == 6
    assert [argv[argv.index("--seed") + 1] for argv in calls] == ["301"] * 4 + ["2101", "101"]
    assert calls[-1][calls[-1].index("--fold") + 1] == "2"
    assert calls[-1][calls[-1].index("--source-checkpoint") + 1].endswith("B/model.pt")
    with pytest.raises(ValueError, match="seed"):
        workflow.train_fold(prepared, tmp_path / "wrong", 202, 2, "cpu")
    def fail(argv, check):
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(workflow.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        workflow.train_fold(prepared, tmp_path / "failed", 101, 2, "cpu")
    assert not (tmp_path / "failed/B").exists()
