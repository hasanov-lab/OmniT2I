from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("sksurv")

from omnit2i.models import MODALITIES, OmicsHEEndToEndCox
from omnit2i.training.fusion import build_fusion_model, fit_fusion, load_molecular_encoders, main, predict_survival
from omnit2i.training.multimodal_data import load_multimodal_data


def test_sparse_patient_alignment_and_core_only_he_scaling(multimodal_files):
    cohort_path, paths, _, availability = multimodal_files
    data = load_multimodal_data(cohort_path, paths)
    np.testing.assert_array_equal(data.availability, availability)
    with np.load(paths[-1], allow_pickle=False) as saved:
        raw, patients = saved["embeddings"], saved["patient"]
    mapping = data.cohort.set_index("patient")["partition"]
    core = mapping.loc[patients].eq("core").to_numpy()
    np.testing.assert_allclose(data.he_standardization["mean"], raw[core].mean(axis=0, dtype=np.float64).astype(np.float32))
    changed = raw.copy()
    changed[~core] *= 100
    np.savez_compressed(paths[-1], patient=patients, embeddings=changed)
    second = load_multimodal_data(cohort_path, paths)
    np.testing.assert_array_equal(data.he_standardization["mean"], second.he_standardization["mean"])
    np.testing.assert_array_equal(data.he_standardization["std"], second.he_standardization["std"])
    np.testing.assert_array_equal(data.inputs[-1][core], second.inputs[-1][core])


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "partition"])
def test_loader_rejects_misaligned_archives(multimodal_files, bad):
    cohort_path, paths, _, _ = multimodal_files
    with np.load(paths[0], allow_pickle=False) as saved:
        payload = dict(saved)
    if bad == "duplicate":
        payload["patient"][0] = payload["patient"][1]
    elif bad == "unknown":
        payload["patient"][0] = "zzz"
    else:
        payload["partition"][:] = "test"
    np.savez_compressed(paths[0], **payload)
    with pytest.raises(ValueError):
        load_multimodal_data(cohort_path, paths)


def test_fusion_ignores_test_data_and_keeps_batchnorm_statistics(multimodal_files):
    cohort_path, paths, checkpoints, _ = multimodal_files
    data = load_multimodal_data(cohort_path, paths)
    args = Namespace(seed=101, device="cpu", encoder_lr=1e-4, fusion_lr=1e-3, weight_decay=1e-4, epochs=2, patience=2, encoder_chunk_size=4, modality_dropout=0.3, aux_weight=0.25, alignment_weight=1.0, gradient_clip=5.0)
    model = build_fusion_model(101, 0.25, 96, "gated")
    load_molecular_encoders(model, checkpoints)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model, metadata, history = fit_fusion(model, data, args)
    for key, value in model.state_dict().items():
        if "running_" in key or "num_batches_tracked" in key:
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not torch.equal(model.fusion.fused_head[0].weight, before["fusion.fused_head.0.weight"])
    assert metadata["best_epoch"] > 0 and len(history) == 2
    for index, positions in enumerate(data.master_positions):
        data.inputs[index][data.cohort.iloc[positions]["partition"].eq("test").to_numpy()] *= 10
    data.cohort.loc[data.cohort["partition"].eq("test"), "event"] = 0
    replay = build_fusion_model(101, 0.25, 96, "gated")
    load_molecular_encoders(replay, checkpoints)
    replay, _, _ = fit_fusion(replay, data, args)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, replay.state_dict()[key], rtol=0, atol=0)


def test_fusion_command_saves_reloadable_predictions(multimodal_files, tmp_path, monkeypatch):
    cohort_path, paths, checkpoints, availability = multimodal_files
    output = tmp_path / "fusion"
    argv = ["omnit2i-train-fusion", "--cohort", str(cohort_path), "--output-dir", str(output), "--epochs", "1", "--encoder-chunk-size", "4"]
    for name, path in zip(MODALITIES, paths):
        argv += [f"--{name}", str(path)]
    for name, path in zip(MODALITIES[:4], checkpoints):
        argv += [f"--{name}-checkpoint", str(path)]
    monkeypatch.setattr("sys.argv", argv)
    main()
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model = OmicsHEEndToEndCox(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    data = load_multimodal_data(cohort_path, paths, he_standardization=checkpoint["he_standardization"])
    risk, weights = predict_survival(model, data, 4, "cpu")
    saved = pd.read_csv(output / "predictions.csv")
    np.testing.assert_allclose(risk, saved["risk_raw"], atol=1e-7)
    assert np.isfinite(saved["risk"]).all()
    np.testing.assert_array_equal(weights[~availability], 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
    with pytest.raises(FileExistsError):
        main()
