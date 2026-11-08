from argparse import Namespace
import copy

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("sksurv")

from omnit2i.models import MODALITIES, FiveModalCrossModalMultitaskModel, OmicsHEEndToEndCox, TargetConditionedCrossModalTranslator
from omnit2i.training import crossmodal
from omnit2i.training.fusion import build_fusion_model, load_molecular_encoders, predict_survival
from omnit2i.training.multimodal_data import load_multimodal_data


@pytest.fixture
def joint_files(multimodal_files, tmp_path):
    cohort_path, paths, checkpoints, availability = multimodal_files
    data = load_multimodal_data(cohort_path, paths)
    rng = np.random.default_rng(9)
    order = rng.permutation(len(data.cohort))
    targets = {
        "patient": data.cohort["patient"].to_numpy(str)[order],
        "cancer_type": data.cohort["cancer_type"].to_numpy(str)[order],
        "partition": data.cohort["partition"].to_numpy(str)[order],
        "availability": availability[order], "modalities": np.array(MODALITIES),
        "target_scale": np.array(["synthetic_z"] * 5),
    }
    for index, name in enumerate(MODALITIES):
        values = rng.normal(size=(len(data.cohort), 6 + index)).astype(np.float32)
        values[~availability[:, index]] = np.nan
        targets[f"{name}_target"] = values[order]
        targets[f"{name}_feature_ids"] = np.array([f"f{i}" for i in range(values.shape[1])])
    target_path = tmp_path / "targets.npz"
    np.savez_compressed(target_path, **targets)
    survival = build_fusion_model(101, 0.25, 96, "gated")
    load_molecular_encoders(survival, checkpoints)
    source = {
        "model_kwargs": {"dropout": 0.25, "latent_dim": 96, "fusion_mode": "gated", **{f"{name}_encoder_architecture": "masked_cnn_spatial_projector" for name in MODALITIES[:4]}},
        "model_state": survival.state_dict(),
        "he_standardization": {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value for key, value in data.he_standardization.items()},
        "risk_calibration": {"mean": 0.0, "std": 1.0, "n_reference": 4},
    }
    source_path = tmp_path / "source.pt"
    torch.save(source, source_path)
    return data, target_path, source, source_path, cohort_path, paths


def joint_args():
    return Namespace(
        seed=101, fold=0, device="cpu", mapper_hidden_dim=192, decoder_hidden_dim=384,
        linear_rank=16, translator_dropout=0.1, warmup_epochs=1, joint_epochs=1,
        warmup_lr=5e-4, translator_lr=5e-4, fusion_lr=5e-5, encoder_lr=1e-5,
        weight_decay=1e-4, gradient_clip=5.0, modality_dropout=0.3, source_dropout=0.35,
        aux_weight=0.25, alignment_weight=0.1, latent_weight=0.25, feature_weight=1.0,
        risk_consistency_weight=0.1, latent_cosine_weight=0.1, feature_cosine_weight=0.05,
        survival_guardrail=0.005, encoder_chunk_size=4, crossmodal_batch_size=32,
        crossmodal_validation_batch_size=32, crossmodal_ramp_epochs=10,
        validation_frequency=1, patience=2,
    )


def test_target_references_use_only_observed_core(joint_files):
    data, path, *_ = joint_files
    state = crossmodal.load_target_state(path, data)
    with np.load(path, allow_pickle=False) as saved:
        payload = dict(saved)
    core = payload["partition"] == "core"
    for index, name in enumerate(MODALITIES):
        values = payload[f"{name}_target"]
        observed_core = core & payload["availability"][:, index]
        np.testing.assert_allclose(state.feature_baselines[name]["global_mean"], values[observed_core].mean(axis=0), atol=1e-7)
        values[~core] *= 100
    np.savez_compressed(path, **payload)
    changed = crossmodal.load_target_state(path, data)
    for index, name in enumerate(MODALITIES):
        for key in ("global_mean", "cancer_means"):
            np.testing.assert_array_equal(state.feature_baselines[name][key], changed.feature_baselines[name][key])
        np.testing.assert_array_equal(state.residual_targets[index][state.core_mask], changed.residual_targets[index][state.core_mask])
        assert not changed.residual_targets[index][data.cohort["partition"].eq("test")].any()


@pytest.mark.parametrize("bad", ["patient", "availability", "nonfinite"])
def test_target_loader_rejects_bad_archives(joint_files, bad):
    data, path, *_ = joint_files
    with np.load(path, allow_pickle=False) as saved:
        payload = dict(saved)
    if bad == "patient":
        payload["patient"][0] = payload["patient"][1]
    elif bad == "availability":
        payload["availability"] = ~payload["availability"]
    else:
        payload["rna_target"][payload["availability"][:, 0]] = np.nan
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError):
        crossmodal.load_target_state(path, data)


def test_source_mask_and_risk_consistency_are_target_safe():
    available = torch.ones((20, 5), dtype=torch.bool)
    available[:10, 2:] = False
    mask = crossmodal.sample_source_mask(available, 0, 0.99, torch.Generator().manual_seed(9))
    assert not mask[:, 0].any() and mask.any(dim=1).all()
    assert not (mask & ~available).any()
    fusion = build_fusion_model(3, 0.75, 96, "gated").fusion.train()
    projected = torch.randn(20, 5, 96)
    assert crossmodal.risk_consistency_loss(fusion, projected, projected, available).item() == 0


def test_translation_teacher_receives_no_gradient(monkeypatch):
    eligible = crossmodal.eligible_translation_rows
    monkeypatch.setattr(crossmodal, "eligible_translation_rows", lambda population, available, target: eligible(population, available, target) if target == 0 else np.zeros(len(available), dtype=bool))
    projected = torch.randn(3, 5, 96, requires_grad=True)
    translator = TargetConditionedCrossModalTranslator([6] * 5, dropout=0.0)
    losses, _ = crossmodal.crossmodal_objective(
        translator, build_fusion_model(3, 0.0, 96, "gated").fusion,
        projected, torch.ones(3, 5, dtype=torch.bool), np.arange(3), np.ones(3, dtype=bool),
        [np.zeros((3, 6), dtype=np.float32)] * 5, 3, 0.0, 0.1, 0.05, None,
    )
    losses["latent"].backward()
    assert not projected.grad[:, 0].any()
    assert projected.grad[:, 1:].abs().sum() > 0


def test_joint_freezing_and_test_isolation(joint_files):
    data, path, source, *_ = joint_files
    state = crossmodal.load_target_state(path, data)
    args = joint_args()
    dims = {name: values.shape[1] for name, values in zip(MODALITIES, state.residual_targets)}
    initial = crossmodal.build_joint_model(source, dims, args)
    groups = crossmodal.configure_joint_trainable_parameters(initial)
    trainable = {id(value) for group in groups.values() for value in group}
    assert trainable == {id(value) for value in initial.parameters() if value.requires_grad}
    assert not initial.survival.encoders[0].trunk.network[0].weight.requires_grad
    assert initial.survival.encoders[0].trunk.network[8].weight.requires_grad
    before = copy.deepcopy(initial.state_dict())
    model, metadata, history, fallback = crossmodal._fit_fold(initial, data.inputs, data.local_lookup, data.availability, data.cohort, 0, state, args)
    assert history["stage"].tolist() == ["warmup", "joint_partial_unfreeze"]
    assert np.isfinite(history["train_total"]).all()
    assert metadata["selected_stage"] in ("warmup_frozen_survival", "joint_partial_unfreeze")
    assert any(not torch.equal(value, before[f"translator.{key}"]) for key, value in fallback.items())
    for key, value in model.state_dict().items():
        if "running_" in key or "num_batches_tracked" in key or key == "survival.encoders.0.trunk.network.0.weight":
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for index, positions in enumerate(data.master_positions):
        data.inputs[index][data.cohort.iloc[positions]["partition"].eq("test").to_numpy()] *= 10
    data.cohort.loc[data.cohort["partition"].eq("test"), "event"] = 0
    with np.load(path, allow_pickle=False) as saved:
        payload = dict(saved)
    for name in MODALITIES:
        payload[f"{name}_target"][payload["partition"] == "test"] *= 100
    np.savez_compressed(path, **payload)
    replay_state = crossmodal.load_target_state(path, data)
    replay = crossmodal.build_joint_model(source, dims, args)
    replay, _, _, _ = crossmodal._fit_fold(replay, data.inputs, data.local_lookup, data.availability, data.cohort, 0, replay_state, args)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, replay.state_dict()[key], rtol=0, atol=0)


def test_joint_command_saves_selected_and_frozen_models(joint_files, tmp_path, monkeypatch):
    data, path, source, source_path, cohort_path, paths = joint_files
    output = tmp_path / "joint"
    argv = ["omnit2i-train-joint", "--cohort", str(cohort_path), "--source-checkpoint", str(source_path), "--targets", str(path), "--output-dir", str(output), "--warmup-epochs", "1", "--joint-epochs", "1", "--validation-frequency", "1", "--encoder-chunk-size", "4"]
    for name, input_path in zip(MODALITIES, paths):
        argv += [f"--{name}", str(input_path)]
    monkeypatch.setattr("sys.argv", argv)
    crossmodal.main()
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model = FiveModalCrossModalMultitaskModel(OmicsHEEndToEndCox(**checkpoint["survival_model_kwargs"]), TargetConditionedCrossModalTranslator(**checkpoint["translator_kwargs"]))
    model.load_state_dict(checkpoint["model_state"])
    risk, _ = predict_survival(model.survival, data, 4, "cpu")
    np.testing.assert_allclose(risk, pd.read_csv(output / "predictions.csv")["risk_raw"], atol=1e-7)
    fallback = torch.load(output / "frozen_fallback.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(fallback["model_state"])
    for key, value in model.survival.state_dict().items():
        torch.testing.assert_close(value, source["model_state"][key], rtol=0, atol=0)
    with np.load(output / "target_baselines.npz", allow_pickle=False) as saved:
        assert "rna_feature_ids" in saved and "he_cancer_means" in saved
    with pytest.raises(FileExistsError):
        crossmodal.main()
