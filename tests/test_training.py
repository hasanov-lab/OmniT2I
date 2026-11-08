from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("sksurv")

from omnit2i.preprocessing.cnv_t2i import snake_pixel_indices
from omnit2i.training.rna import (
    build_spatial_cox_model, cox_partial_likelihood, main, mask_measured_features,
    masked_reconstruction_loss, measured_values, predict, train_rna,
)


def training_inputs(modality="rna"):
    side = 17 if modality == "protein" else 45
    images = np.random.default_rng(713).normal(size=(14, 3, side, side)).astype(np.float32)
    cohort = pd.DataFrame({
        "patient": [f"p{i}" for i in range(14)], "sample": [f"s{i}" for i in range(14)],
        "cancer_type": ["A"] * 14, "time": np.arange(1, 15), "event": [1] * 14,
        "partition": ["core"] * 8 + ["validation"] * 3 + ["test"] * 3,
    })
    args = Namespace(
        seed=101, dropout=0.25, device="cpu", pretrain_epochs=1, epochs=2, patience=2,
        pretrain_batch_size=4, inference_batch_size=5, mask_fraction=0.15,
        pretrain_lr=1e-3, lr=1e-3, weight_decay=1e-4,
    )
    return images, cohort, args


def test_cox_loss_and_gradient_match_eventwise_breslow_with_ties():
    risk = torch.tensor([0.2, -0.3, 1.2, -0.1, 0.6], dtype=torch.float64, requires_grad=True)
    time = torch.tensor([2, 2, 2, 4, 1])
    event = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
    expected = torch.stack([
        -risk[i] + torch.logsumexp(risk[time >= time[i]], dim=0)
        for i in torch.nonzero(event).flatten()
    ]).mean()
    actual = cox_partial_likelihood(risk, time, event)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, risk)[0], torch.autograd.grad(expected, risk)[0])
    with pytest.raises(ValueError, match="observed event"):
        cox_partial_likelihood(risk, time, torch.zeros_like(event))


def test_masking_preserves_padding_and_historical_top2000_order():
    image = torch.arange(3 * 45 * 45, dtype=torch.float32).reshape(1, 3, 45, 45) + 1
    torch.testing.assert_close(measured_values(image, 2000), image.flatten(2)[:, :, :2000], rtol=0, atol=0)
    masked, mask = mask_measured_features(image, 0.5, 60)
    pixels = np.sort(snake_pixel_indices(45, 45)[:60])
    padding = np.setdiff1d(np.arange(2025), pixels)
    torch.testing.assert_close(masked.flatten(2)[:, :, padding], image.flatten(2)[:, :, padding])
    target = measured_values(image, 60)
    torch.testing.assert_close(measured_values(masked, 60), target.masked_fill(mask.unsqueeze(1), 0))
    loss = masked_reconstruction_loss(target.clone().requires_grad_(), target, mask)
    assert loss.item() == 0 and mask.any(dim=1).all()
    loss.backward()
    with pytest.raises(ValueError, match="strictly"):
        mask_measured_features(image, 0, 60)


@pytest.mark.parametrize("modality", ["rna", "cnv", "protein", "methylation"])
def test_training_ignores_test_values_and_outcomes_and_reloads_checkpoint(modality):
    images, cohort, args = training_inputs(modality)
    first = train_rna(images, cohort, 60, args, modality=modality)
    changed = images.copy()
    changed[-3:] *= 10
    changed_cohort = cohort.copy()
    changed_cohort.loc[11:, "time"] *= 10
    changed_cohort.loc[11:, "event"] = 0
    second = train_rna(changed, changed_cohort, 60, args, modality=modality)
    for key in first["checkpoint"]["model_state"]:
        torch.testing.assert_close(first["checkpoint"]["model_state"][key], second["checkpoint"]["model_state"][key], rtol=0, atol=0)
    assert first["checkpoint"]["best_epoch"] == second["checkpoint"]["best_epoch"]
    assert len(first["pretraining_history"]) == 1 and len(first["history"]) == 2
    np.testing.assert_allclose(first["risk"][8:11].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(first["risk"][8:11].std(), 1, atol=1e-12)
    side = 17 if modality == "protein" else 45
    model = build_spatial_cox_model(args.seed, args.dropout, grid_height=side, grid_width=side)
    model.load_state_dict(first["checkpoint"]["model_state"])
    replayed, _ = predict(model, images, args.inference_batch_size, args.device)
    np.testing.assert_array_equal(replayed, first["risk_raw"])


@pytest.mark.parametrize("modality", ["rna", "cnv", "protein", "methylation"])
def test_training_command_writes_reloadable_artifacts(modality, tmp_path, monkeypatch):
    images, cohort, _ = training_inputs(modality)
    inputs = tmp_path / "images.npz"
    np.savez_compressed(inputs, images=images, feature_ids=np.array([f"g{i}" for i in range(60)]), **{
        column: cohort[column].to_numpy(dtype=str if column not in ("time", "event") else float)
        for column in cohort.columns
    })
    output = tmp_path / "trained"
    monkeypatch.setattr("sys.argv", [
        "omnit2i-train-rna", "--images", str(inputs), "--output-dir", str(output),
        "--modality", modality,
        "--pretrain-epochs", "1", "--epochs", "1",
    ])
    main()
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    predictions = pd.read_csv(output / "predictions.csv")
    model = build_spatial_cox_model(checkpoint["seed"], checkpoint["dropout"], grid_height=checkpoint["grid_height"], grid_width=checkpoint["grid_width"])
    model.load_state_dict(checkpoint["model_state"])
    risk, _ = predict(model, images, 512, "cpu")
    np.testing.assert_allclose(predictions["risk_raw"], risk)
    assert set(predictions["partition"]) == {"core", "validation", "test"}
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["modality"] == modality and checkpoint["n_features"] == 60
    with pytest.raises(FileExistsError):
        main()
