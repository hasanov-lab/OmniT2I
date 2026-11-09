from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("sksurv")

from omnit2i.models import MODALITIES
from omnit2i.preprocessing.cnv_t2i import snake_pixel_indices
from omnit2i.training import rna, workflow
from omnit2i.training.ablations import ablate_images, prepare_inputs


def test_input_ablations_preserve_gene_values_padding_and_historical_random_order():
    order = np.random.default_rng(1).permutation(60)
    pixels = snake_pixel_indices(45, 45)[:60]
    genes = np.arange(2 * 3 * 60).reshape(2, 3, 60).astype(np.float32)
    images = np.zeros((2, 3, 45, 45), np.float32)
    images.reshape(2, 3, -1)[:, :, pixels] = genes[:, :, order]
    original = images.copy()
    for variant, channel in [("no_c2", 1), ("no_c3", 2)]:
        result, unchanged = ablate_images(images, order, variant, 2, "rna")
        assert (result[:, channel] == 0).all()
        np.testing.assert_array_equal(result[:, [c for c in range(3) if c != channel]], images[:, [c for c in range(3) if c != channel]])
        np.testing.assert_array_equal(unchanged, order)
    result, random_order = ablate_images(images, order, "random_layout", 2, "rna")
    expected_order = np.random.default_rng(np.random.SeedSequence([20261014, 2, 0])).permutation(60)
    np.testing.assert_array_equal(random_order, expected_order)
    np.testing.assert_array_equal(result.reshape(2, 3, -1)[:, :, pixels], genes[:, :, expected_order])
    padding = np.setdiff1d(np.arange(2025), pixels)
    assert (result.reshape(2, 3, -1)[:, :, padding] == 0).all()
    np.testing.assert_array_equal(images, original)
    assert not np.array_equal(random_order, ablate_images(images, order, "random_layout", 2, "cnv")[1])


def test_channel_ablation_loss_uses_two_active_channels_only():
    prediction = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3).requires_grad_()
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    loss = rna.masked_reconstruction_loss(prediction, target, mask, omitted_channel=1)
    selected = torch.nn.functional.smooth_l1_loss(prediction[:, [0, 2]], target[:, [0, 2]], reduction="none")
    torch.testing.assert_close(loss, selected[mask[:, None].expand_as(selected)].mean())
    loss.backward()
    assert (prediction.grad[:, 1] == 0).all() and prediction.grad[:, [0, 2]].abs().sum() > 0


def prepared_inputs(root):
    root.mkdir()
    pd.DataFrame(dict(seed=[101], outer_fold=[2])).to_csv(root / "cohort.csv", index=False)
    for name in ("he.npz", "targets.npz"):
        np.savez(root / name, values=np.ones(1))
    for name in MODALITIES[:4]:
        folder = root / name
        folder.mkdir()
        np.savez(folder / "images.npz", images=np.ones((1, 3, 45, 45), np.float32), feature_ids=np.array(["g1", "g2"]), patient=np.array(["p1"]))
        np.savez(folder / "preprocessing.npz", feature_ids=np.array(["g1", "g2"]), layout_order=np.array([1, 0]))
    return root


@pytest.mark.parametrize("variant,calls", [("no_pretraining", 6), ("no_c2", 6), ("no_c3", 6), ("random_layout", 6),
    ("no_risk_consistency", 1), ("mean_fusion", 2), ("no_modality_dropout", 2), ("simplified", 1)])
def test_ablation_training_reuses_correct_stages_and_shared_inputs(tmp_path, monkeypatch, variant, calls):
    prepared = prepared_inputs(tmp_path / "prepared")
    reference = tmp_path / "reference"
    (reference / "B").mkdir(parents=True)
    torch.save({"model_kwargs": {"fusion_mode": "mean"}}, reference / "B/model.pt")
    for name in MODALITIES[:4]:
        (reference / "A" / name).mkdir(parents=True)
        (reference / "A" / name / "model.pt").touch()
    commands = []
    monkeypatch.setattr(workflow.subprocess, "run", lambda argv, check: commands.append(argv))
    output = tmp_path / "ablation"
    workflow.train_fold(prepared, output, 101, 2, "cpu", variant=variant, reference=reference)
    assert len(commands) == calls
    stage_c = commands[-1]
    assert stage_c[stage_c.index("--seed") + 1] == "101"
    if variant in {"no_risk_consistency", "simplified"}:
        assert stage_c[stage_c.index("--source-checkpoint") + 1] == str(reference / "B/model.pt")
        assert stage_c[stage_c.index("--risk-consistency-weight") + 1] == "0"
    if variant == "no_pretraining":
        assert all(argv[argv.index("--pretrain-epochs") + 1] == "0" for argv in commands[:4])
    if variant == "mean_fusion":
        assert commands[0][commands[0].index("--fusion-mode") + 1] == "mean"
    if variant == "no_modality_dropout":
        assert all(argv[argv.index("--modality-dropout") + 1] == "0" for argv in commands)
    if variant in {"no_c2", "no_c3", "random_layout"}:
        with np.load(output / "inputs/rna/images.npz") as saved:
            if variant in {"no_c2", "no_c3"}:
                assert (saved["images"][:, 1 if variant == "no_c2" else 2] == 0).all()
                assert all(argv[argv.index("--input-ablation") + 1] == variant for argv in commands[:4])
        assert (output / "inputs/targets.npz").resolve() == (prepared / "targets.npz").resolve()
    with pytest.raises(FileExistsError):
        workflow.train_fold(prepared, output, 101, 2, "cpu", variant=variant, reference=reference)


def test_no_pretraining_never_constructs_autoencoder(tmp_path, monkeypatch):
    images = np.random.default_rng(7).normal(size=(14, 3, 45, 45)).astype(np.float32)
    cohort = pd.DataFrame(dict(patient=[f"p{i}" for i in range(14)], sample=[f"s{i}" for i in range(14)],
        cancer_type="A", time=np.arange(14) + 1., event=1, partition=["core"] * 8 + ["validation"] * 3 + ["test"] * 3))
    args = Namespace(seed=101, dropout=.25, device="cpu", pretrain_epochs=0, epochs=2, patience=2,
        pretrain_batch_size=4, inference_batch_size=5, mask_fraction=.15, pretrain_lr=.001, lr=.001, weight_decay=.0001)
    def fail(*args, **kwargs):
        raise AssertionError("No-pretraining must not instantiate a reconstruction model")
    monkeypatch.setattr(rna, "MaskedRNAImageAutoencoder", fail)
    result = rna.train_molecular(images, cohort, 60, args)
    assert result["pretraining_history"].empty and len(result["history"]) == 2
    assert np.isfinite(result["risk"]).all()
