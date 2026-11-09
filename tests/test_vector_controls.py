from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("sksurv")

from omnit2i.models import MODALITIES, CNNSpatialProjectorEncoder, OmicsHEEndToEndCox, VectorEncoder
from omnit2i.training import crossmodal
from omnit2i.training.fusion import build_fusion_model, fit_fusion, load_molecular_encoders, predict_survival
from omnit2i.training.multimodal_data import load_multimodal_data
from omnit2i.training.rna import build_spatial_cox_model, train_molecular


@pytest.mark.parametrize("variant", ["mlp_matched", "mlp_wide"])
@pytest.mark.parametrize("side,n", [(45, 2000), (17, 217)])
def test_vector_capacity_padding_and_shared_initialization(variant, side, n):
    reference = build_spatial_cox_model(101, .25, grid_height=side, grid_width=side)
    rng_state = torch.get_rng_state().clone()
    model = build_spatial_cox_model(101, .25, grid_height=side, grid_width=side, encoder_architecture=variant)
    torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)
    for key, value in reference.head.state_dict().items():
        torch.testing.assert_close(value, model.head.state_dict()[key], rtol=0, atol=0)
    encoder = model.encoder.eval()
    assert not any(isinstance(module, (nn.Conv2d, nn.Dropout)) for module in encoder.modules())
    if variant == "mlp_matched":
        ratio = sum(p.numel() for p in encoder.parameters()) / sum(p.numel() for p in CNNSpatialProjectorEncoder(side, side).parameters())
        assert abs(ratio - 1) < .2
    else:
        assert encoder.width >= 96
    image = torch.randn(3, 3, side, side)
    padded = image.clone()
    padded.flatten(2)[:, :, n:] = 1e4
    torch.testing.assert_close(encoder(image), encoder(padded), rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["mlp_matched", "mlp_wide"])
def test_vector_abc_training_loading_and_joint_freezing(multimodal_files, tmp_path, variant):
    cohort_path, paths, _, availability = multimodal_files
    data = load_multimodal_data(cohort_path, paths)
    args = Namespace(seed=101, dropout=.25, device="cpu", pretrain_epochs=1, epochs=1, patience=1,
                     pretrain_batch_size=4, inference_batch_size=4, mask_fraction=.15,
                     pretrain_lr=1e-3, lr=1e-3, weight_decay=1e-4, encoder_architecture=variant)
    checkpoints = []
    for name, path in zip(MODALITIES[:4], paths[:4]):
        with np.load(path) as saved:
            images, patients = saved["images"], saved["patient"]
        cohort = data.cohort.set_index("patient").loc[patients].reset_index()
        result = train_molecular(images, cohort, 217 if name == "protein" else 2000, args, modality=name)
        assert result["checkpoint"]["architecture"] == variant
        checkpoint = tmp_path / f"{name}_vector.pt"
        torch.save(result["checkpoint"], checkpoint)
        checkpoints.append(checkpoint)
    fusion = build_fusion_model(101, .25, 96, "gated", encoder_architecture=variant)
    load_molecular_encoders(fusion, checkpoints)
    fusion_args = Namespace(seed=101, device="cpu", encoder_lr=1e-4, fusion_lr=1e-3, weight_decay=1e-4,
                            epochs=1, patience=1, encoder_chunk_size=4, modality_dropout=.3,
                            aux_weight=.25, alignment_weight=1., gradient_clip=5.)
    fusion, _, _ = fit_fusion(fusion, data, fusion_args)
    kwargs = dict(dropout=.25, latent_dim=96, fusion_mode="gated",
                  **{f"{name}_encoder_architecture": variant for name in MODALITIES[:4]})
    restored = OmicsHEEndToEndCox(**kwargs)
    restored.load_state_dict(fusion.state_dict())
    np.testing.assert_array_equal(predict_survival(fusion, data, 4, "cpu")[0], predict_survival(restored, data, 4, "cpu")[0])
    assert all(isinstance(encoder, VectorEncoder) for encoder in restored.encoders[:4])

    targets = dict(patient=data.cohort.patient.to_numpy(str), partition=data.cohort.partition.to_numpy(str),
                   cancer_type=data.cohort.cancer_type.to_numpy(str), availability=availability,
                   modalities=np.asarray(MODALITIES), target_scale=np.asarray(["synthetic_z"] * 5))
    for i, name in enumerate(MODALITIES):
        targets[f"{name}_target"] = np.random.default_rng(i).normal(size=(len(data.cohort), 6)).astype(np.float32)
        targets[f"{name}_feature_ids"] = np.asarray([f"f{j}" for j in range(6)])
    target_path = tmp_path / "targets.npz"
    np.savez_compressed(target_path, **targets)
    state = crossmodal.load_target_state(target_path, data)
    joint_args = Namespace(seed=101, fold=0, device="cpu", mapper_hidden_dim=192, decoder_hidden_dim=384,
        linear_rank=16, translator_dropout=.1, warmup_epochs=1, joint_epochs=1, warmup_lr=5e-4,
        translator_lr=5e-4, fusion_lr=5e-5, encoder_lr=1e-5, weight_decay=1e-4, gradient_clip=5.,
        modality_dropout=.3, source_dropout=.35, aux_weight=.25, alignment_weight=.1, latent_weight=.25,
        feature_weight=1., risk_consistency_weight=.1, latent_cosine_weight=.1, feature_cosine_weight=.05,
        survival_guardrail=.005, encoder_chunk_size=4, crossmodal_batch_size=32,
        crossmodal_validation_batch_size=32, crossmodal_ramp_epochs=10, validation_frequency=1, patience=2)
    joint = crossmodal.build_joint_model(dict(model_kwargs=kwargs, model_state=fusion.state_dict()), {name: 6 for name in MODALITIES}, joint_args)
    groups = crossmodal.configure_joint_trainable_parameters(joint)
    ids = [id(p) for group in groups.values() for p in group]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in joint.parameters() if p.requires_grad}
    for encoder in joint.survival.encoders[:4]:
        assert not any(p.requires_grad for p in encoder.input_block.parameters())
        assert all(p.requires_grad for p in encoder.output.parameters())
    before = {key: value.clone() for key, value in joint.state_dict().items() if "input_block" in key}
    joint, _, history, _ = crossmodal._fit_fold(joint, data.inputs, data.local_lookup, data.availability,
                                               data.cohort, 0, state, joint_args)
    assert history.stage.tolist() == ["warmup", "joint_partial_unfreeze"]
    for key, value in before.items():
        torch.testing.assert_close(value, joint.state_dict()[key], rtol=0, atol=0)
