"""Translator warm-up and partial-unfreeze joint training from prepared inputs."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from omnit2i.models import (
    MODALITIES, CNNSpatialProjectorEncoder, CNNSpatialGlobalEncoder, VectorEncoder,
    OmicsHEEndToEndCox, FiveModalCrossModalMultitaskModel,
    TargetConditionedCrossModalTranslator,
)
from .fusion import add_data_arguments, predict_survival
from .losses import _survival_loss, c_index, fit_risk_calibration, apply_risk_calibration
from .multimodal_data import MultimodalData, load_multimodal_data
from .rna import set_seed


@dataclass
class FoldTargetState:
    """One inner-core reference shared by every residual/add-back operation."""

    core_mask: np.ndarray
    validation_mask: np.ndarray
    residual_targets: list[np.ndarray]
    feature_baselines: dict[str, dict[str, object]]


def eligible_translation_rows(
    population: np.ndarray,
    availability: np.ndarray,
    target: int,
) -> np.ndarray:
    """Observed target rows having at least one genuinely observed other source."""
    available = np.asarray(availability, dtype=bool)
    if available.ndim != 2 or available.shape[1] != len(MODALITIES):
        raise ValueError("availability must have shape [patients, five modalities]")
    rows = np.asarray(population, dtype=bool).copy()
    if rows.shape != (len(available),):
        raise ValueError("population mask does not match availability")
    non_target = available.copy()
    non_target[:, int(target)] = False
    return rows & available[:, int(target)] & non_target.any(axis=1)


def sample_source_mask(
    availability: torch.Tensor,
    target: int,
    dropout_probability: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop complete source modalities per patient while retaining one source.

    The target is always removed before sampling.  This is modality-level
    masking, not element-wise feature masking, and it is applied only to
    modalities that are naturally observed for that patient.
    """
    if availability.ndim != 2 or availability.shape[1] != len(MODALITIES):
        raise ValueError("availability must have shape [patients, five modalities]")
    if not 0.0 <= float(dropout_probability) < 1.0:
        raise ValueError("dropout_probability must be in [0, 1)")
    natural = availability.bool().clone()
    natural[:, int(target)] = False
    if (~natural.any(dim=1)).any():
        raise ValueError("Every selected row must have a non-target source")
    if dropout_probability == 0.0:
        return natural
    keep = torch.rand(
        natural.shape,
        device=natural.device,
        generator=generator,
    ) >= float(dropout_probability)
    sampled = natural & keep
    empty = ~sampled.any(dim=1)
    if empty.any():
        # Uniformly select one of the patient's naturally available sources.
        scores = torch.rand(
            natural.shape,
            device=natural.device,
            generator=generator,
        ).masked_fill(~natural, -1.0)
        replacement = scores.argmax(dim=1)
        sampled[empty, replacement[empty]] = True
    sampled[:, int(target)] = False
    return sampled


def fused_risk_from_projected(
    fusion: nn.Module,
    projected: torch.Tensor,
    availability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the existing fusion gate/head to already projected embeddings."""
    if projected.ndim != 3:
        raise ValueError("projected must have shape [patients, modalities, latent]")
    if availability.shape != projected.shape[:2]:
        raise ValueError("availability shape does not match projected embeddings")
    available = availability.bool()
    if (~available.any(dim=1)).any():
        raise ValueError("Every patient needs at least one available modality")
    if getattr(fusion, "fusion_mode", None) == "mean":
        weights = available.to(projected.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
    else:
        gates = getattr(fusion, "gates", None)
        if gates is None or len(gates) != projected.shape[1]:
            raise ValueError("Fusion module has incompatible gates")
        logits = torch.cat(
            [gate(projected[:, index]) for index, gate in enumerate(gates)],
            dim=1,
        )
        logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
    fused = (weights.unsqueeze(-1) * projected).sum(dim=1)
    # Risk consistency must compare the two representations with the *same*
    # deterministic risk map.  Calling fused_head twice in train mode would
    # otherwise draw two independent dropout masks and create a non-zero loss
    # even when the predicted target exactly equals the observed target.
    risk = fused
    for module in fusion.fused_head:
        if isinstance(module, nn.Dropout):
            continue
        risk = module(risk)
    return risk.flatten(), weights


def risk_consistency_loss(
    fusion: nn.Module,
    true_projected: torch.Tensor,
    predicted_projected: torch.Tensor,
    availability: torch.Tensor,
) -> torch.Tensor:
    """Deterministic survival-risk agreement for true vs imputed target latent."""
    true_risk, _ = fused_risk_from_projected(
        fusion, true_projected, availability
    )
    predicted_risk, _ = fused_risk_from_projected(
        fusion, predicted_projected, availability
    )
    return F.smooth_l1_loss(predicted_risk, true_risk.detach())


def _smooth_cosine_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    cosine_weight: float,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, truth)
    if cosine_weight:
        loss = loss + float(cosine_weight) * (
            1.0 - F.cosine_similarity(prediction, truth, dim=1).mean()
        )
    return loss


def crossmodal_objective(
    translator: TargetConditionedCrossModalTranslator,
    fusion: nn.Module,
    projected: torch.Tensor,
    availability: torch.Tensor,
    global_rows: np.ndarray,
    population: np.ndarray,
    residual_targets: list[np.ndarray],
    batch_size: int,
    source_dropout: float,
    latent_cosine_weight: float,
    feature_cosine_weight: float,
    generator: torch.Generator | None,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Compute latent, measured-profile, and risk-consistency objectives.

    The observed target embedding is detached and used only as supervision.
    The target itself is excluded from the translator input.  Feature truth is
    the fold-safe within-cancer residual in the measured target feature space.
    """
    if projected.ndim != 3 or projected.shape[1] != len(MODALITIES):
        raise ValueError("projected must have shape [patients, five, latent]")
    if len(global_rows) != len(projected):
        raise ValueError("global_rows and projected patient counts differ")
    if np.asarray(population, dtype=bool).shape != (len(projected),):
        raise ValueError("population mask and projected patient counts differ")
    device = projected.device
    latent_losses: list[torch.Tensor] = []
    feature_losses: list[torch.Tensor] = []
    consistency_losses: list[torch.Tensor] = []
    counts: dict[str, int] = {}

    available_np = availability.detach().cpu().numpy().astype(bool)
    population_np = np.asarray(population, dtype=bool)
    for target, target_name in enumerate(MODALITIES):
        eligible = eligible_translation_rows(population_np, available_np, target)
        positions = np.flatnonzero(eligible)
        if len(positions) == 0:
            counts[target_name] = 0
            continue
        if len(positions) > int(batch_size):
            if generator is None:
                positions = positions[: int(batch_size)]
            else:
                permutation = torch.randperm(
                    len(positions), generator=generator, device=device
                )[: int(batch_size)].cpu().numpy()
                positions = positions[permutation]
        selected = torch.from_numpy(positions.astype(np.int64)).to(device)
        selected_projected = projected[selected]
        selected_availability = availability[selected]
        source_mask = sample_source_mask(
            selected_availability,
            target,
            source_dropout,
            generator,
        )
        translated = translator(selected_projected, source_mask, target)
        target_latent = selected_projected[:, target].detach()
        feature_truth = torch.from_numpy(
            residual_targets[target][global_rows[positions]]
        ).to(device=device, dtype=projected.dtype)

        latent_losses.append(
            _smooth_cosine_loss(
                translated["latent_prediction"],
                target_latent,
                latent_cosine_weight,
            )
        )
        feature_losses.append(
            _smooth_cosine_loss(
                translated["feature_prediction"],
                feature_truth,
                feature_cosine_weight,
            )
        )

        completed_availability = source_mask.clone()
        completed_availability[:, target] = True
        predicted_projected = selected_projected.clone()
        predicted_projected[:, target] = translated["latent_prediction"]
        consistency_losses.append(
            risk_consistency_loss(
                fusion,
                selected_projected,
                predicted_projected,
                completed_availability,
            )
        )
        counts[target_name] = int(len(positions))

    if not latent_losses:
        zero = projected.sum() * 0.0
        return {"latent": zero, "feature": zero, "risk_consistency": zero}, counts
    return {
        "latent": torch.stack(latent_losses).mean(),
        "feature": torch.stack(feature_losses).mean(),
        "risk_consistency": torch.stack(consistency_losses).mean(),
    }, counts


def configure_joint_trainable_parameters(
    model: FiveModalCrossModalMultitaskModel,
) -> dict[str, list[nn.Parameter]]:
    """Freeze early CNNs and expose only the prescribed joint-training groups."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    groups: dict[str, list[nn.Parameter]] = {
        "translator": [],
        "fusion": [],
        "encoder": [],
    }
    for parameter in model.translator.parameters():
        parameter.requires_grad_(True)
        groups["translator"].append(parameter)
    for parameter in model.survival.fusion.parameters():
        parameter.requires_grad_(True)
        groups["fusion"].append(parameter)

    # Four omics encoders: unfreeze only the final 96-channel convolution block
    # (Conv2d + BatchNorm affine parameters). BatchNorm running statistics stay
    # fixed through freeze_batchnorm_statistics() in every training epoch.
    for modality, encoder in enumerate(model.survival.encoders[:4]):
        if isinstance(encoder, VectorEncoder):
            trainable_modules = [encoder.output]
        elif isinstance(encoder, CNNSpatialProjectorEncoder):
            trainable_modules = list(encoder.trunk.network.children())[8:]
        else:
            trainable_modules = list(encoder.network.children())[8:]
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                groups["encoder"].append(parameter)
        if modality == 0 and isinstance(encoder, CNNSpatialGlobalEncoder):
            encoder.global_direction.requires_grad_(True)
            groups["encoder"].append(encoder.global_direction)
    # The H&E foundation model is not present here; only its lightweight
    # 1,536->96 adapter is updated.
    for parameter in model.survival.encoders[4].parameters():
        parameter.requires_grad_(True)
        groups["encoder"].append(parameter)
    if any(not values for values in groups.values()):
        raise ValueError("A joint optimizer parameter group is unexpectedly empty")
    return groups


def guardrail_accepts(
    validation_c_index: float,
    baseline_c_index: float,
    tolerance: float,
) -> bool:
    return bool(
        np.isfinite(validation_c_index)
        and validation_c_index >= baseline_c_index - float(tolerance)
    )


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _validation_state(
    model: FiveModalCrossModalMultitaskModel,
    images: list[torch.Tensor],
    local: list[torch.Tensor],
    availability: torch.Tensor,
    master: pd.DataFrame,
    validation_rows: np.ndarray,
    residual_targets: list[np.ndarray],
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[float, float, dict[str, float]]:
    device = availability.device
    selected = torch.from_numpy(validation_rows.astype(np.int64)).to(device)
    validation_generator = torch.Generator(device=device)
    validation_generator.manual_seed(
        args.seed + int(validation_rows[0]) * 37 + 19
    )
    model.eval()
    with torch.no_grad():
        result = model.survival.forward_selected(
            images,
            [indices[selected] for indices in local],
            availability[selected],
            args.encoder_chunk_size,
        )
        projected = torch.stack(result["projected"], dim=1)
        population = np.ones(len(validation_rows), dtype=bool)
        losses, _ = crossmodal_objective(
            model.translator,
            model.survival.fusion,
            projected,
            availability[selected],
            validation_rows,
            population,
            residual_targets,
            args.crossmodal_validation_batch_size or batch_size,
            0.0,
            args.latent_cosine_weight,
            args.feature_cosine_weight,
            generator=validation_generator,
        )
        risk = result["risk"].cpu().numpy()
    clinical = master.iloc[validation_rows]
    validation_c = c_index(
        clinical["event"].to_numpy(bool),
        clinical["time"].to_numpy(float),
        risk,
    )
    cross = {
        key: float(value.detach().cpu().item()) for key, value in losses.items()
    }
    score = (
        args.latent_weight * cross["latent"]
        + args.feature_weight * cross["feature"]
        + args.risk_consistency_weight * cross["risk_consistency"]
    )
    return float(validation_c), float(score), cross


def _fit_fold(
    model: FiveModalCrossModalMultitaskModel,
    images_np: list[np.ndarray],
    local_lookup: list[np.ndarray],
    availability_np: np.ndarray,
    master: pd.DataFrame,
    fold: int,
    target_state: FoldTargetState,
    args: argparse.Namespace,
) -> tuple[
    FiveModalCrossModalMultitaskModel,
    dict[str, object],
    pd.DataFrame,
    dict[str, torch.Tensor],
]:
    device = torch.device(args.device)
    model = model.to(device)
    images = [torch.from_numpy(values).to(device) for values in images_np]
    local = [torch.from_numpy(values).to(device) for values in local_lookup]
    availability = torch.from_numpy(availability_np).to(device)
    time = torch.from_numpy(master["time"].to_numpy(np.float32)).to(device)
    event = torch.from_numpy(master["event"].to_numpy(np.float32)).to(device)

    core_rows = np.flatnonzero(target_state.core_mask)
    validation_rows = np.flatnonzero(target_state.validation_mask)
    residual_targets = target_state.residual_targets
    core_tensor = torch.from_numpy(core_rows.astype(np.int64)).to(device)
    history: list[dict[str, object]] = []

    # Freeze the complete survival model for translator/decoder warm-up.
    for parameter in model.survival.parameters():
        parameter.requires_grad_(False)
    for parameter in model.translator.parameters():
        parameter.requires_grad_(True)
    warm_optimizer = torch.optim.AdamW(
        model.translator.parameters(),
        lr=args.warmup_lr,
        weight_decay=args.weight_decay,
    )
    model.survival.eval()
    with torch.no_grad():
        core_result = model.survival.forward_selected(
            images,
            [indices[core_tensor] for indices in local],
            availability[core_tensor],
            args.encoder_chunk_size,
        )
        core_projected = torch.stack(core_result["projected"], dim=1).detach()

    best_warm_state = copy.deepcopy(model.translator.state_dict())
    best_warm_cross = np.inf
    warm_stale = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + fold * 1000 + 41)
    for epoch in range(1, args.warmup_epochs + 1):
        model.translator.train()
        warm_optimizer.zero_grad(set_to_none=True)
        losses, counts = crossmodal_objective(
            model.translator,
            model.survival.fusion,
            core_projected,
            availability[core_tensor],
            core_rows,
            np.ones(len(core_rows), dtype=bool),
            residual_targets,
            args.crossmodal_batch_size,
            args.source_dropout,
            args.latent_cosine_weight,
            args.feature_cosine_weight,
            generator,
        )
        objective = (
            args.latent_weight * losses["latent"]
            + args.feature_weight * losses["feature"]
            + args.risk_consistency_weight * losses["risk_consistency"]
        )
        if not torch.isfinite(objective):
            raise ValueError(f"Non-finite warm-up loss at epoch {epoch}")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.translator.parameters(), args.gradient_clip, error_if_nonfinite=True)
        warm_optimizer.step()
        row: dict[str, object] = {
            "stage": "warmup",
            "epoch": epoch,
            "train_total": float(objective.detach().cpu()),
            **{f"train_{key}": float(value.detach().cpu()) for key, value in losses.items()},
            **{f"n_{key}": value for key, value in counts.items()},
        }
        if epoch % args.validation_frequency == 0 or epoch == args.warmup_epochs:
            validation_c, validation_cross, validation_parts = _validation_state(
                model,
                images,
                local,
                availability,
                master,
                validation_rows,
                residual_targets,
                args.crossmodal_batch_size,
                args,
            )
            row.update(
                {
                    "validation_c_index": validation_c,
                    "validation_crossmodal": validation_cross,
                    **{f"validation_{key}": value for key, value in validation_parts.items()},
                }
            )
            if validation_cross < best_warm_cross - 1e-6:
                best_warm_cross = validation_cross
                best_warm_state = copy.deepcopy(model.translator.state_dict())
                warm_stale = 0
            else:
                warm_stale += 1
        history.append(row)
        if warm_stale >= args.patience:
            break
    model.translator.load_state_dict(best_warm_state)

    baseline_c, best_cross, validation_parts = _validation_state(
        model,
        images,
        local,
        availability,
        master,
        validation_rows,
        residual_targets,
        args.crossmodal_batch_size,
        args,
    )
    frozen_fallback_state = {
        key: value.detach().cpu().clone()
        for key, value in best_warm_state.items()
    }
    frozen_fallback_validation = {
        "frozen_fallback_validation_c_index": baseline_c,
        "frozen_fallback_validation_crossmodal": best_cross,
        **{
            f"frozen_fallback_validation_{key}": value
            for key, value in validation_parts.items()
        },
    }
    best_state = _copy_state(model)
    best_epoch = 0
    selected_stage = "warmup_frozen_survival"

    parameter_groups = configure_joint_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": parameter_groups["translator"], "lr": args.translator_lr},
            {"params": parameter_groups["fusion"], "lr": args.fusion_lr},
            {"params": parameter_groups["encoder"], "lr": args.encoder_lr},
        ],
        weight_decay=args.weight_decay,
    )
    stale = 0
    for epoch in range(1, args.joint_epochs + 1):
        model.train()
        model.survival.freeze_batchnorm_statistics()
        optimizer.zero_grad(set_to_none=True)
        result = model.survival.forward_selected(
            images,
            [indices[core_tensor] for indices in local],
            availability[core_tensor],
            args.encoder_chunk_size,
            modality_dropout_probability=args.modality_dropout,
        )
        survival_loss, survival_parts = _survival_loss(
            result,
            time[core_tensor],
            event[core_tensor],
            availability[core_tensor],
            args.aux_weight,
            args.alignment_weight,
        )
        projected = torch.stack(result["projected"], dim=1)
        cross_losses, counts = crossmodal_objective(
            model.translator,
            model.survival.fusion,
            projected,
            availability[core_tensor],
            core_rows,
            np.ones(len(core_rows), dtype=bool),
            residual_targets,
            args.crossmodal_batch_size,
            args.source_dropout,
            args.latent_cosine_weight,
            args.feature_cosine_weight,
            generator,
        )
        ramp = min(1.0, epoch / max(1, args.crossmodal_ramp_epochs))
        cross_total = (
            args.latent_weight * cross_losses["latent"]
            + args.feature_weight * cross_losses["feature"]
            + args.risk_consistency_weight * cross_losses["risk_consistency"]
        )
        total = survival_loss + ramp * cross_total
        if not torch.isfinite(total):
            raise ValueError(f"Non-finite joint loss at epoch {epoch}")
        total.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        row = {
            "stage": "joint_partial_unfreeze",
            "epoch": epoch,
            "ramp": ramp,
            "train_total": float(total.detach().cpu()),
            "train_survival": float(survival_loss.detach().cpu()),
            "train_crossmodal": float(cross_total.detach().cpu()),
            **{f"train_{key}": float(value.detach().cpu()) for key, value in survival_parts.items()},
            **{f"train_{key}": float(value.detach().cpu()) for key, value in cross_losses.items()},
            **{f"n_{key}": value for key, value in counts.items()},
        }
        if epoch % args.validation_frequency == 0 or epoch == args.joint_epochs:
            validation_c, validation_cross, parts = _validation_state(
                model,
                images,
                local,
                availability,
                master,
                validation_rows,
                residual_targets,
                args.crossmodal_batch_size,
                args,
            )
            accepted = guardrail_accepts(
                validation_c, baseline_c, args.survival_guardrail
            )
            row.update(
                {
                    "validation_c_index": validation_c,
                    "validation_crossmodal": validation_cross,
                    "survival_guardrail_accepted": accepted,
                    **{f"validation_{key}": value for key, value in parts.items()},
                }
            )
            if accepted and validation_cross < best_cross - 1e-6:
                best_cross = validation_cross
                best_state = _copy_state(model)
                best_epoch = epoch
                selected_stage = "joint_partial_unfreeze"
                validation_parts = parts
                stale = 0
            else:
                stale += 1
        history.append(row)
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    metadata: dict[str, object] = {
        "selected_stage": selected_stage,
        "best_joint_epoch": best_epoch,
        "baseline_validation_c_index": baseline_c,
        "selected_validation_crossmodal": best_cross,
        **{f"selected_validation_{key}": value for key, value in validation_parts.items()},
        "n_inner_core": int(len(core_rows)),
        "n_inner_validation": int(len(validation_rows)),
        **frozen_fallback_validation,
    }
    return model, metadata, pd.DataFrame(history), frozen_fallback_state


def _baseline_for_cancers(
    baseline: dict[str, object],
    cancer_type: np.ndarray,
) -> np.ndarray:
    """Map cancer labels to one fitted inner-core mean with global fallback."""
    labels = np.asarray(baseline["cancer_labels"], dtype=str)
    means = np.asarray(baseline["cancer_means"], dtype=np.float32)
    global_mean = np.asarray(baseline["global_mean"], dtype=np.float32)
    lookup = {label: means[index] for index, label in enumerate(labels)}
    return np.stack(
        [lookup.get(label, global_mean) for label in np.asarray(cancer_type, dtype=str)]
    ).astype(np.float32)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("modality_dropout", "source_dropout", "translator_dropout"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    for name in (
        "aux_weight",
        "alignment_weight",
        "latent_weight",
        "feature_weight",
        "risk_consistency_weight",
        "latent_cosine_weight",
        "feature_cosine_weight",
        "survival_guardrail",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in (
        "validation_frequency",
        "patience",
        "crossmodal_batch_size",
        "crossmodal_validation_batch_size",
        "encoder_chunk_size",
        "crossmodal_ramp_epochs",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    for name in (
        "warmup_lr",
        "translator_lr",
        "fusion_lr",
        "encoder_lr",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmup_epochs < 0 or args.joint_epochs < 0:
        raise ValueError("warmup_epochs and joint_epochs must be non-negative")


def load_target_state(path: Path, data: MultimodalData) -> FoldTargetState:
    """Fit the production within-cancer residual references on observed core rows."""
    core = data.cohort["partition"].eq("core").to_numpy()
    validation = data.cohort["partition"].eq("validation").to_numpy()
    cancers = data.cohort["cancer_type"].to_numpy(str)
    residuals, baselines = [], {}
    with np.load(path, allow_pickle=False) as saved:
        patients = saved["patient"].astype(str)
        if patients.ndim != 1 or len(np.unique(patients)) != len(patients):
            raise ValueError("Target patient IDs must be one-dimensional and unique")
        order = pd.Index(patients).get_indexer(data.cohort["patient"])
        if len(patients) != len(data.cohort) or (order < 0).any():
            raise ValueError("Target archive must cover exactly the master cohort")
        if not np.array_equal(saved["modalities"].astype(str), MODALITIES):
            raise ValueError("Target modality order must be RNA, CNV, protein, methylation, H&E")
        if not np.array_equal(saved["availability"][order], data.availability):
            raise ValueError("Target availability disagrees with the observed inputs")
        for column in ("cancer_type", "partition"):
            if column in saved and not np.array_equal(saved[column][order], data.cohort[column].to_numpy()):
                raise ValueError(f"Target {column} disagrees with the cohort")
        if "outer_test" in saved and not np.array_equal(saved["outer_test"][order], data.cohort["partition"].eq("test")):
            raise ValueError("Target outer_test disagrees with the cohort")
        scales = saved["target_scale"].astype(str)
        if scales.shape != (5,) or any(not value.strip() for value in scales):
            raise ValueError("Record one target scale per modality")
        for target, name in enumerate(MODALITIES):
            values = np.asarray(saved[f"{name}_target"], dtype=np.float32)
            features = saved[f"{name}_feature_ids"].astype(str)
            if features.ndim != 1 or not len(features) or len(np.unique(features)) != len(features):
                raise ValueError(f"{name} feature IDs must be nonempty and unique")
            if values.shape != (len(patients), len(features)):
                raise ValueError(f"{name} target shape disagrees with patients or feature IDs")
            values = values[order]
            observed = data.availability[:, target]
            if not np.isfinite(values[observed]).all():
                raise ValueError(f"Observed {name} targets must be finite")
            observed_core = core & observed
            labels = np.unique(cancers[observed_core])
            baseline = {
                "feature_ids": features, "target_scale": str(scales[target]),
                "global_mean": values[observed_core].mean(axis=0).astype(np.float32),
                "cancer_labels": labels,
                "cancer_means": np.stack([values[observed_core & (cancers == label)].mean(axis=0) for label in labels]).astype(np.float32),
                "n_inner_core_observed": int(observed_core.sum()),
            }
            transformed = np.zeros_like(values)
            for population in (core, validation):
                rows = population & observed
                if rows.any():
                    transformed[rows] = values[rows] - _baseline_for_cancers(baseline, cancers[rows])
            baselines[name] = baseline
            residuals.append(transformed)
    for name, population in (("core", core), ("validation", validation)):
        if not (population & (data.availability.sum(axis=1) >= 2)).any():
            raise ValueError(f"The {name} partition needs patients with at least two observed modalities")
    return FoldTargetState(core, validation, residuals, baselines)


def build_joint_model(source: dict, output_dims: dict[str, int], args: argparse.Namespace) -> FiveModalCrossModalMultitaskModel:
    """Preserve the production translator RNG after the pooled reference model."""
    set_seed(args.seed + args.fold * 1000)
    model_kwargs = source["model_kwargs"]
    reference = OmicsHEEndToEndCox(
        latent_dim=model_kwargs["latent_dim"], dropout=model_kwargs["dropout"],
        fusion_mode=model_kwargs["fusion_mode"],
    )
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    del reference
    survival = OmicsHEEndToEndCox(**model_kwargs)
    survival.load_state_dict(source["model_state"])
    torch.set_rng_state(cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    translator = TargetConditionedCrossModalTranslator(
        output_dims=output_dims, latent_dim=model_kwargs["latent_dim"],
        mapper_hidden_dim=args.mapper_hidden_dim, decoder_hidden_dim=args.decoder_hidden_dim,
        linear_rank=args.linear_rank, dropout=args.translator_dropout,
    )
    return FiveModalCrossModalMultitaskModel(survival, translator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--source-checkpoint", type=Path, required=True, help="Model saved by omnit2i-train-fusion")
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0, help="Fold index used in the production random seed")
    for name, default in {
        "mapper-hidden-dim": 192, "decoder-hidden-dim": 384, "linear-rank": 16,
        "warmup-epochs": 40, "joint-epochs": 80, "validation-frequency": 5,
        "patience": 10, "crossmodal-batch-size": 768, "crossmodal-validation-batch-size": 2048,
        "crossmodal-ramp-epochs": 10,
    }.items():
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in {
        "translator-dropout": 0.10, "warmup-lr": 5e-4, "translator-lr": 5e-4,
        "fusion-lr": 5e-5, "encoder-lr": 1e-5, "weight-decay": 1e-4,
        "gradient-clip": 5.0, "modality-dropout": 0.30, "source-dropout": 0.35,
        "aux-weight": 0.25, "alignment-weight": 0.10, "latent-weight": 0.25,
        "feature-weight": 1.0, "risk-consistency-weight": 0.10,
        "latent-cosine-weight": 0.10, "feature-cosine-weight": 0.05,
        "survival-guardrail": 0.005,
    }.items():
        parser.add_argument(f"--{name}", type=float, default=default)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    _validate_args(args)
    if args.fold < 0 or min(args.mapper_hidden_dim, args.decoder_hidden_dim, args.linear_rank) < 1:
        raise ValueError("Fold must be nonnegative and translator dimensions positive")
    if args.warmup_epochs + args.joint_epochs < 1:
        raise ValueError("At least one warm-up or joint epoch is required")
    for name, value in vars(args).items():
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if args.gradient_clip <= 0 or args.weight_decay < 0:
        raise ValueError("Gradient clip must be positive and weight decay nonnegative")
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    data = load_multimodal_data(args.cohort, [getattr(args, name) for name in MODALITIES], he_standardization=source["he_standardization"])
    target_state = load_target_state(args.targets, data)
    core = target_state.core_mask
    validation = target_state.validation_mask
    if not data.cohort.loc[core, "event"].any():
        raise ValueError("Core patients must include an observed event")
    c_index(data.cohort.loc[validation, "event"], data.cohort.loc[validation, "time"], np.zeros(validation.sum()))
    output_dims = {name: len(target_state.feature_baselines[name]["feature_ids"]) for name in MODALITIES}
    model = build_joint_model(source, output_dims, args)
    model, metadata, history, fallback_translator = _fit_fold(
        model, data.inputs, data.local_lookup, data.availability, data.cohort,
        args.fold, target_state, args,
    )
    risk, weights = predict_survival(model.survival, data, args.encoder_chunk_size, args.device)
    calibration = fit_risk_calibration(risk, validation)
    translator_kwargs = {
        "output_dims": output_dims, "latent_dim": source["model_kwargs"]["latent_dim"],
        "mapper_hidden_dim": args.mapper_hidden_dim, "decoder_hidden_dim": args.decoder_hidden_dim,
        "linear_rank": args.linear_rank, "dropout": args.translator_dropout,
    }
    common = {
        "survival_model_kwargs": source["model_kwargs"], "translator_kwargs": translator_kwargs,
        "he_standardization": source["he_standardization"], "seed": args.seed, "fold": args.fold,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({**common, "model_state": _copy_state(model), "risk_calibration": calibration, **metadata}, args.output_dir / "model.pt")
    fallback_state = {f"survival.{key}": value for key, value in source["model_state"].items()}
    fallback_state.update({f"translator.{key}": value for key, value in fallback_translator.items()})
    torch.save({**common, "model_state": fallback_state, "risk_calibration": source["risk_calibration"]}, args.output_dir / "frozen_fallback.pt")
    np.savez_compressed(args.output_dir / "target_baselines.npz", **{
        f"{name}_{key}": value for name, baseline in target_state.feature_baselines.items()
        for key, value in baseline.items()
    })
    predictions = data.cohort.copy()
    predictions["risk_raw"], predictions["risk"] = risk, apply_risk_calibration(risk, calibration)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(weights, index=pd.Index(data.cohort["patient"], name="patient"), columns=MODALITIES).to_csv(args.output_dir / "fusion_weights.csv")
    history.to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "options.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(f"Saved cross-modal joint model: {args.output_dir}")


if __name__ == "__main__":
    main()
