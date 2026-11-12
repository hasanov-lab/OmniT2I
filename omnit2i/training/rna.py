"""Masked spatial molecular pretraining followed by full-risk-set Cox training."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sksurv.metrics import concordance_index_censored

from omnit2i.data import validate_cohort
from omnit2i.models import CNVT2ICoxModel, CNNSpatialProjectorEncoder, MaskedRNAImageAutoencoder, VectorEncoder
from omnit2i.preprocessing.cnv_t2i import snake_pixel_indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_spatial_cox_model(seed: int, dropout: float, device: str = "cpu", *, grid_height: int = 45, grid_width: int = 45, encoder_architecture: str = "masked_cnn_spatial_projector") -> CNVT2ICoxModel:
    """Preserve the production spatial encoder, common-head initialization, and RNG stream."""
    set_seed(seed)
    model = CNVT2ICoxModel(in_channels=3, dropout=dropout)
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    set_seed(seed)
    if encoder_architecture == "masked_cnn_spatial_projector":
        encoder = CNNSpatialProjectorEncoder(grid_height=grid_height, grid_width=grid_width)
        encoder.trunk.load_state_dict(model.encoder.state_dict())
    else:
        encoder = VectorEncoder(encoder_architecture, grid_height, grid_width)
    model.encoder = encoder
    torch.set_rng_state(cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    return model.to(device)


def cox_partial_likelihood(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    risk_sorted = risk[order]
    time_sorted = time[order]
    event_sorted = event[order].bool()
    event_positions = torch.nonzero(event_sorted, as_tuple=False).flatten()
    if event_positions.numel() == 0:
        raise ValueError("Cox loss requires at least one observed event")
    exp_risk = torch.exp(torch.clamp(risk_sorted, min=-30.0, max=30.0))
    cumulative_exp = torch.cumsum(exp_risk, dim=0)
    _, inverse, counts = torch.unique_consecutive(time_sorted, return_inverse=True, return_counts=True)
    group_last = torch.cumsum(counts, dim=0) - 1
    denominators = group_last[inverse[event_positions]]
    return (-risk_sorted[event_positions].sum() + torch.log(cumulative_exp[denominators]).sum()) / event_positions.numel()


def _measured_pixels(image: torch.Tensor, n_features: int) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Image must have shape [patients, 3, height, width]")
    if not 0 < n_features <= image.shape[2] * image.shape[3]:
        raise ValueError("Measured feature count must fit the image grid")
    # Select occupied serpentine pixels in ascending index order; padding stays excluded.
    pixels = np.sort(snake_pixel_indices(image.shape[2], image.shape[3])[:n_features])
    return torch.as_tensor(pixels, device=image.device)


def measured_values(image: torch.Tensor, n_features: int) -> torch.Tensor:
    return image.flatten(start_dim=2).index_select(2, _measured_pixels(image, n_features))


def mask_measured_features(image: torch.Tensor, mask_fraction: float, n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must lie strictly between zero and one")
    pixels = _measured_pixels(image, n_features)
    flat = image.flatten(start_dim=2).clone()
    mask = torch.rand(image.shape[0], n_features, device=image.device) < mask_fraction
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        mask[empty_rows, 0] = True
    flat[:, :, pixels] = flat[:, :, pixels].masked_fill(mask.unsqueeze(1), 0.0)
    return flat.reshape_as(image), mask


def masked_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, omitted_channel: int | None = None) -> torch.Tensor:
    if target.ndim != 3 or target.shape[1] != 3 or prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same [batch, 3, features] shape")
    if mask.shape != (target.shape[0], target.shape[2]) or not mask.any():
        raise ValueError("Mask must match the feature axis and select at least one value")
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    if omitted_channel is not None:
        if omitted_channel not in (1, 2):
            raise ValueError("Only C2 or C3 may be omitted")
        loss = loss[:, [channel for channel in range(3) if channel != omitted_channel]]
    return loss[mask.unsqueeze(1).expand_as(loss)].mean()


def predict(model: CNVT2ICoxModel, images: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    risks, embeddings = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start:start + batch_size]).to(device)
            embedding = model.encode(batch)
            embeddings.append(embedding.cpu().numpy().astype(np.float32))
            risks.append(model.head(embedding).flatten().cpu().numpy().astype(np.float64))
    return np.concatenate(risks), np.concatenate(embeddings)


def train_molecular(images: np.ndarray, cohort: pd.DataFrame, n_features: int, args: argparse.Namespace, *, modality: str = "rna") -> dict:
    """Fit the core partition and select the checkpoint using validation C-index."""
    cohort = validate_cohort(cohort)
    if modality not in {"rna", "cnv", "protein", "methylation"}:
        raise ValueError("Unknown molecular modality")
    side = 17 if modality == "protein" else 45
    if images.shape != (len(cohort), 3, side, side) or not np.isfinite(images).all():
        raise ValueError(f"{modality} images must be finite with shape [patients, 3, {side}, {side}]")
    if not 1 <= n_features <= side * side:
        raise ValueError("Measured feature count must fit the image grid")
    if args.pretrain_epochs < 0 or min(args.epochs, args.patience, args.pretrain_batch_size, args.inference_batch_size) < 1:
        raise ValueError("Pretraining epochs must be nonnegative; other epoch counts, patience, and batch sizes must be positive")
    if not 0 < args.mask_fraction < 1 or not 0 <= args.dropout < 1:
        raise ValueError("mask_fraction must lie in (0, 1) and dropout in [0, 1)")
    if not np.isfinite([args.lr, args.pretrain_lr, args.weight_decay]).all() or min(args.lr, args.pretrain_lr) <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rates must be finite and positive; weight decay must be finite and nonnegative")
    core = cohort["partition"].eq("core").to_numpy()
    validation = cohort["partition"].eq("validation").to_numpy()
    if not cohort.loc[core, "event"].any():
        raise ValueError("Core patients must include an observed event")
    val_time = cohort.loc[validation, "time"].to_numpy(float)
    val_event = cohort.loc[validation, "event"].to_numpy(bool)
    # Fail before fitting if validation cannot select a survival checkpoint.
    concordance_index_censored(val_event, val_time, np.zeros(validation.sum()))
    core_images = torch.from_numpy(images[core])
    architecture = getattr(args, "encoder_architecture", "masked_cnn_spatial_projector")
    if architecture != "masked_cnn_spatial_projector" and n_features != (217 if side == 17 else 2000):
        raise ValueError("Vector controls require the original 2000/217 selected features")
    model = build_spatial_cox_model(args.seed, args.dropout, args.device, grid_height=side, grid_width=side, encoder_architecture=architecture)
    pretrain_history = []
    if args.pretrain_epochs:
        autoencoder = MaskedRNAImageAutoencoder(model.encoder, n_features).to(args.device)
        set_seed(args.seed + 90_000)
        optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.pretrain_lr, weight_decay=args.weight_decay)
        for epoch in range(1, args.pretrain_epochs + 1):
            autoencoder.train()
            permutation = torch.randperm(len(core_images))
            losses = []
            for start in range(0, len(permutation), args.pretrain_batch_size):
                batch = core_images[permutation[start:start + args.pretrain_batch_size]].to(args.device)
                masked, mask = mask_measured_features(batch, args.mask_fraction, n_features)
                optimizer.zero_grad(set_to_none=True)
                loss = masked_reconstruction_loss(autoencoder(masked), measured_values(batch, n_features), mask,
                    omitted_channel={"no_c2": 1, "no_c3": 2}.get(getattr(args, "input_ablation", None)))
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite pretraining loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            pretrain_history.append({"epoch": epoch, "masked_reconstruction_loss": float(np.mean(losses))})
        encoder_state = {key: value.detach().cpu().clone() for key, value in autoencoder.encoder.state_dict().items()}
        del autoencoder, optimizer
        model = build_spatial_cox_model(args.seed, args.dropout, args.device, grid_height=side, grid_width=side, encoder_architecture=architecture)
        model.encoder.load_state_dict(encoder_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    core_images = core_images.to(args.device)
    time = torch.as_tensor(cohort.loc[core, "time"].to_numpy(np.float32), device=args.device)
    event = torch.as_tensor(cohort.loc[core, "event"].to_numpy(np.float32), device=args.device)
    history, best_state = [], None
    best_score, best_epoch, stale = -np.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # One forward and one loss over the entire core risk set, including ties.
        loss = cox_partial_likelihood(model(core_images), time, event)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite Cox loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        val_risk, _ = predict(model, images[validation], args.inference_batch_size, args.device)
        score = float(concordance_index_censored(val_event, val_time, val_risk)[0])
        history.append({"epoch": epoch, "cox_loss": float(loss.detach().cpu()), "validation_c_index": score})
        if score > best_score + 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    risk, embeddings = predict(model, images, args.inference_batch_size, args.device)
    mean, std = float(risk[validation].mean()), float(risk[validation].std())
    if not np.isfinite(risk).all() or not np.isfinite(std) or std < 1e-8:
        raise ValueError("Predictions are non-finite or validation risk scale is degenerate")
    return {
        "checkpoint": {"model_state": best_state, "architecture": "cnn_spatial_projector" if architecture == "masked_cnn_spatial_projector" else architecture, "modality": modality, "grid_height": side, "grid_width": side, "n_features": n_features, "dropout": args.dropout, "seed": args.seed, "best_epoch": best_epoch, "validation_c_index": best_score, "risk_mean": mean, "risk_std": std},
        "risk_raw": risk, "risk": (risk - mean) / std, "embeddings": embeddings,
        "pretraining_history": pd.DataFrame(pretrain_history, columns=["epoch", "masked_reconstruction_loss"]), "history": pd.DataFrame(history),
    }


train_rna = train_molecular


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("rna", "cnv", "protein", "methylation"), default="rna")
    parser.add_argument("--images", type=Path, required=True, help="images.npz from omnit2i-prepare-rna or omnit2i-prepare-omics")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=101, help="Final training seed; supplied per fold/run")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--input-ablation", choices=("no_c2", "no_c3"))
    parser.add_argument("--encoder-architecture", choices=("masked_cnn_spatial_projector", "mlp_matched", "mlp_wide"), default="masked_cnn_spatial_projector")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    with np.load(args.images, allow_pickle=False) as saved:
        images = np.asarray(saved["images"], dtype=np.float32)
        cohort = pd.DataFrame({key: saved[key] for key in ("patient", "sample", "cancer_type", "time", "event", "partition")})
        feature_ids = saved["feature_ids"]
    if feature_ids.ndim != 1 or len(np.unique(feature_ids)) != len(feature_ids):
        raise ValueError("feature_ids must be a one-dimensional array of unique identifiers")
    result = train_molecular(images, cohort, len(feature_ids), args, modality=args.modality)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(result["checkpoint"], args.output_dir / "model.pt")
    predictions = cohort.copy()
    predictions["risk_raw"], predictions["risk"] = result["risk_raw"], result["risk"]
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    np.savez_compressed(args.output_dir / "embeddings.npz", patient=cohort["patient"].to_numpy(str), embeddings=result["embeddings"])
    result["pretraining_history"].to_csv(args.output_dir / "pretraining_history.csv", index=False)
    result["history"].to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "options.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(f"Saved {args.modality} model and predictions: {args.output_dir}")


if __name__ == "__main__":
    main()
