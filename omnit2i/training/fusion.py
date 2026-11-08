"""Train the five-modality survival backbone from prepared, observed inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from omnit2i.models import MODALITIES, OmicsHEEndToEndCox
from .losses import _survival_loss, apply_risk_calibration, c_index, fit_risk_calibration
from .multimodal_data import MultimodalData, load_multimodal_data
from .rna import set_seed


def build_fusion_model(seed: int, dropout: float, latent_dim: int, fusion_mode: str) -> OmicsHEEndToEndCox:
    """Keep the production H&E/fusion initialization when replacing pooled encoders."""
    set_seed(seed)
    reference = OmicsHEEndToEndCox(dropout=dropout, latent_dim=latent_dim, fusion_mode=fusion_mode)
    he_state = reference.encoders[4].state_dict()
    fusion_state = reference.fusion.state_dict()
    cpu_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    set_seed(seed)
    model = OmicsHEEndToEndCox(
        dropout=dropout, latent_dim=latent_dim, fusion_mode=fusion_mode,
        **{f"{name}_encoder_architecture": "masked_cnn_spatial_projector" for name in MODALITIES[:4]},
    )
    model.encoders[4].load_state_dict(he_state)
    model.fusion.load_state_dict(fusion_state)
    torch.set_rng_state(cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    return model


def load_molecular_encoders(model: OmicsHEEndToEndCox, paths: list[Path]) -> None:
    if len(paths) != 4:
        raise ValueError("Four pretrained molecular encoder checkpoints are required")
    for encoder, path in zip(model.encoders[:4], paths):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        state = payload["model_state"] if "model_state" in payload else payload["state_dict"]
        encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}, strict=True)


def extract_embeddings(model: OmicsHEEndToEndCox, data: MultimodalData, chunk_size: int, device: str) -> list[torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return model.encode_all(
            [torch.from_numpy(values).to(device) for values in data.inputs],
            [torch.from_numpy(values).to(device) for values in data.master_positions],
            len(data.cohort), chunk_size,
        )


def predict_survival(model: OmicsHEEndToEndCox, data: MultimodalData, chunk_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    embeddings = extract_embeddings(model, data, chunk_size, device)
    with torch.no_grad():
        result = model.fusion(embeddings, torch.from_numpy(data.availability).to(device))
    return result["risk"].cpu().numpy(), result["weights"].cpu().numpy()


def fit_fusion(model: OmicsHEEndToEndCox, data: MultimodalData, args: argparse.Namespace) -> tuple[OmicsHEEndToEndCox, dict, pd.DataFrame]:
    set_seed(args.seed)
    model = model.to(args.device)
    images = [torch.from_numpy(values).to(args.device) for values in data.inputs]
    local = [torch.from_numpy(values).to(args.device) for values in data.local_lookup]
    available = torch.from_numpy(data.availability).to(args.device)
    time = torch.as_tensor(data.cohort["time"].to_numpy(np.float32), device=args.device)
    event = torch.as_tensor(data.cohort["event"].to_numpy(np.float32), device=args.device)
    core_rows = np.flatnonzero(data.cohort["partition"].eq("core"))
    validation_rows = np.flatnonzero(data.cohort["partition"].eq("validation"))
    core = torch.as_tensor(core_rows, device=args.device)
    validation = torch.as_tensor(validation_rows, device=args.device)
    clinical = data.cohort.iloc[validation_rows]
    if not event[core].any():
        raise ValueError("Core patients must include an observed event")
    c_index(clinical["event"], clinical["time"], np.zeros(len(clinical)))
    optimizer = torch.optim.AdamW([
        {"params": model.encoders.parameters(), "lr": args.encoder_lr},
        {"params": model.fusion.parameters(), "lr": args.fusion_lr},
    ], weight_decay=args.weight_decay)
    best_c, best_epoch, stale = -np.inf, 0, 0
    best_state, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.freeze_batchnorm_statistics()
        optimizer.zero_grad()
        result = model.forward_selected(images, [indices[core] for indices in local], available[core], args.encoder_chunk_size, modality_dropout_probability=args.modality_dropout)
        loss, parts = _survival_loss(result, time[core], event[core], available[core], args.aux_weight, args.alignment_weight)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite fusion loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            risk = model.forward_selected(images, [indices[validation] for indices in local], available[validation], args.encoder_chunk_size)["risk"].cpu().numpy()
        score = c_index(clinical["event"], clinical["time"], risk)
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "validation_c_index": score, **{key: float(value.detach().cpu()) for key, value in parts.items()}})
        if np.isfinite(score) and score > best_c + 1e-5:
            best_c, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise ValueError("No finite validation C-index was obtained")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "validation_c_index": float(best_c)}, pd.DataFrame(history)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cohort", type=Path, required=True)
    for name in MODALITIES:
        parser.add_argument(f"--{name}", type=Path, required=True, help=f"Observed {name} input NPZ with patient IDs")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--encoder-chunk-size", type=int, default=320)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    for name in MODALITIES[:4]:
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-mode", choices=("gated", "mean"), default="gated")
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--fusion-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--modality-dropout", type=float, default=0.30)
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--he-clip", type=float, default=8.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if min(args.epochs, args.patience, args.encoder_chunk_size, args.latent_dim) < 1:
        raise ValueError("Epochs, patience, chunk size, and latent dimension must be positive")
    if not 0 <= args.dropout < 1 or not 0 <= args.modality_dropout < 1:
        raise ValueError("Dropout probabilities must lie in [0, 1)")
    for name in ("encoder_lr", "fusion_lr", "gradient_clip"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "aux_weight", "alignment_weight"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    data = load_multimodal_data(args.cohort, [getattr(args, name) for name in MODALITIES], he_clip=args.he_clip)
    model = build_fusion_model(args.seed, args.dropout, args.latent_dim, args.fusion_mode)
    load_molecular_encoders(model, [getattr(args, f"{name}_checkpoint") for name in MODALITIES[:4]])
    model, metadata, history = fit_fusion(model, data, args)
    risk, weights = predict_survival(model, data, args.encoder_chunk_size, args.device)
    calibration = fit_risk_calibration(risk, data.cohort["partition"].eq("validation").to_numpy())
    model_kwargs = {"dropout": args.dropout, "latent_dim": args.latent_dim, "fusion_mode": args.fusion_mode, **{f"{name}_encoder_architecture": "masked_cnn_spatial_projector" for name in MODALITIES[:4]}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_kwargs": model_kwargs, "risk_calibration": calibration, "seed": args.seed,
        "he_standardization": {key: torch.as_tensor(value) if isinstance(value, np.ndarray) else value for key, value in data.he_standardization.items()},
        **metadata,
    }, args.output_dir / "model.pt")
    predictions = data.cohort.copy()
    predictions["risk_raw"], predictions["risk"] = risk, apply_risk_calibration(risk, calibration)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(weights, index=pd.Index(data.cohort["patient"], name="patient"), columns=MODALITIES).to_csv(args.output_dir / "fusion_weights.csv")
    history.to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "options.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(f"Saved five-modality survival model: {args.output_dir}")


if __name__ == "__main__":
    main()
