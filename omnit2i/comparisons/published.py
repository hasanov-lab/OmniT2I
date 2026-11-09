"""The manuscript's common-data adaptations of three published model families."""

from __future__ import annotations

import argparse
import math
import random
import resource
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from omnit2i.models import MODALITIES
from omnit2i.training.losses import c_index
from .estimators import stable_cox_partial_likelihood, FitResult

MODELS = ("adapted_multisurv", "adapted_senmo", "adapted_bib_gated")
MODEL_FAMILIES = {
    "adapted_multisurv": "MultiSurv (Vale-Silva and Rohr, 2021)",
    "adapted_senmo": "SeNMo (Ma et al., 2024)",
    "adapted_bib_gated": "Cancer-aware gated multimodal survival model (BIB, 2026)",
}

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)



class AdaptedMultiSurv(nn.Module):
    """Modality subnetworks followed by missing-safe elementwise max fusion."""

    loss_kind = "discrete"

    def __init__(self, input_dims: Sequence[int], latent_dim: int = 256) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, latent_dim),
                    nn.ReLU(),
                    nn.Dropout(0.20),
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                )
                for input_dim in input_dims
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(
        self,
        blocks: Sequence[torch.Tensor],
        availability: torch.Tensor,
        cancer_index: torch.Tensor,
    ) -> torch.Tensor:
        del cancer_index
        representations = torch.stack(
            [encoder(block) for encoder, block in zip(self.encoders, blocks, strict=True)],
            dim=1,
        )
        mask = availability.bool().unsqueeze(-1)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("MultiSurv requires at least one observed modality")
        representations = representations.masked_fill(~mask, torch.finfo(representations.dtype).min)
        fused = representations.max(dim=1).values
        return self.head(fused)



class AdaptedSeNMo(nn.Module):
    """Early-concatenation self-normalizing MLP with a Cox risk head."""

    loss_kind = "cox"

    def __init__(self, input_dims: Sequence[int]) -> None:
        super().__init__()
        dimensions = [sum(input_dims) + len(input_dims), 1024, 512, 256, 128, 48, 48, 48]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:], strict=True):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU(), nn.AlphaDropout(0.10)))
        self.encoder = nn.Sequential(*layers)
        self.risk_head = nn.Linear(dimensions[-1], 1)
        self.apply(self._lecun_initialization)

    @staticmethod
    def _lecun_initialization(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        blocks: Sequence[torch.Tensor],
        availability: torch.Tensor,
        cancer_index: torch.Tensor,
    ) -> torch.Tensor:
        del cancer_index
        masked = [
            block * availability[:, index : index + 1]
            for index, block in enumerate(blocks)
        ]
        values = torch.cat([*masked, availability], dim=1)
        return self.risk_head(self.encoder(values)).squeeze(1)



class AdaptedBIBGated(nn.Module):
    """Availability-gated fusion with shared encoder and cancer-specific heads."""

    loss_kind = "discrete"

    def __init__(self, input_dims: Sequence[int], n_cancers: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.20)
                )
                for input_dim in input_dims
            ]
        )
        self.gate = nn.Linear(hidden_dim * len(input_dims), len(input_dims))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.20,
            batch_first=True,
        )
        self.shared_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.cancer_heads = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(n_cancers)])

    def forward(
        self,
        blocks: Sequence[torch.Tensor],
        availability: torch.Tensor,
        cancer_index: torch.Tensor,
    ) -> torch.Tensor:
        representations = torch.stack(
            [projection(block) for projection, block in zip(self.projections, blocks, strict=True)],
            dim=1,
        )
        gate_input = (representations * availability.unsqueeze(-1)).flatten(start_dim=1)
        gate_logits = self.gate(gate_input).masked_fill(~availability.bool(), -1e9)
        weights = torch.softmax(gate_logits, dim=1)
        fused = (representations * weights.unsqueeze(-1)).sum(dim=1)
        encoded = self.shared_encoder(fused.unsqueeze(1)).squeeze(1)
        all_logits = torch.stack([head(encoded) for head in self.cancer_heads], dim=1)
        return all_logits[torch.arange(len(encoded), device=encoded.device), cancer_index]



def build_model(model: str, input_dims: Sequence[int], n_cancers: int) -> nn.Module:
    if model == "adapted_multisurv":
        return AdaptedMultiSurv(input_dims)
    if model == "adapted_senmo":
        return AdaptedSeNMo(input_dims)
    if model == "adapted_bib_gated":
        return AdaptedBIBGated(input_dims, n_cancers)
    raise ValueError(model)



def discrete_time_loss(
    logits: torch.Tensor, time_bin: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    hazards = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
    survival = torch.cumprod(1.0 - hazards, dim=1).clamp_min(1e-7)
    previous = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
    row = torch.arange(len(logits), device=logits.device)
    event_loss = -(torch.log(previous[row, time_bin]) + torch.log(hazards[row, time_bin]))
    censored_loss = -torch.log(survival[row, time_bin])
    return torch.where(event.bool(), event_loss, censored_loss).mean()



def discrete_risk(logits: torch.Tensor) -> torch.Tensor:
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return -survival.sum(dim=1)



def _time_cutpoints(clinical: pd.DataFrame) -> np.ndarray:
    event_times = clinical.loc[clinical["event"].astype(bool), "time"].to_numpy(float)
    if len(event_times) < 20:
        raise ValueError("Too few observed events to define four survival intervals")
    cutpoints = np.asarray(np.quantile(event_times, [0.25, 0.50, 0.75]), dtype=np.float64)
    for index in range(1, len(cutpoints)):
        if cutpoints[index] <= cutpoints[index - 1]:
            cutpoints[index] = np.nextafter(cutpoints[index - 1], np.inf)
    return cutpoints



def _time_bins(times: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    return np.searchsorted(cutpoints, np.asarray(times, dtype=np.float64), side="right").astype(
        np.int64
    )



def _batch_tensors(
    bundle,
    rows: np.ndarray,
    cancer_indices: np.ndarray,
    device: torch.device,
    forced_mask: np.ndarray | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    blocks = [
        torch.as_tensor(np.asarray(target[rows], dtype=np.float32), device=device)
        for target in bundle.targets
    ]
    if forced_mask is None:
        availability = np.asarray(bundle.availability[rows], dtype=np.float32)
    else:
        availability = np.broadcast_to(
            np.asarray(forced_mask, dtype=np.float32), (len(rows), len(MODALITIES))
        ).copy()
    return (
        blocks,
        torch.as_tensor(availability, dtype=torch.float32, device=device),
        torch.as_tensor(cancer_indices[rows], dtype=torch.long, device=device),
    )



@torch.no_grad()
def predict_rows(
    model: nn.Module,
    bundle,
    rows: np.ndarray,
    cancer_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    forced_mask: np.ndarray | None = None,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        blocks, availability, cancer_index = _batch_tensors(
            bundle, batch_rows, cancer_indices, device, forced_mask
        )
        values = model(blocks, availability, cancer_index)
        risk = discrete_risk(values) if model.loss_kind == "discrete" else values
        output.append(risk.detach().cpu().numpy().astype(np.float64))
    risks = np.concatenate(output)
    if len(risks) != len(rows) or not np.isfinite(risks).all():
        raise RuntimeError("Model produced invalid survival risks")
    return risks



def fit_model(
    model_name: str,
    bundle,
    core_rows: np.ndarray,
    validation_rows: np.ndarray,
    master: pd.DataFrame,
    cancer_indices: np.ndarray,
    args: argparse.Namespace,
    training_seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    _seed_everything(training_seed)
    device = torch.device(args.device)
    input_dims = [int(target.shape[1]) for target in bundle.targets]
    n_cancers = int(cancer_indices.max()) + 1
    model = build_model(model_name, input_dims, n_cancers).to(device)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 5.8e-4 if model_name == "adapted_senmo" else 1e-3
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    cutpoints = _time_cutpoints(master.iloc[core_rows])
    core_time_bins = _time_bins(master.iloc[core_rows]["time"].to_numpy(float), cutpoints)
    core_events = master.iloc[core_rows]["event"].to_numpy(bool)
    core_times = master.iloc[core_rows]["time"].to_numpy(np.float32)
    rng = np.random.default_rng(training_seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = -np.inf
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(core_rows))
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            positions = order[start : start + args.batch_size]
            batch_rows = core_rows[positions]
            if not core_events[positions].any():
                continue
            blocks, availability, cancer_index = _batch_tensors(
                bundle, batch_rows, cancer_indices, device
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(blocks, availability, cancer_index)
            if model.loss_kind == "discrete":
                loss = discrete_time_loss(
                    output,
                    torch.as_tensor(core_time_bins[positions], dtype=torch.long, device=device),
                    torch.as_tensor(core_events[positions], dtype=torch.bool, device=device),
                )
            else:
                loss = stable_cox_partial_likelihood(
                    output,
                    torch.as_tensor(core_times[positions], dtype=torch.float32, device=device),
                    torch.as_tensor(core_events[positions], dtype=torch.bool, device=device),
                )
                if args.senmo_l1 > 0:
                    loss = loss + args.senmo_l1 * sum(
                        parameter.abs().sum() for parameter in model.parameters()
                    )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_risk = predict_rows(
            model,
            bundle,
            validation_rows,
            cancer_indices,
            device,
            args.prediction_batch_size,
        )
        validation_c = float(
            c_index(
                master.iloc[validation_rows]["event"].to_numpy(bool),
                master.iloc[validation_rows]["time"].to_numpy(float),
                validation_risk,
            )
        )
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "training_loss": mean_loss, "validation_c_index": validation_c})
        if validation_c > best_validation + args.min_delta:
            best_validation = validation_c
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None or not np.isfinite(best_validation):
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    metadata = {
        "source_family": MODEL_FAMILIES[model_name],
        "adaptation_status": "common-data adapted implementation; not exact published-code reproduction",
        "loss_kind": model.loss_kind,
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(history)),
        "validation_c_index": float(best_validation),
        "learning_rate": float(learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "time_cutpoint_1": float(cutpoints[0]),
        "time_cutpoint_2": float(cutpoints[1]),
        "time_cutpoint_3": float(cutpoints[2]),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "fit_seconds": float(time.perf_counter() - started),
        "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "last_training_loss": float(history[-1]["training_loss"]),
    }
    return model, metadata



def fit(name, targets, availability, master, rows, args, training_seed):
    if min(args.epochs, args.patience, args.batch_size, args.prediction_batch_size) <= 0:
        raise ValueError("Epochs, patience and batch sizes must be positive")
    bundle = SimpleNamespace(targets=targets, availability=availability)
    categories = sorted(master.cancer_type.unique())
    cancer_indices = master.cancer_type.map({name: i for i, name in enumerate(categories)}).to_numpy(np.int64)
    model, metadata = fit_model(name, bundle, rows[0], rows[1], master, cancer_indices, args, training_seed)
    validation_risk = predict_rows(model, bundle, rows[1], cancer_indices, torch.device(args.device), args.prediction_batch_size)
    test_risk = predict_rows(model, bundle, rows[2], cancer_indices, torch.device(args.device), args.prediction_batch_size)
    return FitResult(validation_risk, test_risk, metadata)
