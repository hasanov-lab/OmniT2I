"""Original conventional-survival fits and validation-based selection."""

from __future__ import annotations

import argparse
import resource
import time
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.exceptions import ConvergenceWarning
from sksurv.ensemble import (RandomSurvivalForest, ComponentwiseGradientBoostingSurvivalAnalysis,
                            ExtraSurvivalTrees, GradientBoostingSurvivalAnalysis)
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from xgboost import XGBRegressor

from omnit2i.training.losses import c_index

@dataclass
class CoxPathFit:
    """Selected model from a fold-local Cox elastic-net path."""

    model: CoxnetSurvivalAnalysis
    retained_columns: np.ndarray
    core_mean: np.ndarray
    core_scale: np.ndarray
    selected_alpha: float
    validation_c_index: float
    validation_risk_std: float
    alphas: np.ndarray
    warning_messages: tuple[str, ...]
    fit_seconds: float
    peak_rss_mb: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        matrix = _cox_matrix(
            values,
            self.retained_columns,
            self.core_mean,
            self.core_scale,
        )
        return np.asarray(
            self.model.predict(matrix, alpha=self.selected_alpha),
            dtype=np.float64,
        )



def _outcome(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty or not np.isfinite(frame["time"].to_numpy(float)).all():
        raise ValueError("Survival frame is empty or has non-finite times")
    if (frame["time"].to_numpy(float) <= 0).any():
        raise ValueError("Survival times must be positive")
    return np.asarray(
        list(
            zip(
                frame["event"].astype(bool).to_numpy(),
                frame["time"].astype(float).to_numpy(),
            )
        ),
        dtype=[("event", "?"), ("time", "<f8")],
    )



def _cox_matrix(
    values: np.ndarray,
    retained: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] != len(retained):
        raise ValueError("Cox input and retained-column mask disagree")
    if mean.shape != scale.shape or mean.shape != (int(retained.sum()),):
        raise ValueError("Cox scaling vectors disagree with retained columns")
    selected = (
        np.asarray(matrix[:, retained], dtype=np.float64) - mean[None, :]
    ) / scale[None, :]
    if not np.isfinite(selected).all():
        raise ValueError("Cox input contains non-finite values")
    return np.asfortranarray(selected, dtype=np.float64)



def fit_cox_path(
    x_core: np.ndarray,
    core_clinical: pd.DataFrame,
    x_validation: np.ndarray,
    validation_clinical: pd.DataFrame,
    *,
    l1_ratio: float,
    n_alphas: int,
    alpha_min_ratio: float,
    max_iter: int,
    tolerance: float,
) -> CoxPathFit:
    """Fit one inner-core Cox path and select alpha by validation C-index."""
    core = np.asarray(x_core)
    validation = np.asarray(x_validation)
    if core.ndim != 2 or validation.ndim != 2 or core.shape[1] != validation.shape[1]:
        raise ValueError("Core and validation feature matrices are incompatible")
    if len(core) != len(core_clinical) or len(validation) != len(validation_clinical):
        raise ValueError("Feature and clinical row counts differ")
    if len(core) < 20 or len(validation) < 10:
        raise ValueError("Cox path requires at least 20 core and 10 validation rows")
    if not 0 < l1_ratio <= 1:
        raise ValueError("l1_ratio must be in (0, 1]")
    if n_alphas < 2 or not 0 < alpha_min_ratio < 1:
        raise ValueError("Invalid Cox alpha-path settings")

    # Refit scaling on the inner core.  The outer-fold target bundle fixes the
    # outcome-free feature set shared by every A1 method, while no inner
    # validation predictor contributes to Cox regularization or scaling.
    core_mean_all = np.mean(core, axis=0, dtype=np.float64)
    core_std = np.std(core, axis=0, dtype=np.float64, ddof=0)
    retained = np.isfinite(core_std) & (core_std > 1e-8)
    if not retained.any():
        raise ValueError("All Cox input features are constant in the inner core")
    core_mean = core_mean_all[retained]
    core_scale = core_std[retained]
    core_matrix = _cox_matrix(core, retained, core_mean, core_scale)
    validation_matrix = _cox_matrix(
        validation, retained, core_mean, core_scale
    )

    model = CoxnetSurvivalAnalysis(
        l1_ratio=float(l1_ratio),
        n_alphas=int(n_alphas),
        alpha_min_ratio=float(alpha_min_ratio),
        normalize=False,
        copy_X=False,
        max_iter=int(max_iter),
        tol=float(tolerance),
        fit_baseline_model=False,
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(core_matrix, _outcome(core_clinical))
    fit_seconds = float(time.perf_counter() - started)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        messages = " | ".join(str(item.message) for item in caught)
        raise RuntimeError(f"Cox path did not converge: {messages}")
    warning_messages = tuple(str(item.message) for item in caught)
    if len(model.alphas_) < 2:
        raise RuntimeError("Cox path returned fewer than two alpha values")

    candidates: list[tuple[float, float, float]] = []
    event = validation_clinical["event"].astype(bool).to_numpy()
    validation_time = validation_clinical["time"].astype(float).to_numpy()
    for alpha in np.asarray(model.alphas_, dtype=float):
        risk = np.asarray(model.predict(validation_matrix, alpha=float(alpha)))
        risk_std = float(risk.std(ddof=0))
        if not np.isfinite(risk).all() or risk_std < 1e-8:
            continue
        score = c_index(event, validation_time, risk)
        if np.isfinite(score):
            candidates.append((float(score), float(alpha), risk_std))
    if not candidates:
        raise RuntimeError("Every Cox alpha produced a constant or invalid validation risk")

    # Prefer the stronger penalty when validation C-indices tie.
    validation_c, selected_alpha, risk_std = max(
        candidates, key=lambda item: (round(item[0], 12), item[1])
    )
    return CoxPathFit(
        model=model,
        retained_columns=retained,
        core_mean=core_mean,
        core_scale=core_scale,
        selected_alpha=selected_alpha,
        validation_c_index=validation_c,
        validation_risk_std=risk_std,
        alphas=np.asarray(model.alphas_, dtype=float),
        warning_messages=warning_messages,
        fit_seconds=fit_seconds,
        peak_rss_mb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    )



@dataclass
class ModelFitResult:
    """Validation/test risks and auditable training metadata for one model."""

    validation_risk: np.ndarray
    test_risk: np.ndarray
    metadata: dict[str, Any]



class DeepSurvMLP(nn.Module):
    """The established project DeepSurv multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim < 1 or not hidden_dims:
            raise ValueError("DeepSurv requires positive input and hidden dimensions")
        layers: list[nn.Module] = []
        current = int(input_dim)
        for hidden in hidden_dims:
            if int(hidden) < 1:
                raise ValueError("DeepSurv hidden dimensions must be positive")
            layers.extend(
                [
                    nn.Linear(current, int(hidden)),
                    nn.BatchNorm1d(int(hidden)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
            current = int(hidden)
        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).flatten()



def parse_hidden_dims(value: str | Sequence[int]) -> tuple[int, ...]:
    try:
        if isinstance(value, str):
            dims = tuple(
                int(item.strip()) for item in value.split(",") if item.strip()
            )
        else:
            dims = tuple(int(item) for item in value)
    except ValueError as error:
        raise ValueError("DeepSurv hidden dimensions must be integers") from error
    if not dims or any(item < 1 for item in dims):
        raise ValueError("DeepSurv hidden dimensions must be positive")
    return dims



def stable_cox_partial_likelihood(
    risk: torch.Tensor,
    time_values: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Breslow Cox loss with tied times and stable log-cumulative sums."""
    order = torch.argsort(time_values, descending=True, stable=True)
    risk_sorted = risk[order]
    time_sorted = time_values[order]
    event_sorted = event[order].bool()
    event_positions = torch.nonzero(event_sorted, as_tuple=False).flatten()
    if event_positions.numel() == 0:
        raise ValueError("Cox loss requires at least one observed event")
    log_cumulative_risk = torch.logcumsumexp(risk_sorted, dim=0)
    _, inverse, counts = torch.unique_consecutive(
        time_sorted,
        return_inverse=True,
        return_counts=True,
    )
    group_last = torch.cumsum(counts, dim=0) - 1
    denominators = group_last[inverse[event_positions]]
    return (
        -risk_sorted[event_positions].sum()
        + log_cumulative_risk[denominators].sum()
    ) / event_positions.numel()



def _validate_arrays(
    x_core: np.ndarray,
    core_clinical: pd.DataFrame,
    x_validation: np.ndarray,
    validation_clinical: pd.DataFrame,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = tuple(
        np.ascontiguousarray(np.asarray(values, dtype=np.float32))
        for values in (x_core, x_validation, x_test)
    )
    if any(values.ndim != 2 for values in matrices):
        raise ValueError("Model inputs must be two-dimensional")
    if len({values.shape[1] for values in matrices}) != 1:
        raise ValueError("Core, validation, and test feature dimensions differ")
    if len(matrices[0]) != len(core_clinical):
        raise ValueError("Core feature and clinical row counts differ")
    if len(matrices[1]) != len(validation_clinical):
        raise ValueError("Validation feature and clinical row counts differ")
    if len(matrices[0]) < 20 or len(matrices[1]) < 10 or len(matrices[2]) < 1:
        raise ValueError("Insufficient core, validation, or test rows")
    if any(not np.isfinite(values).all() for values in matrices):
        raise ValueError("Model inputs contain non-finite values")
    _outcome(core_clinical)
    _outcome(validation_clinical)
    return matrices



def fit_rsf(
    x_core: np.ndarray,
    core_clinical: pd.DataFrame,
    x_validation: np.ndarray,
    validation_clinical: pd.DataFrame,
    x_test: np.ndarray,
    args: argparse.Namespace | SimpleNamespace,
    seed: int,
) -> ModelFitResult:
    """Fit the fixed Random Survival Forest and return validation/test risks."""
    core, validation, test = _validate_arrays(
        x_core,
        core_clinical,
        x_validation,
        validation_clinical,
        x_test,
    )
    if args.rsf_n_estimators < 1:
        raise ValueError("RSF requires at least one tree")
    model = RandomSurvivalForest(
        n_estimators=int(args.rsf_n_estimators),
        min_samples_split=int(args.rsf_min_samples_split),
        min_samples_leaf=int(args.rsf_min_samples_leaf),
        max_features=args.rsf_max_features,
        max_depth=None,
        bootstrap=True,
        low_memory=True,
        n_jobs=int(args.rsf_n_jobs),
        random_state=int(seed),
    )
    started = time.perf_counter()
    model.fit(core, _outcome(core_clinical))
    fit_seconds = float(time.perf_counter() - started)
    validation_risk = np.asarray(model.predict(validation), dtype=np.float64)
    test_risk = np.asarray(model.predict(test), dtype=np.float64)
    if (
        not np.isfinite(validation_risk).all()
        or not np.isfinite(test_risk).all()
        or validation_risk.std(ddof=0) < 1e-8
    ):
        raise RuntimeError("RSF produced invalid or constant validation risk")
    validation_c = c_index(
        validation_clinical["event"].to_numpy(bool),
        validation_clinical["time"].to_numpy(float),
        validation_risk,
    )
    metadata = {
        "validation_c_index": float(validation_c),
        "validation_risk_std": float(validation_risk.std(ddof=0)),
        "fit_seconds": fit_seconds,
        "peak_rss_mb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "n_estimators": int(args.rsf_n_estimators),
        "min_samples_split": int(args.rsf_min_samples_split),
        "min_samples_leaf": int(args.rsf_min_samples_leaf),
        "max_features": str(args.rsf_max_features),
        "n_jobs": int(args.rsf_n_jobs),
        "random_state": int(seed),
        "low_memory": True,
    }
    return ModelFitResult(validation_risk, test_risk, metadata)



def _seed_torch(seed: int, device: torch.device) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)



def fit_deepsurv(
    x_core: np.ndarray,
    core_clinical: pd.DataFrame,
    x_validation: np.ndarray,
    validation_clinical: pd.DataFrame,
    x_test: np.ndarray,
    args: argparse.Namespace | SimpleNamespace,
    seed: int,
) -> ModelFitResult:
    """Fit fixed DeepSurv with inner-validation early stopping."""
    core, validation, test = _validate_arrays(
        x_core,
        core_clinical,
        x_validation,
        validation_clinical,
        x_test,
    )
    hidden_dims = parse_hidden_dims(args.deepsurv_hidden_dims)
    if not 0 <= args.deepsurv_dropout < 1:
        raise ValueError("DeepSurv dropout must be in [0, 1)")
    if args.deepsurv_epochs < 1 or args.deepsurv_patience < 1:
        raise ValueError("DeepSurv epochs and patience must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DeepSurv requested CUDA but no CUDA device is available")
    _seed_torch(seed, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = DeepSurvMLP(core.shape[1], hidden_dims, args.deepsurv_dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.deepsurv_lr),
        weight_decay=float(args.deepsurv_weight_decay),
    )
    core_tensor = torch.from_numpy(core).to(device)
    validation_tensor = torch.from_numpy(validation).to(device)
    test_tensor = torch.from_numpy(test).to(device)
    core_time = torch.from_numpy(
        core_clinical["time"].to_numpy(dtype=np.float32)
    ).to(device)
    core_event = torch.from_numpy(
        core_clinical["event"].to_numpy(dtype=np.float32)
    ).to(device)
    validation_event = validation_clinical["event"].to_numpy(dtype=bool)
    validation_time = validation_clinical["time"].to_numpy(dtype=float)

    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_validation_c = -np.inf
    best_epoch = 0
    stale = 0
    epochs_completed = 0
    started = time.perf_counter()
    for epoch in range(1, int(args.deepsurv_epochs) + 1):
        epochs_completed = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = model(core_tensor)
        loss = stable_cox_partial_likelihood(risk, core_time, core_event)
        if not torch.isfinite(loss):
            raise RuntimeError(f"DeepSurv loss became non-finite at epoch {epoch}")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_risk = model(validation_tensor).detach().cpu().numpy()
        if not np.isfinite(validation_risk).all():
            raise RuntimeError(
                f"DeepSurv validation risk became non-finite at epoch {epoch}"
            )
        validation_c = c_index(
            validation_event,
            validation_time,
            validation_risk,
        )
        if (
            np.isfinite(validation_c)
            and validation_c > best_validation_c + float(args.deepsurv_min_delta)
        ):
            best_validation_c = float(validation_c)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(args.deepsurv_patience):
            break
    fit_seconds = float(time.perf_counter() - started)
    if best_epoch < 1 or not np.isfinite(best_validation_c):
        raise RuntimeError("DeepSurv did not produce a valid validation checkpoint")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_risk = model(validation_tensor).detach().cpu().numpy().astype(float)
        test_risk = model(test_tensor).detach().cpu().numpy().astype(float)
    if (
        not np.isfinite(validation_risk).all()
        or not np.isfinite(test_risk).all()
        or validation_risk.std(ddof=0) < 1e-8
    ):
        raise RuntimeError("DeepSurv produced invalid or constant validation risk")
    peak_gpu_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        if device.type == "cuda"
        else 0.0
    )
    metadata = {
        "validation_c_index": float(best_validation_c),
        "validation_risk_std": float(validation_risk.std(ddof=0)),
        "validation_risk_abs_max": float(np.abs(validation_risk).max()),
        "test_risk_abs_max": float(np.abs(test_risk).max()),
        "fit_seconds": fit_seconds,
        "peak_rss_mb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "peak_gpu_memory_mb": peak_gpu_mb,
        "hidden_dims": ",".join(map(str, hidden_dims)),
        "dropout": float(args.deepsurv_dropout),
        "learning_rate": float(args.deepsurv_lr),
        "weight_decay": float(args.deepsurv_weight_decay),
        "max_epochs": int(args.deepsurv_epochs),
        "patience": int(args.deepsurv_patience),
        "min_delta": float(args.deepsurv_min_delta),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epochs_completed),
        "early_stopped": bool(epochs_completed < int(args.deepsurv_epochs)),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "deterministic_algorithms": True,
        "random_state": int(seed),
    }
    return ModelFitResult(validation_risk, test_risk, metadata)



@dataclass
class FitResult:
    validation_risk: np.ndarray
    test_risk: np.ndarray
    metadata: dict[str, Any]



def _standardize(
    x_core: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    core = np.asarray(x_core, dtype=np.float64)
    validation = np.asarray(x_validation, dtype=np.float64)
    test = np.asarray(x_test, dtype=np.float64)
    if any(values.ndim != 2 for values in (core, validation, test)):
        raise ValueError("Inputs must be two-dimensional")
    if len({values.shape[1] for values in (core, validation, test)}) != 1:
        raise ValueError("Core, validation and test dimensions differ")
    mean = core.mean(axis=0, dtype=np.float64)
    scale = core.std(axis=0, dtype=np.float64, ddof=0)
    retained = np.isfinite(scale) & (scale > 1e-8)
    if not retained.any():
        raise ValueError("All input features are constant in the inner core")
    output = tuple(
        np.ascontiguousarray((values[:, retained] - mean[retained]) / scale[retained])
        for values in (core, validation, test)
    )
    if any(not np.isfinite(values).all() for values in output):
        raise ValueError("Standardized inputs contain non-finite values")
    return output[0], output[1], output[2], int(retained.sum())



def _finish_result(
    validation_risk: np.ndarray,
    test_risk: np.ndarray,
    validation_clinical: pd.DataFrame,
    metadata: dict[str, Any],
) -> FitResult:
    validation = np.asarray(validation_risk, dtype=np.float64).reshape(-1)
    test = np.asarray(test_risk, dtype=np.float64).reshape(-1)
    if not np.isfinite(validation).all() or not np.isfinite(test).all():
        raise RuntimeError("Model produced non-finite risks")
    if validation.std(ddof=0) < 1e-8:
        raise RuntimeError("Model produced constant validation risk")
    validation_c = c_index(
        validation_clinical["event"].to_numpy(bool),
        validation_clinical["time"].to_numpy(float),
        validation,
    )
    return FitResult(
        validation,
        test,
        {
            "validation_c_index": float(validation_c),
            "validation_risk_std": float(validation.std(ddof=0)),
            "peak_rss_mb": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
            **metadata,
        },
    )



def _fit_coxnet(
    model_name: str,
    x_core: np.ndarray,
    core_clinical: pd.DataFrame,
    x_validation: np.ndarray,
    validation_clinical: pd.DataFrame,
    x_test: np.ndarray,
    args: argparse.Namespace,
) -> FitResult:
    l1_ratio = 1.0 if model_name == "zero_fill_lasso_cox" else 0.5
    fit = fit_cox_path(
        x_core,
        core_clinical,
        x_validation,
        validation_clinical,
        l1_ratio=l1_ratio,
        n_alphas=int(args.coxnet_n_alphas),
        alpha_min_ratio=float(args.coxnet_alpha_min_ratio),
        max_iter=int(args.coxnet_max_iter),
        tolerance=float(args.coxnet_tol),
    )
    return _finish_result(
        fit.predict(x_validation),
        fit.predict(x_test),
        validation_clinical,
        {
            "fit_seconds": float(fit.fit_seconds),
            "n_retained_features": int(fit.retained_columns.sum()),
            "l1_ratio": l1_ratio,
            "selected_alpha": float(fit.selected_alpha),
            "alpha_path_max": float(np.max(fit.alphas)),
            "alpha_path_min": float(np.min(fit.alphas)),
            "n_alpha_path": int(len(fit.alphas)),
            "warning_count": int(len(fit.warning_messages)),
            "warnings": " | ".join(fit.warning_messages),
        },
    )



def _fit_sksurv_model(
    model_name: str,
    x_core: np.ndarray,
    core_clinical: pd.DataFrame,
    x_validation: np.ndarray,
    validation_clinical: pd.DataFrame,
    x_test: np.ndarray,
    args: argparse.Namespace,
    training_seed: int,
) -> FitResult:
    if model_name in {"zero_fill_coxboost", "zero_fill_survival_svm"}:
        core, validation, test, retained = _standardize(
            x_core, x_validation, x_test
        )
    else:
        core = np.ascontiguousarray(x_core, dtype=np.float32)
        validation = np.ascontiguousarray(x_validation, dtype=np.float32)
        test = np.ascontiguousarray(x_test, dtype=np.float32)
        retained = int(core.shape[1])

    if model_name == "zero_fill_coxboost":
        estimator = ComponentwiseGradientBoostingSurvivalAnalysis(
            loss="coxph",
            learning_rate=float(args.coxboost_learning_rate),
            n_estimators=int(args.coxboost_n_estimators),
            subsample=1.0,
            random_state=int(training_seed),
        )
    elif model_name == "zero_fill_survival_svm":
        estimator = FastSurvivalSVM(
            alpha=float(args.svm_alpha),
            rank_ratio=1.0,
            fit_intercept=False,
            max_iter=int(args.svm_max_iter),
            random_state=int(training_seed),
        )
    elif model_name == "zero_fill_extra_survival_trees":
        estimator = ExtraSurvivalTrees(
            n_estimators=int(args.extra_n_estimators),
            min_samples_split=int(args.extra_min_samples_split),
            min_samples_leaf=int(args.extra_min_samples_leaf),
            max_features=args.extra_max_features,
            bootstrap=True,
            low_memory=True,
            n_jobs=int(args.n_jobs),
            random_state=int(training_seed),
        )
    elif model_name == "zero_fill_survival_gbm":
        estimator = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            learning_rate=float(args.gbm_learning_rate),
            n_estimators=int(args.gbm_n_estimators),
            subsample=float(args.gbm_subsample),
            max_depth=int(args.gbm_max_depth),
            max_features=args.gbm_max_features,
            random_state=int(training_seed),
        )
    else:
        raise AssertionError(model_name)

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(core, _outcome(core_clinical))
    fit_seconds = float(time.perf_counter() - started)
    return _finish_result(
        estimator.predict(validation),
        estimator.predict(test),
        validation_clinical,
        {
            "fit_seconds": fit_seconds,
            "n_retained_features": retained,
            "warning_count": len(caught),
            "warnings": " | ".join(str(item.message) for item in caught),
        },
    )



def _xgb_labels(clinical: pd.DataFrame) -> np.ndarray:
    time_values = clinical["time"].to_numpy(dtype=np.float32)
    event = clinical["event"].to_numpy(dtype=bool)
    if not np.isfinite(time_values).all() or (time_values <= 0).any():
        raise ValueError("XGBoost survival times must be finite and positive")
    return np.where(event, time_values, -time_values).astype(np.float32)



def _fit_xgboost(
    x_core: np.ndarray,
    core_clinical: pd.DataFrame,
    x_validation: np.ndarray,
    validation_clinical: pd.DataFrame,
    x_test: np.ndarray,
    args: argparse.Namespace,
    training_seed: int,
) -> FitResult:
    core = np.ascontiguousarray(x_core, dtype=np.float32)
    validation = np.ascontiguousarray(x_validation, dtype=np.float32)
    test = np.ascontiguousarray(x_test, dtype=np.float32)
    estimator = XGBRegressor(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        n_estimators=int(args.xgb_n_estimators),
        learning_rate=float(args.xgb_learning_rate),
        max_depth=int(args.xgb_max_depth),
        min_child_weight=float(args.xgb_min_child_weight),
        subsample=float(args.xgb_subsample),
        colsample_bytree=float(args.xgb_colsample_bytree),
        reg_lambda=float(args.xgb_reg_lambda),
        tree_method="hist",
        early_stopping_rounds=int(args.xgb_early_stopping_rounds),
        n_jobs=int(args.n_jobs),
        random_state=int(training_seed),
        verbosity=0,
    )
    started = time.perf_counter()
    estimator.fit(
        core,
        _xgb_labels(core_clinical),
        eval_set=[(validation, _xgb_labels(validation_clinical))],
        verbose=False,
    )
    fit_seconds = float(time.perf_counter() - started)
    return _finish_result(
        estimator.predict(validation),
        estimator.predict(test),
        validation_clinical,
        {
            "fit_seconds": fit_seconds,
            "n_retained_features": int(core.shape[1]),
            "best_iteration": int(estimator.best_iteration),
            "best_score": float(estimator.best_score),
            "warning_count": 0,
            "warnings": "",
        },
    )

