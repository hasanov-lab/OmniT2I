"""Frozen complete-case evaluation of 30 observed modality subsets."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from omnit2i.metrics import _feature_metric_row
from omnit2i.models import MODALITIES, FiveModalCrossModalMultitaskModel, OmicsHEEndToEndCox, TargetConditionedCrossModalTranslator
from omnit2i.training.crossmodal import _baseline_for_cancers, fused_risk_from_projected
from omnit2i.training.fusion import extract_embeddings
from omnit2i.training.losses import apply_risk_calibration, fit_risk_calibration
from omnit2i.training.multimodal_data import load_multimodal_data


CONDITIONS = ("source_only", "cancer_mean", "virtual", "measured")


@torch.no_grad()
def evaluate_subsets(model, projected, cohort, availability, targets, feature_baselines):
    """Compare four conditions on identical complete-case held-out patients.

    Each missing target is predicted independently from the original source
    subset. Calibration uses complete-case validation patients for each
    condition; cancer-mean latent references use observed core patients.
    """
    model.eval()
    if projected.ndim != 3 or projected.shape[:2] != (len(cohort), 5) or availability.shape != (len(cohort), 5):
        raise ValueError("Projected embeddings and availability must match the five-modality cohort")
    if not torch.isfinite(projected).all():
        raise ValueError("Projected embeddings must be finite")
    core = cohort["partition"].eq("core").to_numpy()
    validation = cohort["partition"].eq("validation").to_numpy()
    test = cohort["partition"].eq("test").to_numpy()
    rows = np.flatnonzero(availability.all(axis=1) & (validation | test))
    test_local, val_local = test[rows], validation[rows]
    if not test_local.any() or val_local.sum() < 2:
        raise ValueError("Evaluation requires complete-case test patients and at least two complete-case validation patients")
    cancers = cohort["cancer_type"].to_numpy(str)
    selected = projected[rows]
    all_available = torch.ones((len(rows), 5), dtype=torch.bool, device=projected.device)
    means = []
    for target in range(5):
        observed = core & availability[:, target]
        if not observed.any():
            raise ValueError(f"Core has no observed {MODALITIES[target]} patients")
        values = projected[:, target].cpu().numpy()
        labels = np.unique(cancers[observed])
        means.append({
            "global_mean": values[observed].mean(axis=0).astype(np.float32),
            "cancer_labels": labels,
            "cancer_means": np.stack([values[observed & (cancers == label)].mean(axis=0) for label in labels]).astype(np.float32),
        })

    def risk(values, mask):
        return fused_risk_from_projected(model.survival.fusion, values, mask)[0].cpu().numpy().astype(np.float32)

    measured = risk(selected, all_available)
    predictions, metrics = [], []
    test_rows = rows[test_local]
    for size in range(1, 5):
        for subset in combinations(range(5), size):
            name = "+".join(MODALITIES[index] for index in subset)
            source_mask = torch.zeros_like(all_available)
            source_mask[:, list(subset)] = True
            mean_projected, virtual_projected = selected.clone(), selected.clone()
            missing = [index for index in range(5) if index not in subset]
            for target in missing:
                mean_projected[:, target] = torch.from_numpy(_baseline_for_cancers(means[target], cancers[rows])).to(projected.device)
                translated = model.translator(selected, source_mask, target)
                virtual_projected[:, target] = translated["latent_prediction"]
                baseline = _baseline_for_cancers(feature_baselines[MODALITIES[target]], cancers[test_rows])
                prediction = (translated["feature_prediction"].cpu().numpy()[test_local] + baseline).astype(np.float32)
                metrics.append({
                    "source_subset": name, "observed_count": size, "target": MODALITIES[target],
                    "target_kind": "histopathology_representation" if target == 4 else "measured_omics",
                    "target_scale": str(feature_baselines[MODALITIES[target]]["target_scale"]),
                    "n_test": len(test_rows), "n_features": targets[target].shape[1],
                    **_feature_metric_row(targets[target][test_rows], prediction, baseline, cancers[test_rows]),
                })
            risks = {"source_only": risk(selected, source_mask), "cancer_mean": risk(mean_projected, all_available), "virtual": risk(virtual_projected, all_available), "measured": measured}
            output = cohort.iloc[test_rows].copy()
            output["source_subset"], output["observed_count"] = name, size
            output["missing_modalities"] = "+".join(MODALITIES[index] for index in missing)
            for condition, values in risks.items():
                calibration = fit_risk_calibration(values, val_local)
                output[f"risk_{condition}_raw"] = values[test_local]
                output[f"risk_{condition}"] = apply_risk_calibration(values, calibration)[test_local]
                output[f"calibration_n_{condition}"] = calibration["n_reference"]
            predictions.append(output)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def load_evaluation_targets(target_path, baseline_path, data, output_dims):
    """Align held-out truth to patient IDs and reuse saved training feature means."""
    targets, baselines = [], {}
    with np.load(target_path, allow_pickle=False) as saved, np.load(baseline_path, allow_pickle=False) as fitted:
        patients = saved["patient"].astype(str)
        if patients.ndim != 1 or len(np.unique(patients)) != len(patients):
            raise ValueError("Target patient IDs must be one-dimensional and unique")
        order = pd.Index(patients).get_indexer(data.cohort["patient"])
        if len(patients) != len(data.cohort) or (order < 0).any():
            raise ValueError("Targets must cover exactly the evaluation cohort")
        if not np.array_equal(saved["modalities"].astype(str), MODALITIES) or not np.array_equal(saved["availability"][order], data.availability):
            raise ValueError("Target modalities or availability disagree with the inputs")
        for key in ("cancer_type", "partition"):
            if key in saved and not np.array_equal(saved[key][order], data.cohort[key]):
                raise ValueError(f"Target {key} disagrees with the cohort")
        for index, name in enumerate(MODALITIES):
            baseline = {key: fitted[f"{name}_{key}"] for key in ("feature_ids", "target_scale", "global_mean", "cancer_labels", "cancer_means")}
            features = saved[f"{name}_feature_ids"].astype(str)
            if features.ndim != 1 or len(features) != output_dims[name] or len(np.unique(features)) != len(features) or not np.array_equal(features, baseline["feature_ids"]):
                raise ValueError(f"{name} feature IDs disagree with the trained decoder or saved means")
            if str(baseline["target_scale"]) != str(saved["target_scale"][index]):
                raise ValueError(f"{name} target scale differs from training")
            values = np.asarray(saved[f"{name}_target"], dtype=np.float32)
            if values.shape != (len(patients), len(features)) or not np.isfinite(values[order][data.availability[:, index]]).all():
                raise ValueError(f"Observed {name} targets must be finite and match patient/feature dimensions")
            if baseline["global_mean"].shape != (len(features),) or baseline["cancer_means"].shape != (len(baseline["cancer_labels"]), len(features)) or not np.isfinite(baseline["global_mean"]).all() or not np.isfinite(baseline["cancer_means"]).all():
                raise ValueError(f"Invalid saved {name} feature means")
            targets.append(values[order])
            baselines[name] = baseline
    return targets, baselines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    for name in MODALITIES:
        parser.add_argument(f"--{name}", type=Path, required=True, help=f"Observed {name} input NPZ with patient IDs")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder-chunk-size", type=int, default=320)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage C model.pt or frozen_fallback.pt")
    parser.add_argument("--targets", type=Path, required=True, help="Measured feature archive used for Stage C")
    parser.add_argument("--target-baselines", type=Path, required=True, help="Stage C target_baselines.npz")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if args.encoder_chunk_size < 1:
        raise ValueError("Encoder chunk size must be positive")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FiveModalCrossModalMultitaskModel(OmicsHEEndToEndCox(**saved["survival_model_kwargs"]), TargetConditionedCrossModalTranslator(**saved["translator_kwargs"]))
    model.load_state_dict(saved["model_state"], strict=True)
    model.to(args.device).eval()
    data = load_multimodal_data(args.cohort, [getattr(args, name) for name in MODALITIES], he_standardization=saved["he_standardization"])
    targets, baselines = load_evaluation_targets(args.targets, args.target_baselines, data, saved["translator_kwargs"]["output_dims"])
    with torch.no_grad():
        embeddings = extract_embeddings(model.survival, data, args.encoder_chunk_size, args.device)
        projected = torch.stack([model.survival.fusion.shared_projection(values) for values in embeddings], dim=1)
    predictions, metrics = evaluate_subsets(model, projected, data.cohort, data.availability, targets, baselines)
    for frame in (predictions, metrics):
        frame["fold"], frame["seed"] = saved["fold"], saved["seed"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output_dir / "predictions.tsv.gz", sep="\t", index=False)
    metrics.to_csv(args.output_dir / "feature_metrics.tsv.gz", sep="\t", index=False)
    print(f"Evaluated 30 source subsets, {len(predictions)} predictions and {len(metrics)} reconstruction summaries: {args.output_dir}")


if __name__ == "__main__":
    main()
