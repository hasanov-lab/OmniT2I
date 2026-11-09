"""Aggregate frozen OOF evaluations with the manuscript metric definitions."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omnit2i.models import MODALITIES
from .concordance import fold_metrics

SEEDS = (101, 202, 303, 404, 505)
METRICS = ("pooled_c_index", "within_cancer_c_index", "cross_cancer_c_index", "within_cancer_pair_fraction")
CONDITIONS = ("source_only", "cancer_mean", "virtual", "measured")
ROUTE = ["source_subset", "virtual_targets", "observed_count", "virtual_count", "final_count"]


def _mean_sd(
    frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]
) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, sort=True, dropna=False)[value_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_sd")
    count = grouped.size().rename("n_seeds")
    return pd.concat([mean, std, count], axis=1).reset_index()


def aggregate_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "seed",
        "fold",
        "source_subset",
        "observed_count",
        "target",
        "target_kind",
        "target_scale",
        "n_test",
        "n_features",
        "model_squared_error_sum",
        "cancer_baseline_squared_error_sum",
        "n_feature_values",
    }
    missing = required - set(features)
    if missing:
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    identifiers = [
        "seed",
        "source_subset",
        "observed_count",
        "target",
        "target_kind",
        "target_scale",
        "n_features",
    ]
    excluded = set(identifiers) | {
        "fold",
        "n_test",
        "model_squared_error_sum",
        "cancer_baseline_squared_error_sum",
        "n_feature_values",
    }
    weighted_metrics = [
        column
        for column in features.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    rows: list[dict[str, Any]] = []
    for keys, frame in features.groupby(identifiers, sort=True, dropna=False):
        weights = frame["n_test"].to_numpy(float)
        model_sse = float(frame["model_squared_error_sum"].sum())
        baseline_sse = float(frame["cancer_baseline_squared_error_sum"].sum())
        n_values = int(frame["n_feature_values"].sum())
        row: dict[str, Any] = dict(zip(identifiers, keys, strict=True))
        row.update(
            {
                "n_test": int(frame["n_test"].sum()),
                "n_folds": int(len(frame)),
                "model_mse": model_sse / n_values,
                "cancer_baseline_mse": baseline_sse / n_values,
                "residual_mse_improvement": (
                    1.0 - model_sse / baseline_sse
                    if baseline_sse > 0
                    else float("nan")
                ),
            }
        )
        for metric in weighted_metrics:
            values = frame[metric].to_numpy(float)
            finite = np.isfinite(values)
            row[f"{metric}_fold_weighted"] = (
                float(np.average(values[finite], weights=weights[finite]))
                if finite.any()
                else float("nan")
            )
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    if len(per_seed) != 5 * 75:
        raise ValueError("Expected 75 subset-target rows for each of five seeds")
    value_columns = [
        column
        for column in per_seed.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "observed_count", "n_features"}
    ]
    summary = _mean_sd(
        per_seed,
        [
            "source_subset",
            "observed_count",
            "target",
            "target_kind",
            "target_scale",
            "n_features",
        ],
        value_columns,
    )
    return per_seed, summary


def subset_manifest():
    subsets = [source for size in range(1, 6) for source in combinations(MODALITIES, size)]
    measured = {"+".join(source) for source in subsets}
    selective = set()
    for source in subsets[:-1]:
        missing = [name for name in MODALITIES if name not in source]
        for count in range(1, len(missing) + 1):
            for chosen in combinations(missing, count):
                selective.add(("+".join(source), "+".join(chosen), len(source), count, len(source) + count))
    return measured, selective


def load_seed(root, seed):
    parts = {name: [] for name in ("predictions", "measured_predictions", "selective_predictions", "feature_metrics")}
    for fold in range(5):
        folder = root / f"seed_{seed}" / f"fold{fold}"
        for name in parts:
            frame = pd.read_csv(folder / f"{name}.tsv.gz", sep="\t", dtype={"patient": str})
            if frame.empty or not frame.seed.eq(seed).all() or not frame.fold.eq(fold).all():
                raise ValueError(f"Incorrect or empty seed/fold table: {folder}/{name}")
            if name == "feature_metrics" and len(frame) != 75:
                raise ValueError(f"Incomplete feature metrics: {folder}")
            parts[name].append(frame)
    return {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}


def aggregate_predictions(frame, keys, risks, expected_routes, reference):
    actual = set(map(tuple, frame[keys].drop_duplicates().to_numpy()))
    if actual != expected_routes:
        raise ValueError("Incomplete or incorrect evaluation route coverage")
    if not frame.event.isin([0, 1]).all() or not np.isfinite(frame.time).all() or not frame.time.gt(0).all():
        raise ValueError("Invalid evaluation survival outcomes")
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        if group.patient.duplicated().any():
            raise ValueError("Duplicate OOF patient within a route")
        clinical = group.set_index("patient")[list(reference.columns)].sort_index()
        pd.testing.assert_frame_equal(clinical, reference, check_dtype=False)
        if not np.isfinite(group[list(risks.values())]).all().all():
            raise ValueError("Nonfinite risk predictions")
        for condition, column in risks.items():
            metrics = fold_metrics(group.rename(columns={column: "risk_score"})[["event", "time", "cancer_type", "risk_score"]])
            rows.append(dict(seed=int(group.seed.iloc[0]), **dict(zip(keys, values)),
                             condition=condition, n_patients=len(group), n_events=int(group.event.sum()), **metrics))
    return pd.DataFrame(rows)


def survival_tables(per_seed, keys):
    summary = _mean_sd(per_seed, keys + ["condition"], ["n_patients", "n_events", *METRICS])
    wide = per_seed.pivot(index=["seed", *keys], columns="condition", values=list(METRICS))
    deltas = pd.DataFrame({f"delta_{a}_minus_{b}_{metric}": wide[(metric, a)] - wide[(metric, b)]
                          for a, b in (("virtual", "source_only"), ("virtual", "cancer_mean"),
                                       ("measured", "virtual"), ("measured", "source_only"))
                          for metric in METRICS}).reset_index()
    delta_columns = [column for column in deltas if column.startswith("delta_")]
    counts = [column for column in keys if column.endswith("count")]
    by_count = per_seed.groupby(["seed", *counts, "condition"])[list(METRICS)].mean().reset_index()
    delta_by_count = deltas.groupby(["seed", *counts])[delta_columns].mean().reset_index()
    return dict(survival_per_seed=per_seed, survival_summary=summary,
                survival_deltas_per_seed=deltas, survival_deltas_summary=_mean_sd(deltas, keys, delta_columns),
                survival_by_counts_per_seed=by_count,
                survival_by_counts_summary=_mean_sd(by_count, counts + ["condition"], list(METRICS)),
                survival_deltas_by_counts_per_seed=delta_by_count,
                survival_deltas_by_counts_summary=_mean_sd(delta_by_count, counts, delta_columns))


def aggregate(root, output):
    if output.exists():
        raise FileExistsError(output)
    measured_names, routes = subset_manifest()
    proper = {(name, len(name.split("+"))) for name in measured_names if len(name.split("+")) < 5}
    all_measured = {(name, len(name.split("+"))) for name in measured_names}
    collected = {name: [] for name in ("measured", "fill_all", "selective", "features")}
    reference = None
    for seed in SEEDS:
        frames = load_seed(root, seed)
        if reference is None:
            reference = frames["measured_predictions"].drop_duplicates("patient").set_index("patient")[["cancer_type", "time", "event", "fold"]].sort_index()
        measured = aggregate_predictions(frames["measured_predictions"], ["source_subset", "observed_count"], {"measured": "risk_score"}, all_measured, reference)
        collected["measured"].append(measured.drop(columns="condition"))
        for kind, filename, keys, expected in (("fill_all", "predictions", ["source_subset", "observed_count"], proper),
                                                ("selective", "selective_predictions", ROUTE, routes)):
            table = aggregate_predictions(frames[filename], keys, {name: f"risk_{name}" for name in CONDITIONS}, expected, reference)
            collected[kind].append(table)
        features = frames["feature_metrics"]
        expected_features = {(name, target) for name, _ in proper for target in MODALITIES if target not in name.split("+")}
        if set(map(tuple, features[["source_subset", "target"]].drop_duplicates().to_numpy())) != expected_features or features.duplicated(["source_subset", "target", "fold"]).any():
            raise ValueError("Incomplete or duplicate feature routes")
        collected["features"].append(features)
        print(f"Aggregated seed {seed}: 31 measured subsets and 180 selective routes", flush=True)
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
    full_fill = tables["selective"].loc[tables["selective"].final_count.eq(5)].merge(
        tables["fill_all"], on=["seed", "source_subset", "observed_count", "condition"], validate="one_to_one", suffixes=("", "_original"))
    np.testing.assert_allclose(full_fill[list(METRICS)], full_fill[[metric + "_original" for metric in METRICS]], atol=1e-12, rtol=0, equal_nan=True)
    output.mkdir(parents=True, exist_ok=False)
    def write(folder, values):
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        for name, frame in values.items():
            frame.to_csv(destination / f"{name}.tsv", sep="\t", index=False)
    features, feature_summary = aggregate_features(tables["features"])
    write("summary", {**survival_tables(tables["fill_all"], ["source_subset", "observed_count"]), "feature_per_seed": features, "feature_summary": feature_summary})
    write("selective", survival_tables(tables["selective"], ROUTE))
    write("measured", {"omnit2i_all_31_subsets_per_seed": tables["measured"],
                       "omnit2i_all_31_subsets_summary": _mean_sd(tables["measured"], ["source_subset", "observed_count"], ["n_patients", "n_events", *METRICS])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.input_root, args.output_dir)


if __name__ == "__main__":
    main()
