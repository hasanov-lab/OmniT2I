"""Pool five outer folds and reproduce the original paired comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omnit2i.analysis.aggregation import SEEDS, _mean_sd
from omnit2i.analysis.concordance import fold_metrics
from omnit2i.data import validate_cohort
from .run import ALL_MODELS, AVAILABILITY_COLUMNS, modality_subsets, subset_slug

METRICS = ["pooled_c_index", "within_cancer_c_index", "cross_cancer_c_index"]

def require(condition: bool, message: str) -> None:
    """Raise a readable error when a cross-run comparability check fails."""

    if not condition:
        raise ValueError(message)


def _read_fold(
    root: Path,
    model: str,
    subset: tuple[str, ...],
    subset_index: int,
    seed: int,
    fold: int,
    expected: pd.DataFrame,
    n_complete_cases: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    directory = root / model / subset_slug(subset) / f"seed_{seed}" / f"fold{fold}"
    marker_path = directory / "complete.json"
    prediction_path = directory / "predictions.tsv.gz"
    tuning_path = directory / "tuning.tsv"
    for path in (marker_path, prediction_path, tuning_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    marker = json.loads(marker_path.read_text())
    expected_subset = "+".join(subset)
    if (
        marker.get("status") != "complete"
        or marker.get("model") != model
        or marker.get("source_subset") != expected_subset
        or int(marker.get("subset_index", -1)) != subset_index
        or int(marker.get("seed", -1)) != seed
        or int(marker.get("fold", -1)) != fold
        or int(marker.get("n_complete_cases", -1)) != n_complete_cases
        or int(marker.get("availability_indicators", -1)) != 0
    ):
        raise ValueError(f"Invalid completion marker: {marker_path}")
    predictions = pd.read_csv(prediction_path, sep="\t")
    tuning = pd.read_csv(tuning_path, sep="\t")
    required = {
        "patient",
        "cancer_type",
        "time",
        "event",
        "fold",
        "seed",
        "baseline",
        "source_subset",
        "observed_count",
        "risk_score_calibrated",
        "complete_case",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"{prediction_path} lacks columns {sorted(missing)}")
    if len(predictions) != len(expected) or predictions["patient"].duplicated().any():
        raise ValueError(f"Wrong row count or duplicate patient: {prediction_path}")
    if set(predictions["patient"].astype(str)) != set(expected["patient"].astype(str)):
        raise ValueError(f"Outer-fold patient coverage differs: {prediction_path}")
    if set(predictions["baseline"].astype(str)) != {model}:
        raise ValueError(f"Model label differs: {prediction_path}")
    if set(predictions["source_subset"].astype(str)) != {expected_subset}:
        raise ValueError(f"Subset label differs: {prediction_path}")
    if set(predictions["observed_count"].astype(int)) != {len(subset)}:
        raise ValueError(f"Observed count differs: {prediction_path}")
    if set(predictions["seed"].astype(int)) != {seed} or set(
        predictions["fold"].astype(int)
    ) != {fold}:
        raise ValueError(f"Seed/fold differs: {prediction_path}")
    complete = predictions["complete_case"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if complete.isna().any() or not complete.all():
        raise ValueError(f"Non-complete patient entered benchmark: {prediction_path}")
    risk = predictions["risk_score_calibrated"].to_numpy(dtype=float)
    if not np.isfinite(risk).all() or risk.std(ddof=0) < 1e-8:
        raise ValueError(f"Invalid calibrated risk: {prediction_path}")
    if (
        len(tuning) != 1
        or set(tuning["baseline"].astype(str)) != {model}
        or set(tuning["component"].astype(str)) != {expected_subset}
        or int(tuning.iloc[0]["availability_indicators"]) != 0
    ):
        raise ValueError(f"Invalid tuning row: {tuning_path}")
    aligned = predictions.set_index("patient").loc[expected["patient"]]
    for column in ("time", "event", "cancer_type", "fold"):
        if not np.array_equal(aligned[column].to_numpy(), expected[column].to_numpy()):
            raise ValueError(f"Prediction {column} disagrees with the cohort: {prediction_path}")
    return predictions, tuning, marker


def compare_with_omnit2i(
    baselines: pd.DataFrame,
    omnIt2i: pd.DataFrame,
    label: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    baseline_columns = ["model", "seed", "source_subset", "observed_count", *METRICS]
    baseline = baselines.loc[:, baseline_columns].rename(
        columns={metric: f"baseline_{metric}" for metric in METRICS}
    )
    omni_columns = ["seed", "source_subset", "observed_count", *METRICS]
    omni = omnIt2i.loc[:, omni_columns].rename(
        columns={metric: f"omnit2i_{metric}" for metric in METRICS}
    )

    paired = baseline.merge(
        omni,
        on=["seed", "source_subset", "observed_count"],
        how="inner",
        validate="many_to_one",
    )
    expected_rows = int(baselines["model"].nunique()) * 31 * 5
    require(len(paired) == expected_rows, f"{label}: joined row-count mismatch")
    for metric in METRICS:
        paired[f"delta_{metric}"] = (
            paired[f"omnit2i_{metric}"] - paired[f"baseline_{metric}"]
        )

    aggregation: dict[str, tuple[str, str]] = {}
    for metric in METRICS:
        aggregation[f"baseline_{metric}_mean"] = (f"baseline_{metric}", "mean")
        aggregation[f"baseline_{metric}_sd"] = (f"baseline_{metric}", "std")
        aggregation[f"omnit2i_{metric}_mean"] = (f"omnit2i_{metric}", "mean")
        aggregation[f"omnit2i_{metric}_sd"] = (f"omnit2i_{metric}", "std")
        aggregation[f"delta_{metric}_mean"] = (f"delta_{metric}", "mean")
        aggregation[f"delta_{metric}_sd"] = (f"delta_{metric}", "std")
        aggregation[f"positive_seeds_{metric}"] = (
            f"delta_{metric}",
            lambda values: int((values > 0).sum()),
        )

    summary = (
        paired.groupby(
            ["model", "source_subset", "observed_count"],
            as_index=False,
            sort=False,
        )
        .agg(**aggregation)
        .sort_values(["observed_count", "source_subset", "model"])
    )

    pooled_indices = summary.groupby("source_subset")[
        "baseline_pooled_c_index_mean"
    ].idxmax()
    pooled_best = summary.loc[pooled_indices].copy()
    pooled_best["selection_metric"] = "pooled_c_index"

    within_indices = summary.groupby("source_subset")[
        "baseline_within_cancer_c_index_mean"
    ].idxmax()
    within_best = summary.loc[within_indices].copy()
    within_best["selection_metric"] = "within_cancer_c_index"

    best = pd.concat([pooled_best, within_best], ignore_index=True).sort_values(
        ["selection_metric", "observed_count", "source_subset"]
    )

    per_seed_count = (
        paired.groupby(["model", "observed_count", "seed"], as_index=False)
        .agg(
            baseline_pooled_c_index=("baseline_pooled_c_index", "mean"),
            omnit2i_pooled_c_index=("omnit2i_pooled_c_index", "mean"),
            delta_pooled_c_index=("delta_pooled_c_index", "mean"),
            baseline_within_cancer_c_index=(
                "baseline_within_cancer_c_index",
                "mean",
            ),
            omnit2i_within_cancer_c_index=(
                "omnit2i_within_cancer_c_index",
                "mean",
            ),
            delta_within_cancer_c_index=(
                "delta_within_cancer_c_index",
                "mean",
            ),
        )
    )
    by_count = (
        per_seed_count.groupby(["model", "observed_count"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            baseline_pooled_c_index_mean=("baseline_pooled_c_index", "mean"),
            baseline_pooled_c_index_sd=("baseline_pooled_c_index", "std"),
            omnit2i_pooled_c_index_mean=("omnit2i_pooled_c_index", "mean"),
            omnit2i_pooled_c_index_sd=("omnit2i_pooled_c_index", "std"),
            delta_pooled_c_index_mean=("delta_pooled_c_index", "mean"),
            delta_pooled_c_index_sd=("delta_pooled_c_index", "std"),
            baseline_within_cancer_c_index_mean=(
                "baseline_within_cancer_c_index",
                "mean",
            ),
            baseline_within_cancer_c_index_sd=(
                "baseline_within_cancer_c_index",
                "std",
            ),
            omnit2i_within_cancer_c_index_mean=(
                "omnit2i_within_cancer_c_index",
                "mean",
            ),
            omnit2i_within_cancer_c_index_sd=(
                "omnit2i_within_cancer_c_index",
                "std",
            ),
            delta_within_cancer_c_index_mean=(
                "delta_within_cancer_c_index",
                "mean",
            ),
            delta_within_cancer_c_index_sd=(
                "delta_within_cancer_c_index",
                "std",
            ),
        )
        .sort_values(["observed_count", "model"])
    )

    model_summary = (
        summary.groupby("model", as_index=False)
        .agg(
            subsets=("source_subset", "nunique"),
            baseline_pooled_c_index_mean=("baseline_pooled_c_index_mean", "mean"),
            omnit2i_pooled_c_index_mean=("omnit2i_pooled_c_index_mean", "mean"),
            delta_pooled_c_index_mean=("delta_pooled_c_index_mean", "mean"),
            omnit2i_pooled_wins=(
                "delta_pooled_c_index_mean",
                lambda values: int((values > 0).sum()),
            ),
            baseline_within_cancer_c_index_mean=(
                "baseline_within_cancer_c_index_mean",
                "mean",
            ),
            omnit2i_within_cancer_c_index_mean=(
                "omnit2i_within_cancer_c_index_mean",
                "mean",
            ),
            delta_within_cancer_c_index_mean=(
                "delta_within_cancer_c_index_mean",
                "mean",
            ),
            omnit2i_within_cancer_wins=(
                "delta_within_cancer_c_index_mean",
                lambda values: int((values > 0).sum()),
            ),
        )
        .sort_values("baseline_pooled_c_index_mean", ascending=False)
    )
    return paired, summary, best, by_count, model_summary


def comparison_tables(baselines, omni):
    baselines = baselines.rename(columns={"patients": "n_patients", "events": "n_events"})
    routes = {(seed, "+".join(subset), len(subset)) for seed in SEEDS for subset in modality_subsets()}
    models = set(baselines.model)
    require(bool(models) and models <= set(ALL_MODELS), "Unknown or empty comparison models")
    for frame, keys, expected in (
        (omni, ["seed", "source_subset", "observed_count"], routes),
        (baselines, ["model", "seed", "source_subset", "observed_count"],
         {(model, *route) for model in models for route in routes}),
    ):
        require(not frame.duplicated(keys).any(), "Duplicate model/subset/seed rows")
        require(set(frame[keys].itertuples(index=False, name=None)) == expected,
                "Incomplete model/subset/seed coverage")
        require(frame[METRICS].notna().all().all() and frame[METRICS].ge(0).all().all()
                and frame[METRICS].le(1).all().all(), "Invalid comparison C-indices")
    counts = baselines.merge(omni, on=["seed", "source_subset", "observed_count"],
                             suffixes=("_baseline", "_omni"), validate="many_to_one")
    for column in ("n_patients", "n_events"):
        require(counts[f"{column}_baseline"].eq(counts[f"{column}_omni"]).all(),
                "Comparison patient/event counts differ")
    names = ("paired_per_seed", "vs_all_models", "subsetwise_best_descriptive", "by_observed_count", "model_summary")
    return dict(zip(names, compare_with_omnit2i(baselines, omni, "Shared-core specialists")))


def aggregate(root, cohort, output, omni, models=ALL_MODELS):
    if output.exists():
        raise FileExistsError(output)
    cohort = validate_cohort(cohort)
    require(set(cohort.fold) == set(range(5)), "All five outer folds are required")
    require(cohort[AVAILABILITY_COLUMNS].isin([True, False]).all().all(), "Invalid modality availability")
    require(bool(models) and len(set(models)) == len(models) and set(models) <= set(ALL_MODELS), "Unknown or duplicate models")
    master = cohort.loc[cohort[AVAILABILITY_COLUMNS].all(axis=1)].reset_index(drop=True)
    require(set(master.fold) == set(range(5)), "Each fold needs complete-case test patients")
    rows, tuning_rows = [], []
    for model in models:
        for index, subset in enumerate(modality_subsets()):
            for seed in SEEDS:
                folds = []
                for fold in range(5):
                    predictions, tuning, _ = _read_fold(root, model, subset, index, seed, fold,
                                                        master.loc[master.fold.eq(fold)], len(master))
                    folds.append(predictions)
                    tuning_rows.append(tuning)
                predictions = pd.concat(folds, ignore_index=True).set_index("patient").loc[master.patient]
                metric_frame = predictions[["event", "time", "cancer_type", "risk_score_calibrated"]].rename(
                    columns={"risk_score_calibrated": "risk_score"})
                rows.append(dict(model=model, source_subset="+".join(subset), subset_slug=subset_slug(subset),
                                 subset_index=index, observed_count=len(subset), seed=seed,
                                 patients=len(predictions), events=int(predictions.event.sum()), **fold_metrics(metric_frame)))
        print(f"Aggregated {model}: 31 subsets × 5 seeds", flush=True)
    per_seed = pd.DataFrame(rows)
    summary_metrics = ["patients", "events", *METRICS, "within_cancer_comparable_pairs",
                       "cross_cancer_comparable_pairs", "within_cancer_pair_fraction"]
    tables = dict(model_subset_per_seed=per_seed,
                  model_subset_summary=_mean_sd(per_seed, ["model", "source_subset", "subset_slug", "subset_index", "observed_count"], summary_metrics),
                  tuning_all_folds=pd.concat(tuning_rows, ignore_index=True),
                  **comparison_tables(per_seed, omni))
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.tsv", sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True, help="Shared cohort.csv (all patients)")
    parser.add_argument("--omnit2i-per-seed", type=Path, required=True, help="Measured omnit2i_all_31_subsets_per_seed.tsv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    args = parser.parse_args()
    aggregate(args.input_root, pd.read_csv(args.cohort, dtype={"patient": str}), args.output_dir,
              pd.read_csv(args.omnit2i_per_seed, sep="\t"), args.models)


if __name__ == "__main__":
    main()

