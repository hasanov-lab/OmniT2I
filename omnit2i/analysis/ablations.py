"""Original Stage C OOF and frozen-subset differences for manuscript ablations."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from omnit2i.data import validate_cohort
from omnit2i.training.ablations import VARIANTS
from omnit2i.training.losses import fit_risk_calibration, apply_risk_calibration
from .aggregation import SEEDS, ROUTE, _mean_sd, subset_manifest
from .concordance import fold_metrics

FLAGS = ["has_rna", "has_cnv", "has_rppa", "has_methylation", "has_he"]
METRICS = ["pooled_c_index", "within_cancer_c_index", "raw_merged_pooled_c_index",
           "raw_merged_within_cancer_c_index", "within_fold_pooled_c_index_pair_weighted",
           "within_fold_within_cancer_c_index_pair_weighted"]

def _metrics_for_risk(frame: pd.DataFrame, risk_column: str) -> dict[str, float | int]:
    current = frame.copy()
    current["risk_score"] = current[risk_column].astype(float)
    return fold_metrics(current)



def _within_fold_pair_weighted(frame: pd.DataFrame) -> dict[str, float]:
    """C-index diagnostics that never compare risk scores across CV folds."""
    fold_rows = []
    for _, current in frame.groupby("fold", sort=True):
        metrics = _metrics_for_risk(current, "risk_score_raw")
        fold_rows.append(metrics)

    def weighted(metric: str, pair_count: str) -> float:
        values = np.asarray([row[metric] for row in fold_rows], dtype=float)
        weights = np.asarray([row[pair_count] for row in fold_rows], dtype=float)
        valid = np.isfinite(values) & (weights > 0)
        return float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")

    pooled_weights = np.asarray(
        [
            row["within_cancer_comparable_pairs"]
            + row["cross_cancer_comparable_pairs"]
            for row in fold_rows
        ],
        dtype=float,
    )
    pooled_values = np.asarray([row["pooled_c_index"] for row in fold_rows], dtype=float)
    valid = np.isfinite(pooled_values) & (pooled_weights > 0)
    return {
        "within_fold_pooled_c_index_pair_weighted": (
            float(np.average(pooled_values[valid], weights=pooled_weights[valid]))
            if valid.any()
            else float("nan")
        ),
        "within_fold_within_cancer_c_index_pair_weighted": weighted(
            "within_cancer_c_index", "within_cancer_comparable_pairs"
        ),
        "within_fold_cross_cancer_c_index_pair_weighted": weighted(
            "cross_cancer_c_index", "cross_cancer_comparable_pairs"
        ),
    }



def survival_metrics_for_seed(
    predictions: pd.DataFrame,
    seed: int,
    expected_patients: int | None,
    expected_complete_cases: int | None,
) -> list[dict[str, object]]:
    """Compute C-indices after merging the five OOF folds for one seed."""
    if set(predictions["fold"].astype(int)) != set(range(5)):
        raise ValueError(f"Seed {seed}: not all five held-out folds are present")
    if expected_patients is not None and len(predictions) != expected_patients:
        raise ValueError(
            f"Seed {seed}: expected {expected_patients} OOF patients, found {len(predictions)}"
        )
    scenarios = {
        "natural_availability": predictions,
        "complete_case": predictions.loc[predictions["complete_case"]].copy(),
    }
    if (
        expected_complete_cases is not None
        and len(scenarios["complete_case"]) != expected_complete_cases
    ):
        raise ValueError(
            f"Seed {seed}: expected {expected_complete_cases} complete cases, "
            f"found {len(scenarios['complete_case'])}"
        )
    rows = []
    calibrated_columns = {
        "natural_availability": "risk_score_calibrated",
        "complete_case": "risk_score_complete_case_calibrated",
    }
    calibration_sources = {
        "natural_availability": "inner_validation_natural",
        "complete_case": "inner_validation_complete_case",
    }
    for scenario, frame in scenarios.items():
        primary = _metrics_for_risk(frame, calibrated_columns[scenario])
        raw = _metrics_for_risk(frame, "risk_score_raw")
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "primary_risk": calibrated_columns[scenario],
                "calibration_reference": calibration_sources[scenario],
                "n_patients": len(frame),
                "n_events": int(frame["event"].astype(bool).sum()),
                **primary,
                **{f"raw_merged_{key}": value for key, value in raw.items()},
                **_within_fold_pair_weighted(frame),
            }
        )
    return rows



def paired_differences(table):
    keys = ["seed", "scenario"]
    reference = table.loc[table.variant.eq("full"), keys + METRICS]
    pieces = []
    for variant, frame in table.loc[table.variant.ne("full")].groupby("variant", sort=False):
        merged = reference.merge(frame[keys + METRICS], on=keys, how="outer",
                                 suffixes=("_full", "_ablation"),
                                 validate="one_to_one", indicator=True)
        if not merged._merge.eq("both").all():
            raise ValueError(f"Unpaired seed/scenario: {variant}")
        for metric in METRICS:
            row = merged[keys].copy()
            row["variant"], row["metric"] = variant, metric
            row["full"] = merged[f"{metric}_full"]
            row["ablation"] = merged[f"{metric}_ablation"]
            row["full_minus_ablation"] = row.full - row.ablation
            pieces.append(row)
    paired = pd.concat(pieces, ignore_index=True)
    summary = paired.groupby(["variant", "scenario", "metric"], as_index=False).agg(
        n_seeds=("seed", "nunique"), full_mean=("full", "mean"),
        ablation_mean=("ablation", "mean"),
        delta_mean=("full_minus_ablation", "mean"),
        delta_sd=("full_minus_ablation", "std"),
        delta_min=("full_minus_ablation", "min"),
        delta_max=("full_minus_ablation", "max"),
        full_higher_seeds=("full_minus_ablation", lambda x: int(x.gt(0).sum())),
        ablation_higher_seeds=("full_minus_ablation", lambda x: int(x.lt(0).sum())))
    return paired, summary



def paired_tables(full, ablation, keys, metrics):
    joined = full[keys + metrics].merge(ablation[keys + metrics], on=keys, how="outer",
        suffixes=("_full", "_ablation"), validate="one_to_one", indicator=True)
    require(joined._merge.eq("both").all(), "Missing full/ablation route or seed pair")
    rows = []
    for metric in metrics:
        part = joined[keys].copy()
        part["metric"] = metric
        part["full"] = joined[f"{metric}_full"]
        part["ablation"] = joined[f"{metric}_ablation"]
        require(np.isfinite(part[["full", "ablation"]].to_numpy()).all(), "Nonfinite paired metric")
        part["full_minus_ablation"] = part.full - part.ablation
        rows.append(part)
    pairs = pd.concat(rows, ignore_index=True)
    summary = pairs.groupby([k for k in keys if k != "seed"] + ["metric"], as_index=False).agg(
        n_seeds=("seed", "nunique"), full_mean=("full", "mean"), ablation_mean=("ablation", "mean"),
        delta_mean=("full_minus_ablation", "mean"), delta_sd=("full_minus_ablation", "std"),
        delta_min=("full_minus_ablation", "min"), delta_max=("full_minus_ablation", "max"),
        full_higher_seeds=("full_minus_ablation", lambda x: int(x.gt(0).sum())),
        ablation_higher_seeds=("full_minus_ablation", lambda x: int(x.lt(0).sum())))
    return pairs, summary


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_training_fold(root, master, seed, fold):
    folder = root / f"seed_{seed}" / f"fold{fold}" / "C"
    options = json.loads((folder / "options.json").read_text())
    require(options["seed"] == seed and options["fold"] == fold, "Training seed/fold differs")
    frame = validate_cohort(pd.read_csv(folder / "predictions.csv", dtype={"patient": str}))
    require(set(frame.patient) == set(master.patient), "Training cohort patient coverage differs")
    reference = master.set_index("patient").loc[frame.patient]
    for column in ("time", "event", "cancer_type", "fold"):
        require(np.array_equal(frame[column].to_numpy(), reference[column].to_numpy()),
                f"Training {column} differs from the master")
    require(np.array_equal(frame.partition.eq("test"), frame.fold.eq(fold)), "Training test partition differs")
    availability = reference[FLAGS]
    require(availability.isin([True, False]).all().all(), "Invalid master availability")
    complete = availability.all(axis=1).to_numpy()
    validation = frame.partition.eq("validation").to_numpy()
    raw = frame.risk_raw.to_numpy(np.float32)
    require(np.isfinite(raw).all() and np.isfinite(frame.risk).all(), "Nonfinite training risk")
    natural = fit_risk_calibration(raw, validation)
    require(np.allclose(apply_risk_calibration(raw, natural), frame.risk, rtol=0, atol=1e-6),
            "Saved natural risk disagrees with validation standardization")
    complete_calibration = fit_risk_calibration(raw, validation & complete)
    frame["risk_score_raw"], frame["risk_score_calibrated"] = raw, frame.risk
    frame["risk_score_complete_case_calibrated"] = np.where(
        complete, apply_risk_calibration(raw, complete_calibration), np.nan)
    frame["complete_case"], frame["seed"] = complete, seed
    return frame.loc[frame.partition.eq("test")].copy()


def training_tables(reference, roots, master):
    require(master.patient.is_unique and set(master.fold) == set(range(5)), "Master needs unique patients and all five folds")
    records = []
    complete_count = int(master[FLAGS].all(axis=1).sum())
    for variant, root in {"full": reference, **roots}.items():
        for seed in SEEDS:
            frame = pd.concat([read_training_fold(root, master, seed, fold) for fold in range(5)], ignore_index=True)
            require(frame.patient.is_unique, "Duplicate OOF patients")
            records.extend(dict(variant=variant, **row) for row in survival_metrics_for_seed(frame, seed, len(master), complete_count))
        print(f"Aggregated Stage C: {variant}", flush=True)
    survival = pd.DataFrame(records)
    summary = _mean_sd(survival, ["variant", "scenario"], ["n_patients", "n_events", *METRICS])
    paired, paired_summary = paired_differences(survival)
    tables = dict(survival_per_seed=survival, survival_summary=summary,
                  paired_survival_per_seed=paired, paired_survival_summary=paired_summary)
    groups = dict(encoder=("no_pretraining", "no_c2", "no_c3", "random_layout"),
                  component=("no_risk_consistency", "mean_fusion", "no_modality_dropout"),
                  simplified=("simplified",))
    for group, variants in groups.items():
        part = paired.loc[paired.variant.isin(variants)].copy()
        if group == "simplified":
            part["variant"] = "mean_fusion"  # Keep the original Figure 6 source-table label.
        if not part.empty:
            tables[f"{group}_paired_survival_per_seed"] = part
    return tables


def frozen_tables(reference, roots):
    metrics = ["pooled_c_index", "within_cancer_c_index", "cross_cancer_c_index"]
    measured, routes = subset_manifest()
    specifications = (
        ("measured", "measured/omnit2i_all_31_subsets_per_seed.tsv", ["seed", "source_subset", "observed_count"], metrics),
        ("selective", "selective/survival_per_seed.tsv", ["seed", *ROUTE, "condition"], metrics),
        ("virtual_gain", "selective/survival_deltas_per_seed.tsv", ["seed", *ROUTE],
         [f"delta_virtual_minus_source_only_{metric}" for metric in metrics]),
    )
    tables = {}
    for label, filename, keys, values in specifications:
        full = pd.read_csv(reference / filename, sep="\t")
        pairs, summaries = [], []
        for variant, root in roots.items():
            ablation = pd.read_csv(root / filename, sep="\t")
            for frame in (full, ablation):
                if label == "measured":
                    expected = {(seed, name, len(name.split("+"))) for seed in SEEDS for name in measured}
                else:
                    expected = {(seed, *route, condition) for seed in SEEDS for route in routes
                                for condition in ("source_only", "cancer_mean", "virtual", "measured")} if label == "selective" else {
                                    (seed, *route) for seed in SEEDS for route in routes}
                require(not frame.duplicated(keys).any() and set(frame[keys].itertuples(index=False, name=None)) == expected,
                        "Incomplete or duplicate frozen route/seed rows")
            if label != "virtual_gain":
                counts = full[keys + ["n_patients", "n_events"]].merge(
                    ablation[keys + ["n_patients", "n_events"]], on=keys, validate="one_to_one", suffixes=("_full", "_ablation"))
                require(all(counts[f"{name}_full"].eq(counts[f"{name}_ablation"]).all() for name in ("n_patients", "n_events")),
                        "Full/ablation populations differ")
            paired, summary = paired_tables(full, ablation, keys, values)
            pairs.append(paired.assign(variant=variant))
            summaries.append(summary.assign(variant=variant))
        tables[f"paired_{label}_per_seed"] = pd.concat(pairs, ignore_index=True)
        tables[f"paired_{label}_summary"] = pd.concat(summaries, ignore_index=True)
    return tables


def export_main():
    """Export public Stage C outputs for the manuscript survival analyses."""
    parser = argparse.ArgumentParser(description=export_main.__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True, help="seed_SEED/foldFOLD/C outputs")
    parser.add_argument("--output-dir", type=Path, required=True, help="New WORK/Inputs directory")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    master = pd.read_csv(args.master, dtype={"patient": str})
    require(master.patient.is_unique and set(master.fold) == set(range(5)), "Master needs unique patients and all five folds")
    require(master[FLAGS + ["event"]].isin([True, False]).all().all(), "Invalid master event/availability")
    master[FLAGS + ["event"]] = master[FLAGS + ["event"]].astype(bool)
    complete = master[FLAGS].all(axis=1)
    records = []
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output_dir.parent, prefix=f".{args.output_dir.name}-") as temporary:
        output = Path(temporary) / args.output_dir.name
        output.mkdir()
        for seed in SEEDS:
            folds = []
            for fold in range(5):
                frame = read_training_fold(args.training_root, master, seed, fold)
                frame["event"] = frame.event.astype(bool)  # R readers expect True/False, as in the original OOF files.
                folder = output / "oof" / f"seed_{seed}" / f"fold{fold}"
                folder.mkdir(parents=True)
                frame.to_csv(folder / "survival_oof_predictions.tsv.gz", sep="\t", index=False)
                folds.append(frame)
            oof = pd.concat(folds, ignore_index=True)
            require(oof.patient.is_unique, "Duplicate OOF patients")
            records.extend(survival_metrics_for_seed(oof, seed, len(master), int(complete.sum())))
            print(f"Exported survival OOF: seed {seed}", flush=True)
        master.to_csv(output / "master_folds.csv", index=False)
        master.loc[complete, ["patient", "cancer_type", "time", "event", "fold"]].sort_values("patient").to_csv(
            output / "complete_case_patients.tsv", sep="\t", index=False)
        pd.DataFrame(records).to_csv(output / "performance_per_seed.tsv", sep="\t", index=False)
        output.rename(args.output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-training", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True, help="VARIANT/seed_SEED/foldFOLD/C outputs")
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, required=True)
    parser.add_argument("--reference-evaluation", type=Path, help="Full-model aggregated frozen evaluations")
    parser.add_argument("--evaluation-root", type=Path, help="VARIANT/ aggregated frozen evaluations")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    require(len(args.variants) == len(set(args.variants)), "Duplicate variants")
    require(bool(args.reference_evaluation) == bool(args.evaluation_root), "Supply both frozen evaluation roots")
    master = pd.read_csv(args.master, dtype={"patient": str})
    tables = training_tables(args.reference_training, {name: args.training_root / name for name in args.variants}, master)
    if args.reference_evaluation:
        tables.update(frozen_tables(args.reference_evaluation, {name: args.evaluation_root / name for name in args.variants}))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
