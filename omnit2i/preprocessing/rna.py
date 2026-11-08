"""RNA preprocessing with an explicit patient-level core/validation/test split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from omnit2i.data import validate_cohort
from .rna_t2i import (
    apply_cancer_standardization,
    corr_hclust_order,
    cox_score_z,
    fit_cancer_standardization,
    render_option2,
    sample_gamma,
    select_invariant_genes,
)


def fit_rna_preprocessing(
    expression: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    top_n: int = 2000,
    expression_bins: int = 3,
    invariant_per_bin: int = 50,
    height: int = 45,
    width: int = 45,
) -> dict[str, np.ndarray]:
    """Fit outer-training references and core-only C3, as in the manuscript version.

    Expression rows are genes and columns are samples. Outer training includes
    core and validation patients; only C3 uses core outcomes.
    Input values must already use the intended upstream expression scale;
    this function does not log-transform counts.
    """
    cohort = validate_cohort(cohort)
    if expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError("Expression gene and sample identifiers must be unique")
    if not all(isinstance(item, str) for item in expression.index):
        raise ValueError("Expression gene identifiers must be strings")
    if not 2 <= top_n <= min(len(expression), height * width) or min(height, width) < 4:
        raise ValueError("top_n must be at least 2 and fit both the feature set and image grid")
    if not 1 <= expression_bins <= len(expression) or invariant_per_bin < 1:
        raise ValueError("Invalid invariant gene panel size")
    train = cohort.loc[cohort["partition"].ne("test")]
    core = train["partition"].eq("core").to_numpy()
    samples = train["sample"].tolist()
    values = expression.loc[:, samples]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Outer-training RNA expression must be finite; supply a complete expression matrix")
    panel = select_invariant_genes(expression, samples, expression_bins, invariant_per_bin)
    invariant = panel["gene"].astype(str).tolist()
    corrected = values.sub(sample_gamma(expression, samples, invariant), axis=1)
    variance = corrected.var(axis=1, ddof=0).sort_values(ascending=False)
    genes = variance.head(top_n).index.tolist()
    matrix = corrected.loc[genes].T
    means = matrix.mean(axis=0).to_numpy(dtype=np.float32)
    stds = matrix.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    stds[stds < 1e-6] = 1.0
    standardized = ((matrix.to_numpy(dtype=np.float32) - means) / stds).astype(np.float32)
    labels, cancer_means, cancer_stds, global_mean, global_std = fit_cancer_standardization(
        standardized, train["cancer_type"]
    )
    return {
        "feature_ids": np.asarray(genes, dtype=str),
        "invariant_genes": np.asarray(invariant, dtype=str),
        "train_means": means,
        "train_stds": stds,
        "cancer_labels": labels,
        "cancer_means": cancer_means,
        "cancer_stds": cancer_stds,
        "global_cancer_mean": global_mean,
        "global_cancer_std": global_std,
        "cox_score_z": cox_score_z(
            standardized[core], train.loc[core, "time"].to_numpy(float), train.loc[core, "event"].to_numpy(bool)
        ),
        "layout_order": corr_hclust_order(standardized),
        "grid_height": np.asarray(height),
        "grid_width": np.asarray(width),
    }


def transform_rna(
    expression: pd.DataFrame, cohort: pd.DataFrame, artifact: dict[str, np.ndarray]
) -> np.ndarray:
    """Apply saved parameters to samples in cohort order, without refitting.

    All selected and invariant genes must be present and finite. New cancer
    types use the outer-training global mean and standard deviation for channel 2.
    """
    if expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError("Expression gene and sample identifiers must be unique")
    samples = cohort["sample"].astype(str).tolist()
    required_genes = np.union1d(artifact["feature_ids"], artifact["invariant_genes"])
    if not np.isfinite(expression.loc[required_genes, samples].to_numpy()).all():
        raise ValueError("Selected and invariant RNA expression must be finite")
    gamma = sample_gamma(expression, samples, artifact["invariant_genes"].tolist())
    matrix = expression.loc[artifact["feature_ids"], samples].sub(gamma, axis=1).T
    standardized = (
        (matrix.to_numpy(dtype=np.float32) - artifact["train_means"]) / artifact["train_stds"]
    ).astype(np.float32)
    cancer = apply_cancer_standardization(
        standardized, cohort["cancer_type"], artifact["cancer_labels"],
        artifact["cancer_means"], artifact["cancer_stds"],
        artifact["global_cancer_mean"], artifact["global_cancer_std"],
    )
    return render_option2(
        standardized, cancer, artifact["cox_score_z"], artifact["layout_order"],
        int(artifact["grid_height"]), int(artifact["grid_width"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path, required=True, help="Gene x sample TSV, optionally gzip; first column is gene ID")
    parser.add_argument("--cohort", type=Path, required=True, help="CSV: patient,sample,cancer_type,time,event,partition")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--top-n", type=int, default=2000)
    parser.add_argument("--expression-bins", type=int, default=3)
    parser.add_argument("--invariant-per-bin", type=int, default=50)
    args = parser.parse_args()
    cohort = validate_cohort(pd.read_csv(args.cohort, dtype={"patient": str, "sample": str}))
    expression = pd.read_csv(args.expression, sep="\t", index_col=0)
    expression.index = expression.index.astype(str)
    expression = expression.loc[:, cohort["sample"].tolist()].astype(np.float32)
    artifact = fit_rna_preprocessing(
        expression, cohort, top_n=args.top_n, expression_bins=args.expression_bins,
        invariant_per_bin=args.invariant_per_bin,
    )
    images = transform_rna(expression, cohort, artifact)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "preprocessing.npz", **artifact)
    np.savez_compressed(
        args.output_dir / "images.npz", images=images,
        patient=cohort["patient"].to_numpy(dtype=str),
        sample=cohort["sample"].to_numpy(dtype=str),
        cancer_type=cohort["cancer_type"].to_numpy(dtype=str),
        time=cohort["time"].to_numpy(float), event=cohort["event"].to_numpy(int),
        partition=cohort["partition"].to_numpy(dtype=str), feature_ids=artifact["feature_ids"],
    )
    cohort.to_csv(args.output_dir / "cohort.csv", index=False)
    print(f"Prepared {len(cohort)} patients, {len(artifact['feature_ids'])} genes: {args.output_dir}")


if __name__ == "__main__":
    main()
