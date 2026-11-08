"""Manuscript CNV, RPPA/protein, and promoter-gene methylation preprocessing."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from omnit2i.data import validate_cohort
from . import cnv_t2i, methylation_t2i, rppa_t2i


def _matrix(matrix: pd.DataFrame, samples: list[str]) -> np.ndarray:
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("Feature and sample identifiers must be unique")
    if not all(isinstance(item, str) and item for item in matrix.index):
        raise ValueError("Feature identifiers must be nonempty strings")
    values = matrix.loc[:, samples].T.to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("Matrix values must not contain infinity; use NaN for missing measurements")
    return values


def fit_omics_preprocessing(
    matrix: pd.DataFrame,
    cohort: pd.DataFrame,
    modality: str,
    *,
    top_n: int = 2000,
    chromosomes: pd.Series | None = None,
) -> dict[str, np.ndarray]:
    """Fit outer-training references and core-only C3 for one observed modality.

    Rows are features; columns are sample IDs. CNV uses the supplied gene-level
    scale, protein uses RPPA measurements, and methylation uses promoter-gene
    beta values with chromosome annotations. Protein retains every feature
    passing QC; top_n applies only to CNV and methylation.
    """
    cohort = validate_cohort(cohort)
    if modality not in {"cnv", "protein", "methylation"}:
        raise ValueError("modality must be cnv, protein, or methylation")
    height = width = 17 if modality == "protein" else 45
    if modality != "protein" and not 2 <= top_n <= min(len(matrix), height * width):
        raise ValueError("top_n must be at least 2 and fit both the feature set and image grid")
    train = cohort.loc[cohort["partition"].ne("test")]
    values = _matrix(matrix, train["sample"].tolist())
    identifiers = matrix.index.to_numpy(dtype=str)
    if modality == "cnv":
        # The manuscript Option2 channels do not use the signed clipping bounds.
        ranking = pd.DataFrame({"feature": identifiers, "variance": np.nanvar(values, axis=0, ddof=0)})
        positions = ranking.sort_values(["variance", "feature"], ascending=[False, True], kind="stable").index[:top_n].to_numpy()
        selected = values[:, positions]
        if not np.isfinite(selected).all():
            raise ValueError("Selected CNV measurements must be finite")
        means, stds = cnv_t2i.fit_gene_standardization(selected)
        medians = np.empty(0, dtype=np.float32)
        standardized = cnv_t2i.transform_gene_standardization(selected, means, stds)
    else:
        if modality == "protein":
            keep, _, _ = rppa_t2i.fit_feature_filter(values, 0.20, 1e-8)
            positions = np.flatnonzero(keep)
        else:
            if chromosomes is None or chromosomes.index.has_duplicates:
                raise ValueError("Methylation requires unique chromosome annotations for every gene")
            chrom = chromosomes.reindex(matrix.index)
            if chrom.isna().any() or chrom.astype(str).str.strip().eq("").any():
                raise ValueError("Every methylation gene must have a chromosome annotation")
            values = methylation_t2i.beta_to_m_value(values)
            keep, _, variance = methylation_t2i.fit_feature_filter(values, chrom.ne("chrY").to_numpy(), 0.01, 1e-8)
            positions = methylation_t2i.select_top_variance(identifiers, keep, variance, top_n)
        if not 2 <= len(positions) <= height * width:
            raise ValueError("At least two retained features must fit the image grid")
        # RPPA and methylation use the same median-imputation and z-scoring.
        medians, means, stds = rppa_t2i.fit_impute_standardize(values[:, positions])
        standardized = rppa_t2i.transform_impute_standardize(values[:, positions], medians, means, stds)
    labels, cancer_means, cancer_stds, global_mean, global_std = cnv_t2i.fit_cancer_standardization(standardized, train["cancer_type"])
    core = train["partition"].eq("core").to_numpy()
    artifact = {
        "modality": np.asarray(modality),
        "feature_ids": identifiers[positions],
        "train_medians": medians,
        "train_means": means,
        "train_stds": stds,
        "cancer_labels": labels,
        "cancer_means": cancer_means,
        "cancer_stds": cancer_stds,
        "global_cancer_mean": global_mean,
        "global_cancer_std": global_std,
        "cox_score_z": cnv_t2i.cox_score_z(standardized[core], train.loc[core, "time"].to_numpy(float), train.loc[core, "event"].to_numpy(bool)),
        "layout_order": cnv_t2i.corr_hclust_snake_order(standardized),
        "grid_height": np.asarray(height),
        "grid_width": np.asarray(width),
    }
    if modality == "methylation":
        artifact["feature_chromosomes"] = chromosomes.loc[artifact["feature_ids"]].to_numpy(dtype=str)
    return artifact


def transform_omics_features(matrix: pd.DataFrame, cohort: pd.DataFrame, artifact: dict[str, np.ndarray]) -> np.ndarray:
    """Apply fitted parameters in cohort order; missing selected features raise an error."""
    values = _matrix(matrix, cohort["sample"].astype(str).tolist())
    positions = matrix.index.get_indexer(artifact["feature_ids"])
    if (positions < 0).any():
        raise ValueError("Matrix is missing selected features")
    values = values[:, positions]
    modality = str(artifact["modality"])
    if modality == "methylation":
        values = methylation_t2i.beta_to_m_value(values)
    if modality == "cnv":
        if not np.isfinite(values).all():
            raise ValueError("Selected CNV measurements must be finite")
        standardized = cnv_t2i.transform_gene_standardization(values, artifact["train_means"], artifact["train_stds"])
    elif modality in {"protein", "methylation"}:
        standardized = rppa_t2i.transform_impute_standardize(values, artifact["train_medians"], artifact["train_means"], artifact["train_stds"])
    else:
        raise ValueError("Unknown artifact modality")
    return standardized


def transform_omics(matrix: pd.DataFrame, cohort: pd.DataFrame, artifact: dict[str, np.ndarray]) -> np.ndarray:
    """Render the three manuscript channels from saved preprocessing parameters."""
    standardized = transform_omics_features(matrix, cohort, artifact)
    cancer = cnv_t2i.apply_cancer_standardization(
        standardized, cohort["cancer_type"], artifact["cancer_labels"], artifact["cancer_means"], artifact["cancer_stds"], artifact["global_cancer_mean"], artifact["global_cancer_std"],
    )
    return methylation_t2i.render_option2(standardized, cancer, artifact["cox_score_z"], artifact["layout_order"], int(artifact["grid_height"]), int(artifact["grid_width"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("cnv", "protein", "methylation"), required=True)
    parser.add_argument("--matrix", type=Path, required=True, help="Feature x sample TSV, optionally gzip; first column is feature ID")
    parser.add_argument("--cohort", type=Path, required=True, help="CSV: patient,sample,cancer_type,time,event,partition; observed patients only")
    parser.add_argument("--chromosomes", type=Path, help="Methylation TSV with feature_id,chrom columns; chrY is excluded")
    parser.add_argument("--top-n", type=int, default=2000, help="CNV/methylation only; protein retains all QC-passing features")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    cohort = validate_cohort(pd.read_csv(args.cohort, dtype={"patient": str, "sample": str}))
    matrix = pd.read_csv(args.matrix, sep="\t", index_col=0)
    matrix.index = matrix.index.astype(str)
    chromosomes = None
    if args.chromosomes:
        chromosomes = pd.read_csv(args.chromosomes, sep="\t", dtype=str).set_index("feature_id")["chrom"]
    artifact = fit_omics_preprocessing(matrix, cohort, args.modality, top_n=args.top_n, chromosomes=chromosomes)
    images = transform_omics(matrix, cohort, artifact)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "preprocessing.npz", **artifact)
    np.savez_compressed(
        args.output_dir / "images.npz", images=images, feature_ids=artifact["feature_ids"],
        **{key: cohort[key].to_numpy(dtype=str if key in {"patient", "sample", "cancer_type", "partition"} else float) for key in ("patient", "sample", "cancer_type", "time", "event", "partition")},
    )
    np.savez_compressed(
        args.output_dir / "features.npz", features=transform_omics_features(matrix, cohort, artifact),
        feature_ids=artifact["feature_ids"],
        **{key: cohort[key].to_numpy(dtype=str if key in {"patient", "cancer_type", "partition"} else float)
           for key in ("patient", "cancer_type", "time", "event", "partition")},
    )
    cohort.to_csv(args.output_dir / "cohort.csv", index=False)
    print(f"Prepared {len(cohort)} {args.modality} patients, {len(artifact['feature_ids'])} features: {args.output_dir}")


if __name__ == "__main__":
    main()
