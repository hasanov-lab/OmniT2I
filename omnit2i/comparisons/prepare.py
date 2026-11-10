"""Conventional outer-training RNA z-scores for the original specialists."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from omnit2i.models import MODALITIES
from omnit2i.data import validate_cohort
from omnit2i.preprocessing.rna_t2i import load_expression

RNA_SCALE = "xena_log_expression_train_variance_top2000_z_no_invariant"

def select_standard_rna(
    expression: pd.DataFrame,
    cohort: pd.DataFrame,
    held_out_fold: int,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select and standardize RNA without invariant-gene correction."""
    if top_n <= 0 or top_n > len(expression):
        raise ValueError("rna_top_n must be positive and no larger than RNA genes")
    samples = cohort["sample"].astype(str).tolist()
    missing = sorted(set(samples) - set(expression.columns))
    if missing:
        raise ValueError(f"Expression matrix is missing {len(missing)} RNA samples")
    train_mask = cohort["fold"].ne(held_out_fold).to_numpy(dtype=bool)
    if not train_mask.any() or train_mask.all():
        raise ValueError("RNA cohort does not define a valid held-out fold")
    train_samples = cohort.loc[train_mask, "sample"].astype(str).tolist()

    variance = expression.loc[:, train_samples].var(axis=1, ddof=0)
    feature_ids = variance.index.astype(str).to_numpy()
    variance_values = variance.to_numpy(dtype=np.float64)
    if not np.isfinite(variance_values).all():
        raise ValueError("RNA training variances contain non-finite values")
    order = np.lexsort((feature_ids, -variance_values))
    selected_positions = order[:top_n]
    selected_ids = feature_ids[selected_positions]
    selected_variance = variance_values[selected_positions].astype(np.float32)

    train = (
        expression.loc[selected_ids, train_samples]
        .T.to_numpy(dtype=np.float32, copy=True)
    )
    means = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = train.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    stds[~np.isfinite(stds) | (stds < 1e-6)] = 1.0
    observed = (
        expression.loc[selected_ids, samples]
        .T.to_numpy(dtype=np.float32, copy=True)
    )
    standardized = ((observed - means) / stds).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("Standardized RNA contains non-finite values")
    return selected_ids, selected_variance, means, stds, standardized



def master_align_rna(
    standardized: np.ndarray,
    master: pd.DataFrame,
    cohort: pd.DataFrame,
) -> np.ndarray:
    """Map observed RNA rows into the master cohort; missing RNA stays zero."""
    lookup = {
        patient: row
        for row, patient in enumerate(master["patient"].astype(str).tolist())
    }
    patients = cohort["patient"].astype(str).tolist()
    if len(set(patients)) != len(patients):
        raise ValueError("RNA cohort contains duplicate patients")
    missing = sorted(set(patients) - set(lookup))
    if missing:
        raise ValueError(f"RNA cohort has {len(missing)} patients outside master")
    positions = np.asarray([lookup[patient] for patient in patients], dtype=int)
    output = np.zeros((len(master), standardized.shape[1]), dtype=np.float32)
    output[positions] = standardized
    expected = master["has_rna"].to_numpy(dtype=bool)
    observed = np.zeros(len(master), dtype=bool)
    observed[positions] = True
    if not np.array_equal(expected, observed):
        raise ValueError("RNA cohort disagrees with master has_rna indicator")
    return output



def prepare(cohort, rna_cohort, expression, source, fold, top_n=2000):
    cohort = validate_cohort(cohort)
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("RNA gene and sample IDs must be unique")
    if cohort.patient.duplicated().any() or not np.array_equal(source["patient"].astype(str), cohort.patient):
        raise ValueError("Source target patient order disagrees with the shared cohort")
    if not np.array_equal(source["modalities"].astype(str), MODALITIES):
        raise ValueError("Source target modality order differs")
    if not np.array_equal(source["outer_test"] if "outer_test" in source else source["partition"] == "test", cohort.fold.eq(fold)):
        raise ValueError("Source target outer test fold differs")
    if rna_cohort.patient.duplicated().any():
        raise ValueError("RNA patients must be unique")
    reference = cohort.set_index("patient").loc[rna_cohort.patient]
    if not np.array_equal(rna_cohort.fold, reference.fold):
        raise ValueError("RNA patient folds disagree with the shared cohort")
    selected, variance, means, stds, values = select_standard_rna(expression, rna_cohort, fold, top_n)
    payload = dict(source)
    scales = source["target_scale"].astype(str).tolist()
    scales[0] = RNA_SCALE
    payload.update(rna_target=master_align_rna(values, cohort, rna_cohort), rna_feature_ids=np.asarray(selected, dtype=str),
                   rna_train_variance=variance, rna_train_mean=means, rna_train_std=stds,
                   patient_fold=cohort.fold.to_numpy(int), outer_test=cohort.fold.eq(fold).to_numpy(),
                   target_scale=np.asarray(scales))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="Shared cohort.csv")
    parser.add_argument("--rna-cohort", type=Path, required=True, help="RNA-observed patients: patient,sample,fold")
    parser.add_argument("--expression", type=Path, required=True, help="Original logged Xena gene-by-sample TSV")
    parser.add_argument("--targets", type=Path, required=True, help="Prepared five-modal targets.npz")
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--top-n", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.targets, allow_pickle=False) as saved:
        source = dict(saved)
    cohort = pd.read_csv(args.cohort, dtype={"patient": str})
    rna = pd.read_csv(args.rna_cohort, dtype={"patient": str, "sample": str})
    expression = load_expression(args.expression, rna["sample"].astype(str).tolist())
    payload = prepare(cohort, rna, expression, source, args.fold, args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)


if __name__ == "__main__":
    main()
