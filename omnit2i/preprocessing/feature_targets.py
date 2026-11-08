"""Align prepared molecular feature targets and standardized H&E by patient ID."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from omnit2i.data import validate_cohort
from omnit2i.models import MODALITIES
from omnit2i.training.multimodal_data import standardize_he_embeddings


SCALES = (
    "invariant_corrected_expression_z", "gistic2_gene_cnv_z",
    "median_imputed_rppa_z", "median_imputed_promoter_gene_m_value_z",
    "gigapath_patient_embedding_z",
)


def prepare_targets(cohort, paths):
    cohort = validate_cohort(cohort)
    if len(paths) != 5:
        raise ValueError("Supply RNA, CNV, protein, methylation and H&E in that order")
    payload = {key: cohort[key].to_numpy(str) for key in ("patient", "cancer_type", "partition")}
    payload.update(modalities=np.asarray(MODALITIES), target_scale=np.asarray(SCALES))
    availability = np.zeros((len(cohort), 5), dtype=bool)
    for index, (name, path) in enumerate(zip(MODALITIES, paths)):
        with np.load(path, allow_pickle=False) as saved:
            patients = saved["patient"].astype(str)
            if patients.ndim != 1 or len(np.unique(patients)) != len(patients) or any(not p.strip() for p in patients):
                raise ValueError(f"{name} patient IDs must be nonempty and unique")
            positions = pd.Index(cohort.patient).get_indexer(patients)
            if (positions < 0).any():
                raise ValueError(f"{name} contains unknown patients")
            for field in ("partition", "cancer_type", "time", "event"):
                if name != "he" and field not in saved:
                    raise ValueError(f"{name} features must record {field}")
                if field in saved and not np.array_equal(saved[field], cohort.iloc[positions][field].to_numpy()):
                    raise ValueError(f"{name} {field} disagrees with the master cohort")
            values = np.asarray(saved["embeddings" if name == "he" else "features"], dtype=np.float32)
            features = np.asarray([f"gigapath_{i:04d}" for i in range(1536)]) if name == "he" else saved["feature_ids"].astype(str)
            if features.ndim != 1 or not len(features) or len(np.unique(features)) != len(features):
                raise ValueError(f"{name} feature IDs must be nonempty and unique")
            if values.shape != (len(patients), len(features)) or not np.isfinite(values).all():
                raise ValueError(f"{name} targets must be finite and match patients and features")
            if name == "he":
                outer_training = cohort.iloc[positions].partition.ne("test").to_numpy()
                if not outer_training.any():
                    raise ValueError("H&E requires observed outer-training patients")
                values, parameters = standardize_he_embeddings(values, outer_training)
                payload.update(he_train_mean=parameters["mean"], he_train_std=parameters["std"])
            target = np.zeros((len(cohort), len(features)), dtype=np.float32)
            target[positions] = values
            payload[f"{name}_target"], payload[f"{name}_feature_ids"] = target, features
            availability[positions, index] = True
    flags = ["has_rna", "has_cnv", "has_rppa", "has_methylation", "has_he"]
    for index, flag in enumerate(flags):
        if flag in cohort and not np.array_equal(cohort[flag], availability[:, index]):
            raise ValueError(f"Target membership disagrees with master {flag}")
    if not availability.any(axis=1).all():
        raise ValueError("Every patient needs at least one observed modality")
    payload["availability"] = availability
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    for name in MODALITIES:
        parser.add_argument(f"--{name}", type=Path, required=True, help="Molecular features.npz or frozen H&E embeddings.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    payload = prepare_targets(pd.read_csv(args.cohort, dtype={"patient": str, "sample": str}), [getattr(args, name) for name in MODALITIES])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Prepared {len(payload['patient'])} patients: {args.output}")


if __name__ == "__main__":
    main()
