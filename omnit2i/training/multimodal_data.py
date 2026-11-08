"""Patient-ID alignment for prepared molecular images and frozen H&E embeddings."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from omnit2i.data import validate_cohort
from omnit2i.models import MODALITIES


@dataclass
class MultimodalData:
    cohort: pd.DataFrame
    inputs: list[np.ndarray]
    local_lookup: list[np.ndarray]
    master_positions: list[np.ndarray]
    availability: np.ndarray
    he_standardization: dict[str, np.ndarray | float]


def load_multimodal_data(
    cohort_path: Path,
    modality_paths: list[Path],
    *,
    he_clip: float = 8.0,
    he_standardization: dict | None = None,
) -> MultimodalData:
    cohort = validate_cohort(pd.read_csv(cohort_path, dtype={"patient": str, "sample": str}))
    if len(modality_paths) != len(MODALITIES):
        raise ValueError("Supply RNA, CNV, protein, methylation, and H&E archives in that order")
    if not np.isfinite(he_clip) or he_clip <= 0:
        raise ValueError("H&E standardization clip must be finite and positive")
    master_index = pd.Index(cohort["patient"])
    inputs, local_lookup, master_positions = [], [], []
    shapes = [(3, 45, 45), (3, 45, 45), (3, 17, 17), (3, 45, 45), (1536,)]
    for name, path, shape in zip(MODALITIES, modality_paths, shapes):
        with np.load(path, allow_pickle=False) as saved:
            patients = saved["patient"].astype(str)
            if patients.ndim != 1 or len(np.unique(patients)) != len(patients):
                raise ValueError(f"{name} patient IDs must be one-dimensional and unique")
            positions = master_index.get_indexer(patients)
            if (positions < 0).any():
                raise ValueError(f"{name} archive contains patients absent from the cohort")
            values = np.asarray(saved["embeddings" if name == "he" else "images"], dtype=np.float32)
            if values.shape != (len(patients), *shape) or not np.isfinite(values).all():
                raise ValueError(f"{name} inputs must be finite and have shape [patients, {shape}]")
            if name != "he" and "partition" not in saved:
                raise ValueError(f"{name} images must record the partition used for preprocessing")
            for column in ("partition", "time", "event", "cancer_type"):
                if column in saved and not np.array_equal(saved[column], cohort.iloc[positions][column].to_numpy()):
                    raise ValueError(f"{name} archive {column} disagrees with the master cohort")
        lookup = np.full(len(cohort), -1, dtype=np.int64)
        lookup[positions] = np.arange(len(patients))
        inputs.append(values)
        local_lookup.append(lookup)
        master_positions.append(positions)
    availability = np.stack([lookup >= 0 for lookup in local_lookup], axis=1)
    if not availability.any(axis=1).all():
        raise ValueError("Every cohort patient needs at least one observed modality")
    for partition in ("core", "validation"):
        represented = availability[cohort["partition"].eq(partition)].any(axis=0)
        if not represented.all():
            raise ValueError(f"The {partition} partition must include an observation of every modality")
    train = cohort.iloc[master_positions[4]]["partition"].ne("test").to_numpy()
    inputs[4], he_standardization = standardize_he_embeddings(inputs[4], train, he_clip, he_standardization)
    return MultimodalData(cohort, inputs, local_lookup, master_positions, availability, he_standardization)


def standardize_he_embeddings(values, outer_training, clip=8.0, parameters=None):
    """Share the manuscript outer-training H&E transform between inputs and targets."""
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("H&E standardization clip must be finite and positive")
    if parameters is None:
        mean = values[outer_training].mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values[outer_training].std(axis=0, dtype=np.float64).astype(np.float32)
        parameters = {"mean": mean, "std": np.where(std < 1e-6, 1.0, std).astype(np.float32), "clip": clip}
    mean, std = np.asarray(parameters["mean"]), np.asarray(parameters["std"])
    clip = float(parameters["clip"])
    if mean.shape != (1536,) or std.shape != (1536,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any() or not np.isfinite(clip) or clip <= 0:
        raise ValueError("Invalid saved H&E standardization parameters")
    return np.clip((values - mean) / std, -clip, clip).astype(np.float32), parameters
