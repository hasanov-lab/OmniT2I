"""Summarize the original 75 measured add-one transitions from saved seed tables."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

MODALITIES = ["rna", "cnv", "protein", "methylation", "he"]
SEEDS = [101, 202, 303, 404, 505]
METRICS = ["within_cancer_c_index", "pooled_c_index"]


def canonical(source):
    selected = set(source.split("+"))
    if not selected or selected - set(MODALITIES):
        raise ValueError(f"Unknown source subset: {source}")
    return "+".join(modality for modality in MODALITIES if modality in selected)


def transition_tables(frame):
    frame = frame.copy()
    frame["source_subset"] = frame.source_subset.map(canonical)
    subsets = ["+".join(subset) for size in range(1, 6) for subset in combinations(MODALITIES, size)]
    assert len(frame) == 155 and not frame.duplicated(["seed", "source_subset"]).any()
    assert set(frame.seed) == set(SEEDS)
    for _, group in frame.groupby("seed"):
        assert set(group.source_subset) == set(subsets)
    assert np.isfinite(frame[METRICS]).all().all()
    lookup = frame.set_index(["seed", "source_subset"])
    rows = []
    for start in subsets:
        for added in MODALITIES:
            if added in start.split("+"):
                continue
            target = canonical(start + "+" + added)
            for seed in SEEDS:
                before, after = lookup.loc[(seed, start)], lookup.loc[(seed, target)]
                assert before.n_patients == after.n_patients and before.n_events == after.n_events
                row = dict(start_subset=start, added_modality=added, target_subset=target,
                    start_count=len(start.split("+")), seed=seed,
                    patients=int(before.n_patients), events=int(before.n_events))
                for metric in METRICS:
                    row[f"{metric}_before"] = float(before[metric])
                    row[f"{metric}_after"] = float(after[metric])
                    row[f"{metric}_delta"] = float(after[metric] - before[metric])
                rows.append(row)
    per_seed = pd.DataFrame(rows)
    assert len(per_seed) == 375
    combined = []
    for (start, added, target), group in per_seed.groupby(["start_subset", "added_modality", "target_subset"], sort=False):
        assert len(group) == 5 and set(group.seed) == set(SEEDS)
        row = dict(start_subset=start, added_modality=added, target_subset=target,
            start_count=int(group.start_count.iloc[0]), patients=int(group.patients.iloc[0]),
            events=int(group.events.iloc[0]), seeds=5)
        for metric in METRICS:
            for suffix in ["before", "after", "delta"]:
                row[f"{metric}_{suffix}_mean"] = float(group[f"{metric}_{suffix}"].mean())
            row[f"{metric}_positive_seeds"] = int((group[f"{metric}_delta"] > 0).sum())
        combined.append(row)
    summary = pd.DataFrame(combined)
    assert len(summary) == 75 and summary.groupby("added_modality").size().eq(15).all()
    return per_seed, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    per_seed, summary = transition_tables(pd.read_csv(args.per_seed, sep="\t"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "conditional_gains_per_seed.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / "conditional_gains_75.tsv", sep="\t", index=False)
