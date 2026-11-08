"""Original four RNA-start paired patient-bootstrap intervals, using saved risks."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from omnit2i.analysis import paired_bootstrap as bootstrap
from measured_gains import canonical, transition_tables, MODALITIES, SEEDS, METRICS

KEYS = ["patient", "cancer_type", "time", "event", "fold"]
ADDITIONS = ["cnv", "protein", "methylation", "he"]

def aligned_risk(frame, cohort, column):
    assert frame.patient.is_unique and len(frame) == len(cohort)
    frame = frame.sort_values("patient").reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[KEYS], cohort, check_dtype=False, check_exact=True)
    risk = frame[column].to_numpy(float)
    assert np.isfinite(risk).all()
    return risk

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=Path, required=True)
    parser.add_argument("--per-seed", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    saved = pd.read_csv(args.per_seed, sep="\t")
    per_seed, summary = transition_tables(saved)
    per_seed.to_csv(out / "conditional_gains_per_seed.tsv", sep="\t", index=False)
    summary.to_csv(out / "conditional_gains_75.tsv", sep="\t", index=False)
    cohort = pd.read_csv(args.patients, sep="\t")
    assert len(cohort) == 5605 and cohort.patient.is_unique and cohort.event.sum() == 1546
    assert cohort.patient.tolist() == sorted(cohort.patient.tolist())
    subsets = ["rna"] + [canonical("rna+" + added) for added in ADDITIONS]
    risks = {subset: [] for subset in subsets}
    for seed in SEEDS:
        chunks = []
        for fold in range(5):
            path = args.predictions_root / f"seed_{seed}/fold{fold}/predictions.tsv.gz"
            frame = pd.read_csv(path, sep="\t", usecols=KEYS + ["seed", "source_subset", "risk_source_only"])
            assert frame.seed.eq(seed).all() and frame.fold.eq(fold).all()
            frame["source_subset"] = frame.source_subset.map(canonical)
            chunks.append(frame.loc[frame.source_subset.isin(subsets)])
        frame = pd.concat(chunks, ignore_index=True)
        assert set(frame.source_subset) == set(subsets)
        for subset in subsets:
            risks[subset].append(aligned_risk(frame.loc[frame.source_subset.eq(subset)], cohort, "risk_source_only"))
        print(f"Loaded five measured-only subsets, seed {seed}", flush=True)

    cancer = cohort.cancer_type.to_numpy()
    left, right = bootstrap.comparable_pairs(cohort.time.to_numpy(), cohort.event.to_numpy(bool), np.zeros(len(cohort)))
    within = cancer[left] == cancer[right]
    assert len(left) == 5096363 and within.sum() == 219280
    scores = np.empty((len(left), len(subsets)))
    saved["source_subset"] = saved.source_subset.map(canonical)
    lookup = saved.set_index(["seed", "source_subset"])
    replay = []
    for index, subset in enumerate(subsets):
        # Here each model axis is one seed; the singleton seed axis preserves its C.
        risk = np.array(risks[subset])[:, None, :]
        seed_scores = np.empty((len(left), len(SEEDS)))
        for start in range(0, len(left), 10000):
            stop = start + 10000
            seed_scores[start:stop] = bootstrap.pair_scores(risk, left[start:stop], right[start:stop])
        for metric, selection in [(METRICS[0], within), (METRICS[1], slice(None))]:
            observed = seed_scores[selection].mean(axis=0)
            expected = np.array([lookup.loc[(seed, subset), metric] for seed in SEEDS])
            np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0)
            for j, seed in enumerate(SEEDS):
                replay.append(dict(source_subset=subset, seed=seed, metric=metric, calculated=float(observed[j]),
                    saved=float(expected[j]), absolute_difference=float(abs(observed[j] - expected[j]))))
        scores[:, index] = seed_scores.mean(axis=1)
        print(f"Replayed both C metrics for all five seeds: {subset}", flush=True)
    del seed_scores
    pd.DataFrame(replay).to_csv(out / "point_estimate_replay.tsv", sep="\t", index=False)
    differences = np.ascontiguousarray(scores[:, 1:] - scores[:, :1])
    del scores
    rng = np.random.default_rng(20261019)
    weights = np.zeros((2000, len(cohort)), dtype=np.uint16)
    for label in np.unique(cancer):
        ids = np.flatnonzero(cancer == label)
        weights[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=len(weights))
    rna = summary.loc[summary.start_subset.eq("rna")].set_index("added_modality")
    intervals = []
    for metric, selection in [(METRICS[0], within), (METRICS[1], slice(None))]:
        pair_differences = np.ascontiguousarray(differences[selection])
        point = pair_differences.mean(axis=0)
        expected = rna.loc[ADDITIONS, f"{metric}_delta_mean"].to_numpy()
        np.testing.assert_allclose(point, expected, atol=1e-12, rtol=0)
        draws = bootstrap.weighted_c(weights, left[selection], right[selection], pair_differences)
        assert np.isfinite(draws).all(), "Undefined bootstrap draw: do not silently discard"
        pd.DataFrame(draws, columns=ADDITIONS).to_csv(out / f"{metric}_draws.tsv.gz", sep="\t", index=False)
        for i, added in enumerate(ADDITIONS):
            intervals.append(dict(start_subset="rna", added_modality=added, target_subset=subsets[i + 1], metric=metric,
                patients=len(cohort), events=int(cohort.event.sum()), seeds=5, bootstrap_draws=2000,
                c_before_mean=float(rna.loc[added, f"{metric}_before_mean"]),
                c_after_mean=float(rna.loc[added, f"{metric}_after_mean"]), delta_c=float(point[i]),
                positive_seeds=int(rna.loc[added, f"{metric}_positive_seeds"]),
                ci_lower=float(np.quantile(draws[:, i], .025)), ci_upper=float(np.quantile(draws[:, i], .975))))
        print(f"Finished RNA-start paired intervals: {metric}", flush=True)
    ci = pd.DataFrame(intervals)
    ci.to_csv(out / "rna_addition_ci.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
