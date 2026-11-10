"""Frozen RNA-only external inference with saved TCGA preprocessing."""

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored

from omnit2i.models import OmicsHEEndToEndCox
from omnit2i.preprocessing.cnv_t2i import transform_gene_standardization
from omnit2i.preprocessing.rna_t2i import apply_cancer_standardization, render_option2, sample_gamma
from omnit2i.training.crossmodal import fused_risk_from_projected


SEEDS = (101, 202, 303, 404, 505)


def scanb_clinical(paths):
    frames = []
    for path in paths:
        fields = {}
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if not row or not row[0].startswith("!Sample_"):
                    continue
                if row[0] == "!Sample_characteristics_ch1":
                    keys = {value.split(": ", 1)[0] for value in row[1:]}
                    if len(keys) != 1:
                        raise ValueError("Inconsistent SCAN-B characteristic labels")
                    fields[keys.pop()] = [value.split(": ", 1)[1] for value in row[1:]]
                else:
                    fields[row[0]] = row[1:]
        frames.append(pd.DataFrame(fields))
    frame = pd.concat(frames, ignore_index=True)
    frame["patient"] = frame["scan-b external id"].str.extract(r"\.(C\d+)\.")[0]
    if frame.patient.isna().any() or not frame["!Sample_title"].is_unique:
        raise ValueError("Missing patient IDs or duplicate SCAN-B sample titles")
    for column in ("overall survival days", "overall survival event"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if frame.groupby("patient")[column].nunique(dropna=False).gt(1).any():
            raise ValueError("Discordant SCAN-B replicate survival outcomes")
    selected = frame.loc[~frame["!Sample_title"].str.endswith("repl")].copy()
    if not selected.patient.is_unique or set(selected.patient) != set(frame.patient):
        raise ValueError("Require exactly one original SCAN-B sample per patient")
    selected = selected.rename(columns={"!Sample_title": "sample", "overall survival days": "time", "overall survival event": "event"})
    selected["cancer_type"], selected["cohort"] = "BRCA", "SCAN-B"
    selected = selected[["patient", "sample", "cancer_type", "cohort", "time", "event"]]
    validate_clinical(selected)
    return selected


def clinical_main():
    parser = argparse.ArgumentParser(description="Prepare original SCAN-B patients from GEO series matrices")
    parser.add_argument("--series-matrix", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = scanb_clinical(args.series_matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        frame.to_csv(handle, index=False)


def validate_clinical(clinical):
    required = {"patient", "sample", "cancer_type", "time", "event"}
    if required - set(clinical) or clinical[list(required)].isna().any().any() or clinical.empty:
        raise ValueError("External clinical table is empty or missing required fields")
    if clinical.patient.duplicated().any() or clinical["sample"].duplicated().any():
        raise ValueError("External patients and samples must be unique; exclude SCAN-B replicate columns")
    if not clinical.event.isin([0, 1]).all() or not np.isfinite(clinical.time).all() or not clinical.time.gt(0).all():
        raise ValueError("External OS requires positive finite days and binary events")


def map_expression(expression, mapping, saved):
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("External gene and sample IDs must be unique")
    required = list(dict.fromkeys([*saved["feature_ids"].astype(str), *saved["invariant_genes"].astype(str)]))
    if mapping is None:
        return expression.loc[expression.index.intersection(required)].copy()
    needed = {"saved_gene_symbol", "source_gene_id", "match_status"}
    if needed - set(mapping) or mapping.saved_gene_symbol.duplicated().any():
        raise ValueError("Mapping requires unique saved_gene_symbol, source_gene_id and match_status")
    indexed = mapping.set_index("saved_gene_symbol")
    if not set(required).issubset(indexed.index):
        raise ValueError("Mapping must document every selected and invariant gene, including unresolved genes")
    accepted = indexed.loc[required]
    accepted = accepted.loc[accepted.match_status.isin(["exact", "safe-renaming"])]
    if not accepted.source_gene_id.is_unique or not accepted.source_gene_id.isin(expression.index).all():
        raise ValueError("Accepted mappings must be one-to-one and present in the expression matrix")
    mapped = expression.loc[accepted.source_gene_id].copy()
    mapped.index = accepted.index
    return mapped


def transform_external(mapped, saved, cancers):
    genes = saved["feature_ids"].astype(str)
    invariant = saved["invariant_genes"].astype(str)
    patients = mapped.columns.astype(str).tolist()
    if not np.isfinite(mapped.to_numpy()).all():
        raise ValueError("Observed external RNA expression must be finite")
    gamma = sample_gamma(mapped, patients, invariant)
    measured = np.isin(genes, mapped.index)
    corrected = mapped.reindex(genes).sub(gamma, axis=1).T.to_numpy(np.float32)
    if not np.array_equal(np.isnan(corrected), np.broadcast_to(~measured, corrected.shape)):
        raise ValueError("Only unmapped genes may be missing")
    # Fill on the saved gamma-corrected scale, preserving fixed gene positions.
    corrected[:, ~measured] = saved["train_means"][~measured]
    standardized = transform_gene_standardization(corrected, saved["train_means"], saved["train_stds"])
    cancer = apply_cancer_standardization(standardized, pd.Series(cancers), saved["cancer_labels"],
        saved["cancer_means"], saved["cancer_stds"], saved["global_cancer_mean"], saved["global_cancer_std"])
    images = render_option2(standardized, cancer, saved["cox_score_z"], saved["layout_order"],
        int(saved.get("grid_height", 45)), int(saved.get("grid_width", 45)))
    if not np.isfinite(images).all():
        raise ValueError("Frozen external transform produced nonfinite values")
    audit = dict(measured_input_genes=int(measured.sum()), mean_filled_input_genes=int((~measured).sum()),
                 available_invariant_genes=int(np.isin(invariant, mapped.index).sum()), gamma_median=float(np.median(gamma)))
    return images, audit


def scanb_expression(path, samples, needed):
    """Invert deposited log2(FPKM + .1), using all genes in the TPM denominator."""
    totals = np.zeros(len(samples), dtype=np.float64)
    selected, gene_count = [], 0
    observed_genes = set()
    for chunk in pd.read_csv(path, index_col=0, chunksize=512):
        if not chunk.index.is_unique or observed_genes.intersection(chunk.index):
            raise ValueError("SCAN-B gene IDs must be unique")
        observed_genes.update(chunk.index)
        values = chunk.loc[:, samples].to_numpy(np.float64)
        raw = np.exp2(values) - .1
        if not np.isfinite(raw).all() or raw.min() < -1e-7:
            raise ValueError("Invalid deposited SCAN-B expression scale")
        raw = np.maximum(raw, 0.)
        totals += raw.sum(axis=0)
        keep = chunk.index.isin(needed)
        if keep.any():
            selected.append(pd.DataFrame(raw[keep], index=chunk.index[keep], columns=samples))
        gene_count += len(chunk)
    if not selected or not np.all(totals > 0):
        raise ValueError("SCAN-B needs mapped genes and positive all-gene FPKM sums")
    expression = np.log2(pd.concat(selected).div(totals, axis=1) * 1e6 + 1).astype(np.float32)
    return expression, gene_count


@torch.inference_mode()
def predict_rna(model, images):
    model.eval()
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(images), 32):
        batch = images[start:start + 32]
        size = len(batch)
        if size < 32:
            batch = np.concatenate([batch, np.repeat(batch[-1:], 32 - size, axis=0)])
        embedding = model.encoders[0](torch.from_numpy(batch).to(device))
        rna = model.fusion.shared_projection(embedding)
        projected = torch.zeros((32, 5, rna.shape[1]), device=device)
        projected[:, 0] = rna
        mask = torch.zeros((32, 5), dtype=torch.bool, device=device)
        mask[:, 0] = True
        results.append(fused_risk_from_projected(model.fusion, projected, mask)[0].cpu().numpy()[:size])
    result = np.concatenate(results).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite frozen RNA risks")
    return result


def c_index(frame):
    try:
        return float(concordance_index_censored(frame.event.to_numpy(bool), frame.time.to_numpy(float), frame.risk.to_numpy(float))[0])
    except (NoComparablePairException, ValueError):
        # Bootstrap draws can contain no events or no comparable pairs.
        if frame.event.any() and any((frame.time > time).any() for time in frame.loc[frame.event.astype(bool), "time"]):
            raise
        return float("nan")


def bootstrap(frame, repetitions=2000):
    estimate = c_index(frame)
    rng = np.random.default_rng(101)
    values = []
    for _ in range(repetitions):
        value = c_index(frame.iloc[rng.integers(0, len(frame), len(frame))])
        if np.isfinite(value):
            values.append(value)
    if not values or not np.isfinite(estimate):
        raise ValueError("External survival estimate is unevaluable")
    lo, hi = np.quantile(values, [.025, .975])
    return dict(c_index=estimate, ci_low=lo, ci_high=hi, bootstrap_attempted=repetitions,
                bootstrap_valid=len(values), bootstrap_unevaluable=repetitions - len(values))


def summarize_members(root, output):
    if output.exists():
        raise FileExistsError(output)
    parts, reference = [], None
    for seed in SEEDS:
        for fold in range(5):
            frame = pd.read_csv(root / f"seed_{seed}/fold{fold}/predictions.csv", dtype={"patient": str, "sample": str})
            validate_clinical(frame)
            if frame.empty or frame.patient.duplicated().any() or not frame.seed.eq(seed).all() or not frame.fold.eq(fold).all():
                raise ValueError("Incorrect frozen member identifiers or duplicate patients")
            if not np.isfinite(frame[["risk_raw", "risk"]]).all().all():
                raise ValueError("Nonfinite frozen member predictions")
            clinical = frame.drop(columns=["seed", "fold", "risk_raw", "risk"]).set_index("patient").sort_index()
            if reference is None:
                reference = clinical
            pd.testing.assert_frame_equal(clinical, reference)
            parts.append(frame)
    members = pd.concat(parts, ignore_index=True)
    keys = [column for column in members if column not in {"seed", "fold", "risk_raw", "risk"}]
    averaged = members.groupby(keys, as_index=False, dropna=False).risk.mean()
    seeds = members.groupby(["seed", *keys], as_index=False, dropna=False).risk.mean()
    populations = {"primary": averaged}
    if "endpoint_flag" in averaged:
        if not averaged.endpoint_flag.isin([True, False]).all():
            raise ValueError("endpoint_flag must be boolean")
        populations["exclude_endpoint_flags"] = averaged.loc[~averaged.endpoint_flag.astype(bool)]
    if {"rna_timing_complete", "rna_any_after_origin"}.issubset(averaged):
        if not averaged[["rna_timing_complete", "rna_any_after_origin"]].isin([True, False]).all().all():
            raise ValueError("RNA timing flags must be boolean")
        timing = averaged.rna_timing_complete.astype(bool) & ~averaged.rna_any_after_origin.astype(bool)
        populations["timing_restricted"] = averaged.loc[timing]
        if "endpoint_flag" in averaged:
            populations["timing_and_endpoint_restricted"] = averaged.loc[timing & ~averaged.endpoint_flag.astype(bool)]
    summaries = [dict(analysis=name, patients=len(frame), deaths=int(frame.event.sum()), **bootstrap(frame)) for name, frame in populations.items()]
    seed_metrics = [dict(seed=seed, patients=len(frame), deaths=int(frame.event.sum()), c_index=c_index(frame)) for seed, frame in seeds.groupby("seed")]
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (("patient_predictions", averaged), ("seed_predictions", seeds),
                         ("summary", pd.DataFrame(summaries)), ("seed_metrics", pd.DataFrame(seed_metrics))):
        frame.to_csv(output / f"{name}.csv", index=False)


def summary_main():
    parser = argparse.ArgumentParser(description="Average the 25 frozen RNA members and calculate 2,000 patient-bootstrap intervals")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize_members(args.input_root, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path, required=True, help="Logged gene x sample TSV; SCAN-B deposited CSV with --scanb")
    parser.add_argument("--clinical", type=Path, required=True, help="CSV: patient,sample,cancer_type,time,event")
    parser.add_argument("--mapping", type=Path, help="CSV: saved_gene_symbol,source_gene_id,match_status; omit for exact symbol matches")
    parser.add_argument("--scanb", action="store_true")
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Released Stage C model.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    clinical = pd.read_csv(args.clinical, dtype={"patient": str, "sample": str})
    validate_clinical(clinical)
    with np.load(args.preprocessing, allow_pickle=False) as archive:
        saved = dict(archive)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    mapping = pd.read_csv(args.mapping, keep_default_na=False) if args.mapping else None
    if mapping is not None:
        for field in ("seed", "fold"):
            if field in mapping:
                mapping = mapping.loc[mapping[field].eq(checkpoint[field])]
    needed = np.union1d(saved["feature_ids"], saved["invariant_genes"])
    if mapping is not None:
        needed = mapping.loc[mapping.match_status.isin(["exact", "safe-renaming"]), "source_gene_id"]
    if args.scanb:
        expression, denominator_genes = scanb_expression(args.expression, clinical["sample"].tolist(), needed)
    else:
        expression = pd.read_csv(args.expression, sep="\t", index_col=0).loc[:, clinical["sample"]].astype(np.float32)
        denominator_genes = None
    mapped = map_expression(expression, mapping, saved)
    images, audit = transform_external(mapped, saved, clinical.cancer_type.to_numpy(str))
    model = OmicsHEEndToEndCox(**checkpoint["survival_model_kwargs"])
    model.load_state_dict({key.removeprefix("survival."): value for key, value in checkpoint["model_state"].items() if key.startswith("survival.")}, strict=True)
    model.to(args.device).eval().requires_grad_(False)
    calibration = checkpoint["risk_calibration"]
    if not np.isfinite([calibration["mean"], calibration["std"]]).all() or calibration["std"] <= 0:
        raise ValueError("Invalid saved TCGA risk calibration")
    clinical["risk_raw"] = predict_rna(model, images)
    clinical["risk"] = (clinical.risk_raw - calibration["mean"]) / calibration["std"]
    clinical["seed"], clinical["fold"] = checkpoint["seed"], checkpoint["fold"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    clinical.to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame([dict(seed=checkpoint["seed"], fold=checkpoint["fold"], denominator_genes=denominator_genes, **audit)]).to_csv(args.output_dir / "input_coverage.csv", index=False)
    print(f"Frozen RNA inference: {len(clinical)} patients; {audit['measured_input_genes']} measured model genes")


if __name__ == "__main__":
    main()
