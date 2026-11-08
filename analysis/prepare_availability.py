"""Prepare the final Figure 2 aggregate tables from the shared master cohort."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    master = pd.read_csv(args.master)
    columns = ["has_rna", "has_cnv", "has_rppa", "has_methylation", "has_he"]
    names = ["RNA", "CNV", "Protein", "Meth", "H&E"]
    modalities = ["rna", "cnv", "rppa", "methylation", "he"]
    assert master.patient.is_unique and master[columns].isin([True, False]).all().all()
    mask = master[columns].astype(bool)
    assert mask.any(axis=1).all()
    n = len(master)
    coverage = pd.DataFrame({"modality": names, "n": mask.sum().to_numpy()})
    coverage["percent"] = coverage.n / n * 100
    counts = mask.sum(axis=1).value_counts().sort_index().rename_axis("observed_modalities").reset_index(name="n")
    counts["percent"] = counts.n / n * 100
    availability = mask.apply(lambda row: "+".join(m for m, value in zip(modalities, row) if value), axis=1)
    combinations = availability.value_counts().rename_axis("availability").reset_index(name="patients")
    combinations = combinations.sort_values(["patients", "availability"], ascending=[False, True]).reset_index(drop=True)
    combinations.insert(0, "combination_id", range(1, len(combinations) + 1))
    combinations["percent"] = combinations.patients / n * 100
    for label, modality in zip(names, modalities):
        combinations[label] = combinations.availability.str.split("+").map(lambda parts: int(modality in parts))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in {"coverage": coverage, "modality_counts": counts, "Fig2c_source_data": combinations}.items():
        table.to_csv(args.output_dir / (name + ".tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
