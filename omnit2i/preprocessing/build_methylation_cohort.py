"""Select the original primary-tumour 450K OS cohort before promoter aggregation."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .cohort_samples import read_matrix_header, build_primary_os_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("matrix", "survival", "phenotype", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    _, samples = read_matrix_header(args.matrix)
    cohort = build_primary_os_cohort(list(dict.fromkeys(samples)),
        pd.read_csv(args.survival, sep="\t", dtype=str),
        pd.read_csv(args.phenotype, sep="\t", dtype=str))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
