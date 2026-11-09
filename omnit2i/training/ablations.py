"""The manuscript's input, component and joint-simplification experiments."""

import argparse
from pathlib import Path

import numpy as np

from omnit2i.models import MODALITIES
from omnit2i.preprocessing.cnv_t2i import snake_pixel_indices

VARIANTS = ("no_pretraining", "no_c2", "no_c3", "random_layout", "no_risk_consistency",
            "mean_fusion", "no_modality_dropout", "simplified", "mlp_matched", "mlp_wide")
LAYOUT_SEED = 20261014


def ablate_images(images, order, variant, fold, modality):
    if variant not in {"no_c2", "no_c3", "random_layout"} or fold not in range(5):
        raise ValueError("Unknown input ablation or outer fold")
    if images.ndim != 4 or images.shape[1] != 3 or not np.isfinite(images).all():
        raise ValueError("Input ablations require finite three-channel images")
    order = np.asarray(order)
    if order.ndim != 1 or not len(order) or not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError("Layout must permute the selected features")
    result, new_order = images.copy(), order.copy()
    if variant == "random_layout":
        rng = np.random.default_rng(np.random.SeedSequence([LAYOUT_SEED, fold, MODALITIES.index(modality)]))
        new_order = rng.permutation(len(order))
        pixels = snake_pixel_indices(*images.shape[2:])[:len(order)]
        source_positions = np.argsort(order)[new_order]
        result.reshape(len(result), 3, -1)[:, :, pixels] = images.reshape(len(images), 3, -1)[:, :, pixels[source_positions]]
    else:
        result[:, 1 if variant == "no_c2" else 2] = 0
    return result, new_order


def prepare_inputs(source, destination, variant, fold):
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ("cohort.csv", "he.npz", "targets.npz"):
        (destination / filename).symlink_to((source / filename).resolve(strict=True))
    for name in MODALITIES[:4]:
        with np.load(source / name / "images.npz", allow_pickle=False) as saved:
            payload = dict(saved)
        with np.load(source / name / "preprocessing.npz", allow_pickle=False) as saved:
            parameters = dict(saved)
        if not np.array_equal(payload["feature_ids"], parameters["feature_ids"]):
            raise ValueError(f"{name} image features disagree with saved preprocessing")
        payload["images"], parameters["layout_order"] = ablate_images(
            payload["images"], parameters["layout_order"], variant, fold, name)
        folder = destination / name
        folder.mkdir()
        np.savez_compressed(folder / "images.npz", **payload)
        np.savez_compressed(folder / "preprocessing.npz", **parameters)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, help="Main trained fold; mean-fusion trained fold for simplified")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(101, 202, 303, 404, 505), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from .workflow import train_fold
    train_fold(args.prepared_dir, args.output_dir, args.seed, args.fold, args.device,
               variant=args.variant, reference=args.reference_dir)


if __name__ == "__main__":
    main()
