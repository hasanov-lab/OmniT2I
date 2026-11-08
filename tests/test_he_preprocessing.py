import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from omnit2i.preprocessing import he_case_embeddings, he_extra_patches, he_multislide, he_patches, he_slide_embeddings


def test_tissue_mask_and_sampling():
    image = Image.new("RGB", (32, 32), (180, 80, 120))
    mask = he_patches.tissue_mask_from_rgb(image)
    assert mask.all()
    assert not he_patches.tissue_mask_from_rgb(Image.new("RGB", (32, 32), "white")).any()
    kwargs = dict(mask=mask, slide_width=1024, slide_height=1024, patch_size=128,
                  n_patches=8, min_tissue_fraction=0.45, max_attempts=800)
    first = he_patches.sample_locations(**kwargs, rng=np.random.default_rng(123))
    second = he_patches.sample_locations(**kwargs, rng=np.random.default_rng(123))
    assert first == second and len(first) == 8
    assert len({(x // 128, y // 128) for x, y, _ in first}) == 8
    assert all(0 <= x <= 896 and 0 <= y <= 896 for x, y, _ in first)
    assert he_extra_patches.sample_locations(**kwargs, rng=np.random.default_rng(123)) == first


def test_multislide_equal_weight_not_patch_weight(tmp_path, monkeypatch):
    columns = [f"he_emb_{i:04d}" for i in range(1536)]
    baseline = pd.DataFrame(np.array([[1] * 1536, [7] * 1536]), columns=columns)
    baseline.insert(0, "case_id", ["A", "B"])
    baseline["n_patches"] = [64, 50]
    extra = pd.DataFrame([[3] * 1536], columns=columns)
    extra["case_id"], extra["file_id"], extra["file_name"], extra["n_patches"] = "A", "extra", "extra.svs", 2
    cohort = pd.DataFrame({"patient": ["B", "A"]})
    target = pd.DataFrame({"case_id": ["A", "A", "B"]})
    paths = []
    for name, frame, sep in [("baseline.tsv", baseline, "\t"), ("extra.tsv", extra, "\t"),
                             ("cohort.csv", cohort, ","), ("target.tsv", target, "\t")]:
        path = tmp_path / name
        frame.to_csv(path, sep=sep, index=False)
        paths.append(str(path))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["omnit2i-pool-he", "--baseline-case-embeddings", paths[0],
        "--extra-slide-embeddings", paths[1], "--cohort", paths[2], "--target-table", paths[3],
        "--output-dir", str(out)])
    he_multislide.main()
    result = pd.read_csv(out / "multislide_patient_mean_embeddings.tsv", sep="\t")
    assert result.case_id.tolist() == ["A", "B"]
    np.testing.assert_allclose(result[columns], [[2] * 1536, [7] * 1536])
    assert result.n_patches.tolist() == [66, 50]
    assert result.n_slides.tolist() == [2, 1]


@pytest.mark.parametrize("extra", [False, True])
def test_embedding_pipeline_with_synthetic_encoder(tmp_path, monkeypatch, extra):
    class Encoder(torch.nn.Module):
        def forward(self, images):
            return {"features": images.mean((2, 3))}

    def transform(image):
        return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)

    module = he_slide_embeddings if extra else he_case_embeddings
    monkeypatch.setattr(module, "build_transform", lambda: transform)
    if extra:
        monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=lambda *a, **k: Encoder()))
    else:
        monkeypatch.setattr(module, "load_gigapath_tile_encoder", lambda device: Encoder())
    rows = []
    for i, (case, value, tissue) in enumerate([("B", 50, 1), ("A", 10, 1), ("A", 30, 1), ("A", 200, .1)]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (8, 8), (value, value, value)).save(path)
        rows.append(dict(case_id=case, file_id=case, file_name=f"{case}.svs", patch_index=i,
                         patch_path=str(path), status="ok", patch_tissue_fraction=tissue))
    path = tmp_path / "patches.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    out = tmp_path / "embeddings"
    monkeypatch.setattr(sys, "argv", ["embed", "--patch-table", str(path),
        "--output-dir" if extra else "--outdir", str(out), "--device", "cpu", "--num-workers", "0", "--batch-size", "2"])
    module.main()
    name = "extra_slide_mean_embeddings.tsv" if extra else "full_he_gigapath_64patch_mean_case_mean_embeddings.tsv"
    result = pd.read_csv(out / name, sep="\t")
    assert result.case_id.tolist() == ["A", "B"]
    assert result.n_patches.tolist() == [2, 1]
    np.testing.assert_allclose(result[["he_emb_0000", "he_emb_0001", "he_emb_0002"]], [[20]*3, [50]*3])


def test_patch_bag_command_on_synthetic_tiff(tmp_path, monkeypatch):
    pytest.importorskip("tiffslide")
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "synthetic.svs"
    tifffile.imwrite(path, np.full((1024, 1024, 3), [180, 80, 120], dtype=np.uint8),
                     photometric="rgb", tile=(128, 128))
    table = tmp_path / "slides.tsv"
    pd.DataFrame({"case_id": ["A"], "file_name": [path.name]}).to_csv(table, sep="\t", index=False)
    out = tmp_path / "patches"
    monkeypatch.setattr(sys, "argv", ["sample", "--table", str(table), "--svs-root", str(tmp_path),
        "--outdir", str(out), "--report-dir", str(tmp_path), "--patches-per-case", "4"])
    he_patches.main()
    patches = pd.read_csv(out / "full_he_64patch_bag_patch_table.tsv", sep="\t")
    assert len(patches) == 4 and patches.status.eq("ok").all()
    assert Image.open(patches.patch_path.iloc[0]).size == (128, 128)
