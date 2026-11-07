import csv
import random

import cv2
import numpy as np
import pytest

from data.dataset import PolypSegDataset, random_dihedral_augment
from data.prepare_video_dataset import apply_frame_stride, build_manifest, split_case_ids, write_manifest_csv


def write_image(path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(40, 180, size=(120, 160, 3), dtype=np.uint8))


@pytest.mark.parametrize("normalize_color", [True, False])
def test_dataset_returns_normalized_image_and_binary_mask(tmp_path, normalize_color):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    write_image(images / "a.png", seed=0)
    mask = np.zeros((120, 160), dtype=np.uint8)
    mask[30:90, 40:120] = 255
    cv2.imwrite(str(masks / "a.png"), mask)

    image_tensor, mask_tensor = PolypSegDataset(images, masks, normalize_color=normalize_color)[0]

    assert image_tensor.shape == (3, 352, 352)
    assert 0.0 <= image_tensor.min() and image_tensor.max() <= 1.0
    assert mask_tensor.shape == (1, 352, 352)
    assert set(mask_tensor.unique().tolist()) == {0.0, 1.0}


def test_augmentation_transforms_image_and_mask_identically():
    random.seed(0)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[1:5, 2:12] = 255
    image = np.stack([mask] * 3, axis=-1)
    for _ in range(50):
        augmented_image, augmented_mask = random_dihedral_augment(image, mask)
        assert np.array_equal(augmented_image[..., 0], augmented_mask)
        assert augmented_mask.sum() == mask.sum()


def test_case_split_is_deterministic_and_disjoint():
    case_ids = [f"Positive/case{i}" for i in range(20)]
    train, test = split_case_ids(case_ids, ratio=0.8, seed=42)
    assert (train, test) == split_case_ids(list(reversed(case_ids)), ratio=0.8, seed=42)
    assert (len(train), len(test)) == (16, 4)
    assert not set(train) & set(test)


def test_frame_stride_keeps_every_nth_frame():
    assert apply_frame_stride(list(range(10)), 4) == [0, 4, 8]
    with pytest.raises(ValueError):
        apply_frame_stride([1], 0)


def test_manifest_keeps_cases_together_and_paths_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for class_name, n_cases in [("Positive", 5), ("negative", 3)]:
        for case in range(n_cases):
            case_dir = tmp_path / "datasets" / class_name / f"case{case}"
            case_dir.mkdir(parents=True)
            for frame in range(3):
                write_image(case_dir / f"image{frame}.jpg", seed=frame)

    rows = build_manifest(video_root=tmp_path.joinpath("datasets").relative_to(tmp_path))

    splits_by_case = {}
    for row in rows:
        splits_by_case.setdefault(row["case_id"], set()).add(row["split"])
        assert not row["image_path"].startswith(str(tmp_path))
        assert "\\" not in row["image_path"]
    assert all(len(splits) == 1 for splits in splits_by_case.values())

    out = tmp_path / "manifest.csv"
    write_manifest_csv(rows, out)
    with out.open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == len(rows)
