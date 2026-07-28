"""
data/video_classification_dataset.py
======================================
PyTorch Dataset for phase-1 binary classification on the video dataset,
reading from the manifest produced by data/prepare_video_dataset.py.

Not a PolypSegDataset subclass (data/dataset.py) since it's manifest-driven
across two label-partitioned directory trees (datasets/Positive/*,
datasets/negative/*) rather than a single images_dir/masks_dir pair -- but it
reuses the exact same preprocessing (preprocess_image, _to_image_tensor,
load_config) so every image goes through the identical FOV-crop/specular-
inpaint/color-norm/resize pipeline as every other dataset in this project,
guaranteeing uniform (config image_size) tensors before the network sees
them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from data.dataset import _to_image_tensor, load_config
from data.preprocessing import preprocess_image


def _read_manifest_csv(manifest_path: Path) -> List[dict]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class VideoClassificationDataset(Dataset):
    """Args:
        manifest_path: CSV from data/prepare_video_dataset.py (columns:
            image_path, case_id, label, split).
        split: "train" or "test" -- rows are filtered to this split.
        size/normalize_color/minkowski_p: as in PolypSegDataset, default to
            configs/config.yaml when left None.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        split: Literal["train", "test"],
        config_path: Union[str, Path, None] = None,
        size: Optional[Tuple[int, int]] = None,
        normalize_color: Optional[bool] = None,
        minkowski_p: Optional[float] = None,
    ):
        config = load_config(config_path)
        self.size: Tuple[int, int] = tuple(size or config["image_size"])
        self.normalize_color: bool = (
            normalize_color if normalize_color is not None else config["color_normalization"]["enabled"]
        )
        self.minkowski_p: float = (
            minkowski_p if minkowski_p is not None else config["color_normalization"]["minkowski_p"]
        )
        self.fov_crop_kwargs = config.get("fov_crop", {})
        self.specular_kwargs = config.get("specular_inpaint", {})

        rows = _read_manifest_csv(Path(manifest_path))
        self.samples: List[Tuple[Path, int]] = [
            (Path(row["image_path"]), int(row["label"])) for row in rows if row["split"] == split
        ]
        if not self.samples:
            raise ValueError(f"No rows found for split={split!r} in manifest {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = preprocess_image(
            image_path,
            size=self.size,
            normalize_color=self.normalize_color,
            minkowski_p=self.minkowski_p,
            fov_crop_kwargs=self.fov_crop_kwargs,
            specular_kwargs=self.specular_kwargs,
        )
        return _to_image_tensor(image), torch.tensor(label, dtype=torch.float32)
