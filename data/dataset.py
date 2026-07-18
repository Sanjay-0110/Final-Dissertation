"""
data/dataset.py
================
Generic PyTorch Dataset wrapping the Stage 1 preprocessing pipeline
(data/preprocessing.py). Dataset-agnostic: works on any images-dir (+
optional masks-dir) pair. Per-dataset folder-naming conventions live in
data/datasets.py, which subclasses this.

Ground-truth masks are training/eval labels for loss/metric computation only
-- they are never fed into the network as input (PROJECT_BRIEF.md §2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from data.preprocessing import SUPPORTED_EXTENSIONS, preprocess_image, preprocess_image_and_mask

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def load_config(config_path: Union[str, Path, None] = None) -> dict:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    images.sort(key=lambda p: p.name)
    if not images:
        raise ValueError(f"No image files found in {images_dir}")
    return images


def _to_image_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB -> 3xHxW float32 tensor in [0, 1]."""
    tensor = torch.from_numpy(image_rgb).float() / 255.0
    return tensor.permute(2, 0, 1).contiguous()


def _to_mask_tensor(mask: np.ndarray) -> torch.Tensor:
    """HxW uint8 mask -> 1xHxW float32 tensor, binarized to {0, 1}."""
    tensor = torch.from_numpy(mask).float()
    tensor = (tensor > 127).float()
    return tensor.unsqueeze(0)


class PolypSegDataset(Dataset):
    """
    Args:
        images_dir: folder of input images.
        masks_dir: folder of ground-truth masks (same filename as the
            corresponding image), or None for mask-less data (e.g. video
            frames with unconfirmed ground truth -- §2, §12).
        size: (width, height) resize target. Defaults to configs/config.yaml.
        normalize_color: §7a ablation switch. Defaults to configs/config.yaml.
        minkowski_p: Shades-of-Gray Minkowski order. Defaults to config.
        config_path: config file to pull defaults from when an argument above
            is left as None.
        mask_filename_fn: maps an image Path to its expected mask filename.
            Defaults to an identical filename in masks_dir (Kvasir-SEG /
            CVC-ClinicDB convention) -- override for datasets that differ.
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Optional[Union[str, Path]] = None,
        size: Optional[Tuple[int, int]] = None,
        normalize_color: Optional[bool] = None,
        minkowski_p: Optional[float] = None,
        config_path: Union[str, Path, None] = None,
        mask_filename_fn: Optional[Callable[[Path], str]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir is not None else None
        self.mask_filename_fn = mask_filename_fn or (lambda image_path: image_path.name)

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

        self.image_paths = collect_images(self.images_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        if self.masks_dir is not None:
            mask_path = self.masks_dir / self.mask_filename_fn(image_path)
            image, mask = preprocess_image_and_mask(
                image_path,
                mask_path,
                size=self.size,
                normalize_color=self.normalize_color,
                minkowski_p=self.minkowski_p,
                fov_crop_kwargs=self.fov_crop_kwargs,
                specular_kwargs=self.specular_kwargs,
            )
            return _to_image_tensor(image), _to_mask_tensor(mask)

        image = preprocess_image(
            image_path,
            size=self.size,
            normalize_color=self.normalize_color,
            minkowski_p=self.minkowski_p,
            fov_crop_kwargs=self.fov_crop_kwargs,
            specular_kwargs=self.specular_kwargs,
        )
        return _to_image_tensor(image), image_path.name
