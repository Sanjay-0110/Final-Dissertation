"""
data/datasets.py
=================
Thin per-dataset subclasses of PolypSegDataset (data/dataset.py) that supply
each dataset's folder-naming convention, keyed off configs/config.yaml
`datasets:` entries.

ASSUMPTION (stated per PROJECT_BRIEF.md §9, since none of these datasets are
downloaded into this repo yet): folder-naming conventions below are the
common public-release layouts. Verify/adjust once the real data is placed --
see the `# TODO` markers in configs/config.yaml.
    - Kvasir-SEG: `images/` and `masks/`, same filename in both.
    - CVC-ClinicDB: `Original/` and `Ground Truth/`, same filename in both.
    - ETIS-LaribPolypDB: `images/` and `masks/` (naming varies by mirror;
      some releases use `GT`/`GTS` for masks -- override via masks_dir if so).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from data.dataset import PolypSegDataset, load_config


def _resolve_dataset_dirs(config: dict, key: str) -> tuple[Path, Path]:
    entry = config["datasets"][key]
    if entry.get("root") is None:
        raise ValueError(
            f"datasets.{key}.root is not set in config.yaml -- update it once the "
            f"{key} dataset has been placed on disk."
        )
    root = Path(entry["root"])
    return root / entry["images_dir"], root / entry["masks_dir"]


class KvasirSEGDataset(PolypSegDataset):
    def __init__(self, config_path: Union[str, Path, None] = None, **kwargs):
        config = load_config(config_path)
        images_dir, masks_dir = _resolve_dataset_dirs(config, "kvasir_seg")
        super().__init__(images_dir=images_dir, masks_dir=masks_dir, config_path=config_path, **kwargs)


class CVCClinicDBDataset(PolypSegDataset):
    def __init__(self, config_path: Union[str, Path, None] = None, **kwargs):
        config = load_config(config_path)
        images_dir, masks_dir = _resolve_dataset_dirs(config, "cvc_clinicdb")
        super().__init__(images_dir=images_dir, masks_dir=masks_dir, config_path=config_path, **kwargs)


class ETISLaribDataset(PolypSegDataset):
    def __init__(self, config_path: Union[str, Path, None] = None, **kwargs):
        config = load_config(config_path)
        images_dir, masks_dir = _resolve_dataset_dirs(config, "etis_larib")
        super().__init__(images_dir=images_dir, masks_dir=masks_dir, config_path=config_path, **kwargs)


class VideoFrameDataset(PolypSegDataset):
    """External test-only, ground-truth mask availability unconfirmed (§2,
    §12) -- masks_dir is optional and left unset by default."""

    def __init__(
        self,
        case_dir: Union[str, Path],
        masks_dir: Optional[Union[str, Path]] = None,
        config_path: Union[str, Path, None] = None,
        **kwargs,
    ):
        super().__init__(images_dir=case_dir, masks_dir=masks_dir, config_path=config_path, **kwargs)
