"""
evaluation/generate_qualitative_panel.py
===========================================
Fig 6.2 (report): input image / ground-truth mask / predicted mask, one row
in-domain (Kvasir-SEG val) and one row cross-domain (CVC-ClinicDB) --
requires matplotlib (not in requirements.txt -- `pip install matplotlib`
first if not already present in the conda env), the real trained checkpoint,
and torch, so this runs on CSF3, not locally.

Usage:
    python -m evaluation.generate_qualitative_panel --checkpoint checkpoints/proposed_segmentation_best.pt --output results/fig6_2_qualitative.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import PolypSegDataset, load_config
from models.proposed_model import ProposedSegmentationModel


def load_model(checkpoint_path: Path, device: torch.device) -> ProposedSegmentationModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = {"model": checkpoint["model_config"]}
    model = ProposedSegmentationModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalize_color = checkpoint.get("color_normalization", {}).get("enabled", True)
    return model, normalize_color


@torch.no_grad()
def predict(model, image: torch.Tensor, device: torch.device) -> np.ndarray:
    logits = model(image.unsqueeze(0).to(device))
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (probs >= 0.5).astype(np.uint8)


def to_display(image_tensor: torch.Tensor) -> np.ndarray:
    return (image_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Fig 6.2 qualitative prediction panel")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/fig6_2_qualitative.png"))
    parser.add_argument("--seed", type=int, default=0, help="Which example to pick from each dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, normalize_color = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint}, color_normalization={normalize_color}")

    kvasir_cfg = config["datasets"]["kvasir_seg"]
    kvasir_root = Path(kvasir_cfg["root"])
    kvasir_ds = PolypSegDataset(
        kvasir_root / kvasir_cfg["images_dir"], kvasir_root / kvasir_cfg["masks_dir"],
        config_path=args.config, augment=False, normalize_color=normalize_color,
    )

    cvc_cfg = config["datasets"]["cvc_clinicdb"]
    cvc_root = Path(cvc_cfg["root"])
    cvc_ds = PolypSegDataset(
        cvc_root / cvc_cfg["images_dir"], cvc_root / cvc_cfg["masks_dir"],
        config_path=args.config, augment=False, normalize_color=normalize_color,
    )

    random.seed(args.seed)
    kvasir_idx = random.randrange(len(kvasir_ds))
    cvc_idx = random.randrange(len(cvc_ds))

    rows = [
        ("Kvasir-SEG (in-domain)", *kvasir_ds[kvasir_idx]),
        ("CVC-ClinicDB (cross-domain)", *cvc_ds[cvc_idx]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i, (name, image, mask) in enumerate(rows):
        pred = predict(model, image, device)
        gt = mask[0].numpy()

        axes[i, 0].imshow(to_display(image))
        axes[i, 0].set_title("Input image" if i == 0 else "")
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground truth" if i == 0 else "")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction" if i == 0 else "")

        axes[i, 0].set_ylabel(name, fontsize=10, fontweight="bold")
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    fig.suptitle("Fig 6.2 -- Qualitative predictions: in-domain vs. cross-domain", fontsize=12, y=1.02)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
