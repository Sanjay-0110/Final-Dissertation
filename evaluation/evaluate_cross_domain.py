"""
evaluation/evaluate_cross_domain.py
=====================================
Cross-domain segmentation evaluation (PROJECT_BRIEF.md §5/§6): loads an
already-trained checkpoint (training/train_segmentation.py) and evaluates it,
INFERENCE ONLY -- no gradient updates, no fine-tuning -- on a dataset's full
image set. This is the actual core experiment: measuring how much Dice/IoU
degrades when the model sees a domain (scanner/hospital/color profile) it
never trained on.

Usage:
    python -m evaluation.evaluate_cross_domain --checkpoint checkpoints/proposed_segmentation_best.pt --dataset kvasir_seg
    python -m evaluation.evaluate_cross_domain --checkpoint checkpoints/proposed_segmentation_best.pt --dataset cvc_clinicdb
    python -m evaluation.evaluate_cross_domain --checkpoint checkpoints/proposed_segmentation_best.pt --dataset etis_larib
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader

from data.dataset import PolypSegDataset, load_config
from models.proposed_model import ProposedSegmentationModel
from training.losses import DiceBCELoss, dice_score, iou_score

DATASET_KEYS = ("kvasir_seg", "cvc_clinicdb", "etis_larib")


def build_dataset(
    config: dict, dataset_name: str, config_path: Union[str, Path, None], normalize_color: bool,
) -> PolypSegDataset:
    if dataset_name not in DATASET_KEYS:
        raise ValueError(f"Unknown dataset {dataset_name!r}, expected one of {DATASET_KEYS}")
    ds_cfg = config["datasets"][dataset_name]
    if ds_cfg.get("root") is None:
        raise ValueError(f"datasets.{dataset_name}.root is not set in config -- dataset not placed yet?")
    root = Path(ds_cfg["root"])
    return PolypSegDataset(
        root / ds_cfg["images_dir"],
        root / ds_cfg["masks_dir"],
        config_path=config_path,
        augment=False,  # eval only, never augmented -- these numbers must be reproducible
        normalize_color=normalize_color,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    n_samples = 0
    dice_total = 0.0
    iou_total = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        dice_total += dice_score(logits, masks).sum().item()
        iou_total += iou_score(logits, masks).sum().item()
    return {
        "loss": running_loss / n_samples if n_samples else float("nan"),
        "dice": dice_total / n_samples if n_samples else float("nan"),
        "iou": iou_total / n_samples if n_samples else float("nan"),
        "n_samples": n_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-domain segmentation evaluation (inference only, no training)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (defaults to configs/config.yaml)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained segmentation checkpoint (training/train_segmentation.py)")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_KEYS, help="Which dataset to evaluate on")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--results-csv", type=Path, default=None, help="Append results here (defaults to <results_dir>/cross_domain_eval.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Built from the CHECKPOINT's own saved model_config, not the current
    # configs/config.yaml -- guarantees the architecture matches the weights
    # being loaded even if config.yaml has since changed (e.g. a different
    # compression_ratio for a later run).
    model_config = {"model": checkpoint["model_config"]}
    model = ProposedSegmentationModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded {args.checkpoint} (trained epoch {checkpoint['epoch']}, "
          f"in-domain metrics at save time: {checkpoint['metrics']})")

    # Preprocessing must match what the checkpoint was actually trained
    # under, not whatever the live config.yaml currently says -- checkpoints
    # saved before this field existed fall back to the live config, with a
    # loud warning, since that's a real ambiguity worth flagging rather than
    # silently guessing.
    if "color_normalization" in checkpoint:
        normalize_color = checkpoint["color_normalization"]["enabled"]
    else:
        normalize_color = config["color_normalization"]["enabled"]
        print(
            "WARNING: checkpoint has no recorded color_normalization setting "
            f"(older checkpoint) -- falling back to the live config value "
            f"({normalize_color}). Verify this matches what the checkpoint was trained with."
        )
    print(f"Color normalization enabled: {normalize_color}")

    dataset = build_dataset(config, args.dataset, config_path=args.config, normalize_color=normalize_color)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"Evaluating on {args.dataset}: {len(dataset)} samples")

    criterion = DiceBCELoss()
    metrics = evaluate(model, loader, criterion, device)
    print(
        f"Dataset={args.dataset} loss={metrics['loss']:.4f} dice={metrics['dice']:.4f} "
        f"iou={metrics['iou']:.4f} n_samples={metrics['n_samples']}"
    )

    results_dir = Path(config["segmentation_training"]["results_dir"])
    results_csv = args.results_csv or (results_dir / "cross_domain_eval.csv")
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["checkpoint", "dataset", "loss", "dice", "iou", "n_samples"])
        writer.writerow([str(args.checkpoint), args.dataset, metrics["loss"], metrics["dice"], metrics["iou"], metrics["n_samples"]])
    print(f"Appended result to {results_csv}")


if __name__ == "__main__":
    main()
