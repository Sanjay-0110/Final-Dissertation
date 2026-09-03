"""
training/train_segmentation.py
===============================
Phase-2 training loop: ProposedSegmentationModel on Kvasir-SEG (real pixel
masks) -- PROJECT_BRIEF.md §5/§6's actual training source, unlike phase-1's
video-dataset classification pilot (training/train.py).

Kvasir-SEG has no case/video structure -- every image is independent, so
(unlike data/prepare_video_dataset.py's case-level split) a simple seeded
random image-level train/val split is appropriate here.

Usage:
    python -m training.train_segmentation --config configs/config.yaml
    python -m training.train_segmentation --epochs 5
    python -m training.train_segmentation --resume checkpoints/proposed_segmentation_last.pt
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset import PolypSegDataset, load_config
from models.compression import compression_summary
from models.proposed_model import (
    ProposedSegmentationModel,
    _BLOCK_TYPES,
    _KERNEL_SIZES,
    _NUM_CONVS,
    _USE_POOL_DOWNSAMPLE,
)
from training.losses import DiceBCELoss, dice_score, iou_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_compression_summary(config: dict) -> dict:
    """Records the achieved channel plan / compression ratio into checkpoints
    for later §7b ablation bookkeeping (encoder only -- decoder channel plan
    is derived from this, not independently compressed)."""
    encoder_cfg = config["model"]["encoder"]
    return compression_summary(
        base_channels=encoder_cfg["base_channels"],
        compression_ratio=encoder_cfg["compression_ratio"],
        mode=encoder_cfg["compression_mode"],
        block_types=_BLOCK_TYPES,
        num_convs=_NUM_CONVS,
        kernel_sizes=_KERNEL_SIZES,
        use_pool_downsample=_USE_POOL_DOWNSAMPLE,
    )


def build_dataloaders(config: dict, config_path: Union[str, Path, None] = None) -> Tuple[DataLoader, DataLoader]:
    kvasir_cfg = config["datasets"]["kvasir_seg"]
    root = Path(kvasir_cfg["root"])

    # normalize_color passed explicitly (not left to PolypSegDataset's own
    # config_path reload) so an in-memory override -- e.g. main()'s
    # --disable-color-normalization for the §7a ablation -- actually reaches
    # the dataset instead of being silently discarded in favor of whatever
    # configs/config.yaml says on disk.
    normalize_color = config["color_normalization"]["enabled"]

    # Two separate PolypSegDataset instances (not one shared + Subset) so
    # augmentation can be train-only -- image_paths ordering is deterministic
    # (collect_images sorts by filename), so both instances index identically
    # and the same split indices apply to both.
    train_full_ds = PolypSegDataset(
        root / kvasir_cfg["images_dir"], root / kvasir_cfg["masks_dir"],
        config_path=config_path, augment=True, normalize_color=normalize_color,
    )
    val_full_ds = PolypSegDataset(
        root / kvasir_cfg["images_dir"], root / kvasir_cfg["masks_dir"],
        config_path=config_path, augment=False, normalize_color=normalize_color,
    )

    split_cfg = config["segmentation_split"]
    n = len(train_full_ds)
    indices = list(range(n))
    random.Random(split_cfg["seed"]).shuffle(indices)
    n_train = round(n * split_cfg["train_ratio"])
    train_ds = Subset(train_full_ds, indices[:n_train])
    val_ds = Subset(val_full_ds, indices[n_train:])

    train_cfg = config["segmentation_training"]
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=pin_memory,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return running_loss / n_samples if n_samples else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
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
    }


def load_encoder_weights(model: ProposedSegmentationModel, classifier_checkpoint_path: Path, device: torch.device) -> None:
    """Warm-starts the segmentation encoder from an already-trained
    ProposedClassifier checkpoint (training/train.py). Both models build an
    identical CompressedEncoder from the same config.model.encoder block, so
    the "encoder." keys in the classifier's state_dict load directly --
    strict=True intentionally, since a shape mismatch here means the two
    configs' encoder settings (base_channels/compression_ratio/mode) drifted
    apart, and that should fail loudly rather than silently skip weights.

    This only initializes the encoder -- the segmentation decoder still
    trains from scratch, since the classifier never had one. Rationale: the
    video dataset has thousands of labeled frames vs. Kvasir-SEG's 800
    training images, so encoder features learned there are a reasonable
    warm start, addressing the "no pretrained weights" gap in the first
    from-scratch segmentation run (best Dice 0.5733)."""
    checkpoint = torch.load(classifier_checkpoint_path, map_location=device)
    encoder_state = {
        k[len("encoder."):]: v
        for k, v in checkpoint["model_state_dict"].items()
        if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=True)
    print(f"Initialized encoder from {classifier_checkpoint_path} (epoch {checkpoint['epoch']})")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "model_config": config["model"],
        "compression_summary": get_compression_summary(config),
        # Preprocessing must match at eval time or Dice/IoU numbers are not
        # comparable -- recorded here (not just left to the live config)
        # so evaluation/evaluate_cross_domain.py can't silently evaluate a
        # checkpoint with a DIFFERENT normalization setting than it trained
        # under, e.g. after configs/config.yaml changes for a later run
        # (§7a ablation: this is exactly the setting that run compares).
        "color_normalization": config["color_normalization"],
    }, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the proposed segmentation model on Kvasir-SEG (phase 2)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (defaults to configs/config.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Override segmentation_training.epochs from config")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--init-encoder-from", type=Path, default=None,
        help="Warm-start the encoder from a ProposedClassifier checkpoint (e.g. checkpoints/proposed_classifier_best.pt). "
             "Ignored if --resume is also set (resume already restores full model+optimizer state).",
    )
    parser.add_argument(
        "--disable-color-normalization", action="store_true",
        help="§7a ablation: override color_normalization.enabled to False for this run, regardless of config.yaml.",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Suffix for checkpoint/history filenames (e.g. 'no_color_norm'), so ablation runs don't overwrite "
             "checkpoints/proposed_segmentation_{best,last}.pt. Produces proposed_segmentation_{tag}_{best,last}.pt.",
    )
    parser.add_argument(
        "--compression-mode", type=str, default=None, choices=["asymmetric", "uniform"],
        help="§7b ablation: override model.encoder.compression_mode for this run, regardless of config.yaml. "
             "NOTE: changes the encoder's per-stage channel widths, so --init-encoder-from a checkpoint trained "
             "under a DIFFERENT compression_mode will fail (shape mismatch, by design -- see load_encoder_weights).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.disable_color_normalization or args.compression_mode:
        config = dict(config)  # shallow copy is enough -- only these nested dicts are mutated below
        if args.disable_color_normalization:
            config["color_normalization"] = dict(config["color_normalization"])
            config["color_normalization"]["enabled"] = False
        if args.compression_mode:
            config["model"] = dict(config["model"])
            config["model"]["encoder"] = dict(config["model"]["encoder"])
            config["model"]["encoder"]["compression_mode"] = args.compression_mode
    seed_everything(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Color normalization enabled: {config['color_normalization']['enabled']}")
    print(f"Compression mode: {config['model']['encoder']['compression_mode']}")

    train_loader, val_loader = build_dataloaders(config, config_path=args.config)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = ProposedSegmentationModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    if args.init_encoder_from and not args.resume:
        load_encoder_weights(model, args.init_encoder_from, device)

    train_cfg = config["segmentation_training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    criterion = DiceBCELoss()

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    results_dir = Path(train_cfg["results_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    name_suffix = f"_{args.tag}" if args.tag else ""
    best_ckpt_path = checkpoint_dir / f"proposed_segmentation{name_suffix}_best.pt"
    last_ckpt_path = checkpoint_dir / f"proposed_segmentation{name_suffix}_last.pt"
    history_path = results_dir / f"proposed_segmentation{name_suffix}_history.csv"
    write_header = not history_path.exists()
    best_dice = -1.0
    epochs_since_best = 0
    early_stop_patience = train_cfg.get("early_stop_patience")
    metrics: Optional[dict] = None

    with history_path.open("a", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "dice", "iou"])

        for epoch in range(start_epoch, epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, val_loader, criterion, device)
            print(
                f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
                f"dice={metrics['dice']:.4f} iou={metrics['iou']:.4f}"
            )
            writer.writerow([epoch, train_loss, metrics["loss"], metrics["dice"], metrics["iou"]])
            history_file.flush()

            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                epochs_since_best = 0
                save_checkpoint(model, optimizer, epoch, metrics, config, best_ckpt_path)
            else:
                epochs_since_best += 1
                if early_stop_patience is not None and epochs_since_best >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1} (no val Dice improvement in {early_stop_patience} epochs)")
                    break

    if metrics is not None:
        save_checkpoint(model, optimizer, epochs - 1, metrics, config, last_ckpt_path)
    print(f"Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
