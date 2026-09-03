"""
training/train.py
==================
Phase-1 training loop: binary classification (ProposedClassifier) on the
video dataset manifest (data/prepare_video_dataset.py's output).

Not the final training protocol (PROJECT_BRIEF.md §5 trains only on
Kvasir-SEG, with the video set held out as external test-only) -- this is
the agreed pilot: train on the video dataset now, since it's the data
actually available, with a segmentation head to follow once Kvasir-SEG (real
masks) is in place.

Kept as plain functions (not one monolithic script body) so a later
training/hp_search.py can reuse train_one_epoch/evaluate/build_dataloaders
without duplicating this logic.

Usage:
    python -m training.train --config configs/config.yaml
    python -m training.train --epochs 5
    python -m training.train --resume checkpoints/proposed_classifier_last.pt
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
from torch.utils.data import DataLoader

from data.dataset import load_config
from data.video_classification_dataset import VideoClassificationDataset
from models.compression import compression_summary
from models.proposed_model import (
    ProposedClassifier,
    _BLOCK_TYPES,
    _KERNEL_SIZES,
    _NUM_CONVS,
    _USE_POOL_DOWNSAMPLE,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_compression_summary(config: dict) -> dict:
    """Records the achieved channel plan / compression ratio into checkpoints
    for later §7b ablation bookkeeping."""
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
    manifest_path = config["datasets"]["video"]["manifest_path"]
    train_ds = VideoClassificationDataset(manifest_path, split="train", config_path=config_path)
    test_ds = VideoClassificationDataset(manifest_path, split="test", config_path=config_path)

    train_cfg = config["training"]
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], pin_memory=pin_memory,
    )
    return train_loader, test_loader


def compute_pos_weight(train_dataset: VideoClassificationDataset) -> torch.Tensor:
    """neg_count / pos_count from the actual train-split manifest rows (post
    frame-stride subsampling) -- NOT hardcoded from pre-split, pre-stride
    dataset totals, since those don't reflect what the model actually trains
    on. This addresses frame-level class imbalance (the training unit here);
    case-level imbalance (3 negative vs. 20 positive cases) is a separate
    axis already handled by split_case_ids' min_test/min_train guarantees in
    data/prepare_video_dataset.py, not by this weight."""
    labels = [label for _, label in train_dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Train split must contain both classes; got n_pos={n_pos}, n_neg={n_neg}")
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


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
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
    return running_loss / n_samples if n_samples else float("nan")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    """accuracy/precision/recall/F1 via manual confusion-matrix counts (no
    sklearn dependency). Dice/IoU are explicitly NOT computed here -- the
    video dataset has no pixel masks this phase."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    tp = fp = fn = tn = 0
    for images, labels in loader:
        images = images.to(device)
        labels_dev = labels.to(device).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels_dev)
        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

        preds = (torch.sigmoid(logits) >= 0.5).float().cpu().squeeze(1)
        tp += int(((preds == 1) & (labels == 1)).sum().item())
        fp += int(((preds == 1) & (labels == 0)).sum().item())
        fn += int(((preds == 0) & (labels == 1)).sum().item())
        tn += int(((preds == 0) & (labels == 0)).sum().item())

    accuracy = (tp + tn) / n_samples if n_samples else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "loss": running_loss / n_samples if n_samples else float("nan"),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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
    }, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the proposed classifier on the video dataset (phase 1)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (defaults to configs/config.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs from config")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = build_dataloaders(config, config_path=args.config)
    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    pos_weight = compute_pos_weight(train_loader.dataset).to(device)
    print(f"pos_weight (neg/pos, train split): {pos_weight.item():.4f}")

    model = ProposedClassifier(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

    history_path = results_dir / "proposed_classifier_history.csv"
    write_header = not history_path.exists()
    best_f1 = -1.0
    epochs_since_best = 0
    early_stop_patience = train_cfg.get("early_stop_patience")
    metrics: Optional[dict] = None

    with history_path.open("a", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "test_loss", "accuracy", "precision", "recall", "f1"])

        for epoch in range(start_epoch, epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, test_loader, criterion, device)
            print(
                f"epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} test_loss={metrics['loss']:.4f} "
                f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
                f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
            )
            writer.writerow([
                epoch, train_loss, metrics["loss"], metrics["accuracy"],
                metrics["precision"], metrics["recall"], metrics["f1"],
            ])
            history_file.flush()

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                epochs_since_best = 0
                save_checkpoint(model, optimizer, epoch, metrics, config, checkpoint_dir / "proposed_classifier_best.pt")
            else:
                epochs_since_best += 1
                if early_stop_patience is not None and epochs_since_best >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1} (no test F1 improvement in {early_stop_patience} epochs)")
                    break

    if metrics is not None:
        save_checkpoint(model, optimizer, epochs - 1, metrics, config, checkpoint_dir / "proposed_classifier_last.pt")
    print(f"Best test F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
