"""
training/losses.py
===================
Segmentation loss/metrics for ProposedSegmentationModel (Kvasir-SEG, real
pixel masks). Separate from training/train.py's classification BCE loss --
segmentation's class imbalance is spatial (most pixels are background per
image), not per-sample, so plain BCE alone under-weights the polyp region.
Dice directly optimizes mask overlap and is far less sensitive to that
imbalance; combining both is standard practice for medical segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def soft_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable per-sample Dice from raw logits (sigmoid applied
    here), shape (B,) -- used inside the training loss, not for reporting."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return (2 * intersection + eps) / (union + eps)


class DiceBCELoss(nn.Module):
    """BCEWithLogitsLoss + (1 - soft Dice), averaged over the batch."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = 1.0 - soft_dice(logits, targets).mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Hard (thresholded) per-sample Dice, shape (B,) -- for eval reporting,
    not backprop."""
    preds = (torch.sigmoid(logits) >= threshold).float().flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return (2 * intersection + eps) / (union + eps)


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Hard (thresholded) per-sample IoU, shape (B,) -- for eval reporting."""
    preds = (torch.sigmoid(logits) >= threshold).float().flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)
