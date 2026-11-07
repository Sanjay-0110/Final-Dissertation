import torch

from training.losses import DiceBCELoss, dice_score, iou_score

CONFIDENT = 20.0  # sigmoid(+/-20) is effectively 1/0


def logits_from(mask: torch.Tensor) -> torch.Tensor:
    return (mask * 2 - 1) * CONFIDENT


def half_mask() -> torch.Tensor:
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4] = 1
    return mask


def test_perfect_prediction_scores_one():
    mask = half_mask()
    assert dice_score(logits_from(mask), mask).item() == 1.0
    assert iou_score(logits_from(mask), mask).item() == 1.0


def test_inverted_prediction_scores_zero():
    mask = half_mask()
    assert dice_score(logits_from(1 - mask), mask).item() < 1e-6
    assert iou_score(logits_from(1 - mask), mask).item() < 1e-6


def test_partial_overlap_matches_hand_calculation():
    mask = half_mask()  # 32 positive pixels
    pred = torch.zeros(1, 1, 8, 8)
    pred[..., 2:6] = 1  # 32 positive pixels, 16 overlapping
    assert abs(dice_score(logits_from(pred), mask).item() - 0.5) < 1e-5
    assert abs(iou_score(logits_from(pred), mask).item() - 1 / 3) < 1e-5


def test_metrics_are_per_sample():
    mask = half_mask().repeat(3, 1, 1, 1)
    assert dice_score(logits_from(mask), mask).shape == (3,)
    assert iou_score(logits_from(mask), mask).shape == (3,)


def test_loss_rewards_better_predictions_and_backpropagates():
    mask = half_mask()
    criterion = DiceBCELoss()
    assert criterion(logits_from(mask), mask) < criterion(logits_from(1 - mask), mask)

    logits = torch.zeros_like(mask, requires_grad=True)
    criterion(logits, mask).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
