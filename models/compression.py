"""
models/compression.py
======================
Shared asymmetric-vs-uniform compression utilities (PROJECT_BRIEF.md §1, §7b).

Two conv-block primitives:
  - standard_conv_block: dense conv, used for the full-capacity first/last
    stages (goal 1: domain-shift robustness -- these are the layers doing
    edge/color detection and final decision-relevant feature combination).
  - depthwise_separable_block: MobileNet-style depthwise+pointwise conv, used
    for the compressed middle stages (goal 2: compute reduction).

build_channel_plan() computes a list of per-stage channel widths hitting a
target overall param-compression ratio, in one of two modes:
  - "asymmetric": first and last stage stay at their base (ratio=1.0) width;
    only middle stages are compressed.
  - "uniform": every stage is compressed by the same multiplier.
Both modes are measured against the SAME ratio=1.0 reference (identical
block_types/num_convs/kernel_sizes/use_pool_downsample), so achieved ratios
are directly comparable -- this is the §7b ablation's whole point: same
overall compression budget, different placement.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Literal, Sequence

import torch.nn as nn

BlockType = Literal["standard", "depthwise_separable"]
CompressionMode = Literal["asymmetric", "uniform"]


def standard_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    use_pool_downsample: bool = False,
) -> nn.Sequential:
    """Dense conv -> BN -> ReLU. When use_pool_downsample, spatial
    downsampling is a parameter-free MaxPool2d(2) ahead of a stride-1 conv, so
    a wide "full-capacity" stage doesn't have to pay for a large kernel just
    to downsample (see models/proposed_model.py's bottleneck for why this
    matters at 512 channels)."""
    layers: List[nn.Module] = []
    if use_pool_downsample:
        layers.append(nn.MaxPool2d(2))
        conv_stride = 1
    else:
        conv_stride = stride
    layers.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=conv_stride,
                   padding=kernel_size // 2, bias=False)
    )
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def depthwise_separable_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """Depthwise 3x3 (per-channel spatial filter) + pointwise 1x1 (channel
    mixing) """
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _standard_block_params(in_ch: int, out_ch: int, kernel_size: int) -> int:
    conv = in_ch * out_ch * kernel_size * kernel_size
    bn = 2 * out_ch
    return conv + bn


def _depthwise_separable_block_params(in_ch: int, out_ch: int) -> int:
    depthwise = in_ch * 9 + 2 * in_ch
    pointwise = in_ch * out_ch + 2 * out_ch
    return depthwise + pointwise


def stage_param_count(
    in_channels: int,
    out_channels: int,
    block_type: BlockType,
    num_convs: int = 2,
    kernel_size: int = 3,
    use_pool_downsample: bool = False,
) -> int:
    """Analytic param count for one encoder stage, mirroring the block
    builders above exactly (bias=False throughout; BN affine = 2*out_channels
    per BN layer -- running_mean/var are buffers, not parameters).

    num_convs=1 -> a single in->out block (stem/bottleneck).
    num_convs=2 -> a down (in->out) block followed by a refine (out->out)
    block (s1/s2/s3's downsample-then-refine pattern).
    use_pool_downsample doesn't change the param count (MaxPool2d is
    parameter-free) -- kept as an argument for signature symmetry with
    standard_conv_block.
    """
    del use_pool_downsample  # parameter-free; accepted for signature symmetry only
    if num_convs not in (1, 2):
        raise ValueError(f"num_convs must be 1 or 2, got {num_convs}")

    if block_type == "standard":
        total = _standard_block_params(in_channels, out_channels, kernel_size)
        if num_convs == 2:
            total += _standard_block_params(out_channels, out_channels, kernel_size)
        return total
    elif block_type == "depthwise_separable":
        total = _depthwise_separable_block_params(in_channels, out_channels)
        if num_convs == 2:
            total += _depthwise_separable_block_params(out_channels, out_channels)
        return total
    else:
        raise ValueError(f"Unknown block_type: {block_type}")


def _total_params(
    channels: Sequence[int],
    block_types: Sequence[BlockType],
    num_convs: Sequence[int],
    kernel_sizes: Sequence[int],
    use_pool_downsample: Sequence[bool],
    in_channels: int,
) -> int:
    total = 0
    prev = in_channels
    for i, out_ch in enumerate(channels):
        total += stage_param_count(prev, out_ch, block_types[i], num_convs[i], kernel_sizes[i], use_pool_downsample[i])
        prev = out_ch
    return total


def _validate_stage_args(
    base_channels: Sequence[int],
    block_types: Sequence[BlockType],
    num_convs: Sequence[int],
    kernel_sizes: Sequence[int],
    use_pool_downsample: Sequence[bool],
) -> int:
    n = len(base_channels)
    if not (len(block_types) == len(num_convs) == len(kernel_sizes) == len(use_pool_downsample) == n):
        raise ValueError(
            "base_channels, block_types, num_convs, kernel_sizes, use_pool_downsample must all have the same length"
        )
    if n < 2:
        raise ValueError("Need at least 2 stages (first + last) for asymmetric compression to be meaningful")
    return n


def build_channel_plan(
    base_channels: Sequence[int],
    compression_ratio: float,
    mode: CompressionMode,
    block_types: Sequence[BlockType],
    num_convs: Sequence[int],
    kernel_sizes: Sequence[int],
    use_pool_downsample: Sequence[bool],
    in_channels: int = 3,
    tol: float = 0.02,
    max_iter: int = 40,
) -> List[int]:
    """Bisection search on a single channel-width multiplier so the built
    network's total param count hits `compression_ratio` of the ratio=1.0
    reference (same base_channels, same block plan).

    mode="asymmetric": base_channels[0] and base_channels[-1] are fixed at
    their reference width; the multiplier is searched only over the middle
    stages (base_channels[1:-1]).
    mode="uniform": the multiplier is searched over ALL stages equally.

    Both modes share the same reference total, so their achieved_ratio
    values are directly comparable -- that comparability is the entire point
    of the §7b ablation (same budget, different placement).
    """
    n = _validate_stage_args(base_channels, block_types, num_convs, kernel_sizes, use_pool_downsample)
    if not (0.0 < compression_ratio <= 1.0):
        raise ValueError(f"compression_ratio must be in (0, 1], got {compression_ratio}")

    base_channels = list(base_channels)
    reference_params = _total_params(base_channels, block_types, num_convs, kernel_sizes, use_pool_downsample, in_channels)

    if compression_ratio >= 1.0:
        return base_channels

    target_params = reference_params * compression_ratio

    def channels_at(multiplier: float) -> List[int]:
        if mode == "asymmetric":
            if n <= 2:
                return base_channels  # no middle stages to compress
            mids = [max(1, round(c * multiplier)) for c in base_channels[1:-1]]
            return [base_channels[0], *mids, base_channels[-1]]
        elif mode == "uniform":
            return [max(1, round(c * multiplier)) for c in base_channels]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def params_at(multiplier: float) -> int:
        return _total_params(channels_at(multiplier), block_types, num_convs, kernel_sizes, use_pool_downsample, in_channels)

    lo, hi = 0.01, 1.0
    best_channels = channels_at(hi)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        channels = channels_at(mid)
        params = params_at(mid)
        best_channels = channels
        achieved_ratio = params / reference_params
        if abs(achieved_ratio - compression_ratio) <= tol:
            break
        if params > target_params:
            hi = mid
        else:
            lo = mid
    return best_channels


@dataclass
class CompressionSummary:
    channel_plan: List[int]
    reference_params: int
    achieved_params: int
    achieved_ratio: float


def compression_summary(
    base_channels: Sequence[int],
    compression_ratio: float,
    mode: CompressionMode,
    block_types: Sequence[BlockType],
    num_convs: Sequence[int],
    kernel_sizes: Sequence[int],
    use_pool_downsample: Sequence[bool],
    in_channels: int = 3,
) -> dict:
    """Query helper for logging into checkpoints/ablation manifests (§7b)."""
    channel_plan = build_channel_plan(
        base_channels, compression_ratio, mode, block_types, num_convs,
        kernel_sizes, use_pool_downsample, in_channels=in_channels,
    )
    reference_params = _total_params(list(base_channels), block_types, num_convs, kernel_sizes, use_pool_downsample, in_channels)
    achieved_params = _total_params(channel_plan, block_types, num_convs, kernel_sizes, use_pool_downsample, in_channels)
    return asdict(CompressionSummary(
        channel_plan=channel_plan,
        reference_params=reference_params,
        achieved_params=achieved_params,
        achieved_ratio=achieved_params / reference_params,
    ))


def _verify_param_formulas() -> None:
    """Asserts stage_param_count's analytic formula matches a real
    instantiated nn.Module's actual trainable-parameter count, for both block
    types and both num_convs values. Run via `python -m models.compression`."""
    cases = [
        ("standard", 1, 3, 3, 64),
        ("standard", 2, 3, 3, 64),
        ("standard", 1, 1, 384, 512),
        ("depthwise_separable", 1, 3, 64, 128),
        ("depthwise_separable", 2, 3, 64, 128),
        ("depthwise_separable", 2, 3, 128, 256),
    ]
    for block_type, num_convs, kernel_size, in_ch, out_ch in cases:
        analytic = stage_param_count(in_ch, out_ch, block_type, num_convs, kernel_size, use_pool_downsample=False)

        if block_type == "standard":
            modules = [standard_conv_block(in_ch, out_ch, kernel_size=kernel_size)]
            if num_convs == 2:
                modules.append(standard_conv_block(out_ch, out_ch, kernel_size=kernel_size))
        else:
            modules = [depthwise_separable_block(in_ch, out_ch)]
            if num_convs == 2:
                modules.append(depthwise_separable_block(out_ch, out_ch))

        actual = sum(p.numel() for m in modules for p in m.parameters())
        status = "OK" if analytic == actual else "MISMATCH"
        print(f"[{status}] {block_type} num_convs={num_convs} in={in_ch} out={out_ch} k={kernel_size}: "
              f"analytic={analytic:,} actual={actual:,}")
        assert analytic == actual, f"stage_param_count formula does not match real module params: {analytic} != {actual}"

    print("\nAll stage_param_count formulas match real nn.Module parameter counts.")


if __name__ == "__main__":
    _verify_param_formulas()
