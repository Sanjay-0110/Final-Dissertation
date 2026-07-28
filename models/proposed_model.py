"""
models/proposed_model.py
=========================
Proposed architecture (PROJECT_BRIEF.md §1): asymmetric-compression encoder.

Phase 1 (this file, current scope): binary classification on the video
dataset (datasets/Positive|negative case folders), since that data has no
pixel masks yet. CompressedEncoder returns a list of multi-scale feature
maps precisely so a segmentation decoder can be attached later (U-Net-style
skip connections) without rebuilding the backbone, once Kvasir-SEG (real
masks) is in place at the end of the project.

NOTE: the frontend color/illumination normalizer (frontend/color_normalizer.py,
§3, §7a) is NOT called anywhere in this module. It already runs upstream,
per-item, inside data/preprocessing.py's preprocess_image(), before tensors
are created. Calling it again here would double-normalize the image and would
force a numpy round-trip mid-batch, breaking GPU batching -- it stays an
isolated, toggleable data-pipeline switch (on/off is a config flag, §7a),
not a model layer.

The <1.47M param target (UNeXt, models/baseline_registry.py) is the ceiling
for the FUTURE full system (this encoder + a segmentation decoder). This
phase's encoder+classification-head is deliberately budgeted well under that
(~265K at the default compression_ratio=0.6), reserving headroom for the
decoder.
"""

from __future__ import annotations

from typing import List, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.compression import build_channel_plan, depthwise_separable_block, standard_conv_block

# Fixed across both §7b ablation arms (asymmetric/uniform) -- only the
# resulting channel widths differ between modes, everything else here is
# identical, so the ablation isolates channel-placement, not block design.
_BLOCK_TYPES = ["standard", "depthwise_separable", "depthwise_separable", "depthwise_separable", "standard"]
_NUM_CONVS = [1, 2, 2, 2, 1]
_KERNEL_SIZES = [3, 3, 3, 3, 1]
_USE_POOL_DOWNSAMPLE = [False, False, False, False, True]


class CompressedEncoder(nn.Module):
    """5-stage encoder: stem, s1, s2, s3, bottleneck.

    Stem (dense, full width, stride-2 3x3): goal 1 (§1) -- first layer does
    edge/color detection, exactly where scanner differences manifest; dense
    preserves full cross-channel interaction there.

    s1/s2/s3 (depthwise-separable, compressed width): goal 2 -- standard
    params/FLOPs reduction. This is also the variable the domain-shift
    hypothesis is testing (§7): if the ablation later shows compressing here
    hurts cross-domain robustness, that's evidence for the hypothesis, not a
    design bug.

    Bottleneck (dense but 1x1 kernel + free MaxPool2d downsample, full
    width): goal 1 (full channel-mixing capacity for the "final decision" per
    §1) AND goal 2 -- a literal dense 3x3 at 512 channels alone costs
    512*512*9 = 2,359,296 params, over 1.6x the entire 1.47M UNeXt ceiling by
    itself. 1x1 is the only way to keep this stage "full capacity" (dense
    channel-mixing, not large spatial kernel) in-budget; upstream depthwise
    3x3s already cumulatively provide spatial context by this point.

    forward() returns [f0..f4], the 5 multi-scale feature maps (176/88/44/22/
    11 px at a 352 input) -- these become segmentation-decoder skip
    connections later, and the Grad-CAM (§8) target layer (f4) now.
    """

    def __init__(
        self,
        base_channels: Sequence[int] = (64, 128, 256, 384, 512),
        compression_ratio: float = 0.6,
        compression_mode: Literal["asymmetric", "uniform"] = "asymmetric",
        in_channels: int = 3,
    ):
        super().__init__()
        base_channels = list(base_channels)
        if len(base_channels) != 5:
            raise ValueError(
                f"CompressedEncoder expects exactly 5 stages (stem, s1, s2, s3, "
                f"bottleneck), got {len(base_channels)}"
            )

        plan = build_channel_plan(
            base_channels=base_channels,
            compression_ratio=compression_ratio,
            mode=compression_mode,
            block_types=_BLOCK_TYPES,
            num_convs=_NUM_CONVS,
            kernel_sizes=_KERNEL_SIZES,
            use_pool_downsample=_USE_POOL_DOWNSAMPLE,
            in_channels=in_channels,
        )
        self.out_channels = plan

        self.stem = standard_conv_block(in_channels, plan[0], kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(
            depthwise_separable_block(plan[0], plan[1], stride=2),
            depthwise_separable_block(plan[1], plan[1], stride=1),
        )
        self.stage2 = nn.Sequential(
            depthwise_separable_block(plan[1], plan[2], stride=2),
            depthwise_separable_block(plan[2], plan[2], stride=1),
        )
        self.stage3 = nn.Sequential(
            depthwise_separable_block(plan[2], plan[3], stride=2),
            depthwise_separable_block(plan[3], plan[3], stride=1),
        )
        self.bottleneck = standard_conv_block(plan[3], plan[4], kernel_size=1, use_pool_downsample=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f0 = self.stem(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.bottleneck(f3)
        return [f0, f1, f2, f3, f4]


class ProposedClassifier(nn.Module):
    """Phase-1 deliverable: CompressedEncoder + global-average-pool +
    Linear(-> 1 logit).

    Flagged per PROJECT_BRIEF.md §9: this classification head serves NEITHER
    of §1's two goals (domain-shift robustness or compute reduction) -- it
    exists only because pixel masks aren't available for the video dataset
    yet. It will be replaced/supplemented by a segmentation decoder (reusing
    this same CompressedEncoder as its backbone) once Kvasir-SEG is in place.
    """

    def __init__(self, config: dict):
        super().__init__()
        encoder_cfg = config["model"]["encoder"]
        self.encoder = CompressedEncoder(
            base_channels=encoder_cfg["base_channels"],
            compression_ratio=encoder_cfg["compression_ratio"],
            compression_mode=encoder_cfg["compression_mode"],
        )
        self.classifier_head = nn.Linear(self.encoder.out_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (B, 1) -- pair with BCEWithLogitsLoss."""
        features = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        return self.classifier_head(pooled)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    from data.dataset import load_config

    config = load_config()
    model = ProposedClassifier(config)
    total_params = count_parameters(model)
    target = config["model"]["target_max_params"]

    print(f"Compression mode: {config['model']['encoder']['compression_mode']}")
    print(f"Compression ratio: {config['model']['encoder']['compression_ratio']}")
    print(f"CompressedEncoder channel plan: {model.encoder.out_channels}")
    print(f"ProposedClassifier total params: {total_params:,}")
    print(f"Target ceiling (full future encoder+decoder system, spec section 1): {target:,}")
    print(f"Headroom reserved for future decoder: {target - total_params:,} ({(target - total_params) / target:.0%})")

    width, height = config["image_size"]
    dummy = torch.randn(2, 3, height, width)
    out = model(dummy)
    print(f"Forward pass output shape: {tuple(out.shape)} (expected (2, 1))")
