"""
frontend/color_normalizer.py
=============================
Non-learned color/illumination normalizer (PROJECT_BRIEF.md §1, §3, §7a).

Implements Shades-of-Gray color constancy (Minkowski p-norm generalization of
the gray-world assumption). This corrects per-image illumination/color-cast
differences between endoscopy scanners -- it does NOT convert images to
grayscale. Output stays a full 3-channel RGB image; only the relative channel
gains are adjusted so the estimated scene illuminant is neutralized.

Kept isolated from models/ and from the rest of the preprocessing pipeline so
it is a single, independently toggleable switch for the §7a ablation (frontend
normalizer on vs. off, everything else held fixed).
"""

from __future__ import annotations

import numpy as np


def shades_of_gray(image_rgb: np.ndarray, p: float = 6.0, eps: float = 1e-6) -> np.ndarray:
    """
    Shades-of-Gray color constancy normalization.

    Args:
        image_rgb: HxWx3 uint8 RGB image.
        p: Minkowski norm order. p=1 reduces to the classic gray-world
           assumption; p=inf reduces to max-RGB. p=6 (this project's default,
           per PROJECT_BRIEF.md §3) sits between the two.
        eps: numerical floor to avoid division by zero on near-black images.

    Returns:
        HxWx3 uint8 RGB image with per-channel gain correction applied.
        Always 3-channel color -- never desaturated.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {image_rgb.shape}")

    img = image_rgb.astype(np.float64)

    # Per-channel Minkowski p-norm mean -> illuminant estimate.
    illum = np.power(np.mean(np.power(img, p), axis=(0, 1)), 1.0 / p)
    illum = np.maximum(illum, eps)

    # Normalize the illuminant vector so a perfectly neutral illuminant maps
    # to a unit gain per channel (equal-energy assumption), then correct.
    illum = illum / np.linalg.norm(illum) * np.sqrt(3.0)
    corrected = img / illum

    corrected = np.clip(corrected, 0, 255)
    return corrected.astype(np.uint8)


class ColorNormalizer:
    """Toggleable wrapper so on/off is a single flag threaded through the
    preprocessing pipeline (PROJECT_BRIEF.md §7a), not a code-path change."""

    def __init__(self, enabled: bool = True, p: float = 6.0):
        self.enabled = enabled
        self.p = p

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return image_rgb
        return shades_of_gray(image_rgb, p=self.p)
