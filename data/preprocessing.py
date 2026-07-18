"""
data/preprocessing.py
======================
Stage 1 offline preprocessing pipeline (PROJECT_BRIEF.md §3).

Fixed step order -- do not reorder (§3, §11):
    1. FOV crop            -- remove scanner-specific black borders/UI chrome.
    2. Specular inpainting  -- must run BEFORE color normalization, otherwise
                                 bright specular reflections bias the
                                 illuminant estimate used in step 3.
    3. Color normalization  -- Shades-of-Gray (frontend/color_normalizer.py),
                                 toggleable for the §7a ablation.
    4. Resize               -- images use area/cubic interpolation; masks use
                                 nearest-neighbor ONLY (§2, §11), to preserve
                                 binary label integrity.

This is an offline, one-time pass (per-image cost is CPU-heavy: contour
detection + inpainting), not online augmentation -- see run_preprocessing.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from frontend.color_normalizer import ColorNormalizer

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return mask


def crop_fov(
    image_rgb: np.ndarray,
    gray_thresh: int = 10,
    skip_if_coverage_above: float = 0.98,
    max_inset_fraction: float = 0.35,
    inset_step_fraction: float = 0.01,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Detects and crops scanner-specific black borders/UI chrome.

    Two shapes of border need handling:
      - Rectangular black bars (top/bottom or side letterboxing): a plain
        bounding-rect crop over the non-near-black content handles these.
      - Circular/octagonal endoscope vignettes (the field-of-view mask often
        baked into raw video captures): these touch all four edge midpoints,
        so their bounding rect is ~the whole frame and a bounding-rect crop
        alone leaves the black corners untouched. To handle this, after the
        bounding-rect crop the box is symmetrically inset step-by-step until
        all four of its corners land on non-black content (or the max inset
        budget is exhausted).

    If the initial bounding box already covers most of the frame with all
    four corners non-black (no real border present -- common for Kvasir-SEG,
    less common for CVC-ClinicDB / raw video captures), no cropping happens,
    to avoid cutting into valid full-frame content.

    Returns the (possibly) cropped image and the (x, y, w, h) box applied, so
    the same box can be reused to crop a corresponding mask identically.
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, gray_thresh, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_rgb, (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(largest)
    content_mask = thresh > 0

    def corners_are_content(bx: int, by: int, bw: int, bh: int) -> bool:
        if bw <= 0 or bh <= 0:
            return False
        ys = [by, by + bh - 1]
        xs = [bx, bx + bw - 1]
        return all(content_mask[cy, cx] for cy in ys for cx in xs)

    if not corners_are_content(x, y, box_w, box_h):
        step = max(1, int(round(min(box_w, box_h) * inset_step_fraction)))
        max_inset = int(round(min(box_w, box_h) * max_inset_fraction))
        inset = 0
        while inset < max_inset and not corners_are_content(
            x + inset, y + inset, box_w - 2 * inset, box_h - 2 * inset
        ):
            inset += step
        x, y = x + inset, y + inset
        box_w, box_h = max(1, box_w - 2 * inset), max(1, box_h - 2 * inset)

    coverage = (box_w * box_h) / float(w * h)
    if coverage >= skip_if_coverage_above:
        return image_rgb, (0, 0, w, h)

    cropped = image_rgb[y : y + box_h, x : x + box_w]
    return cropped, (x, y, box_w, box_h)


def apply_crop_box(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Applies a previously-computed FOV crop box to a corresponding mask."""
    x, y, box_w, box_h = box
    return image[y : y + box_h, x : x + box_w]


def inpaint_specular_highlights(
    image_rgb: np.ndarray,
    v_thresh: int = 200,
    s_thresh: int = 60,
    inpaint_radius: int = 5,
    dilate_kernel_size: int = 5,
) -> np.ndarray:
    """
    Detects bright, low-saturation specular reflections and inpaints them.

    Must run before color normalization (§3, §11): specular highlights are
    near-white regardless of the true scene illuminant, so leaving them in
    would bias the Shades-of-Gray illuminant estimate.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    highlight_mask = ((v_channel >= v_thresh) & (s_channel <= s_thresh)).astype(np.uint8) * 255
    if not highlight_mask.any():
        return image_rgb

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    highlight_mask = cv2.dilate(highlight_mask, kernel)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(image_bgr, highlight_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def resize_image(image_rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resizes a color image. size is (width, height). Never used for masks."""
    return cv2.resize(image_rgb, size, interpolation=cv2.INTER_CUBIC)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resizes a binary/label mask. Nearest-neighbor ONLY (§2, §11) -- never
    bilinear/bicubic, to avoid introducing fractional/blended label values."""
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def preprocess_image(
    image_path: Path,
    size: Tuple[int, int] = (352, 352),
    normalize_color: bool = True,
    minkowski_p: float = 6.0,
    fov_crop_kwargs: Optional[dict] = None,
    specular_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """Runs the full fixed-order Stage 1 pipeline on a single image."""
    image = load_image_rgb(image_path)

    image, _crop_box = crop_fov(image, **(fov_crop_kwargs or {}))
    image = inpaint_specular_highlights(image, **(specular_kwargs or {}))

    normalizer = ColorNormalizer(enabled=normalize_color, p=minkowski_p)
    image = normalizer(image)

    image = resize_image(image, size)
    return image


def preprocess_image_and_mask(
    image_path: Path,
    mask_path: Path,
    size: Tuple[int, int] = (352, 352),
    normalize_color: bool = True,
    minkowski_p: float = 6.0,
    fov_crop_kwargs: Optional[dict] = None,
    specular_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the pipeline on an image and its ground-truth mask together, so
    the same FOV crop box and target size apply to both (§2: masks are labels
    for loss computation only, never fed into the network as input)."""
    image = load_image_rgb(image_path)
    mask = load_mask(mask_path)

    image, crop_box = crop_fov(image, **(fov_crop_kwargs or {}))
    mask = apply_crop_box(mask, crop_box)

    image = inpaint_specular_highlights(image, **(specular_kwargs or {}))

    normalizer = ColorNormalizer(enabled=normalize_color, p=minkowski_p)
    image = normalizer(image)

    image = resize_image(image, size)
    mask = resize_mask(mask, size)
    return image, mask
