from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    images.sort(key=lambda p: p.name)
    if not images:
        raise ValueError(f"No image files were found in {input_dir}")
    return images


def preprocess_image(image_path: Path, output_path: Path, size: Tuple[int, int], normalize: bool) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = ImageOps.pad(img, size, color=(0, 0, 0))
        img = img.resize(size, Image.Resampling.LANCZOS)

        if normalize:
            arr = np.array(img, dtype=np.float32) / 255.0
            Image.fromarray(np.uint8(arr * 255.0)).save(output_path)
        else:
            img.save(output_path)


def make_initial_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Create a coarse foreground mask using color contrast and morphology."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Adaptive threshold to capture bright/contrast regions while suppressing noise.
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        10,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(gray, dtype=np.uint8)

    # Keep the largest non-border component as a candidate foreground.
    h, w = gray.shape
    border = 5
    candidates = []
    for label in range(1, num_labels):
        x, y, ww, hh, area = stats[label]
        if area < 200:
            continue
        if x <= border or y <= border or (x + ww) >= (w - border) or (y + hh) >= (h - border):
            continue
        candidates.append((area, label))

    if not candidates:
        return np.zeros_like(gray, dtype=np.uint8)

    _, largest_label = max(candidates, key=lambda item: item[0])
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return mask


def refine_mask_with_grabcut(image_bgr: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
    """Refine the initial mask using GrabCut for a more realistic ROI approximation."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Seed with the coarse mask if available; otherwise fall back to a central rectangle.
    if coarse_mask.any():
        mask[coarse_mask > 0] = cv2.GC_FGD
        mask[coarse_mask == 0] = cv2.GC_BGD
    else:
        margin = max(10, int(min(w, h) * 0.08))
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)
        mask[:] = cv2.GC_BGD
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return mask

    # Use the coarse mask as a foreground prior and let GrabCut refine it.
    cv2.grabCut(image_bgr, mask, None, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_MASK)
    refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    if refined.sum() < 100:
        return coarse_mask

    # Clean up small noise and keep the main object.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    return refined


def create_mask(image_path: Path, output_path: Path) -> None:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    coarse_mask = make_initial_mask(img_bgr)
    refined_mask = refine_mask_with_grabcut(img_bgr, coarse_mask)

    # Keep only the largest connected component for a cleaner ROI.
    if refined_mask.any():
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size > 0:
                largest_label = 1 + int(np.argmax(areas))
                refined_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    cv2.imwrite(str(output_path), refined_mask)


def preprocess_folder(input_dir: Path, output_dir: Path, size: Tuple[int, int], normalize: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(input_dir)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        manifest_path.unlink()

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source_file", "processed_file", "width", "height"])

        for image_path in images:
            output_path = images_dir / image_path.name
            preprocess_image(image_path, output_path, size, normalize)

            with Image.open(output_path) as proc_img:
                width, height = proc_img.size

            writer.writerow([image_path.name, output_path.name, width, height])

    return manifest_path


def create_masks_for_folder(input_dir: Path, output_dir: Path) -> Path:
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    mask_manifest_path = output_dir / "mask_manifest.csv"
    if mask_manifest_path.exists():
        mask_manifest_path.unlink()

    with mask_manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_file", "mask_file"])

        for image_path in images:
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
            create_mask(image_path, mask_path)
            writer.writerow([image_path.name, mask_path.name])

    return mask_manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create realistic approximate masks for colonoscopy-style image frames")
    parser.add_argument("--input", type=Path, default=Path("sample"), help="Folder containing the input frame images")
    parser.add_argument("--output", type=Path, default=Path("sample/realistic_masks"), help="Folder for preprocessed images and masks")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], metavar=("W", "H"), help="Target image size")
    parser.add_argument("--normalize", action="store_true", help="Normalize pixel values to [0, 1]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = (args.size[0], args.size[1])

    manifest_path = preprocess_folder(args.input, args.output, size, args.normalize)
    print(f"Processed images saved to: {args.output / 'images'}")
    print(f"Manifest saved to: {manifest_path}")

    mask_manifest_path = create_masks_for_folder(args.output / "images", args.output)
    print(f"Masks saved to: {args.output / 'masks'}")
    print(f"Mask manifest saved to: {mask_manifest_path}")


if __name__ == "__main__":
    main()
