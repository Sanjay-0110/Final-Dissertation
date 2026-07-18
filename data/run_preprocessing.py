"""
data/run_preprocessing.py
==========================
CLI batch runner for the Stage 1 offline preprocessing pass (§3). Not used
as online augmentation -- per-image cost (contour detection + inpainting) is
CPU-heavy, so this is run once per dataset and the output is what training/
evaluation actually read.

Usage:
    python -m data.run_preprocessing --input sample --output sample/preprocessed_run
    python -m data.run_preprocessing --input path/to/images --masks path/to/masks \
        --output path/to/out --config configs/config.yaml
    python -m data.run_preprocessing --input sample --output out --no-color-norm
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from data.dataset import collect_images, load_config
from data.preprocessing import preprocess_image, preprocess_image_and_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 offline preprocessing batch runner")
    parser.add_argument("--input", type=Path, required=True, help="Folder containing input images")
    parser.add_argument("--masks", type=Path, default=None, help="Folder of ground-truth masks (same filenames as input images); omitted for mask-less data")
    parser.add_argument("--output", type=Path, required=True, help="Folder to write preprocessed images/masks + manifest.csv")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (defaults to configs/config.yaml)")
    parser.add_argument("--no-color-norm", action="store_true", help="Disable the Shades-of-Gray frontend normalizer (§7a ablation: off arm)")
    parser.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"), help="Override target size from config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    size = tuple(args.size) if args.size else tuple(config["image_size"])
    normalize_color = not args.no_color_norm and config["color_normalization"]["enabled"]
    minkowski_p = config["color_normalization"]["minkowski_p"]
    fov_crop_kwargs = config.get("fov_crop", {})
    specular_kwargs = config.get("specular_inpaint", {})

    images_out_dir = args.output / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir = args.output / "masks" if args.masks else None
    if masks_out_dir:
        masks_out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args.input)

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        header = ["source_image", "processed_image", "width", "height", "color_normalized"]
        if masks_out_dir:
            header.append("processed_mask")
        writer.writerow(header)

        for image_path in image_paths:
            out_image_path = images_out_dir / image_path.name

            if masks_out_dir:
                mask_path = args.masks / image_path.name
                image, mask = preprocess_image_and_mask(
                    image_path,
                    mask_path,
                    size=size,
                    normalize_color=normalize_color,
                    minkowski_p=minkowski_p,
                    fov_crop_kwargs=fov_crop_kwargs,
                    specular_kwargs=specular_kwargs,
                )
                out_mask_path = masks_out_dir / image_path.name
                cv2.imwrite(str(out_mask_path), mask)
            else:
                image = preprocess_image(
                    image_path,
                    size=size,
                    normalize_color=normalize_color,
                    minkowski_p=minkowski_p,
                    fov_crop_kwargs=fov_crop_kwargs,
                    specular_kwargs=specular_kwargs,
                )
                out_mask_path = None

            cv2.imwrite(str(out_image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            row = [image_path.name, out_image_path.name, image.shape[1], image.shape[0], normalize_color]
            if masks_out_dir:
                row.append(out_mask_path.name)
            writer.writerow(row)

    print(f"Processed {len(image_paths)} image(s).")
    print(f"Images saved to: {images_out_dir}")
    if masks_out_dir:
        print(f"Masks saved to: {masks_out_dir}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
