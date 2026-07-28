"""
Filter out overexposed / textureless "red-blob" junk frames from a
colonoscopy image dataset (e.g. scope pressed against mucosa wall,
lens smear, near-contact frames with no diagnostic content).

Detection logic (two signals combined):
  1. Red dominance  - fraction of in-view (non-black) pixels that are
                       strongly red/pink in HSV space.
  2. Low texture    - Laplacian variance (a standard blur/detail
                       measure) inside the in-view region. Junk frames
                       are flat and grainy, not textured.

A frame is flagged only if BOTH conditions hold, which keeps genuine
(but reddish) mucosa frames with real texture out of the flagged set.

By default this MOVES flagged files to a review folder rather than
deleting them outright -- check the folder, then delete for real.

Usage:
    python filter_red_blob_frames.py /path/to/dataset
    python filter_red_blob_frames.py /path/to/dataset --delete   # skip review, delete directly
    python filter_red_blob_frames.py /path/to/dataset --dry-run  # just list, don't move/delete
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_in_view_mask(img_bgr):
    """Mask out the black border/octagon vignette common in endoscope frames."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > 15  # treat near-black pixels as border, not content
    return mask


def analyze_frame(path, red_ratio_thresh, texture_thresh, min_in_view_frac):
    img = cv2.imread(str(path))
    if img is None:
        return None  # unreadable file, skip / flag separately

    mask = get_in_view_mask(img)
    total_px = mask.size
    in_view_px = mask.sum()

    if in_view_px < min_in_view_frac * total_px:
        # Almost entirely black frame -- also junk, flag it.
        return {"red_ratio": 1.0, "texture": 0.0, "flag": True, "reason": "mostly_black"}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Red hue wraps around 0/180 in OpenCV's H range (0-179)
    red_hue = ((h <= 10) | (h >= 160)) & (s > 60) & (v > 40)
    red_hue = red_hue & mask
    red_ratio = red_hue.sum() / max(in_view_px, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Median blur suppresses sensor grain/noise while preserving real edges
    # (folds, vessels), so noisy-but-featureless frames don't score as "textured".
    denoised = cv2.medianBlur(gray, 9)
    lap = cv2.Laplacian(denoised, cv2.CV_64F)
    texture = lap[mask].var() if in_view_px > 0 else 0.0

    flag = (red_ratio >= red_ratio_thresh) and (texture <= texture_thresh)
    return {"red_ratio": red_ratio, "texture": texture, "flag": flag, "reason": "red_blob" if flag else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path, help="Folder containing images (searched recursively)")
    ap.add_argument("--red-ratio-thresh", type=float, default=0.55,
                     help="Min fraction of in-view pixels that must be red/pink to count as red-dominant (default 0.55)")
    ap.add_argument("--texture-thresh", type=float, default=25.0,
                     help="Max Laplacian variance (post-denoise) to count as 'low texture' (default 25.0). Lower = stricter/junkier only.")
    ap.add_argument("--min-in-view-frac", type=float, default=0.05,
                     help="Below this fraction of non-black pixels, frame is flagged as mostly-black junk (default 0.05)")
    ap.add_argument("--review-dir", type=Path, default=None,
                     help="Where to move flagged files (default: <dataset_dir>/_flagged_red_blob)")
    ap.add_argument("--delete", action="store_true", help="Delete flagged files directly instead of moving to review dir")
    ap.add_argument("--dry-run", action="store_true", help="Only report what would be flagged, don't move or delete anything")
    args = ap.parse_args()

    if not args.dataset_dir.is_dir():
        sys.exit(f"Not a directory: {args.dataset_dir}")

    review_dir = args.review_dir or (args.dataset_dir / "_flagged_red_blob")
    if not args.dry_run and not args.delete:
        review_dir.mkdir(parents=True, exist_ok=True)

    paths = [p for p in args.dataset_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and review_dir not in p.parents]

    print(f"Scanning {len(paths)} images in {args.dataset_dir} ...")

    flagged = []
    unreadable = []

    for i, p in enumerate(paths, 1):
        result = analyze_frame(p, args.red_ratio_thresh, args.texture_thresh, args.min_in_view_frac)
        if result is None:
            unreadable.append(p)
            continue
        if result["flag"]:
            flagged.append((p, result))
        if i % 500 == 0:
            print(f"  ...{i}/{len(paths)} scanned, {len(flagged)} flagged so far")

    print(f"\nDone. {len(flagged)} / {len(paths)} images flagged as red-blob/junk frames.")
    if unreadable:
        print(f"{len(unreadable)} files could not be read (corrupt/unsupported) -- listed below, not moved.")
        for p in unreadable[:20]:
            print(f"  [unreadable] {p}")

    if args.dry_run:
        print("\n--dry-run set: not moving or deleting anything. Sample of flagged files:")
        for p, r in flagged[:20]:
            print(f"  {p}  (red_ratio={r['red_ratio']:.2f}, texture={r['texture']:.1f}, reason={r['reason']})")
        return

    if args.delete:
        for p, _ in flagged:
            p.unlink()
        print(f"Deleted {len(flagged)} files directly.")
    else:
        for p, _ in flagged:
            dest = review_dir / p.relative_to(args.dataset_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
        print(f"Moved {len(flagged)} files to {review_dir} for review.")
        print("Once you've checked them, delete the folder to permanently remove them:")
        print(f"  rm -rf \"{review_dir}\"")


if __name__ == "__main__":
    main()