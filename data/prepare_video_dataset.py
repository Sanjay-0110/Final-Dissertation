"""
data/prepare_video_dataset.py
==============================
Case-level train/test split + CSV manifest builder for the video dataset
(datasets/Positive/case*, datasets/negative/case*) -- PROJECT_BRIEF.md §2,
§3, §11.

Mandatory rule (§2, §11): all splits are done at the case/video level, never
frame-level -- frames within a case are near-duplicates, so a frame-level
split would leak near-identical frames across train/test and invalidate the
domain-shift evidence. This script never shuffles individual frames across
the boundary; it only ever assigns whole cases to train or test.

Note on scope: PROJECT_BRIEF.md §3/§12 describe this file as already
including perceptual-hash (imagehash) deduplication. That dependency isn't in
requirements.txt yet and case-level splitting already prevents the severe
leakage risk (identical frames straddling train/test), so phash dedup is
deferred here in favor of cheap, dependency-free temporal stride subsampling
(--frame-stride). Add phash dedup as a fast-follow if per-case frame
redundancy turns out to be an actual training problem.

Usage:
    python -m data.prepare_video_dataset --config configs/config.yaml
    python -m data.prepare_video_dataset --frame-stride 5 --train-ratio 0.8 --seed 42
    python -m data.prepare_video_dataset --previous-manifest data/video_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from data.dataset import collect_images, load_config

_NATURAL_SORT_RE = re.compile(r"(\d+)")


def _natural_sort_key(path: Path) -> list:
    """Splits a filename into text/number chunks so e.g. image2.jpg sorts
    before image10.jpg even without zero-padding. Defensive: the actual
    filenames observed on disk are already zero-padded and sort correctly
    lexicographically, but this doesn't assume that holds for future data."""
    parts = _NATURAL_SORT_RE.split(path.name)
    return [int(p) if p.isdigit() else p for p in parts]


@dataclass
class CaseInfo:
    case_id: str  # e.g. "Positive/case20", "negative/case1" -- disambiguates label collisions
    label: int  # 1 = Positive, 0 = negative
    frame_paths: List[Path]


def discover_cases(
    root: Path,
    positive_dirname: str = "Positive",
    negative_dirname: str = "negative",
) -> List[CaseInfo]:
    """Globs root/positive_dirname/case* (label=1) and
    root/negative_dirname/case* (label=0). Exact folder casing matters --
    confirmed on disk as "Positive" (capital P) and "negative" (lowercase n)."""
    cases: List[CaseInfo] = []
    for dirname, label in [(positive_dirname, 1), (negative_dirname, 0)]:
        class_root = root / dirname
        if not class_root.exists():
            raise FileNotFoundError(f"Expected class folder not found: {class_root}")
        for case_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
            frame_paths = collect_images(case_dir)
            frame_paths = sorted(frame_paths, key=_natural_sort_key)
            cases.append(CaseInfo(case_id=f"{dirname}/{case_dir.name}", label=label, frame_paths=frame_paths))
    return cases


def split_case_ids(
    case_ids: Sequence[str],
    ratio: float,
    seed: int,
    min_test: int = 1,
    min_train: int = 1,
) -> Tuple[List[str], List[str]]:
    """Deterministic (seeded) case-level split. Shuffles a SORTED copy of
    case_ids (so the result doesn't depend on filesystem iteration order),
    then takes n_test = clamp(round(n*(1-ratio)), min_test, n-min_train).

    With few cases, exact `ratio` isn't always achievable -- e.g. 3 cases at
    ratio=0.8 gives round(3*0.2)=1 test case, i.e. 2/1 (~67/33), not exactly
    80/20. This is expected and documented, not an error.
    """
    ids = sorted(case_ids)
    n = len(ids)
    if n < min_test + min_train:
        raise ValueError(
            f"Only {n} case(s) available, need at least min_test({min_test}) + "
            f"min_train({min_train}) = {min_test + min_train} to split."
        )
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_test = round(n * (1 - ratio))
    n_test = max(min_test, min(n_test, n - min_train))

    test_ids = sorted(ids[:n_test])
    train_ids = sorted(ids[n_test:])
    return train_ids, test_ids


def apply_frame_stride(frame_paths: Sequence[Path], stride: int) -> List[Path]:
    """Per-case temporal subsampling: keep every `stride`-th frame. stride=1
    keeps every frame (no subsampling)."""
    if stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {stride}")
    return list(frame_paths[::stride])


def _read_previous_splits(previous_manifest_path: Path) -> Dict[str, str]:
    """Returns {case_id: split} from an existing manifest, so already-split
    cases keep their prior assignment on re-runs (see build_manifest's
    `previous_manifest_path` docstring)."""
    splits: Dict[str, str] = {}
    with previous_manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            splits.setdefault(row["case_id"], row["split"])
    return splits


def build_manifest(
    video_root: Path,
    positive_dirname: str = "Positive",
    negative_dirname: str = "negative",
    train_ratio: float = 0.8,
    seed: int = 42,
    frame_stride: int = 1,
    min_test_cases: int = 1,
    min_train_cases: int = 1,
    previous_manifest_path: Optional[Path] = None,
) -> List[dict]:
    """Builds the full manifest row list. Splitting is done independently per
    class (positive cases split among themselves, negative cases split among
    themselves) so every split has both classes represented.

    Sticky re-run behavior (previous_manifest_path): case IDs already present
    in that manifest keep their prior split -- only newly-discovered case IDs
    (e.g. negative cases you add later) go through split_case_ids among
    themselves. This avoids silently reshuffling old cases (and invalidating
    old checkpoints/eval history) when new data shows up. If there are too
    few new cases in a class to satisfy min_test/min_train, they're all
    assigned to "train" (documented default: growing the training set is the
    safer assumption than forcing a too-small class into a test-only split).
    """
    cases = discover_cases(video_root, positive_dirname, negative_dirname)
    previous_splits = _read_previous_splits(previous_manifest_path) if previous_manifest_path and previous_manifest_path.exists() else {}

    case_split: Dict[str, str] = {}
    for label in (1, 0):
        class_case_ids = [c.case_id for c in cases if c.label == label]
        sticky_ids = [cid for cid in class_case_ids if cid in previous_splits]
        new_ids = [cid for cid in class_case_ids if cid not in previous_splits]

        for cid in sticky_ids:
            case_split[cid] = previous_splits[cid]

        if new_ids:
            if len(new_ids) >= min_test_cases + min_train_cases:
                new_train, new_test = split_case_ids(new_ids, train_ratio, seed, min_test_cases, min_train_cases)
                for cid in new_train:
                    case_split[cid] = "train"
                for cid in new_test:
                    case_split[cid] = "test"
            else:
                for cid in new_ids:
                    case_split[cid] = "train"

    rows: List[dict] = []
    for case in cases:
        split = case_split[case.case_id]
        frames = apply_frame_stride(case.frame_paths, frame_stride)
        for frame_path in frames:
            rows.append({
                "image_path": str(frame_path.resolve()),
                "case_id": case.case_id,
                "label": case.label,
                "split": split,
            })
    return rows


def write_manifest_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "case_id", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: List[dict]) -> None:
    cases_by_split_label: Dict[Tuple[str, int], set] = {}
    frames_by_split_label: Dict[Tuple[str, int], int] = {}
    for row in rows:
        key = (row["split"], row["label"])
        cases_by_split_label.setdefault(key, set()).add(row["case_id"])
        frames_by_split_label[key] = frames_by_split_label.get(key, 0) + 1

    print(f"{'split':<8}{'label':<8}{'cases':<8}{'frames':<10}")
    for split in ("train", "test"):
        for label in (1, 0):
            key = (split, label)
            n_cases = len(cases_by_split_label.get(key, set()))
            n_frames = frames_by_split_label.get(key, 0)
            print(f"{split:<8}{label:<8}{n_cases:<8}{n_frames:<10}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case-level train/test split + manifest for the video dataset")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml (defaults to configs/config.yaml)")
    parser.add_argument("--root", type=Path, default=None, help="Override datasets.video.root from config")
    parser.add_argument("--output", type=Path, default=None, help="Override datasets.video.manifest_path from config")
    parser.add_argument("--train-ratio", type=float, default=None, help="Override video_split.train_ratio from config")
    parser.add_argument("--seed", type=int, default=None, help="Override video_split.seed from config")
    parser.add_argument("--frame-stride", type=int, default=None, help="Override video_split.frame_stride from config")
    parser.add_argument("--previous-manifest", type=Path, default=None,
                         help="Existing manifest CSV whose case-split assignments should be kept (sticky re-run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    video_cfg = config["datasets"]["video"]
    split_cfg = config["video_split"]

    root = args.root or Path(video_cfg["root"])
    output = args.output or Path(video_cfg["manifest_path"])
    train_ratio = args.train_ratio if args.train_ratio is not None else split_cfg["train_ratio"]
    seed = args.seed if args.seed is not None else split_cfg["seed"]
    frame_stride = args.frame_stride if args.frame_stride is not None else split_cfg["frame_stride"]

    rows = build_manifest(
        video_root=root,
        positive_dirname=video_cfg["positive_dirname"],
        negative_dirname=video_cfg["negative_dirname"],
        train_ratio=train_ratio,
        seed=seed,
        frame_stride=frame_stride,
        min_test_cases=split_cfg["min_test_cases"],
        min_train_cases=split_cfg["min_train_cases"],
        previous_manifest_path=args.previous_manifest,
    )
    write_manifest_csv(rows, output)

    print(f"Wrote {len(rows)} row(s) to {output}\n")
    _print_summary(rows)


if __name__ == "__main__":
    main()
