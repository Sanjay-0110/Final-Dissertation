"""End-to-end smoke test: train for one epoch, then evaluate that checkpoint,
using the real command-line entry points on a tiny synthetic dataset."""

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from data.dataset import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_dataset(root: Path, n_images: int) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rng = np.random.default_rng(0)
    for i in range(n_images):
        image = rng.integers(40, 180, size=(120, 160, 3), dtype=np.uint8)
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[30:90, 40:120] = 255
        cv2.imwrite(str(root / "images" / f"{i}.png"), image)
        cv2.imwrite(str(root / "masks" / f"{i}.png"), mask)


@pytest.fixture
def smoke_config(tmp_path) -> Path:
    write_dataset(tmp_path / "kvasir", 5)
    write_dataset(tmp_path / "cvc", 2)

    config = load_config()
    config["datasets"]["kvasir_seg"]["root"] = str(tmp_path / "kvasir")
    config["datasets"]["cvc_clinicdb"].update(root=str(tmp_path / "cvc"), images_dir="images", masks_dir="masks")
    config["segmentation_training"].update(
        batch_size=2,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        results_dir=str(tmp_path / "results"),
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("disable_color_normalization", [False, True])
def test_train_then_evaluate(tmp_path, smoke_config, disable_color_normalization):
    flags = ["--disable-color-normalization"] if disable_color_normalization else []
    run("training.train_segmentation", "--config", str(smoke_config), "--epochs", "1", "--tag", "smoke", *flags)

    best = tmp_path / "checkpoints" / "proposed_segmentation_smoke_best.pt"
    assert best.exists()
    assert (tmp_path / "results" / "proposed_segmentation_smoke_history.csv").exists()

    # The eval step must take the colour setting from the checkpoint, not the
    # config -- this is what keeps the §7a comparison valid.
    checkpoint = torch.load(best, map_location="cpu")
    assert checkpoint["color_normalization"]["enabled"] is (not disable_color_normalization)

    results_csv = tmp_path / "results" / "eval.csv"
    run(
        "evaluation.evaluate_cross_domain", "--config", str(smoke_config), "--checkpoint", str(best),
        "--dataset", "cvc_clinicdb", "--num-workers", "0", "--batch-size", "2", "--results-csv", str(results_csv),
    )
    with results_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["n_samples"] == "2"
    assert 0.0 <= float(rows[0]["dice"]) <= 1.0
