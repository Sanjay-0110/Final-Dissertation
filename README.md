# Domain-Robust Polyp Detection

Benchmarking and improving cross-domain generalization of AI-assisted polyp detection in colonoscopy. MSc dissertation project (COMP66060).

**Core hypothesis:** cross-domain polyp-segmentation performance drop is caused by scanner color/illumination differences, not real morphological differences.

**Two-pillar architecture:**
1. A non-learned frontend color/illumination normalizer (Shades-of-Gray, Minkowski p=6.0)
2. Asymmetric compression: full-capacity first/last encoder layers, compressed middle layers

See `PROJECT_BRIEF.md` for the full spec and `ARCHITECTURE.md` for the architecture writeup.

## Repo structure

```
configs/config.yaml       # all hyperparameters, dataset paths, ablation switches
frontend/                 # non-learned color normalizer (Shades-of-Gray)
data/                     # preprocessing pipeline, PyTorch datasets, video-dataset splitter
models/                   # CompressedEncoder, classifier/segmentation heads, baseline registry
training/                 # training loops (classification + segmentation), loss functions
evaluation/               # cross-domain evaluation (inference only, no training)
scripts/slurm/            # CSF3 SLURM jobscripts (env setup + all training/eval jobs)
checkpoints/               # saved model weights (not committed — see below)
results/                   # training history CSVs, cross-domain eval results
```

## Setup

Requires Python 3.11 and a CUDA-capable GPU for training (CPU works for inference/testing on small batches).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs `torch==2.13.0`, `opencv-python`, `pillow`, `numpy`, `pyyaml`. If the pinned torch version doesn't resolve for your CUDA setup, use the install command from [pytorch.org](https://pytorch.org)'s selector instead.

### Running on CSF3 (University of Manchester HPC)

CSF4/ALAN has **no GPUs** — all training must run on CSF3.

```bash
bash scripts/slurm/env_setup.sh    # one-time, run interactively on the login node
```

Then submit jobs with `sbatch scripts/slurm/<job>.sbatch` — **never** run training directly on a login-node shell (no GPU attached there; you'll get `cudaErrorDevicesUnavailable`). See each `.sbatch` file's header comment for what it does.

## Datasets

Place datasets under `datasets/` (not committed — too large for git):

| Dataset | Expected path | Get it from |
|---|---|---|
| Kvasir-SEG | `datasets/Kvasir-SEG/{images,masks}` | [datasets.simula.no/kvasir-seg](https://datasets.simula.no/kvasir-seg) |
| CVC-ClinicDB | `datasets/CVC-ClinicDB/PNG/{Original,"Ground Truth"}` | [Kaggle: balraj98/cvcclinicdb](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) |
| ETIS-LaribPolypDB | `datasets/ETIS-LaribPolypDB/{images,masks}` | [Kaggle: nguyenvoquocduong/etis-laribpolypdb](https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb) |
| Video dataset (Phase-1 pilot) | `datasets/Positive/case*`, `datasets/negative/case*` | project-specific, no public source I collected fmo my Supervisor|

After placing a dataset, update its `root` path in `configs/config.yaml` under `datasets:`, then sanity-check it loads correctly:

```bash
python -c "
from data.dataset import PolypSegDataset
ds = PolypSegDataset('datasets/Kvasir-SEG/images', 'datasets/Kvasir-SEG/masks')
img, mask = ds[0]
print(len(ds), img.shape, mask.shape, mask.unique())
"
```
Expect image shape `(3, 352, 352)` in `[0,1]`, mask shape `(1, 352, 352)` with only values `{0.0, 1.0}`.

## Quickstart — train and test the segmentation model

```bash
# 1. Train (Kvasir-SEG, ~800 train / 200 val split)
python -m training.train_segmentation --config configs/config.yaml

# Optional flags:
#   --init-encoder-from checkpoints/proposed_classifier_best.pt   (warm-start from the video classifier)
#   --disable-color-normalization                                  (§7a ablation: normalizer off)
#   --compression-mode {asymmetric,uniform}                        (§7b ablation)
#   --tag <name>                                                    (avoid overwriting checkpoints)

# 2. Evaluate on any dataset (in-domain or cross-domain, inference only)
python -m evaluation.evaluate_cross_domain \
    --checkpoint checkpoints/proposed_segmentation_best.pt \
    --dataset kvasir_seg        # or: cvc_clinicdb, etis_larib
```

Results append to `results/proposed_segmentation_history.csv` (per-epoch training curve) and `results/cross_domain_eval.csv` (one row per eval run).

## Other entry points

```bash
# Phase-1 classification pilot (video dataset, no masks — case-present/absent only)
python -m data.prepare_video_dataset --config configs/config.yaml   # build train/test split first
python -m training.train --config configs/config.yaml

# Check model architecture + param counts
python -m models.proposed_model

# Verify compression math against real nn.Module parameter counts
python -m models.compression

# Manually inspect the preprocessing pipeline on a folder of images
python -m data.run_preprocessing --input <dir> --output <dir> --config configs/config.yaml
```

## Config

All hyperparameters live in `configs/config.yaml` — dataset paths, image size, preprocessing thresholds, model architecture (`compression_ratio`, `compression_mode`), and training settings (`segmentation_training`, `training`, `video_split`, `segmentation_split`). Most training/eval scripts accept `--config <path>` to point at an alternate config.