# Domain-Robust Polyp Detection — Cross-Domain Generalization Benchmark

This document is the full specification for the project. Read it in its entirety before
writing or editing any code. It captures the research hypothesis, the architecture,
the data pipeline already built, the infrastructure constraints, and the exact file
structure to follow. Every design decision should be traceable back to one of the two
stated goals below — if a proposed change doesn't serve either goal, flag it explicitly
rather than adding it silently.

---

## 1. Core Hypothesis

Most of the "different hospital" / "different scanner" performance drop seen in polyp
segmentation models is caused by **color and illumination differences between
endoscopy scanners**, not by real morphological differences in polyps across
populations. This is a domain-shift problem, not a fundamentally harder
segmentation problem.

The project tests this by building a model with two components, each traceable to a
specific goal:

1. **A cheap, non-learned (or very light) frontend color/illumination normalizer**
   that maps an incoming image's color/brightness statistics toward the training
   distribution *before* it reaches any learned weights.
   → Goal: directly operationalizes and tests the domain-shift hypothesis. Must be
   kept isolatable/toggleable so its contribution can be measured in isolation
   (see Ablation, §7).

2. **Asymmetric compression**: keep the first and last layers full-capacity (they do
   edge/color detection and the final decision), and compress only the middle layers
   (generic feature combination).
   → Goal: reduces compute (params/FLOPs/inference time) without sacrificing the
   layers that matter most for cross-domain robustness.

Every architectural choice must be justified against goal (1) domain-shift reduction
or goal (2) compute reduction. If a choice serves neither, call that out explicitly
instead of quietly including it.

**Open question, resolved by assumption (do not re-litigate without cause):**
"Detection" in this project means **pixel-wise segmentation** (mask output), not
bounding-box detection. This determines the final layer design, loss function
(Dice/IoU-based), and which baselines are comparable.

---

## 2. Data

| Dataset | Role |
|---|---|
| **Kvasir-SEG** | Training/tuning domain. All models are trained and hyperparameter-tuned **only** on this dataset. |
| **Kvasir-SEG** (held-out split) | In-domain baseline for the "drop" metric. |
| **CVC-ClinicDB** | External test-only set. Zero fine-tuning. |
| **ETIS-LaribPolypDB** | External test-only set. Zero fine-tuning. |
| **Video-derived dataset** | External test-only set. Frames organized into positive/negative case folders (~500 frames/case). Treated as test-only to preserve a clean "trained only on Kvasir-SEG" protocol. Ground-truth mask availability for this set is still unconfirmed — affects which metrics (Dice/IoU vs. classification-only) are computable on it. |

**Mandatory rule — case-level splitting:** Frames within a case/video are
near-duplicates. Random frame-level splitting causes data leakage that would
contaminate evidence for the domain-shift hypothesis. All splits (train/val, and any
splits within the video dataset) must be done at the case/video level, never at the
frame level.

**Mask handling:** segmentation masks are resized using **nearest-neighbor
interpolation only** (never bilinear/bicubic) to preserve binary label integrity.

**Architecture clarification:** color images are always the model input. Ground-truth
masks are training labels for loss computation only and are never fed into the
network as input.

---

## 3. Preprocessing Pipeline (Stage 1 — already built)

A three-stage, offline, one-time preprocessing pass (per-image cost is CPU-heavy, so
this is not done as online augmentation):

1. **FOV detection/cropping** — removes scanner-specific black borders/UI elements.
2. **Specular highlight inpainting** (`cv2.inpaint`) — removes bright specular
   reflections. **Must run before color normalization** — reversing this order biases
   the illuminant estimate used in step 3.
3. **Shades-of-Gray color constancy normalization** (Minkowski p=6) — the direct
   operationalization of the domain-shift hypothesis (§1). This is the frontend
   normalizer's statistical basis.

Implemented across:
- `preprocessing.py` — core transform functions (FOV crop, inpainting, color
  normalization).
- `dataset.py` — PyTorch Dataset classes wrapping the preprocessing.
- `run_preprocessing.py` — CLI batch script to run the offline pass over a dataset.

**Design principle:** the frontend normalizer must live in its own file/module,
separate from model weights, so it can be toggled on/off cleanly as an ablation
variable (§7) without touching any learned parameters.

### Video dataset deduplication (already built)

`prepare_video_dataset.py` — complete script handling:
- Natural-sort frame ordering (so frame_2.png sorts before frame_10.png).
- Temporal subsampling (reduce near-duplicate frame density).
- Perceptual-hash deduplication (`imagehash.phash`, Hamming-distance threshold).
- **Case-level** stratified splitting (never frame-level — see §2).
- CSV manifest output describing the resulting split.
- Two modes: `train_val` and `external_test`.

### Baseline registry (already built)

`baseline_registry.py` — catalogs candidate polyp-segmentation papers/models across
design-philosophy categories, with reproducibility flags (code available, weights
available, etc.). Used to select the baselines in §4.

---

## 4. Baselines

Select **3–5 baselines** that are:
(a) strong / competitive on polyp segmentation benchmarks,
(b) have available code and/or pretrained weights (reproducibility matters more than
    marginal performance gains, given limited compute),
(c) span different design philosophies — not just UNet variants. At minimum: one
    UNet-family model, one transformer-based model, and one other lightweight model,
    so the comparison isn't "my model vs. UNet variants."

All baselines are trained through the **same** `training/train.py` +
`training/hp_search.py` path, with the same hyperparameter search budget and search
space, so comparisons are fair. No baseline gets bespoke tuning the others don't get.

---

## 5. Training Protocol

- Train **only** on Kvasir-SEG, with a proper (case/image-level, non-leaking)
  train/val split.
- Every model (proposed + all baselines) goes through an identical HP search
  process, same budget, same search space — see `training/hp_search.py`.
- No model is ever fine-tuned on CVC-ClinicDB, ETIS-LaribPolypDB, or the video
  dataset. Those are zero-fine-tuning, test-only.

---

## 6. Cross-Domain Evaluation

Evaluate all trained models (proposed + baselines) on:
- Kvasir-SEG held-out test split (in-domain baseline number).
- CVC-ClinicDB (external).
- ETIS-LaribPolypDB (external).
- Video-derived dataset (external; metrics depend on mask availability, see §2).

Report per-dataset **and** aggregate:
- Dice, IoU, precision, recall.
- Params, FLOPs, inference time (the efficiency angle).

**The key result is not the absolute external-domain numbers — it's the
performance DROP from in-domain (Kvasir-SEG) to each external set.** Report the
drop explicitly, per dataset and aggregated, for every model. This is the number
that supports or refutes the core hypothesis in §1.

---

## 7. Ablation Study

Isolate, independently, with everything else held fixed:
(a) **Frontend color/illumination normalizer: on vs. off.**
(b) **Asymmetric vs. uniform compression** (same total compression ratio, different
    placement).

Goal: quantify how much of the measured domain robustness (§6) comes from each of
the two architectural pillars in §1. This is the primary evidence for the paper's
contribution — the ablation code path must reuse the same eval pipeline as §6 so
numbers are directly comparable.

---

## 8. Explainability

Use Grad-CAM (or a comparable saliency method) to check whether the model's spatial
attention shifts inappropriately when moving from in-domain (Kvasir-SEG) to
external-domain images, and whether enabling the frontend normalizer (§7a) reduces
that shift. Tie findings back explicitly to the color/illumination hypothesis in
§1 — this is not generic saliency commentary, it's a targeted test of whether
normalization stabilizes attention across domains.

---

## 9. Format / Output Requirements

- Runnable PyTorch code, not pseudocode, unless pseudocode is explicitly requested.
- Flag any step where compute cost could be a problem given the infrastructure
  constraints in §10.
- Where something is ambiguous, state the assumption explicitly and proceed, rather
  than stopping to ask — unless the ambiguity would change the experimental
  protocol itself (e.g., segmentation vs. detection, which has already been
  resolved in §1).

---

## 10. Infrastructure

- **Compute:** University of Manchester HPC.
  - **CSF4 (ALAN) has no GPUs** — it's CPU-only (Cascade Lake nodes), intended for
    multi-node parallel CPU jobs. Do not target CSF4 for training.
  - **CSF3 has the GPUs** (Nvidia V100 and A100) and has migrated to **SLURM**
    (`sbatch`/`squeue`/`scancel`), same as CSF4. Point all GPU job scripts at CSF3's
    login node, using its GPU partition.
  - Exact GPU partition name should be confirmed on the login node via `sinfo` —
    documentation across UoM's site has shown inconsistent partition names, so
    treat `sinfo` output as ground truth over any hardcoded partition name.
- **Environment:** conda, **not yet set up**. On CSF, install packages into a
  self-contained conda environment in the home directory (central Anaconda install
  should not be modified directly). Environment activation inside jobscripts should
  use `source activate <env_name>` (not `conda activate`, which only works
  reliably on the login node).
- **Batch system:** SLURM (`#SBATCH` directives, submitted via `sbatch`).

---

## 11. Key Learnings & Non-Negotiable Principles

- **Case-level splitting is mandatory everywhere** (§2) — this is the single most
  important thing to get right; violating it invalidates the domain-shift evidence.
- **Preprocessing order is fixed:** specular inpainting before color normalization
  (§3), never the reverse.
- **Frontend normalizer stays isolated** in its own module so it's a clean on/off
  ablation switch (§3, §7).
- **Masks always use nearest-neighbor interpolation**, never bilinear/bicubic (§2).
- **CSF3 = GPUs, CSF4/ALAN = no GPUs.** This is an easy mistake to make given the
  naming, and has already been corrected once in this project — do not target CSF4
  for any GPU work.

---

## 12. Current Project State

**Done (Stage 1 — Data Preprocessing):**
- `preprocessing.py`, `dataset.py`, `run_preprocessing.py` — full 3-stage offline
  preprocessing pipeline (§3).
- `prepare_video_dataset.py` — video dataset dedup/split pipeline (§3).
- `baseline_registry.py` — candidate baseline catalog (§3, feeds into §4).

**Not yet done:**
- Stage 2: baseline selection (finalize the 3–5 from the registry) and
  implementation.
- Environment setup on CSF3 (conda env creation, confirming GPU partition name via
  `sinfo`).
- Confirming ground-truth mask availability for the video dataset (affects which
  metrics are computable on it — §2, §6).
- The proposed architecture itself (§1) — frontend normalizer module exists from
  preprocessing but the asymmetric-compression network has not been built.
- Training loop, HP search, baseline training runs (§5).
- Cross-domain evaluation pipeline (§6).
- Ablation pipeline (§7).
- Grad-CAM / explainability pipeline (§8).
- SLURM jobscripts for CSF3 GPU submission (§10).

---

## 13. Project File Structure

```
polyp-domain-robust/
├── PROJECT_BRIEF.md             # this file
├── configs/
│   └── config.yaml              # dataset paths, hyperparams, model list, seeds
│
├── data/
│   ├── datasets.py              # Dataset classes for Kvasir-SEG, CVC-ClinicDB, ETIS-Larib
│   ├── preprocessing.py         # [BUILT] FOV crop, specular inpaint, Shades-of-Gray norm
│   ├── dataset.py               # [BUILT] PyTorch Dataset wrappers around preprocessing
│   ├── run_preprocessing.py     # [BUILT] CLI batch script for offline preprocessing pass
│   └── prepare_video_dataset.py # [BUILT] video frame dedup + case-level split + CSV manifest
│
├── frontend/
│   └── color_normalizer.py      # non-learned color/illum normalizer module (§1, §3, §7a)
│                                 # must be a standalone, toggleable module
│
├── models/
│   ├── proposed_model.py        # frontend normalizer + asymmetric compression net (§1)
│   ├── baseline_registry.py     # [BUILT] catalog of candidate baseline papers/models
│   ├── baselines/
│   │   ├── unet.py
│   │   ├── resunet.py
│   │   ├── transformer_baseline.py   # e.g. lightweight ViT/Segformer-style
│   │   └── lightweight_baseline.py   # e.g. ENet/MobileUNet-style
│   └── compression.py           # shared asymmetric-vs-uniform compression utilities (§7b)
│
├── training/
│   ├── train.py                 # single training run given a model + config (§5)
│   └── hp_search.py             # shared HP search loop, same budget for all models (§5)
│
├── evaluation/
│   ├── metrics.py                # Dice, IoU, precision, recall, params/FLOPs/inference time
│   ├── cross_domain_eval.py      # runs all trained models on all external test sets,
│   │                              # computes and reports in-domain→external DROP (§6)
│   └── ablation.py               # frontend on/off × asymmetric/uniform compression grid (§7)
│
├── explainability/
│   └── gradcam.py                # Grad-CAM + cross-domain attention-shift comparison (§8)
│
├── scripts/
│   ├── run_train_all.sh          # trains proposed model + all baselines, same protocol
│   ├── run_eval_all.sh           # runs cross_domain_eval.py for every checkpoint
│   ├── run_ablation.sh
│   └── slurm/
│       ├── env_setup.sh          # one-time conda env creation on CSF3 login node
│       ├── train_job.sbatch      # CSF3 GPU SLURM jobscript template (§10)
│       ├── eval_job.sbatch
│       └── gradcam_job.sbatch
│
├── results/                      # auto-populated: metrics CSVs, Grad-CAM overlays, logs
└── checkpoints/                  # auto-populated: trained model weights
```

**Structural rules to preserve when adding code:**
- `frontend/color_normalizer.py` stays separate from `models/` — it must be
  independently toggleable for §7a.
- `models/compression.py` stays separate from individual architecture files — it's
  a swappable strategy applied to a base network, needed cleanly for §7b.
- `evaluation/` stays decoupled from `training/` — evaluation is re-run often
  without retraining; don't couple them or recomputation becomes wasteful.
- `models/baselines/` all go through the same `training/train.py` +
  `training/hp_search.py` path — no baseline-specific training scripts.
