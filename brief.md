# Project 10 - Severstal Steel Defect Detection

**Track:** Computer Vision / MLE - **Difficulty:** 8/10 - **Status:** Phase 1 scaffolded (code-only)

## Goal

Build a multi-class semantic segmentation pipeline that detects and localises four types of surface defects on hot-rolled steel sheets from production-line camera images. The output is a per-pixel mask, four binary channels, one per defect class. The downstream business value is automated quality control (QC) on the rolling-mill inspection line: defects flagged early prevent downstream scrap, customer rejects, and warranty claims, and the per-pixel mask supports root-cause attribution back to a specific roller, lubricant batch, or thermal-cycle event.

## Problem framing

| Item | Value |
|------|-------|
| Task | Multi-class semantic segmentation |
| Input | Grayscale steel-sheet image, 256 x 1600 px |
| Output | 4-channel binary mask (defect classes 1, 2, 3, 4) |
| Loss | Dice + BCE (baseline), focal Tversky (advanced) |
| Headline metric | Dice coefficient (per-class and mean) |
| Secondary metrics | IoU, pixel accuracy, per-class precision/recall |
| Class imbalance | Severe: ~85% of pixels are background, class 3 (scratches) is far more common than class 1 |

## Domain context (DACH industrial Mittelstand)

Surface inspection on steel rolling mills has been a textbook computer-vision problem since the early 2000s. Traditional systems used rule-based morphology and SVM classifiers; modern installations (Salzgitter, ThyssenKrupp, voestalpine, ArcelorMittal Eisenhüttenstadt) increasingly run deep encoder-decoder networks. The Severstal Kaggle competition (2019) released the first large public benchmark in this space, 12,568 labelled images covering pitting, scaling, scratches, and inclusion defects. The competition closed but the dataset remains widely used as the standard benchmark for industrial-CV defect-segmentation work, including in DACH-region predictive-QC tooling.

## Datasets

| Source | Scope | Size | Use |
|--------|-------|------|-----|
| [Severstal Steel Defect Detection (Kaggle)](https://www.kaggle.com/competitions/severstal-steel-defect-detection) | Hot-rolled steel sheet images, 4 defect classes, RLE masks | ~2 GB | Primary dataset, document only (Phase 1) |

Dataset is behind Kaggle competition acceptance, file footprint is ~2 GB. Phase 1 documents the `kaggle` CLI command in `data/README.md`; download deferred to main-session execution after Kaggle terms are accepted.

## Models

- **Baseline (`src/model_baseline.py`):** U-Net with ImageNet-pretrained ResNet34 encoder, four-class output head, Dice + BCE loss. PyTorch + segmentation_models_pytorch.
- **Advanced (`src/model_advanced.py`):** SegFormer (mit-b3 backbone) trained from HuggingFace `nvidia/mit-b3` ImageNet-pretrained weights, focal Tversky loss, mixed-precision (AMP), cosine LR schedule with warm restarts. Optional fallback: DeepLabV3+ with EfficientNet-B4 backbone.

## Deliverables (Liora full format, Phase 1 scaffold)

- [x] `brief.md` - this file
- [x] `data/README.md` - Kaggle CLI command, expected file layout, license note
- [x] `notebooks/01_EDA.ipynb` - raw notebook (NOT executed; main-session run)
- [x] `reports/references.md` - 33 verified academic references
- [x] `src/model_baseline.py` - U-Net + ResNet34, runnable, NOT executed
- [x] `src/model_advanced.py` - SegFormer mit-b3, runnable, NOT executed
- [x] `manuscripts/manuscript.md` - IMRaD draft, results placeholders pending model run
- [x] `deliverables/presentation.html` - 10-slide self-contained HTML
- [x] `checkpoint.json` - Phase 1 status JSON

## Open questions (carry into Phase 2)

1. Does the SegFormer mit-b3 advanced model close the 5-10 Dice-point gap to the Kaggle leaderboard ensemble (private test mean Dice 0.91+)?
2. Is the focal Tversky loss the right choice over generalised Dice + BCE for this specific class imbalance, or does it overweight rare-class false positives on production images?
3. Can a 256 x 800 sliding-window inference pipeline meet the 100 ms latency budget of a typical mill QC line, or is mit-b0 (smaller variant) needed?
4. Cross-domain transfer: how much fine-tuning is required before the Severstal-trained model reaches usable Dice on a different-mill image stream (e.g. cold-rolled vs hot-rolled, different illumination)? This is the realistic deployment question for DACH Mittelstand customers.
