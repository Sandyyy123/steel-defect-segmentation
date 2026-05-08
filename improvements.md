# Improvements - Project #10 Severstal Steel Defect Detection (IMPROVER)

Reviewer role: B (IMPROVER). Scope: scaffold-only Phase 1 (no executed metrics yet). All recommendations are non-modifying.

## Top recommendation

**Add a coil-aware, iterative-multilabel-stratified 5-fold cross-validation harness as the default training protocol, replacing the single 70/15/15 image-level split.** The brief and manuscript both flag image-level splits as a known leakage vector, and a single fold provides no confidence interval on per-class Dice (which is the headline metric clients will read). Concrete next step: add `iterstrat.MultilabelStratifiedKFold(n_splits=5)` keyed on the four `class_present` indicators, expose `--fold {0..4}` in both training scripts, persist fold-level metrics to `deliverables/cv_metrics.json`, and report mean +/- SD per-class Dice in the manuscript Results section. This single change converts the benchmark from "one number, one seed" to a defensible client-facing comparison and unlocks paired statistical tests (Wilcoxon signed-rank across folds) for the U-Net vs SegFormer claim.

## Weaknesses and actionable improvements

### 1. Single-fold split with no statistical test [HIGH]
Manuscript Section 4.4 already lists this as a limitation, but it is also fixable in a few lines. Without folds there is no way to attach a CI to "SegFormer beats U-Net by X Dice points", and a 1-3 point mean-Dice gap is well within single-fold noise on Severstal. Action: implement the 5-fold harness in the Top recommendation, then add a Wilcoxon signed-rank test on per-fold mean Dice and report `p` and a 95% bootstrap CI on the per-class delta.

### 2. Severstal-known label noise is not modeled [HIGH]
The Severstal training masks are well documented in the Kaggle community as containing ~5-10% mask-noise (mislabelled or partially-labelled instances), and the leaderboard winners explicitly used label-smoothing and pseudo-labelling on the test set to recover it. The current pipeline ignores this. Action: add (a) BCE label-smoothing of 0.05 on the mask channel inside `DiceBCELoss`, (b) an optional pseudo-label round on `test_images/` after first-pass training, gated by a `--pseudo-labels` flag, and (c) a small-mask-area filter (drop predicted components below ~150 pixels at inference, matching the winning Kaggle post-processing).

### 3. No per-class threshold tuning or calibration [HIGH]
A multi-label sigmoid head with a hardcoded 0.5 cutoff systematically loses 1-3 points of mean Dice on Severstal. The Discussion mentions calibration but the code does not run it. Action: after validation, sweep per-class thresholds in `[0.3, 0.7]` step 0.02 on the validation fold, store the four optima in `deliverables/{baseline,advanced}_thresholds.json`, and use them at test time. Add Platt scaling (`sklearn.linear_model.LogisticRegression` on per-pixel probabilities) on a held-out fold for downstream triage scores. Add a reliability diagram to the deliverables.

### 4. Missing reproducibility and environment pinning [HIGH]
There is no `requirements.txt`, no `environment.yml`, and no recorded library versions; `seed` is set in PyTorch and NumPy but not in CUDA (`torch.use_deterministic_algorithms`, `cudnn.benchmark=False`), albumentations, or Python `random`. AMP further breaks bitwise determinism (manuscript notes this but does not act on it). Action: add `requirements.txt` pinning torch/timm/segmentation-models-pytorch/transformers/albumentations versions, add a `set_global_seed()` helper that seeds Python `random`, NumPy, PyTorch CPU + CUDA, and albumentations, and run the advanced model across `seed in {42, 1337, 2024}` reporting mean +/- SD as the headline number.

### 5. Loss-function bake-off is asserted but not run [MEDIUM]
The manuscript's central claim is that focal Tversky beats Dice + BCE on rare classes, but the scripts hardcode one loss per model. A reader cannot disentangle "loss helped" from "architecture helped". Action: add a `--loss {dice_bce, dice_focal, focal_tversky, generalised_dice, dilated_balanced_ce}` flag wired to a small loss registry, then run a 4x2 grid (4 losses x 2 architectures) on a single fold and add a results table to Section 3.3. Generalised Dice [Sudre 2017, ref 21] and Dilated Balanced CE [Hosseini 2026, ref 22] are already in the reference list; wiring them in costs ~30 lines.

### 6. Anisotropic defect geometry is not exploited [MEDIUM]
The Introduction emphasises that scratches and slivers are 100s of pixels along the rolling axis but tens of pixels across, yet the augmentation pipeline treats horizontal and vertical symmetrically (HFlip + VFlip both at p=0.5) and the architecture uses square receptive fields. Action: (a) drop VFlip to p=0.0 because the rolling-direction axis is not symmetric (lubricant flow direction breaks it), (b) add `A.ShiftScaleRotate(rotate_limit=0, scale_limit=(-0.1, 0.0), shift_limit=0.05)` for axis-only jitter, (c) try a `mit-b3` variant with anisotropic patch sizes (e.g. 4x16 instead of 4x4 in the first stem) or use `timm`'s `convnextv2_tiny` with a U-Net++ decoder which empirically handles thin-elongated masks better at similar parameter cost.

### 7. No latency or memory benchmark, blocking deployment claims [MEDIUM]
The Discussion correctly flags the 50-100 ms QC latency budget as the deployment-relevant constraint, but the deliverables contain zero latency numbers, FLOPs, or parameter counts. A client reading the slide deck cannot decide between mit-b3 and the EfficientNet-B4 fallback. Action: add a `bench.py` script that runs `torch.profiler` over 100 forward passes at batch size 1 on CPU + CUDA + ONNX Runtime, reports p50/p95/p99 latency and peak VRAM for each architecture, and emits `deliverables/latency_benchmark.json` plus a comparison bar chart. Add a row to manuscript Table-1 with these numbers.

### 8. EDA notebook lacks a held-out coil-stratification check and a mask-area sanity audit [MEDIUM]
The EDA notebook scans the first 500 images for shape integrity and computes co-occurrence and area histograms, but does not (a) test the column-major RLE decoder against a few known masks (a frequent silent-bug source), (b) run a duplicate-image check by perceptual hash (Severstal has known near-duplicate pairs that inflate test-set scores when split image-level), or (c) extract any pseudo-coil identifier from filenames or EXIF for stratification. Action: add three new cells: a unit-style RLE round-trip check (encode-decode-compare), an `imagehash.phash` near-duplicate scan with a default threshold of 6 bits, and a perceptual-hash-based coil grouping with cross-fold leakage measurement.

### 9. Presentation is self-contained but business framing is thin [LOW]
The 10-slide deck (266 lines, inline-only) covers method correctly but the Severstal customer for a DACH Mittelstand pitch will care about cost-per-coil-saved, not Dice. Action: add one slide with a back-of-envelope unit economics calculation: `mill_throughput_coils_per_shift x P(defect | coil) x cost_per_missed_defect x P(model_catches | defect)` populated from public industry benchmarks (e.g. Salzgitter annual reports), with the model-recall row sourced from per-class recall in the metrics JSON.

### 10. Ensemble and test-time augmentation absent [LOW]
The Kaggle leaderboard's 0.91 mean-Dice ceiling is from 3-5 model ensembles with HFlip TTA, and the manuscript explicitly acknowledges this as the gap-to-ceiling but does not build any ensembling infrastructure. Action: add a thin `predict_tta.py` that loads N saved checkpoints, predicts with HFlip TTA (mean-of-probabilities), and writes ensemble metrics to `deliverables/ensemble_metrics.json`. This is ~50 lines and lets the manuscript Results section close the gap-to-ceiling argument with a real number.

## Summary

| # | Improvement | Priority |
|---|-------------|----------|
| 1 | 5-fold CV + Wilcoxon test on per-class Dice | HIGH |
| 2 | Label-smoothing, small-component filter, optional pseudo-labels | HIGH |
| 3 | Per-class threshold tuning + Platt calibration | HIGH |
| 4 | requirements.txt + multi-seed determinism harness | HIGH |
| 5 | Loss-function 4x2 ablation grid | MEDIUM |
| 6 | Anisotropic augmentation + thin-mask-friendly architecture | MEDIUM |
| 7 | Latency/FLOPs benchmark per architecture | MEDIUM |
| 8 | EDA: RLE round-trip, perceptual-hash duplicates, coil grouping | MEDIUM |
| 9 | Unit-economics slide for client framing | LOW |
| 10 | TTA + checkpoint-ensemble script | LOW |

All ten changes are additive (new files or new flags); none require modifying existing code. Items 1, 2, 3, 4 together convert the manuscript from "literature-anchored expectations with placeholders" to a defensible single-author benchmark; items 5-7 sharpen the deployment story; items 8-10 polish.
