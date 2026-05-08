# Validation Report - Project #10 Severstal Steel Defect Detection

**Role:** A (VALIDATOR)
**Date:** 2026-05-08
**Project folder:** `/root/AI/liora_projects/10_severstal_steel/`
**Phase:** Scaffold-only (Phase 1)

## Overall verdict: PASS-WITH-WARNINGS

The scaffold is structurally sound. All scripts parse, the notebook is valid JSON, the manuscript is within the target word range, IMRaD sections are complete, every numbered inline citation maps to a reference entry, all 5 sampled DOIs resolve live on CrossRef with matching titles, no em-dashes anywhere, no AI-tell phrases. Two minor warnings: (1) `presentation.html` contains 6 external `href` links inside slide body text (DOI / Kaggle URLs, not CSS or script resources), so the deck is functionally self-contained but not strictly link-free; (2) the project is scaffold-only so no saved model artefacts (.pt, .png, metrics JSON) exist under `deliverables/` yet, which is expected.

---

## Task-by-task findings

### 1. Notebook validity
- [PASS] `notebooks/01_EDA.ipynb` parses as valid JSON via `json.load`.

### 2. Python script syntax
- [PASS] `src/model_baseline.py` parses cleanly via `ast.parse`.
- [PASS] `src/model_advanced.py` parses cleanly via `ast.parse`.

### 3. Manuscript word count
- [PASS] `wc -w manuscripts/manuscript.md` = 4139. Target band 4000-5000. Inside band.

### 4. Self-contained HTML
- [WARN] `grep -E 'href="http|src="http' deliverables/presentation.html` returns 6 hits.
  - All 6 are `<a href="...">` text hyperlinks inside slide content (Kaggle competition page + 5 reference DOI/arXiv links).
  - Zero external `<link rel="stylesheet">` and zero external `<script src=...>` tags. CSS is one inline `<style>` block.
  - Functional offline use is preserved (deck renders without network); strict zero-link policy is not met. Treating as WARN, not FAIL, because no external assets are loaded at render time.

### 5. IMRaD completeness
- [PASS] All required sections present in `manuscripts/manuscript.md`:
  - Title (heading line 1)
  - Abstract
  - Introduction (Section 1)
  - Methods (Section 2)
  - Results (Section 3)
  - Discussion (Section 4)
  - Conclusion (Section 5)
  - References (Section "References" pointing to `reports/references.md`)

### 6. Method drift (manuscript Methods vs src files)
- [PASS] Methods named in Section 2 of manuscript and confirmed in `src/`:
  - U-Net + ResNet-34 encoder, multi-label sigmoid head -> `model_baseline.py` (`smp.Unet(encoder_name="resnet34", ...)`)
  - Dice + BCE loss -> `model_baseline.py` `DiceBCELoss`
  - SegFormer mit-b3 (HuggingFace `nvidia/mit-b3`) -> `model_advanced.py` `SegformerWrapper`
  - Focal Tversky loss with alpha=0.7, beta=0.3, gamma=0.75 -> `model_advanced.py` `FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)`
  - DeepLabV3+ with EfficientNet-B4 fallback -> `model_advanced.py` `smp.DeepLabV3Plus(encoder_name="efficientnet-b4", ...)`
  - AdamW optimiser, cosine schedule, cosine-warm-restarts (T_0=5, T_mult=2), AMP mixed precision -> all imported and used.
  - No method named in the manuscript is missing from the scripts.

### 7. Citation drift (inline cites vs `reports/references.md`)
- [PASS] Manuscript uses numeric `[N]` citations. Unique inline numbers: {1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33}.
- All 27 unique cited numbers map to existing entries in `reports/references.md` (which has 34 numbered entries).
- Refs not yet cited in manuscript prose (allowed): 6, 7, 15, 16, 25, 28, 34. No orphan citations.

### 8. CrossRef live re-verification (5 random refs)
Sampled refs 2, 4, 13, 19, 29 against `https://api.crossref.org/works/{doi}`:
- [PASS] Ref 2 DOI `10.1007/978-3-319-24574-4_28` -> HTTP 200, title "U-Net: Convolutional Networks for Biomedical Image Segmentation" (matches).
- [PASS] Ref 4 DOI `10.1007/978-3-030-01234-2_49` -> HTTP 200, title "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (matches DeepLabV3+).
- [PASS] Ref 13 DOI `10.1109/CVPR.2016.90` -> HTTP 200, title "Deep Residual Learning for Image Recognition" (matches ResNet).
- [PASS] Ref 19 DOI `10.1109/ICCV.2017.324` -> HTTP 200, title "Focal Loss for Dense Object Detection" (matches).
- [WARN] Ref 29 DOI `10.3390/s26072254` -> HTTP 200, but live CrossRef title is "Defect-Mask2Former: An Improved Semantic Segmentation Model for Precise Small-Sized Defect Detection on Large-Sized Timbers" (timber domain, not steel). The local references.md and manuscript drop the trailing "on Large-Sized Timbers" qualifier and present this as a steel-defect comparator. The DOI is real and the architecture is generic, but the original paper's experimental scope is timber. Suggest adding a "(timber domain)" qualifier in the references entry to avoid implying steel-specific evaluation.

### 9. Em-dash scan
- [PASS] `python3` count of U+2014 across all 7 files: 0.

### 10. AI-tell scan
- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude' .` returns 0 hits.

### 11. Checkpoint schema
- [PASS] `checkpoint.json` keys: `['project_number', 'title', 'methodology', 'phase', 'status', 'needs_main_session_execution', 'blockers']`.
- All four required fields present: `project_number=10`, `title="Severstal Steel Defect Detection"`, `methodology="CV semantic segmentation (industrial QC, multi-class defect masks)"`, `status` (object with sub-flags). Note: top-level `phase` is used instead of a top-level `status` string; the `status` field is an object of completion booleans, which is acceptable but worth flagging. Required-field coverage is satisfied.

---

## Summary table

| # | Task | Result |
|---|------|--------|
| 1 | Notebook valid JSON | PASS |
| 2 | Script syntax (baseline + advanced) | PASS |
| 3 | Manuscript word count (4139, target 4000-5000) | PASS |
| 4 | Self-contained HTML (6 text hyperlinks present) | WARN |
| 5 | IMRaD completeness | PASS |
| 6 | Method drift | PASS |
| 7 | Citation drift | PASS |
| 8 | 5-ref CrossRef re-verification | PASS (Ref 29 domain caveat -> WARN) |
| 9 | Em-dash scan | PASS |
| 10 | AI-tell scan | PASS |
| 11 | Checkpoint schema | PASS |

## Top 3 findings
1. Scaffold is structurally clean: zero em-dashes, zero AI-tell phrases, no syntax errors, no citation orphans, all sampled DOIs live.
2. `deliverables/presentation.html` has 6 in-body hyperlinks (Kaggle URL + 5 DOI links). No external CSS or script tags, so the deck still renders fully offline. Recommend converting these to plain text DOI strings if strict zero-link compliance is required.
3. Reference 29 (Defect-Mask2Former) resolves correctly but the live CrossRef title indicates a timber-defect (not steel) experimental scope. Adding a "(timber domain comparator)" qualifier in `reports/references.md` would prevent overstated claims about direct steel applicability.

## Blockers
None. All checks executed successfully; CrossRef API responded for all 5 sampled DOIs.

Role A complete.
