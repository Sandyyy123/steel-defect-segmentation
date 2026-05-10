![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Segmentation](https://img.shields.io/badge/CV-segmentation-red) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

# Severstal Steel Defect Detection — Semantic Segmentation

Pixel-level semantic segmentation of 4 steel surface defect classes on Severstal manufacturing data using U-Net and EfficientNet encoder.

---

## Task

**Semantic Segmentation (Computer Vision)**

---

## Architecture

```
Steel Strip Images → Augmentation → U-Net (EfficientNet encoder) → 4-class Mask → Dice Score
```

---

## Key Features

- 4-class pixel-level defect segmentation (Classes 1-4)
- U-Net with EfficientNet-B4 encoder (ImageNet pretrained)
- Heavy augmentation pipeline (albumentations)
- Dice loss + BCE combined training objective
- Mean Dice score evaluation per defect class

---

## Dataset

[Severstal: Steel Defect Detection (Kaggle)](https://www.kaggle.com/competitions/severstal-steel-defect-detection)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/steel-defect-segmentation.git
cd steel-defect-segmentation
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`PyTorch · segmentation_models_pytorch · albumentations · OpenCV`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
