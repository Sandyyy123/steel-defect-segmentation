# Severstal Steel Defect Detection - Data

## Source

Kaggle competition: [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection)

Released by Severstal (Russian steelmaker) in collaboration with the Kaggle research team in 2019. The competition closed but the dataset remains the standard public benchmark for multi-class steel-surface defect segmentation.

## Why no v1.0 download

Total payload is approximately 2 GB and the dataset is behind Kaggle competition rules (you must accept the competition terms once via the Kaggle web UI before the CLI can pull the files). Per the Project layout

## Download command

```bash
# 1. One-time: accept the Severstal competition terms via the Kaggle web UI
#    https://www.kaggle.com/competitions/severstal-steel-defect-detection/rules
#    (cannot be done from CLI)

# 2. Download (run from this folder)
cd data/
kaggle competitions download -c severstal-steel-defect-detection

# 3. Extract
unzip severstal-steel-defect-detection.zip
# Produces: train_images/, test_images/, train.csv, sample_submission.csv

# 4. Optional: clean up the zip after extraction
rm severstal-steel-defect-detection.zip
```

Kaggle credentials live at `~/.kaggle/kaggle.json` (already present on this machine).

## Expected file layout after extraction

```
data/
├── README.md                            (this file)
├── train.csv                            (per-defect-instance RLE annotations)
├── sample_submission.csv                (test split image IDs, empty masks)
├── train_images/                        (~12,568 images, 256 x 1600 grayscale JPG)
│   ├── 0002cc93b.jpg
│   ├── 00031f466.jpg
│   └── ...
└── test_images/                         (~5,506 images, no labels)
    ├── 004f40c73.jpg
    └── ...
```

## Annotation format (`train.csv`)

| Column | Type | Notes |
|--------|------|-------|
| `ImageId_ClassId` | string | Format `<image_id>.jpg_<class>` where class in {1, 2, 3, 4} |
| `EncodedPixels` | string | Run-length-encoded mask, space-separated `start length` pairs over column-major flattened pixels. Empty when no defect of that class is present. |

Each image has up to four rows in the CSV (one per defect class). An image with no defects has all four rows present with empty `EncodedPixels`.

## Defect class semantics

The Severstal organisers did not publish the per-class defect names in the competition data. Community consensus from inspection of the masks aligns the four classes loosely with: pitting, slivers/inclusion, scratches, scaling. The exact engineering interpretation is not required for the modelling task: classes are treated as labelled categories.

## Class imbalance summary (from train.csv, prior literature analyses)

- Class 3 ("scratches"): ~50% of all defect pixels, present on ~5,000 images
- Class 4: ~25% of defect pixels, ~750 images
- Class 1: ~17% of defect pixels, ~900 images
- Class 2: ~8% of defect pixels, ~250 images
- Background dominates: ~85% of pixels are non-defect across the corpus

This severe class imbalance motivates the focal Tversky loss in the advanced model. The baseline uses Dice + BCE which gives reasonable performance on the abundant class 3 but tends to under-predict class 2.

## Image properties

- All training and test images: 256 x 1600 px, single grayscale channel (some files have 3-channel RGB stored but with R=G=B), JPEG.
- Sheet orientation is left-to-right along the rolling direction.
- Defects are typically thin elongated artefacts spanning hundreds of pixels along the roll axis but a few dozen pixels across.

## License and attribution

The Severstal competition data is released under the [Kaggle competition rules](https://www.kaggle.com/competitions/severstal-steel-defect-detection/rules), which permit non-commercial research use. Cite the competition page when publishing results derived from this dataset.

## Provenance check after download

After unzipping, sanity check:

```bash
ls data/train_images/ | wc -l       # expect ~12,568
ls data/test_images/  | wc -l       # expect ~5,506
wc -l data/train.csv                # expect 50,273 (4 rows per train image, plus header)
```
