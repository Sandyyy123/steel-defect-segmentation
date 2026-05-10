"""Severstal Steel Defect Detection - Baseline U-Net (ResNet-34 encoder).

Multi-class semantic segmentation, 4 defect channels, sigmoid output, Dice + BCE loss.

Run AFTER `kaggle competitions download -c severstal-steel-defect-detection` is unzipped
into `../data/`. This file is a runnable implementation but is NOT executed during v1.0.

Usage (main session):
    cd .
    python src/model_baseline.py --epochs 20 --batch-size 16

Outputs (saved to ../deliverables/):
    baseline_unet_resnet34.pt          Trained model weights
    baseline_metrics.json              Per-epoch and final test metrics
    baseline_dice_per_class.png        Per-class Dice bar chart
    baseline_sample_predictions.png    Image / mask / prediction triptych
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None  # The main session installs `pip install segmentation-models-pytorch albumentations`

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"

IMG_HEIGHT, IMG_WIDTH = 256, 1600
NUM_CLASSES = 4


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def rle_decode(rle: str, shape: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)) -> np.ndarray:
    """Severstal RLE decoder. Pixels are flattened column-major."""
    if not isinstance(rle, str) or rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)
    parts = rle.split()
    starts = np.array(parts[0::2], dtype=int) - 1
    lengths = np.array(parts[1::2], dtype=int)
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for s, l in zip(starts, lengths):
        flat[s : s + l] = 1
    return flat.reshape(shape[1], shape[0]).T


def load_annotations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[["ImageId", "ClassId"]] = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)
    df["ClassId"] = df["ClassId"].astype(int)
    return df


def build_image_index(df: pd.DataFrame) -> pd.DataFrame:
    """One row per image with four RLE columns."""
    pivot = df.pivot_table(
        index="ImageId",
        columns="ClassId",
        values="EncodedPixels",
        aggfunc="first",
    )
    pivot.columns = [f"rle_{c}" for c in pivot.columns]
    return pivot.reset_index()


class SeverstalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, img_dir: Path, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img_path = self.img_dir / row["ImageId"]
        image = np.array(Image.open(img_path).convert("RGB"))
        masks = np.stack(
            [rle_decode(row[f"rle_{c}"]) for c in range(1, NUM_CLASSES + 1)], axis=-1
        ).astype(np.float32)

        if self.transform is not None:
            t = self.transform(image=image, mask=masks)
            image, masks = t["image"], t["mask"]
            masks = masks.permute(2, 0, 1) if masks.ndim == 3 else masks
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            masks = torch.from_numpy(masks.transpose(2, 0, 1)).float()

        return image, masks


def build_transforms():
    if A is None:
        return None, None
    train_t = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_t = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_t, val_t


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    """Equal-weighted sum of soft Dice and BCE, summed over channels."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return 0.5 * bce_loss + 0.5 * dice_loss


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def dice_per_class(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * target).sum(dim=(0, 2, 3))
    union = preds.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    return dice.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Train / eval loops
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    num_workers: int = 4
    seed: int = 42


def split_frame(frame: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(frame))
    n_test = int(cfg.test_fraction * len(frame))
    n_val = int(cfg.val_fraction * len(frame))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return (
        frame.iloc[train_idx].reset_index(drop=True),
        frame.iloc[val_idx].reset_index(drop=True),
        frame.iloc[test_idx].reset_index(drop=True),
    )


def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    losses = []
    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        optimiser.zero_grad()
        logits = model(image)
        loss = criterion(logits, mask)
        loss.backward()
        optimiser.step()
        losses.append(float(loss))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, dice_acc = [], []
    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        logits = model(image)
        losses.append(float(criterion(logits, mask)))
        dice_acc.append(dice_per_class(logits, mask))
    mean_loss = float(np.mean(losses))
    dice = np.stack(dice_acc).mean(axis=0)
    return mean_loss, dice


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baseline U-Net (ResNet-34) on Severstal")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if smp is None:
        raise SystemExit(
            "segmentation_models_pytorch is not installed. Run: "
            "pip install segmentation-models-pytorch albumentations"
        )

    df = load_annotations(TRAIN_CSV)
    images = build_image_index(df)
    train_df, val_df, test_df = split_frame(images, cfg)
    print(f"Splits: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_t, val_t = build_transforms()
    train_ds = SeverstalDataset(train_df, TRAIN_IMG_DIR, transform=train_t)
    val_ds = SeverstalDataset(val_df, TRAIN_IMG_DIR, transform=val_t)
    test_ds = SeverstalDataset(test_df, TRAIN_IMG_DIR, transform=val_t)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    criterion = DiceBCELoss()
    optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_dice = -1.0
    best_path = DELIVERABLES_DIR / "baseline_unet_resnet34.pt"

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice.tolist())
        mean_val_dice = float(np.mean(val_dice))
        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_dice_mean={mean_val_dice:.4f}  per_class={[round(x, 3) for x in val_dice]}"
        )
        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice
            torch.save(model.state_dict(), best_path)

    # Final test evaluation
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_dice = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  test dice per class={test_dice.tolist()}")

    metrics = {
        "model": "U-Net (ResNet-34)",
        "loss": "Dice + BCE",
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "best_val_dice_mean": best_val_dice,
        "test_dice_per_class": test_dice.tolist(),
        "test_dice_mean": float(np.mean(test_dice)),
        "history": history,
    }
    with (DELIVERABLES_DIR / "baseline_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Per-class Dice plot
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Class 1", "Class 2", "Class 3", "Class 4"], test_dice, color=["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice")
    ax.set_title(f"Baseline U-Net per-class test Dice (mean={np.mean(test_dice):.3f})")
    plt.tight_layout()
    plt.savefig(DELIVERABLES_DIR / "baseline_dice_per_class.png", dpi=140)

    print(f"Done. Artefacts in {DELIVERABLES_DIR}.")


if __name__ == "__main__":
    main()
