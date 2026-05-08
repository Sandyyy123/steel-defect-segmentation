"""Severstal Steel Defect Detection - Advanced model.

Primary architecture: SegFormer with `nvidia/mit-b3` ImageNet-pretrained encoder
(via HuggingFace `transformers`). Mixed precision (AMP), focal Tversky loss,
cosine schedule with warm restarts.

Fallback architecture (if `transformers` is unavailable): DeepLabV3+ with an
EfficientNet-B4 encoder via `segmentation_models_pytorch`.

This file is a runnable scaffold but is NOT executed during Phase 1.

Usage (main session):
    cd /root/AI/liora_projects/10_severstal_steel
    python src/model_advanced.py --epochs 30 --batch-size 8

Outputs (saved to ../deliverables/):
    advanced_segformer_mitb3.pt         Trained model weights
    advanced_metrics.json               Per-epoch and final test metrics
    advanced_dice_per_class.png         Per-class Dice bar chart
    advanced_vs_baseline.png            Side-by-side comparison
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

try:
    from transformers import SegformerForSemanticSegmentation
    HAVE_HF = True
except ImportError:
    HAVE_HF = False

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None

from PIL import Image
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_images"

IMG_HEIGHT, IMG_WIDTH = 256, 1600
NUM_CLASSES = 4


# -----------------------------------------------------------------------------
# Data (shared with baseline; duplicated here so each script is standalone)
# -----------------------------------------------------------------------------
def rle_decode(rle: str, shape: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)) -> np.ndarray:
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
        image = np.array(Image.open(self.img_dir / row["ImageId"]).convert("RGB"))
        masks = np.stack(
            [rle_decode(row[f"rle_{c}"]) for c in range(1, NUM_CLASSES + 1)], axis=-1
        ).astype(np.float32)
        if self.transform is not None:
            t = self.transform(image=image, mask=masks)
            image = t["image"]
            masks = t["mask"].permute(2, 0, 1) if t["mask"].ndim == 3 else t["mask"]
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
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_t = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_t, val_t


# -----------------------------------------------------------------------------
# Loss: Focal Tversky
# -----------------------------------------------------------------------------
class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss (Salehi 2017 + Abraham/Khan 2019 form).

    Tversky index trades false-positive vs false-negative penalty (alpha vs beta).
    Focal exponent gamma down-weights easy examples; helpful for the heavy
    class imbalance on Severstal class 2.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        tp = (probs * target).sum(dim=(2, 3))
        fp = (probs * (1 - target)).sum(dim=(2, 3))
        fn = ((1 - probs) * target).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal = (1.0 - tversky).clamp(min=1e-7) ** self.gamma
        return focal.mean()


# -----------------------------------------------------------------------------
# Model wrappers
# -----------------------------------------------------------------------------
class SegformerWrapper(nn.Module):
    """SegFormer mit-b3 with multi-label sigmoid head and full-resolution upsampling."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(pixel_values=x).logits
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(num_classes: int) -> nn.Module:
    if HAVE_HF:
        print("Using SegFormer mit-b3 (HuggingFace) as advanced model.")
        return SegformerWrapper(num_classes)
    if smp is None:
        raise SystemExit(
            "Neither transformers nor segmentation_models_pytorch is installed. "
            "Run: pip install transformers segmentation-models-pytorch albumentations"
        )
    print("transformers unavailable; falling back to DeepLabV3+ with EfficientNet-B4.")
    return smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )


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
# Train / eval
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 6e-5
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    num_workers: int = 4
    seed: int = 42
    cosine_T0: int = 5
    cosine_Tmult: int = 2


def split_frame(frame: pd.DataFrame, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(frame))
    n_test = int(cfg.test_fraction * len(frame))
    n_val = int(cfg.val_fraction * len(frame))
    return (
        frame.iloc[idx[n_test + n_val :]].reset_index(drop=True),
        frame.iloc[idx[n_test : n_test + n_val]].reset_index(drop=True),
        frame.iloc[idx[:n_test]].reset_index(drop=True),
    )


def train_one_epoch(model, loader, optimiser, criterion, device, scaler):
    model.train()
    losses = []
    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        optimiser.zero_grad()
        with autocast(enabled=scaler is not None):
            logits = model(image)
            loss = criterion(logits, mask)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
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
    return float(np.mean(losses)), np.stack(dice_acc).mean(axis=0)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced SegFormer mit-b3 on Severstal")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
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

    df = load_annotations(TRAIN_CSV)
    images = build_image_index(df)
    train_df, val_df, test_df = split_frame(images, cfg)
    print(f"Splits: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_t, val_t = build_transforms()
    train_loader = DataLoader(SeverstalDataset(train_df, TRAIN_IMG_DIR, train_t), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(SeverstalDataset(val_df, TRAIN_IMG_DIR, val_t), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(SeverstalDataset(test_df, TRAIN_IMG_DIR, val_t), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(NUM_CLASSES).to(device)
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    optimiser = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=cfg.cosine_T0, T_mult=cfg.cosine_Tmult)
    scaler = GradScaler() if device == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_dice = -1.0
    best_path = DELIVERABLES_DIR / "advanced_segformer_mitb3.pt"

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device, scaler)
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

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_dice = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  test dice per class={test_dice.tolist()}")

    metrics = {
        "model": "SegFormer mit-b3" if HAVE_HF else "DeepLabV3+ EfficientNet-B4",
        "loss": "Focal Tversky (alpha=0.7, beta=0.3, gamma=0.75)",
        "amp": device == "cuda",
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
    with (DELIVERABLES_DIR / "advanced_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["Class 1", "Class 2", "Class 3", "Class 4"], test_dice, color=["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice")
    ax.set_title(f"Advanced model per-class test Dice (mean={np.mean(test_dice):.3f})")
    plt.tight_layout()
    plt.savefig(DELIVERABLES_DIR / "advanced_dice_per_class.png", dpi=140)

    print(f"Done. Artefacts in {DELIVERABLES_DIR}.")


if __name__ == "__main__":
    main()
