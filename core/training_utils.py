"""Shared modality-aware training utilities for CAFA-TTA.

Supports high-resolution medical imaging datasets (ImageFolder) as well as
backward-compatible PathMNIST loading, pretrained ImageNet backbones
(ResNet-18, ResNet-50, DenseNet-121), AMP, and gradient accumulation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIMPLE_MEAN = (0.5, 0.5, 0.5)
SIMPLE_STD = (0.5, 0.5, 0.5)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ModalityProfile:
    name: str
    min_size: int
    optimal_size: int
    max_size: int
    default_backbone: str
    micro_batch_4gb: int
    grad_accum_4gb: int


MODALITY_PROFILES: dict[str, ModalityProfile] = {
    "histopathology": ModalityProfile(
        name="histopathology",
        min_size=96,
        optimal_size=224,
        max_size=256,
        default_backbone="resnet18",
        micro_batch_4gb=16,
        grad_accum_4gb=4,
    ),
    "dermatology": ModalityProfile(
        name="dermatology",
        min_size=224,
        optimal_size=224,
        max_size=384,
        default_backbone="resnet18",
        micro_batch_4gb=16,
        grad_accum_4gb=4,
    ),
    "chest_xray": ModalityProfile(
        name="chest_xray",
        min_size=224,
        optimal_size=320,
        max_size=512,
        default_backbone="densenet121",
        micro_batch_4gb=8,
        grad_accum_4gb=8,
    ),
    "fundus": ModalityProfile(
        name="fundus",
        min_size=224,
        optimal_size=384,
        max_size=512,
        default_backbone="resnet50",
        micro_batch_4gb=8,
        grad_accum_4gb=8,
    ),
    # Blood-smear cell crops (e.g. BloodMNIST+ at 224).
    "hematology": ModalityProfile(
        name="hematology",
        min_size=96,
        optimal_size=224,
        max_size=384,
        default_backbone="resnet18",
        micro_batch_4gb=16,
        grad_accum_4gb=4,
    ),
    # Grayscale breast ultrasound (e.g. BreastMNIST+ at 224).
    "breast_ultrasound": ModalityProfile(
        name="breast_ultrasound",
        min_size=128,
        optimal_size=224,
        max_size=384,
        default_backbone="resnet18",
        micro_batch_4gb=16,
        grad_accum_4gb=4,
    ),
}

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "densenet121")

DEFAULT_SHIFT = {"brightness": 1.3, "contrast": 1.3, "saturation": 1.3}


def load_domain_shift(root: Path) -> dict:
    """Return the shift config from experiments/domain_shift/metrics.json, else DEFAULT_SHIFT."""
    shift_file = root / "experiments/domain_shift/metrics.json"
    if shift_file.is_file():
        with open(shift_file, encoding="utf-8") as f:
            return json.load(f)["shift"]
    return DEFAULT_SHIFT


class HEDJitter:
    """Stain augmentation in HED colour space (Tellez et al. 2018) for H&E histopathology."""

    def __init__(self, alpha: float = 0.05, beta: float = 0.01):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, image: Image.Image) -> Image.Image:
        import numpy as np
        from skimage.color import hed2rgb, rgb2hed

        hed = rgb2hed(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)
        scale = np.random.uniform(1 - self.alpha, 1 + self.alpha, size=3)
        shift = np.random.uniform(-self.beta, self.beta, size=3)
        rgb = np.clip(hed2rgb(hed * scale + shift), 0.0, 1.0)
        return Image.fromarray((rgb * 255).round().astype("uint8"))


class RandomRot90:
    """Rotate by a random multiple of 90 degrees (tissue patches have no canonical orientation)."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.rotate(90 * random.randrange(4))


def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    """Resize + modality-specific augmentation (train only) + normalization from `resolve_input_config`."""
    size = cfg["input_size"]
    ops: list = [transforms.Resize((size, size))]
    if train:
        modality = cfg.get("modality")
        if modality == "histopathology" and cfg.get("augment") == "strong":
            # Cross-lab robustness: NCT-CRC-HE-100K is Macenko-normalised, so the light
            # recipe never sees real stain/scanner variety.
            ops += [
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                HEDJitter(alpha=0.2, beta=0.03), RandomRot90(), transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.25, contrast=0.3, saturation=0.2, hue=0.04),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.3),
            ]
        elif modality == "histopathology":
            ops += [HEDJitter(), RandomRot90(), transforms.RandomHorizontalFlip()]
        elif modality == "dermatology":
            ops += [
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.02),
            ]
        elif modality == "chest_xray":
            # No flips: laterality matters on chest radiographs.
            ops += [transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05))]
        elif modality == "hematology":
            # Isolated cells: orientation-free; mild colour jitter for stain variation.
            ops += [RandomRot90(), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)]
        elif modality == "breast_ultrasound":
            # Grayscale: gain/contrast variation and small geometric changes only.
            ops += [
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]
        elif modality == "fundus":
            ops += [
                transforms.RandomResizedCrop(size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.1, 0.1),
            ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["normalization"]["mean"], std=cfg["normalization"]["std"]),
    ]
    return transforms.Compose(ops)


def resolve_input_config(
    modality: str | None,
    size: int | None,
    backbone: str | None,
    pretrained: bool = True,
    batch_size: int | None = None,
    grad_accum: int | None = None,
) -> dict:
    """Resolve and validate input resolution, backbone, normalization, and batching."""
    if modality is not None:
        if modality not in MODALITY_PROFILES:
            raise ValueError(
                f"Unsupported modality '{modality}'. Choose from: {list(MODALITY_PROFILES)}"
            )
        profile = MODALITY_PROFILES[modality]
        resolved_size = profile.optimal_size if size is None else int(size)
        if not (profile.min_size <= resolved_size <= profile.max_size):
            raise ValueError(
                f"Input size {resolved_size} is outside the valid range "
                f"[{profile.min_size}, {profile.max_size}] for modality '{modality}'."
            )
        resolved_backbone = profile.default_backbone if backbone is None else backbone
        resolved_batch = profile.micro_batch_4gb if batch_size is None else int(batch_size)
        resolved_accum = profile.grad_accum_4gb if grad_accum is None else int(grad_accum)
    else:
        resolved_size = 28 if size is None else int(size)
        if resolved_size < 28:
            raise ValueError(f"Input size must be >= 28, got {resolved_size}")
        resolved_backbone = "resnet18" if backbone is None else backbone
        resolved_batch = 64 if batch_size is None else int(batch_size)
        resolved_accum = 1 if grad_accum is None else int(grad_accum)

    if resolved_backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{resolved_backbone}'. Choose from: {SUPPORTED_BACKBONES}"
        )
    if resolved_batch < 1 or resolved_accum < 1:
        raise ValueError("batch_size and grad_accum must be >= 1")

    mean = list(IMAGENET_MEAN if pretrained else SIMPLE_MEAN)
    std = list(IMAGENET_STD if pretrained else SIMPLE_STD)

    return {
        "modality": modality or "legacy_pathmnist",
        "input_size": resolved_size,
        "backbone": resolved_backbone,
        "pretrained": bool(pretrained),
        "batch_size": resolved_batch,
        "grad_accum": resolved_accum,
        "normalization": {"mean": mean, "std": std},
    }


def build_backbone(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Build a classification backbone with optional ImageNet pretrained weights."""
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported backbone '{backbone_name}'. Choose from: {SUPPORTED_BACKBONES}"
    )


class _IndexedImageDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[Path, int]], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        if self.transform is not None:
            rgb = self.transform(rgb)
        return rgb, label


def load_image_datasets(
    data_dir: Path | str,
    train_transform,
    val_transform,
    seed: int = 42,
    val_ratio: float = 0.2,
):
    """Load high-resolution medical image datasets from an ImageFolder directory.

    Supports two layouts:
    1. Pre-split: `<data_dir>/train/<class>/*` and `<data_dir>/val/<class>/*`
    2. Flat class folders: `<data_dir>/<class>/*` (deterministic per-class 80/20 split)
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    train_dir = root / "train"
    val_dir = root / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
        val_ds = datasets.ImageFolder(str(val_dir), transform=val_transform)
        if train_ds.classes != val_ds.classes:
            raise ValueError(
                f"Class mismatch between train ({train_ds.classes}) and val ({val_ds.classes})"
            )
        return train_ds, val_ds, list(train_ds.classes)

    class_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least 2 class subdirectories in {root}, found {len(class_dirs)}"
        )

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    class_names: list[str] = []

    for cls_idx, cls_dir in enumerate(class_dirs):
        images = sorted(
            p for p in cls_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 2:
            raise ValueError(
                f"Class folder '{cls_dir.name}' needs at least 2 images for train/val split, found {len(images)}"
            )
        rng.shuffle(images)
        n_val = max(1, int(round(len(images) * val_ratio)))
        n_val = min(n_val, len(images) - 1)
        val_samples.extend((p, cls_idx) for p in images[:n_val])
        train_samples.extend((p, cls_idx) for p in images[n_val:])
        class_names.append(cls_dir.name)

    return (
        _IndexedImageDataset(train_samples, transform=train_transform),
        _IndexedImageDataset(val_samples, transform=val_transform),
        class_names,
    )


def default_checkpoint_path(
    root: Path,
    stage: str,
    cfg: dict,
    is_legacy: bool,
) -> Path:
    """Compute a non-destructive checkpoint path preserving legacy frozen checkpoints."""
    if is_legacy:
        legacy_names = {
            "baseline": "resnet18_pathmnist_best.pt",
            "enhanced": "resnet18_pathmnist_enhanced_best.pt",
            "robust": "resnet18_pathmnist_robust_best.pt",
            "balanced": "resnet18_pathmnist_balanced_best.pt",
        }
        return root / "checkpoints" / legacy_names[stage]

    return (
        root
        / "checkpoints"
        / f"{cfg['backbone']}_{cfg['modality']}_{cfg['input_size']}_{stage}_best.pt"
    )


def add_common_train_args(
    parser: argparse.ArgumentParser,
    default_epochs: int,
    default_lr: float,
) -> argparse.ArgumentParser:
    """Attach standard modality-aware training CLI flags."""
    parser.add_argument(
        "--modality",
        choices=list(MODALITY_PROFILES.keys()),
        default=None,
        help="Medical imaging modality profile (sets sweet-spot size, backbone, and 4GB VRAM batching).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Input image size (validated against modality min/max range when --modality is set).",
    )
    parser.add_argument(
        "--backbone",
        choices=list(SUPPORTED_BACKBONES),
        default=None,
        help="Model backbone architecture.",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained weights (default: enabled).",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train from scratch (weights=None).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to high-resolution ImageFolder dataset (if omitted, uses legacy PathMNIST).",
    )
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Prerequisite checkpoint for fine-tuning stages.",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=None,
        help="Output path for best checkpoint.",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
