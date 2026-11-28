"""Frozen pathology foundation-model backbone (Phikon, ViT-B/16) for the continual-learning line.

Weights are downloaded separately into `data/foundation/phikon/` (owkin/phikon, Owkin
non-commercial licence, research use only) and are never committed. The backbone stays
frozen: only a linear head is trained, and test-time adaptation works on its features.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
PHIKON_DIR = ROOT / "data/foundation/phikon"
FEATURE_DIM = 768


def phikon_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class PhikonEncoder(nn.Module):
    """Images -> CLS token of the last layer (B, 768), as in the Phikon model card."""

    def __init__(self, weights_dir: Path = PHIKON_DIR):
        super().__init__()
        from transformers import ViTModel

        self.vit = ViTModel.from_pretrained(str(weights_dir), add_pooling_layer=False, local_files_only=True)
        self.vit.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
            return self.vit(pixel_values=x).last_hidden_state[:, 0].float()


class FeatureHead(nn.Module):
    """Linear classifier on standardised features. `mean`/`std` are the source statistics;
    test-time adaptation may replace them with target statistics (feature-level BN-adapt)."""

    def __init__(self, num_classes: int, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
        self.linear = nn.Linear(mean.numel(), num_classes)

    def normalise(self, f, mean=None, std=None):
        return (f - (self.mean if mean is None else mean)) / (self.std if std is None else std)

    def forward(self, f, mean=None, std=None):
        return self.linear(self.normalise(f, mean, std))


def train_head(features: torch.Tensor, labels: torch.Tensor, num_classes: int, *, epochs: int = 30,
               lr: float = 1e-3, weight_decay: float = 1e-4, label_smoothing: float = 0.1,
               device: str = "cpu", seed: int = 0, init: FeatureHead | None = None,
               teacher: FeatureHead | None = None, kd_weight: float = 0.0,
               sample_weights: torch.Tensor | None = None) -> FeatureHead:
    """Full-batch-ish AdamW on precomputed features with class-balanced sampling.

    With `teacher` and `kd_weight`, adds distillation to the teacher's logits (LwF), used by consolidation.
    """
    g = torch.Generator().manual_seed(seed)
    f, y = features.float().to(device), labels.long().to(device)
    head = init if init is not None else FeatureHead(num_classes, f.mean(0), f.std(0) + 1e-6)
    head = head.to(device).train()
    opt = torch.optim.AdamW(head.linear.parameters(), lr=lr, weight_decay=weight_decay)
    counts = torch.bincount(y, minlength=num_classes).clamp(min=1).float()
    weights = (1.0 / counts)[y] if sample_weights is None else sample_weights.to(device)
    steps = max(20, len(y) // 512)  # small sets still get enough updates
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    for _ in range(epochs):
        for _ in range(steps):
            idx = torch.multinomial(weights.cpu(), 512, replacement=True, generator=g).to(device)
            logits = head(f[idx])
            loss = F.cross_entropy(logits, y[idx], label_smoothing=label_smoothing)
            if teacher is not None and kd_weight > 0:
                with torch.no_grad():
                    t = teacher(f[idx])
                loss = loss + kd_weight * 4 * F.kl_div(F.log_softmax(logits / 2, 1), F.softmax(t / 2, 1),
                                                       reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return head.eval()
