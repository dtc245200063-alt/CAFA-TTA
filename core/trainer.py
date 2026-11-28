"""Generic classifier training loop for the multi-modality / continual-learning line.

Used by `scripts/train_modality.py` (initial training) and `scripts/consolidate.py`
(periodic fine-tuning with replay and distillation). The original research
training chain (`scripts/train_*.py`) is intentionally left unchanged.
"""
from __future__ import annotations

import copy
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler


def dataset_targets(dataset) -> list[int]:
    """Integer labels of an ImageFolder, `_IndexedImageDataset`, Subset or ConcatDataset."""
    if isinstance(dataset, ConcatDataset):
        return [t for part in dataset.datasets for t in dataset_targets(part)]
    if isinstance(dataset, Subset):
        parent = dataset_targets(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    if hasattr(dataset, "targets"):
        return [int(t) for t in dataset.targets]
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]
    return [int(dataset[i][1]) for i in range(len(dataset))]


def balanced_sampler(targets: list[int]) -> WeightedRandomSampler:
    counts = Counter(targets)
    weights = [1.0 / counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)


def ema_avg_fn(decay: float):
    """EMA with warmup: effective decay min(decay, (1+n)/(10+n)), so short runs are not stuck at init."""
    def fn(averaged, current, num_averaged):
        n = float(num_averaged)
        d = min(decay, (1.0 + n) / (10.0 + n))
        return d * averaged + (1.0 - d) * current
    return fn


def warmup_cosine(optimizer, total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device)).float().cpu())
        all_labels.append(labels.view(-1).long())
    return torch.cat(all_logits), torch.cat(all_labels)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    pred = logits.argmax(1).numpy()
    true = labels.numpy()
    classes = list(range(num_classes))
    return {
        "n": int(len(true)),
        "accuracy": float((pred == true).mean()) if len(true) else 0.0,
        "macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
        "per_class_recall": [
            float(r) for r in recall_score(true, pred, labels=classes, average=None, zero_division=0)
        ],
    }


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    """Post-hoc temperature scaling (Guo et al. 2017) on held-out logits."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


def train_classifier(
    model: nn.Module,
    train_ds,
    val_ds,
    num_classes: int,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
    grad_accum: int = 1,
    num_workers: int = 0,
    label_smoothing: float = 0.1,
    warmup_epochs: float = 1.0,
    weight_decay: float = 1e-4,
    ema_decay: float | None = 0.999,
    balanced: bool = True,
    teacher: nn.Module | None = None,
    kd_weight: float = 0.0,
    kd_temperature: float = 2.0,
    log=print,
) -> tuple[nn.Module, list[dict], dict]:
    """Train a logits-returning model; return (best model, history, best val metrics).

    Selection is on validation macro-F1 of the EMA weights when `ema_decay` is set.
    With `teacher` and `kd_weight > 0`, adds a distillation term (Learning without Forgetting).
    """
    use_amp = torch.device(device).type == "cuda"
    targets = dataset_targets(train_ds)
    sampler = balanced_sampler(targets) if balanced else None
    # Keep training workers alive across epochs: on Windows each respawn re-imports torch.
    # Validation gets few, short-lived workers: each Windows worker commits ~1.8 GB.
    worker_kw = {"persistent_workers": True} if num_workers > 0 else {}
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
        num_workers=num_workers, pin_memory=use_amp, drop_last=len(train_ds) > batch_size, **worker_kw,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=min(num_workers, 2), pin_memory=use_amp)

    model = model.to(device)
    if teacher is not None:
        teacher = teacher.to(device).eval().requires_grad_(False)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    scheduler = warmup_cosine(optimizer, epochs * steps_per_epoch, int(warmup_epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = (
        AveragedModel(model, avg_fn=ema_avg_fn(ema_decay), use_buffers=True)
        if ema_decay else None
    )

    best_state, best_metrics, history = None, {"macro_f1": -1.0}, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
                if teacher is not None and kd_weight > 0:
                    with torch.no_grad():
                        teacher_logits = teacher(images)
                    t = kd_temperature
                    loss = loss + kd_weight * t * t * F.kl_div(
                        F.log_softmax(logits.float() / t, dim=1),
                        F.softmax(teacher_logits.float() / t, dim=1),
                        reduction="batchmean",
                    )
            scaler.scale(loss / grad_accum).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update_parameters(model)
            total_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)

        evaluated = ema.module if ema is not None else model
        metrics = classification_metrics(*predict_logits(evaluated, val_loader, device), num_classes)
        record = {"epoch": epoch, "train_loss": total_loss / max(seen, 1), **metrics}
        history.append(record)
        log(f"Epoch {epoch}/{epochs} | loss {record['train_loss']:.4f} | "
            f"val acc {metrics['accuracy']:.4f} | val macro-F1 {metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_state = copy.deepcopy(evaluated.state_dict())

    best_model = copy.deepcopy(model)
    best_model.load_state_dict(best_state)
    return best_model.eval(), history, best_metrics


@torch.no_grad()
def extract_features(extractor: nn.Module, loader, device):
    """Run a `(features, logits)` extractor; return CPU features, logits, labels."""
    extractor.eval()
    feats, logits, labels = [], [], []
    for images, y in loader:
        f, l = extractor(images.to(device))
        feats.append(f.float().cpu())
        logits.append(l.float().cpu())
        labels.append(y.view(-1).long())
    return torch.cat(feats), torch.cat(logits), torch.cat(labels)
