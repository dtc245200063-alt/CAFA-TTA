"""Hybrid continual learning: safe online adaptation plus an experience store.

Online (per session): `ContinualSession` wraps CAFA-TTA v3 (`CAFASoftAdapter`) with
source BatchNorm statistics kept fixed, so single images are predicted stably and
only BN affine parameters adapt, on buffered batches of in-distribution images.
Out-of-distribution images are flagged and never used. If the adapted model drifts
from the source model or becomes much more overconfident, it is reset to source.

Persistent learning happens offline in `scripts/consolidate.py`, from the records
saved by `ExperienceStore`, behind an evaluation gate (`core/registry.py`).
"""
from __future__ import annotations

from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.cafa_soft_adapter import CAFASoftAdapter
from core.trainer import extract_features, fit_temperature
from core.training_utils import build_backbone, build_transforms
from kernel.confidence import normalized_entropy
from kernel.feature_alignment import compute_prototypes
from models.feature_extractor import ResNetFeatureExtractor


def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    """Logits-returning torchvision model with the checkpoint's weights."""
    model = build_backbone(checkpoint["architecture"], len(checkpoint["labels"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def eval_transform(checkpoint: dict):
    return build_transforms(
        {"input_size": checkpoint["input_size"], "normalization": checkpoint["normalization"],
         "modality": checkpoint.get("modality")},
        train=False,
    )


class OODGate:
    """Flags inputs unlike the source data: high energy OR low similarity to every class prototype.

    Thresholds are set on in-distribution validation data so that each criterion alone
    keeps `tpr` of it; combined with OR, slightly fewer in-distribution images pass.
    """

    def __init__(self, energy_threshold: float, similarity_threshold: float, temperature: float = 1.0):
        self.energy_threshold = float(energy_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.temperature = float(temperature)

    @staticmethod
    def energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return -temperature * torch.logsumexp(logits / temperature, dim=1)

    @staticmethod
    def max_similarity(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return (F.normalize(features.float(), dim=1) @ F.normalize(prototypes.float(), dim=1).T).max(1).values

    @classmethod
    def calibrate(cls, features, logits, prototypes, temperature: float = 1.0, tpr: float = 0.95) -> "OODGate":
        energy = cls.energy(logits.float(), temperature)
        similarity = cls.max_similarity(features, prototypes)
        return cls(torch.quantile(energy, tpr).item(), torch.quantile(similarity, 1 - tpr).item(), temperature)

    def is_ood(self, features, logits, prototypes) -> torch.Tensor:
        energy = self.energy(logits.float(), self.temperature)
        similarity = self.max_similarity(features, prototypes.to(features.device))
        return (energy > self.energy_threshold) | (similarity < self.similarity_threshold)

    def to_dict(self) -> dict:
        return {"energy_threshold": self.energy_threshold,
                "similarity_threshold": self.similarity_threshold,
                "temperature": self.temperature}

    @classmethod
    def from_dict(cls, data: dict) -> "OODGate":
        return cls(**data)


class CollapseMonitor:
    """Detect harmful drift of the adapted model, judged against the frozen source model."""

    def __init__(self, window: int = 64, min_count: int = 16,
                 max_disagreement: float = 0.3, min_entropy_ratio: float = 0.5):
        self.records = deque(maxlen=window)
        self.min_count = min_count
        self.max_disagreement = max_disagreement
        self.min_entropy_ratio = min_entropy_ratio

    def update(self, adapted_pred, source_pred, adapted_entropy, source_entropy) -> None:
        for a, s, ea, es in zip(adapted_pred.tolist(), source_pred.tolist(),
                                adapted_entropy.tolist(), source_entropy.tolist()):
            self.records.append((a != s, ea, es))

    def collapsed(self) -> bool:
        if len(self.records) < self.min_count:
            return False
        n = len(self.records)
        disagreement = sum(r[0] for r in self.records) / n
        adapted_entropy = sum(r[1] for r in self.records) / n
        source_entropy = sum(r[2] for r in self.records) / n
        overconfident = adapted_entropy < self.min_entropy_ratio * source_entropy
        return disagreement > self.max_disagreement or overconfident

    def clear(self) -> None:
        self.records.clear()


class ExperienceStore:
    """Local, append-only store of seen images and their predictions / user labels.

    Images are re-encoded from pixels as PNG (drops EXIF and other metadata) and named
    by the SHA-256 of their pixels, so the same image is stored once.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.images = self.root / "images"
        self.log = self.root / "records.jsonl"

    @staticmethod
    def image_id(image: Image.Image) -> str:
        rgb = image.convert("RGB")
        return hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()

    def image_path(self, image_id: str) -> Path:
        return self.images / f"{image_id}.png"

    def _append(self, event: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        event["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def add(self, image: Image.Image, record: dict) -> str:
        image_id = self.image_id(image)
        path = self.image_path(image_id)
        if not path.is_file():
            self.images.mkdir(parents=True, exist_ok=True)
            rgb = image.convert("RGB")
            Image.frombytes("RGB", rgb.size, rgb.tobytes()).save(path)
        self._append({"id": image_id, **record})
        return image_id

    def set_label(self, image_id: str, label: str) -> None:
        if not self.image_path(image_id).is_file():
            raise KeyError(f"Unknown image id {image_id}")
        self._append({"id": image_id, "user_label": label})

    def records(self) -> list[dict]:
        """One merged record per image; later events override earlier fields."""
        merged: dict[str, dict] = {}
        if self.log.is_file():
            for line in self.log.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    event = json.loads(line)
                    merged.setdefault(event["id"], {}).update(event)
        return list(merged.values())


def build_checkpoint(model: nn.Module, *, labels: list[str], cfg: dict, prototype_loader, val_loader,
                     device, training_type: str, metrics: dict, extra: dict | None = None) -> dict:
    """Package a trained model with its calibration, class prototypes and OOD thresholds.

    `prototype_loader` must use the evaluation transform (no augmentation).
    """
    extractor = ResNetFeatureExtractor(model).to(device).eval()
    train_features, _, train_labels = extract_features(extractor, prototype_loader, device)
    prototypes, _ = compute_prototypes(train_features, train_labels, len(labels))
    val_features, val_logits, val_labels = extract_features(extractor, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    gate = OODGate.calibrate(val_features, val_logits, prototypes, temperature)
    return {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "labels": list(labels),
        "num_classes": len(labels),
        "architecture": cfg["backbone"],
        "modality": cfg["modality"],
        "input_size": cfg["input_size"],
        "normalization": cfg["normalization"],
        "temperature": temperature,
        "prototypes": prototypes.cpu(),
        "ood": gate.to_dict(),
        "metrics": metrics,
        "training_type": training_type,
        **(extra or {}),
    }


class StableBNFeatures(ResNetFeatureExtractor):
    """Keep source BatchNorm statistics during adaptation (BN layers always in eval mode)."""

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self


class ContinualSession:
    """Online CAFA-TTA v3 session for one modality model from the registry."""

    def __init__(self, checkpoint: dict, device="cpu", *, adapt: bool = True, batch_size: int = 8,
                 lr: float = 1e-4, lambda_align: float = 0.05, entropy_floor: float = 0.2,
                 monitor: CollapseMonitor | None = None):
        self.device = torch.device(device)
        self.labels = list(checkpoint["labels"])
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.prototypes = checkpoint["prototypes"].float().to(self.device)
        self.gate = OODGate.from_dict(checkpoint["ood"])
        self.preprocess = eval_transform(checkpoint)
        self.adapt_enabled = adapt
        self.batch_size = batch_size
        self.adapter_kwargs = {"lr": lr, "lambda_align": lambda_align, "entropy_floor": entropy_floor}
        self.monitor = monitor or CollapseMonitor()
        self.source = ResNetFeatureExtractor(build_model_from_checkpoint(checkpoint)).to(self.device).eval()
        self.resets = 0
        self._build_adapter()

    def _build_adapter(self) -> None:
        backbone = copy.deepcopy(self.source.backbone)
        self.adapter = CAFASoftAdapter(StableBNFeatures(backbone), self.prototypes,
                                       min_samples=2, **self.adapter_kwargs)
        # CAFAAdapter discards BN running statistics; restore the source ones.
        for src, dst in zip(self.source.backbone.modules(), backbone.modules()):
            if isinstance(src, nn.BatchNorm2d):
                dst.track_running_stats = True
                dst.running_mean = src.running_mean.detach().clone()
                dst.running_var = src.running_var.detach().clone()
                dst.num_batches_tracked = src.num_batches_tracked.detach().clone()
        self.pending: list[torch.Tensor] = []
        self.updates = 0
        self.monitor.clear()

    def reset(self) -> None:
        """Discard all online adaptation and return to the source weights."""
        self._build_adapter()
        self.resets += 1

    @torch.no_grad()
    def _forward(self, model: nn.Module, x: torch.Tensor):
        model.eval()
        return model(x)

    def predict(self, images: list[Image.Image]) -> list[dict]:
        x = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        src_features, src_logits = self._forward(self.source, x)
        features, logits = self._forward(self.adapter.model, x)
        ood = self.gate.is_ood(src_features, src_logits, self.prototypes)
        probs = torch.softmax(logits / self.temperature, dim=1)
        pred = probs.argmax(1)
        proto_pred = (F.normalize(features.float(), dim=1) @ self.prototypes.T).argmax(1)
        src_pred = src_logits.argmax(1)

        keep = ~ood
        if keep.any():
            self.monitor.update(pred[keep], src_pred[keep],
                                normalized_entropy(logits[keep]), normalized_entropy(src_logits[keep]))

        reset = False
        if self.adapt_enabled and keep.any():
            self.pending.extend(x[keep])
            while len(self.pending) >= self.batch_size:
                batch = torch.stack(self.pending[:self.batch_size])
                self.pending = self.pending[self.batch_size:]
                if self.adapter.adapt(batch).get("updated"):
                    self.updates += 1
            if self.monitor.collapsed():
                self.reset()
                reset = True

        return [
            {
                "label": self.labels[int(pred[i])],
                "index": int(pred[i]),
                "confidence": float(probs[i, pred[i]]),
                "probabilities": {name: float(p) for name, p in zip(self.labels, probs[i])},
                "ood": bool(ood[i]),
                "agreement": bool(proto_pred[i] == pred[i]),
                "source_label": self.labels[int(src_pred[i])],
                "session_reset": reset,
            }
            for i in range(len(images))
        ]

    def save_state(self, path: Path | str) -> None:
        torch.save({"model": self.adapter.model.state_dict(),
                    "optimizer": self.adapter.optimizer.state_dict(),
                    "updates": self.updates}, path)

    def load_state(self, path: Path | str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.adapter.model.load_state_dict(state["model"])
        self.adapter.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
