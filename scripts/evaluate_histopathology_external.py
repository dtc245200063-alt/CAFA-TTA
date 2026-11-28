"""External evaluation of a registered histopathology model (continual-learning line).

Writes one JSON with:
  - CRC-VAL-HE-7K (independent test set): clean and synthetic colour shift; accuracy, macro-F1,
    per-class recall, ECE (raw and temperature-scaled), OOD-gate flag rate.
  - Kather 2016 (7 classes mapped to NCT names; 03_COMPLEX has no counterpart): online
    batch-prequential stream for source / BN-adapt / CAFA v3 / ContinualSession, 3 shuffled seeds.
    Labels are used only for scoring.
  - Synthetic non-tissue images: OOD-gate rejection rate.

Research use only. Kather 2016 is one external set; seeds are paired, not independent runs.

Example:
    python scripts/evaluate_histopathology_external.py --version v2 --data-root C:/Code/DataSet/histopathology --out runs/final/v2.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cafa_soft_adapter import CAFASoftAdapter  # noqa: E402
from core.continual import ContinualSession, OODGate, build_model_from_checkpoint, eval_transform  # noqa: E402
from core.registry import ModelRegistry  # noqa: E402
from core.trainer import classification_metrics  # noqa: E402
from models.feature_extractor import ResNetFeatureExtractor  # noqa: E402

KATHER_TO_NCT = {"01_TUMOR": "TUM", "02_STROMA": "STR", "04_LYMPHO": "LYM", "05_DEBRIS": "DEB",
                 "06_MUCOSA": "NORM", "07_ADIPOSE": "ADI", "08_EMPTY": "BACK"}
STREAM_BATCH, SEEDS = 64, (0, 1, 2)
# Kather 2016 class definitions (Sci Rep 6:27988): "debris" includes necrosis, haemorrhage and mucus;
# "simple stroma" includes smooth muscle. Harmonised scoring counts NCT MUC as DEB and MUS as STR.
# Strict scoring (the default `accuracy`) counts them as errors. Both are reported.
HARMONISED = {"MUC": "DEB", "MUS": "STR"}


def harmonised_accuracy(pred, true, labels) -> float:
    pred = np.asarray([labels.index(HARMONISED.get(labels[p], labels[p])) for p in np.asarray(pred).tolist()])
    return float((pred == np.asarray(true)).mean())


def colour_shift(x01: torch.Tensor) -> torch.Tensor:
    """Synthetic colour shift used by the research evaluation."""
    return (x01 * 0.65 + 0.18).clamp(0, 1).pow(1.15)


def ece(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return round(float(total), 4)


def synthetic_non_tissue(n: int = 200) -> list[Image.Image]:
    rng, out = np.random.default_rng(0), []
    for _ in range(n // 4):
        out.append(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        out.append(np.full((224, 224, 3), rng.integers(0, 255, 3), np.uint8))
        g = np.linspace(0, 1, 224)[None, :, None] * rng.uniform(50, 255, 3) + rng.uniform(0, 100, 3)
        out.append(np.broadcast_to(g, (224, 224, 3)).clip(0, 255).astype(np.uint8))
        k = int(rng.integers(4, 40))
        out.append((((np.indices((224, 224)).sum(0) // k) % 2)[..., None] * rng.uniform(80, 255, 3)).astype(np.uint8))
    return [Image.fromarray(a) for a in out]


class Evaluator:
    def __init__(self, checkpoint: dict, device: str):
        self.ck, self.device = checkpoint, device
        self.labels, self.T = checkpoint["labels"], float(checkpoint["temperature"])
        self.tf = eval_transform(checkpoint)
        self.source = ResNetFeatureExtractor(build_model_from_checkpoint(checkpoint)).to(device).eval()
        self.protos = checkpoint["prototypes"].float().to(device)
        self.gate = OODGate.from_dict(checkpoint["ood"])
        norm = checkpoint["normalization"]
        self.mean = torch.tensor(norm["mean"])[:, None, None]
        self.std = torch.tensor(norm["std"])[:, None, None]

    def tensors(self, images: list[Image.Image], shifted: bool = False) -> torch.Tensor:
        x = torch.stack([self.tf(im) for im in images])
        if shifted:
            x = (colour_shift((x * self.std + self.mean).clamp(0, 1)) - self.mean) / self.std
        return x

    @torch.no_grad()
    def forward(self, model, images: list[Image.Image], shifted: bool = False):
        """Preprocess in chunks so the whole set never sits in memory as float tensors."""
        model.eval()
        feats, logits = [], []
        for i in range(0, len(images), 256):
            f, l = model(self.tensors(images[i:i + 256], shifted).to(self.device))
            feats.append(f.float()), logits.append(l.float())
        return torch.cat(feats), torch.cat(logits)

    def static_report(self, images: list[Image.Image], y: torch.Tensor, shifted: bool = False) -> dict:
        f, l = self.forward(self.source, images, shifted)
        m = classification_metrics(l.cpu(), y, len(self.labels))
        present = sorted(set(y.tolist()))
        return {"n": len(y), "accuracy": round(m["accuracy"], 4),
                "accuracy_harmonised": round(harmonised_accuracy(l.argmax(1).cpu(), y, self.labels), 4),
                "macro_f1_present_classes": round(float(
                    np.mean([_f1(l.argmax(1).cpu(), y, c) for c in present])), 4),
                "per_class_recall": {self.labels[c]: round(m["per_class_recall"][c], 3) for c in present},
                "ece_temperature_scaled": ece(torch.softmax(l / self.T, 1).cpu(), y),
                "ece_raw": ece(torch.softmax(l, 1).cpu(), y),
                "ood_flag_rate": round(float(self.gate.is_ood(f, l, self.protos).float().mean()), 4)}

    def stream(self, method: str, images: list[Image.Image], y: np.ndarray, seed: int,
               class_ordered: bool = False, keep_predictions: bool = False) -> dict:
        order = np.random.default_rng(seed).permutation(len(y))
        if class_ordered:  # one tissue class after another (label-correlated, non-i.i.d. stream)
            order = order[np.argsort(y[order], kind="stable")]
        session = adapter = model = None
        if method == "source":
            model = self.source
        elif method == "bn_adapt":
            model = copy.deepcopy(self.source)
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats, m.running_mean, m.running_var = False, None, None
            model.train()
        elif method == "cafa_v3":
            adapter = CAFASoftAdapter(copy.deepcopy(self.source), self.protos)
        elif method == "session_b8_recent":  # BN statistics from the most recent images only
            session = ContinualSession(self.ck, self.device, batch_size=8, bn_memory_per_class=0)
        elif method == "session_b8":  # default: class-balanced BN statistics memory
            session = ContinualSession(self.ck, self.device, batch_size=8)
        preds = []
        for i in range(0, len(order), STREAM_BATCH):
            batch = [images[j] for j in order[i:i + STREAM_BATCH]]
            if session is not None:
                preds += [r["index"] for r in session.predict(batch)]
                continue
            x = torch.stack([self.tf(im) for im in batch]).to(self.device)
            with torch.no_grad():
                logits = adapter.predict(x) if adapter else model(x)[1]
            preds += logits.argmax(1).tolist()
            if adapter:
                adapter.adapt(x)
        preds, true = np.array(preds), y[order]
        out = {"accuracy": float((preds == true).mean()),
               "accuracy_harmonised": harmonised_accuracy(preds, true, self.labels),
               "recall": {self.labels[c]: round(float((preds[true == c] == c).mean()), 3)
                          for c in sorted(set(true.tolist()))}}
        if session is not None:
            out.update(updates=session.updates, resets=session.resets)
        if keep_predictions:
            out.update(order=order.tolist(), predictions=preds.tolist())
        return out


def _f1(pred: torch.Tensor, y: torch.Tensor, c: int) -> float:
    tp = ((pred == c) & (y == c)).sum().item()
    fp = ((pred == c) & (y != c)).sum().item()
    fn = ((pred != c) & (y == c)).sum().item()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def load_folder(root: Path, label_of) -> tuple[list[Image.Image], np.ndarray]:
    items = [(p, label_of(p.parent.name)) for p in sorted(root.glob("*/*.tif"))]
    items = [(p, y) for p, y in items if y is not None]
    images = []
    for p, _ in items:
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
    return images, np.array([y for _, y in items])


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=None, help="Registry version (default: current)")
    parser.add_argument("--registry-root", type=Path, default=ROOT / "models_registry")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Folder containing crc_val_he_7k/ and kather2016/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-stream", action="store_true", help="Only static CRC-VAL and OOD checks")
    parser.add_argument("--orders", default="iid,class",
                        help="Comma list of Kather stream orders: iid (shuffled) and/or class (class-ordered)")
    parser.add_argument("--methods", default="source,bn_adapt,cafa_v3,session_b8_recent,session_b8")
    parser.add_argument("--skip-crc", action="store_true", help="Skip CRC-VAL (saves ~1 GB RAM)")
    args = parser.parse_args(argv)

    registry = ModelRegistry(args.registry_root, "histopathology")
    version = args.version or registry.current_version()
    ck = registry.load(version)
    ev = Evaluator(ck, "cuda" if torch.cuda.is_available() else "cpu")
    labels = ev.labels
    result = {"version": version, "architecture": ck["architecture"], "augment": ck.get("augment"),
              "temperature": ev.T, "val_metrics": ck.get("metrics")}

    t0 = time.time()
    if not args.skip_crc:
        _crc(args, ev, labels, result)
    _kather_and_synthetic(args, ev, labels, result, version, t0)
    return result


def _crc(args, ev, labels, result):
    crc_imgs, crc_y = load_folder(args.data_root / "crc_val_he_7k", lambda n: labels.index(n) if n in labels else None)
    y = torch.from_numpy(crc_y)
    result["crc_val_clean"] = ev.static_report(crc_imgs, y)
    result["crc_val_colour_shift"] = ev.static_report(crc_imgs, y, shifted=True)


def _kather_and_synthetic(args, ev, labels, result, version, t0):
    kat_imgs, kat_y = load_folder(args.data_root / "kather2016",
                                  lambda n: labels.index(KATHER_TO_NCT[n]) if n in KATHER_TO_NCT else None)
    result["kather2016_source_static"] = ev.static_report(kat_imgs, torch.from_numpy(kat_y))
    for order in [] if args.skip_stream else args.orders.split(","):
        key = "kather2016_stream" if order == "iid" else "kather2016_stream_class_ordered"
        result[key] = {}
        for method in args.methods.split(","):
            runs = [ev.stream(method, kat_imgs, kat_y, s, class_ordered=order == "class") for s in SEEDS]
            accs = [r["accuracy"] for r in runs]
            result[key][method] = {"acc_mean": round(float(np.mean(accs)), 4),
                                   "acc_std": round(float(np.std(accs)), 4),
                                   "acc_harmonised_mean": round(float(np.mean([r["accuracy_harmonised"] for r in runs])), 4),
                                   "runs": runs}
            print(f"[{version}] kather {order:5s} {method:12s} {np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)

    synth = synthetic_non_tissue()
    f, l = ev.forward(ev.source, synth)
    flag = ev.gate.is_ood(f, l, ev.protos).cpu()
    result["synthetic_non_tissue_rejected"] = round(float(flag.float().mean()), 3)
    result["elapsed_s"] = round(time.time() - t0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if "crc_val_clean" in result:
        c, s = result["crc_val_clean"], result["crc_val_colour_shift"]
        print(f"[{version}] CRC-VAL clean {c['accuracy']:.4f} (ECE {c['ece_temperature_scaled']}) | "
              f"shift {s['accuracy']:.4f}", flush=True)
    print(f"[{version}] Kather static {result['kather2016_source_static']['accuracy']:.4f} | "
          f"non-tissue rejected {result['synthetic_non_tissue_rejected']}", flush=True)


if __name__ == "__main__":
    main()
