"""Phikon (frozen) + linear head: source training on NCT-100K features and external evaluation.

Reads runs/foundation/features_*.pt (scripts/extract_foundation_features.py), trains a linear
head on an 80/20 stratified split of NCT-CRC-HE-100K, and reports CRC-VAL-HE-7K and Kather 2016
(static and online feature-level TTA streams, shuffled and class-ordered, 3 seeds).
Kather is split by patient: group A (01,03,05,06,07,09) is the development / adaptation stream,
group B (02,04,08,10) is held out. Writes runs/foundation/head.pt and results.json.
Research use only. One external dataset; seeds are paired, not independent runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.continual import OODGate  # noqa: E402
from core.feature_tta import FeatureSession  # noqa: E402
from core.foundation import train_head  # noqa: E402
from core.trainer import classification_metrics, fit_temperature  # noqa: E402
from evaluate_histopathology_external import harmonised_accuracy  # noqa: E402

GROUP_A = {"01", "03", "05", "06", "07", "09"}
METHODS = {
    "source": dict(stats="source"),
    "feat_recent": dict(stats="recent"),
    "feat_balanced": dict(stats="balanced"),
    "feat_balanced_t3a": dict(stats="balanced", t3a=True),
    "t3a": dict(stats="source", t3a=True),
}


def load(name):
    return torch.load(ROOT / f"runs/foundation/features_{name}.pt", weights_only=False)


def ece(probs, y, bins=15):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    total = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return round(float(total), 4)


def stream(head, features, y, method, seed, class_ordered, batch=64, names=None):
    """Return a dict: accuracy, accuracy_harmonised (if names), mean_recall, recall per class index."""
    order = np.random.default_rng(seed).permutation(len(y))
    if class_ordered:
        order = order[np.argsort(y[order], kind="stable")]
    session = FeatureSession(head, **METHODS[method])
    preds = []
    for i in range(0, len(order), batch):
        preds.append(session.predict(features[order[i:i + batch]]).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    true = y[order]
    recall = {int(c): float((preds[true == c] == c).mean()) for c in np.unique(true)}
    out = {"accuracy": float((preds == true).mean()), "mean_recall": float(np.mean(list(recall.values()))),
           "recall": recall}
    if names is not None:
        out["accuracy_harmonised"] = harmonised_accuracy(preds, true, names)
    return out


def build_head(nct, device, seed=0):
    """Train the source head on an 80/20 split; return (head, extras dict, val indices)."""
    y = nct["labels"]
    rng = np.random.default_rng(seed)
    val = np.concatenate([rng.permutation(np.flatnonzero(y.numpy() == c))[: int(0.2 * (y == c).sum())]
                          for c in range(len(nct["label_names"]))])
    train = np.setdiff1d(np.arange(len(y)), val)
    f = nct["features"].float()
    head = train_head(f[train], y[train], len(nct["label_names"]), device=device)
    with torch.no_grad():
        val_logits = head(f[val].to(device)).cpu()
        z = head.normalise(f[train].to(device)).cpu()
    temperature = fit_temperature(val_logits, y[val])
    prototypes = torch.stack([F.normalize(z[y[train] == c].mean(0), dim=0) for c in range(len(nct["label_names"]))])
    with torch.no_grad():
        val_z = head.normalise(f[val].to(device)).cpu()
    gate = OODGate.calibrate(val_z, val_logits, prototypes, temperature)
    metrics = classification_metrics(val_logits, y[val], len(nct["label_names"]))
    return head, {"temperature": temperature, "prototypes": prototypes, "ood": gate.to_dict(),
                  "val_metrics": metrics}, val


def static(head, extras, data, device, mask=None):
    f, y = data["features"].float(), data["labels"]
    if mask is not None:
        f, y = f[mask], y[mask]
    with torch.no_grad():
        logits = head(f.to(device)).cpu()
        z = head.normalise(f.to(device)).cpu()
    m = classification_metrics(logits, y, len(data["label_names"]))
    present = sorted(set(y.tolist()))
    gate = OODGate.from_dict(extras["ood"])
    return {"n": len(y), "accuracy": round(m["accuracy"], 4),
            "accuracy_harmonised": round(harmonised_accuracy(logits.argmax(1), y, data["label_names"]), 4),
            "mean_recall_present": round(float(np.mean([m["per_class_recall"][c] for c in present])), 4),
            "per_class_recall": {data["label_names"][c]: round(m["per_class_recall"][c], 3) for c in present},
            "ece": ece(torch.softmax(logits / extras["temperature"], 1), y),
            "ood_flag_rate": round(float(gate.is_ood(z, logits, extras["prototypes"]).float().mean()), 4)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/foundation")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nct, crc, kat = load("nct_crc_100k"), load("crc_val"), load("kather2016")
    head, extras, _ = build_head(nct, device)
    head = head.cpu()
    torch.save({"head_state_dict": head.state_dict(), "label_names": nct["label_names"], **extras,
                "backbone": "owkin/phikon@057cc02958"}, args.out / "head.pt")
    groups = np.array(["A" if p in GROUP_A else "B" for p in kat["patients"]])
    result = {"backbone": "phikon (frozen) + linear head", "val_metrics": extras["val_metrics"],
              "crc_val": static(head, extras, crc, "cpu"), "kather_static": static(head, extras, kat, "cpu"),
              "kather_static_A": static(head, extras, kat, "cpu", torch.from_numpy(groups == "A")),
              "kather_static_B": static(head, extras, kat, "cpu", torch.from_numpy(groups == "B"))}
    print(f"NCT val {extras['val_metrics']['accuracy']:.4f} | CRC-VAL {result['crc_val']['accuracy']:.4f} | "
          f"Kather {result['kather_static']['accuracy']:.4f} (A {result['kather_static_A']['accuracy']:.4f}, "
          f"B {result['kather_static_B']['accuracy']:.4f})", flush=True)

    y_all = kat["labels"].numpy()
    seeds = [int(s) for s in args.seeds.split(",")]
    for subset, mask in (("all", np.ones(len(y_all), bool)), ("A", groups == "A"), ("B", groups == "B")):
        f, y = kat["features"][torch.from_numpy(mask)], y_all[mask]
        for order in ("iid", "class"):
            key = f"kather_{subset}_{order}"
            result[key] = {}
            for method in METHODS:
                runs = [stream(head, f, y, method, s, order == "class", names=nct["label_names"]) for s in seeds]
                result[key][method] = {
                    "acc_mean": round(float(np.mean([r["accuracy"] for r in runs])), 4),
                    "acc_harmonised_mean": round(float(np.mean([r["accuracy_harmonised"] for r in runs])), 4),
                    "mean_recall": round(float(np.mean([r["mean_recall"] for r in runs])), 4),
                    "recall": {nct["label_names"][c]: round(v, 3) for c, v in runs[0]["recall"].items()}}
                print(f"{key:18s} {method:18s} acc {result[key][method]['acc_mean']:.4f} "
                      f"harmonised {result[key][method]['acc_harmonised_mean']:.4f} "
                      f"mean-recall {result[key][method]['mean_recall']:.4f}", flush=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
