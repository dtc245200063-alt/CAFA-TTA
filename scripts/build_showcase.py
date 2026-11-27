"""Before/after showcase on Kather 2016 (continual-learning line, research use only).

Compares, image by image on the full Kather 2016 set (7 classes mapped to NCT names):
  before: the first model (default v1) without adaptation
  after:  the current model (default v2) with ContinualSession online adaptation (shuffled stream, seed 0)
and writes runs/showcase/showcase.json (per-image predictions), summary.png and examples.png.
Labels are used only for scoring. One external dataset; not a clinical evaluation.

    python scripts/build_showcase.py --data-root C:/Code/DataSet/histopathology
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.registry import ModelRegistry  # noqa: E402
from evaluate_histopathology_external import KATHER_TO_NCT, Evaluator, load_folder  # noqa: E402


def predictions(ev: Evaluator, images, y) -> dict:
    _, logits = ev.forward(ev.source, images)
    static = logits.argmax(1).cpu().numpy()
    run = ev.stream("session_b8", images, y, seed=0, keep_predictions=True)
    adapted = np.empty(len(y), dtype=int)
    adapted[run["order"]] = run["predictions"]
    return {"static": static, "adapted": adapted}


def confusion(true, pred, classes):
    m = np.zeros((len(classes), len(classes)))
    for i, c in enumerate(classes):
        row = pred[true == c]
        for j, d in enumerate(classes):
            m[i, j] = (row == d).mean() if len(row) else 0
    return m


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, default=ROOT / "models_registry")
    parser.add_argument("--before", default="v1")
    parser.add_argument("--after", default=None, help="Default: current registry version")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/showcase")
    args = parser.parse_args(argv)

    registry = ModelRegistry(args.registry_root, "histopathology")
    after = args.after or registry.current_version()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev_before = Evaluator(registry.load(args.before), device)
    ev_after = Evaluator(registry.load(after), device)
    assert ev_before.labels == ev_after.labels
    labels = ev_after.labels
    folder = args.data_root / "kather2016"
    paths = [p for p in sorted(folder.glob("*/*.tif")) if p.parent.name in KATHER_TO_NCT]
    images, y = load_folder(folder, lambda n: labels.index(KATHER_TO_NCT[n]) if n in KATHER_TO_NCT else None)

    before = predictions(ev_before, images, y)
    after_p = predictions(ev_after, images, y)
    results = {
        f"{args.before} không thích nghi": before["static"],
        f"{after} không thích nghi": after_p["static"],
        f"{after} + tự thích nghi": after_p["adapted"],
    }
    acc = {k: float((v == y).mean()) for k, v in results.items()}
    for k, v in acc.items():
        print(f"{k:28s} {100 * v:.1f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    records = [{"path": str(p), "true": labels[t], **{k: labels[v[i]] for k, v in results.items()}}
               for i, (p, t) in enumerate(zip(paths, y))]
    (args.out / "showcase.json").write_text(json.dumps(
        {"before": args.before, "after": after, "accuracy": acc, "labels": labels, "images": records},
        ensure_ascii=False, indent=1), encoding="utf-8")
    render(args.out, labels, y, results, images)


def render(out: Path, labels, y, results, images):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(results)
    classes = sorted(set(y.tolist()))
    names = [labels[c] for c in classes]
    colours = ["#9aa0a6", "#4c8bf5", "#e8590c"]

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(grid[0, :])
    accs = [100 * (results[k] == y).mean() for k in keys]
    bars = ax.barh(keys[::-1], accs[::-1], color=colours[::-1])
    ax.bar_label(bars, fmt="%.1f%%", fontsize=16, fontweight="bold", padding=6)
    ax.set_xlim(0, 100)
    ax.set_title(f"Độ chính xác trên Kather 2016 (ảnh từ phòng lab khác, {len(y):,} ảnh, {len(set(y.tolist()))} loại mô)",
                 fontsize=14, loc="left")
    ax.tick_params(labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    width = 0.27
    for k, key in enumerate(keys):
        rec = [100 * (results[key][y == c] == c).mean() for c in classes]
        ax.bar(np.arange(len(classes)) + (k - 1) * width, rec, width, color=colours[k], label=key)
    ax.set_xticks(np.arange(len(classes)), names)
    ax.set_ylim(0, 105)
    ax.set_title("Tỷ lệ nhận đúng theo loại mô (%)", loc="left")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=1, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    for slot, key in ((1, keys[0]), (2, keys[-1])):
        ax = fig.add_subplot(grid[1, slot])
        m = confusion(y, results[key], classes)
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if m[i, j] >= 0.05:
                    ax.text(j, i, f"{100 * m[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if m[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(classes)), names, fontsize=8)
        ax.set_yticks(range(len(classes)), names, fontsize=8)
        ax.set_xlabel("Mô hình dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_title(f"Ma trận nhầm lẫn: {key}", loc="left", fontsize=10)
    fig.savefig(out / "summary.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Rescued examples: wrong before, right after; 3 per class.
    first, last = results[keys[0]], results[keys[-1]]
    per_class = {c: np.flatnonzero((y == c) & (first != c) & (last == c))[:3] for c in classes}
    fig, axes = plt.subplots(3, len(classes), figsize=(2.3 * len(classes), 9.6),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.08})
    for col, c in enumerate(classes):
        for row in range(3):
            ax = axes[row, col]
            ax.axis("off")
            if row < len(per_class[c]):
                i = per_class[c][row]
                ax.imshow(images[i])
                ax.set_title(f"Thật: {labels[c]}\nTrước: {labels[first[i]]} ✗\nSau: {labels[last[i]]} ✓",
                             fontsize=9)
    fig.suptitle(f"Ảnh mô hình ban đầu nhận sai, hệ thống mới nhận đúng ({keys[0]} → {keys[-1]})",
                 fontsize=13, y=0.99)
    fig.savefig(out / "examples.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
