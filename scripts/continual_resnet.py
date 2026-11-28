"""Continual learning rounds for the ResNet line, measured on held-out Kather 2016 patients.

Copies the chosen registry version into a scratch registry (the main registry is never touched).
Each round: stream the unlabelled group-A patients through ContinualSession, store predictions
in a fresh experience store exactly as the demo does, run `scripts/consolidate.py`
(pseudo-labels + NCT replay + distillation + evaluation gate), then score the current version on
group B (never used for learning) and on CRC-VAL-HE-7K (forgetting). Labels only score.
Research use only; one external dataset.

    python scripts/continual_resnet.py --data-root C:/Code/DataSet/histopathology
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import consolidate  # noqa: E402
from core.continual import ContinualSession, ExperienceStore  # noqa: E402
from core.registry import ModelRegistry  # noqa: E402
from evaluate_histopathology_external import KATHER_TO_NCT, Evaluator, harmonised_accuracy, load_folder  # noqa: E402

GROUP_A = {"01", "03", "05", "06", "07", "09"}


def kather(folder: Path, labels):
    images, y, groups = [], [], []
    for p in sorted(folder.glob("*/*.tif")):
        if p.parent.name not in KATHER_TO_NCT:
            continue
        with Image.open(p) as im:
            images.append(im.convert("RGB"))
        y.append(labels.index(KATHER_TO_NCT[p.parent.name]))
        groups.append("A" if re.search(r"HE-(\d+)", p.name).group(1) in GROUP_A else "B")
    return images, np.array(y), np.array(groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--version", default="v2")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-pseudo-ratio", type=float, default=0.3)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/continual_resnet")
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.out.exists():
        shutil.rmtree(args.out)
    registry = ModelRegistry(args.out / "registry", "histopathology")
    source = ModelRegistry(ROOT / "models_registry", "histopathology").load(args.version)
    registry.promote(registry.register(source, notes=f"copy of main registry {args.version}"))
    labels = source["labels"]
    images, y, groups = kather(args.data_root / "kather2016", labels)
    a_idx, b_idx = np.flatnonzero(groups == "A"), np.flatnonzero(groups == "B")
    b_images, b_y = [images[i] for i in b_idx], y[b_idx]
    crc_images, crc_y = load_folder(args.data_root / "crc_val_he_7k",
                                    lambda n: labels.index(n) if n in labels else None)

    def evaluate(round_no, version, pseudo_acc=None, n_pseudo=0):
        ev = Evaluator(registry.load(version), device)
        crc_acc = round(float((ev.forward(ev.source, crc_images)[1].argmax(1).cpu().numpy() == crc_y).mean()), 4)
        _, logits = ev.forward(ev.source, b_images)
        pred = logits.argmax(1).cpu().numpy()
        recall = {labels[c]: round(float((pred[b_y == c] == c).mean()), 3) for c in np.unique(b_y)}
        tta = [ev.stream("session_b8", b_images, b_y, s) for s in (0, 1, 2)]
        tta_class = [ev.stream("session_b8", b_images, b_y, s, class_ordered=True) for s in (0, 1, 2)]
        row = {"round": round_no, "version": version, "pseudo_labels": n_pseudo, "pseudo_label_accuracy": pseudo_acc,
               "crc_val_acc": crc_acc,
               "B_static_acc": round(float((pred == b_y).mean()), 4),
               "B_static_acc_harmonised": round(harmonised_accuracy(pred, b_y, labels), 4),
               "B_static_mean_recall": round(float(np.mean(list(recall.values()))), 4), "B_recall": recall,
               "B_tta_iid_acc": round(float(np.mean([r["accuracy"] for r in tta])), 4),
               "B_tta_iid_acc_harmonised": round(float(np.mean([r["accuracy_harmonised"] for r in tta])), 4),
               "B_tta_class_acc": round(float(np.mean([r["accuracy"] for r in tta_class])), 4),
               "B_tta_class_acc_harmonised": round(float(np.mean([r["accuracy_harmonised"] for r in tta_class])), 4)}
        print(json.dumps({k: v for k, v in row.items() if k != "B_recall"}), flush=True)
        return row

    history = [evaluate(0, registry.current_version())]
    for r in range(1, args.rounds + 1):
        exp_root = args.out / f"experience_round{r}"
        store = ExperienceStore(exp_root / "histopathology")
        version = registry.current_version()
        session = ContinualSession(registry.load(version), device, batch_size=8)
        order = np.random.default_rng(100 + r).permutation(a_idx)
        correct = []
        for i in range(0, len(order), 64):
            batch = order[i:i + 64]
            for j, res in zip(batch, session.predict([images[k] for k in batch])):
                if res["ood"]:
                    continue
                store.add(images[j], {"modality": "histopathology", "model_version": version,
                                      "prediction": res["label"], "confidence": res["confidence"], "ood": False,
                                      "agreement": res["agreement"], "source_label": res["source_label"]})
                if res["agreement"] and res["confidence"] >= 0.9:
                    correct.append(res["label"] == labels[y[j]])
        pseudo_acc = round(float(np.mean(correct)), 4) if correct else None
        new = consolidate.main(["--modality", "histopathology", "--registry-root", str(args.out / "registry"),
                                "--experience-root", str(exp_root), "--source-data-dir",
                                str(args.data_root / "nct_crc_100k"), "--epochs", str(args.epochs),
                                "--batch-size", "32", "--num-workers", str(args.num_workers),
                                "--max-pseudo-ratio", str(args.max_pseudo_ratio)])
        history.append(evaluate(r, registry.current_version(), pseudo_acc, len(correct)) | {"accepted": new is not None})
    (args.out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
