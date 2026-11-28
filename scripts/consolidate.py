"""Consolidate experience into a new model version, behind an evaluation gate.

Training set = images the user labelled in the demo
             + confident pseudo-labels (in-distribution, classifier/prototype agreement), capped
             + class-balanced replay of the original source training data.
Loss = cross-entropy + distillation from the current model (Learning without Forgetting).
The candidate is promoted only if it does not regress on the fixed source validation split.

Examples:
    python scripts/consolidate.py --modality histopathology --source-data-dir D:/DataSet/histopathology/nct_crc_100k
    python scripts/consolidate.py --modality histopathology --rollback
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from core.continual import ExperienceStore, build_checkpoint, build_model_from_checkpoint
from core.registry import ModelRegistry, passes_gate
from core.trainer import classification_metrics, dataset_targets, predict_logits, train_classifier
from core.training_utils import _IndexedImageDataset, build_transforms, load_image_datasets, seed_everything


def select_experience(records: list[dict], labels: list[str], min_confidence: float):
    """Split experience records into user-labelled and pseudo-labelled (id, class index) pairs."""
    labelled, pseudo = [], []
    for r in records:
        user_label = r.get("user_label")
        if user_label is not None:
            if user_label in labels:  # anything else (e.g. "rejected") is never trained on
                labelled.append((r["id"], labels.index(user_label)))
        elif (not r.get("ood", True) and r.get("agreement") and r.get("prediction") in labels
              and r.get("confidence", 0.0) >= min_confidence):
            pseudo.append((r["id"], labels.index(r["prediction"])))
    return labelled, pseudo


def class_balanced_indices(targets: list[int], per_class: int, rng: random.Random) -> list[int]:
    by_class: dict[int, list[int]] = {}
    for idx, t in enumerate(targets):
        by_class.setdefault(t, []).append(idx)
    chosen = []
    for idxs in by_class.values():
        chosen += rng.sample(idxs, min(per_class, len(idxs)))
    return sorted(chosen)


def main(argv=None) -> str | None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modality", required=True)
    p.add_argument("--registry-root", type=Path, default=ROOT / "models_registry")
    p.add_argument("--experience-root", type=Path, default=ROOT / "data/experience")
    p.add_argument("--rollback", action="store_true", help="Restore the previously promoted version and exit")
    p.add_argument("--source-data-dir", type=Path, help="ImageFolder of the original training data")
    p.add_argument("--replay-per-class", type=int, default=200)
    p.add_argument("--min-confidence", type=float, default=0.9)
    p.add_argument("--max-pseudo-ratio", type=float, default=0.3,
                   help="Max share of pseudo-labelled images in the fine-tuning set")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--max-macro-f1-drop", type=float, default=0.005)
    p.add_argument("--max-class-recall-drop", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if not 0.0 <= args.max_pseudo_ratio < 1.0:
        p.error("--max-pseudo-ratio must be in [0, 1)")

    registry = ModelRegistry(args.registry_root, args.modality)
    if args.rollback:
        version = registry.rollback()
        print(f"Rolled back {args.modality} to {version}")
        return version
    if args.source_data_dir is None:
        p.error("--source-data-dir is required (replay data and the fixed evaluation split)")

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_version = registry.current_version()
    checkpoint = registry.load()
    labels = checkpoint["labels"]
    cfg = {"backbone": checkpoint["architecture"], "modality": checkpoint["modality"],
           "input_size": checkpoint["input_size"], "normalization": checkpoint["normalization"]}
    train_tf, eval_tf = build_transforms(cfg, train=True), build_transforms(cfg, train=False)

    store = ExperienceStore(args.experience_root / args.modality)
    labelled, pseudo = select_experience(store.records(), labels, args.min_confidence)

    src_train, src_val, src_classes = load_image_datasets(args.source_data_dir, train_tf, eval_tf, seed=args.seed)
    plain_train, _, _ = load_image_datasets(args.source_data_dir, eval_tf, eval_tf, seed=args.seed)
    if src_classes != labels:
        raise ValueError(f"Source classes {src_classes} do not match model labels {labels}")
    replay_idx = class_balanced_indices(dataset_targets(src_train), args.replay_per_class, rng)

    cap = int(args.max_pseudo_ratio / (1.0 - args.max_pseudo_ratio) * (len(labelled) + len(replay_idx)))
    if len(pseudo) > cap:
        pseudo = rng.sample(pseudo, cap)
    print(f"{args.modality}/{current_version}: {len(labelled)} user-labelled, {len(pseudo)} pseudo-labelled, "
          f"{len(replay_idx)} replay images", flush=True)
    if not labelled and not pseudo:
        print("Nothing new to learn; registry unchanged.")
        return None

    experience = [(store.image_path(i), y) for i, y in labelled + pseudo]
    train_ds = ConcatDataset([_IndexedImageDataset(experience, transform=train_tf), Subset(src_train, replay_idx)])
    eval_loader = DataLoader(src_val, batch_size=args.batch_size, num_workers=args.num_workers)

    teacher = build_model_from_checkpoint(checkpoint).to(device)
    current_metrics = classification_metrics(*predict_logits(teacher, eval_loader, device), len(labels))
    student, history, _ = train_classifier(
        build_model_from_checkpoint(checkpoint), train_ds, src_val, len(labels),
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=device,
        num_workers=args.num_workers, warmup_epochs=0.0, ema_decay=0.99,
        teacher=teacher, kd_weight=args.kd_weight,
    )
    candidate_metrics = classification_metrics(*predict_logits(student, eval_loader, device), len(labels))
    accepted, reasons = passes_gate(candidate_metrics, current_metrics,
                                    args.max_macro_f1_drop, args.max_class_recall_drop)

    log = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "parent": current_version,
           "accepted": accepted, "reasons": reasons, "current": current_metrics, "candidate": candidate_metrics,
           "labelled": len(labelled), "pseudo": len(pseudo), "replay": len(replay_idx)}
    registry.dir.mkdir(parents=True, exist_ok=True)
    with open(registry.dir / "consolidation_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")
    print(f"Current macro-F1 {current_metrics['macro_f1']:.4f} | candidate {candidate_metrics['macro_f1']:.4f}")
    if not accepted:
        print("Candidate REJECTED by the evaluation gate: " + "; ".join(reasons))
        return None

    proto_ds = ConcatDataset([_IndexedImageDataset(experience, transform=eval_tf), Subset(plain_train, replay_idx)])
    new_checkpoint = build_checkpoint(
        student, labels=labels, cfg=cfg,
        prototype_loader=DataLoader(proto_ds, batch_size=args.batch_size, num_workers=args.num_workers),
        val_loader=eval_loader, device=device, training_type="consolidation", metrics=candidate_metrics,
        extra={"history": history, "parent": current_version,
               "experience": {"labelled": len(labelled), "pseudo": len(pseudo), "replay": len(replay_idx)}},
    )
    version = registry.register(new_checkpoint, notes=f"consolidation from {current_version}")
    registry.promote(version)
    print(f"Promoted {args.modality}/{version}")
    return version


if __name__ == "__main__":
    main()
