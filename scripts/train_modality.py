"""Train one modality model with the continual-learning recipe and register it.

Recipe: modality-specific augmentation, label smoothing, warmup + cosine LR, EMA weights,
class-balanced sampling, AMP. After training it fits temperature scaling, computes class
prototypes and calibrates the OOD gate, then registers the model in `models_registry/<modality>/`.

Example:
    python scripts/train_modality.py --modality histopathology --data-dir D:/DataSet/histopathology/nct_crc_100k
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from core.continual import build_checkpoint
from core.registry import ModelRegistry
from core.trainer import train_classifier
from core.training_utils import (
    add_common_train_args,
    build_backbone,
    build_transforms,
    load_image_datasets,
    resolve_input_config,
    seed_everything,
)


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_train_args(parser, default_epochs=15, default_lr=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    parser.add_argument("--no-balanced", action="store_true", help="Disable class-balanced sampling")
    parser.add_argument("--prototype-samples", type=int, default=5000,
                        help="Max training images used to compute class prototypes")
    parser.add_argument("--registry-root", type=Path, default=ROOT / "models_registry")
    parser.add_argument("--promote", action="store_true",
                        help="Promote the new version even if the registry already has a current one")
    parser.add_argument("--augment", choices=["light", "strong"], default="light",
                        help="'strong' adds crop/HED/colour/blur augmentation (histopathology only)")
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    if args.modality is None or args.data_dir is None:
        parser.error("--modality and --data-dir are required")
    if args.augment == "strong" and args.modality != "histopathology":
        parser.error("--augment strong is only defined for histopathology")

    cfg = resolve_input_config(
        modality=args.modality, size=args.size, backbone=args.backbone, pretrained=args.pretrained,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
    )
    cfg["augment"] = args.augment
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, eval_tf = build_transforms(cfg, train=True), build_transforms(cfg, train=False)
    train_ds, val_ds, labels = load_image_datasets(args.data_dir, train_tf, eval_tf, seed=args.seed)
    # Same deterministic split, without augmentation, for prototypes.
    plain_train, _, _ = load_image_datasets(args.data_dir, eval_tf, eval_tf, seed=args.seed)
    n_proto = min(args.prototype_samples, len(plain_train))
    proto_idx = sorted(random.Random(args.seed).sample(range(len(plain_train)), n_proto))

    print(f"Modality {cfg['modality']} | {cfg['backbone']} | {cfg['input_size']}px | "
          f"{len(labels)} classes | train {len(train_ds)} | val {len(val_ds)} | device {device}", flush=True)

    model = build_backbone(cfg["backbone"], num_classes=len(labels), pretrained=cfg["pretrained"])
    model, history, metrics = train_classifier(
        model, train_ds, val_ds, len(labels),
        epochs=args.epochs, lr=args.lr, batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
        device=device, num_workers=args.num_workers, label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs, ema_decay=args.ema_decay or None,
        balanced=not args.no_balanced,
    )

    loader_kw = {"batch_size": cfg["batch_size"], "num_workers": args.num_workers}
    checkpoint = build_checkpoint(
        model, labels=labels, cfg=cfg,
        prototype_loader=DataLoader(Subset(plain_train, proto_idx), **loader_kw),
        val_loader=DataLoader(val_ds, **loader_kw),
        device=device, training_type="modality", metrics=metrics,
        extra={"history": history, "data_dir": str(args.data_dir), "seed": args.seed, "augment": args.augment},
    )
    if args.output_checkpoint:
        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.output_checkpoint)

    registry = ModelRegistry(args.registry_root, cfg["modality"])
    version = registry.register(checkpoint, notes=args.notes or "initial training")
    if registry.current_version() is None or args.promote:
        registry.promote(version)
    print(f"Registered {cfg['modality']}/{version} | val macro-F1 {metrics['macro_f1']:.4f} | "
          f"T={checkpoint['temperature']:.3f} | current={registry.current_version()}", flush=True)
    return version


if __name__ == "__main__":
    main()
