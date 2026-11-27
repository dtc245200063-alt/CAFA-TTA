"""CAFA-TTA | Stage 1: Baseline Training (Modality-Aware + High-Res + Pretrained)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from core.training_utils import (
    add_common_train_args,
    build_backbone,
    default_checkpoint_path,
    load_image_datasets,
    resolve_input_config,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CAFA-TTA Baseline Training")
    add_common_train_args(parser, default_epochs=3, default_lr=1e-3)
    args = parser.parse_args()

    is_legacy = args.modality is None and args.data_dir is None
    pretrained = False if (is_legacy and "--pretrained" not in sys.argv) else args.pretrained
    cfg = resolve_input_config(
        modality=args.modality,
        size=args.size,
        backbone=args.backbone,
        pretrained=pretrained,
        batch_size=args.batch_size if args.batch_size is not None else (32 if is_legacy else None),
        grad_accum=args.grad_accum,
    )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    size = cfg["input_size"]
    mean, std = cfg["normalization"]["mean"], cfg["normalization"]["std"]

    transform_ops = []
    if size != 28 or args.data_dir is not None:
        transform_ops.append(transforms.Resize((size, size)))
    transform_ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    base_transform = transforms.Compose(transform_ops)

    if args.data_dir is not None:
        train_data, val_data, class_names = load_image_datasets(
            args.data_dir, base_transform, base_transform, seed=args.seed
        )
    else:
        from medmnist import PathMNIST

        data_dir = str(ROOT / "data/raw/medmnist")
        train_data = PathMNIST(split="train", root=data_dir, transform=base_transform, download=False, size=28)
        val_data = PathMNIST(split="val", root=data_dir, transform=base_transform, download=False, size=28)
        class_names = [f"class_{i}" for i in range(9)]

    num_classes = len(class_names)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    model = build_backbone(cfg["backbone"], num_classes=num_classes, pretrained=cfg["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_path = args.output_checkpoint or default_checkpoint_path(ROOT, "baseline", cfg, is_legacy)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CAFA-TTA | BASELINE TRAINING")
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Modality: {cfg['modality']} | Size: {size}x{size} | Backbone: {cfg['backbone']} (pretrained={cfg['pretrained']})")
    print(f"Classes: {num_classes} | Batch: {cfg['batch_size']} x Accum {cfg['grad_accum']} | Epochs: {args.epochs}")
    print("=" * 60, flush=True)

    best_val_acc = -1.0
    grad_accum = cfg["grad_accum"]

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum

            scaler.scale(loss).backward()

            if step % grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += (loss.item() * grad_accum) * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.view(-1).long().to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_acc": best_val_acc,
                    "num_classes": num_classes,
                    "labels": class_names,
                    "architecture": cfg["backbone"],
                    "modality": cfg["modality"],
                    "input_size": size,
                    "normalization": cfg["normalization"],
                    "training_type": "baseline",
                },
                output_path,
            )

    print("=" * 60)
    print(f"TRAINING COMPLETED | Best Val Acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {output_path}")
    if torch.cuda.is_available():
        print("Peak GPU VRAM:", round(torch.cuda.max_memory_allocated() / 1024**2, 2), "MB")


if __name__ == "__main__":
    main()
