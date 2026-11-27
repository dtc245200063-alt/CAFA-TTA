"""CAFA-TTA | Stage 2: Enhanced Geometry-Augmented Fine-Tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from core.training_utils import (
    add_common_train_args,
    build_backbone,
    default_checkpoint_path,
    load_image_datasets,
    resolve_input_config,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CAFA-TTA Enhanced Training")
    add_common_train_args(parser, default_epochs=15, default_lr=1e-4)
    args = parser.parse_args()

    is_legacy = args.modality is None and args.data_dir is None
    pretrained = False if (is_legacy and "--pretrained" not in sys.argv) else args.pretrained
    cfg = resolve_input_config(
        modality=args.modality,
        size=args.size,
        backbone=args.backbone,
        pretrained=pretrained,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    size = cfg["input_size"]
    mean, std = cfg["normalization"]["mean"], cfg["normalization"]["std"]

    resize_op = [transforms.Resize((size, size))] if (size != 28 or args.data_dir is not None) else []
    train_transform = transforms.Compose([
        *resize_op,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        *resize_op,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.data_dir is not None:
        train_dataset, val_dataset, class_names = load_image_datasets(
            args.data_dir, train_transform, val_transform, seed=args.seed
        )
    else:
        from medmnist import PathMNIST

        data_root = str(ROOT / "data/raw/medmnist")
        train_dataset = PathMNIST(split="train", root=data_root, transform=train_transform, download=False, size=28)
        val_dataset = PathMNIST(split="val", root=data_root, transform=val_transform, download=False, size=28)
        class_names = [f"class_{i}" for i in range(9)]

    num_classes = len(class_names)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    init_path = args.init_checkpoint or default_checkpoint_path(ROOT, "baseline", cfg, is_legacy)
    source_checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
    backbone_name = source_checkpoint.get("architecture", cfg["backbone"])

    model = build_backbone(backbone_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(source_checkpoint["model_state_dict"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = ROOT / "experiments/enhanced_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_checkpoint or default_checkpoint_path(ROOT, "enhanced", cfg, is_legacy)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CAFA-TTA | ENHANCED TRAINING")
    print(f"Device: {device} | AMP: {use_amp} | Init: {init_path}")
    print(f"Modality: {cfg['modality']} | Size: {size}x{size} | Backbone: {backbone_name}")
    print(f"Batch: {cfg['batch_size']} x Accum {cfg['grad_accum']} | Epochs: {args.epochs} | LR: {args.lr}")
    print("=" * 60, flush=True)

    history = []
    best_val_acc = -1.0
    grad_accum = cfg["grad_accum"]

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, (images, labels) in enumerate(progress, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels) / grad_accum

            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += (loss.item() * grad_accum) * images.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += images.size(0)
            progress.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        train_loss = running_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.view(-1).long().to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / max(val_total, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
        })
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
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
                    "architecture": backbone_name,
                    "modality": cfg["modality"],
                    "input_size": size,
                    "normalization": cfg["normalization"],
                    "training_type": "enhanced_finetuning",
                },
                checkpoint_path,
            )
            print(f"New best checkpoint saved: {best_val_acc:.4f}", flush=True)

        scheduler.step()

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
