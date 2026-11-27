"""CAFA-TTA | Stage 3B: Balanced Moderate-Color-Augmented Fine-Tuning."""
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
from torchvision.transforms import functional as TF
from tqdm import tqdm

from core.training_utils import (
    add_common_train_args,
    build_backbone,
    default_checkpoint_path,
    load_domain_shift,
    load_image_datasets,
    resolve_input_config,
    seed_everything,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="CAFA-TTA Balanced Robust Training")
    add_common_train_args(parser, default_epochs=6, default_lr=3e-5)
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
    normalize = transforms.Normalize(mean, std)
    shift = load_domain_shift(ROOT)

    resize_op = [transforms.Resize((size, size))] if (size != 28 or args.data_dir is not None) else []
    train_transform = transforms.Compose([
        *resize_op,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02)
        ], p=0.6),
        transforms.ToTensor(),
        normalize,
    ])
    original_transform = transforms.Compose([*resize_op, transforms.ToTensor(), normalize])

    def shifted_transform(image):
        if size != 28 or args.data_dir is not None:
            image = TF.resize(image, [size, size])
        image = TF.adjust_brightness(image, shift["brightness"])
        image = TF.adjust_contrast(image, shift["contrast"])
        image = TF.adjust_saturation(image, shift["saturation"])
        return normalize(TF.to_tensor(image))

    if args.data_dir is not None:
        train_ds, orig_val_ds, class_names = load_image_datasets(
            args.data_dir, train_transform, original_transform, seed=args.seed
        )
        _, shifted_val_ds, _ = load_image_datasets(
            args.data_dir, train_transform, shifted_transform, seed=args.seed
        )
    else:
        from medmnist import PathMNIST

        data_root = str(ROOT / "data/raw/medmnist")
        train_ds = PathMNIST(split="train", root=data_root, transform=train_transform, download=False, size=28)
        orig_val_ds = PathMNIST(split="val", root=data_root, transform=original_transform, download=False, size=28)
        shifted_val_ds = PathMNIST(split="val", root=data_root, transform=shifted_transform, download=False, size=28)
        class_names = [f"class_{i}" for i in range(9)]

    num_classes = len(class_names)
    loader_kw = {"batch_size": cfg["batch_size"], "num_workers": args.num_workers, "pin_memory": use_amp}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    original_loader = DataLoader(orig_val_ds, shuffle=False, **loader_kw)
    shifted_loader = DataLoader(shifted_val_ds, shuffle=False, **loader_kw)

    init_path = args.init_checkpoint or default_checkpoint_path(ROOT, "enhanced", cfg, is_legacy)
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
    backbone_name = checkpoint.get("architecture", cfg["backbone"])

    model = build_backbone(backbone_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    @torch.inference_mode()
    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        return correct / max(total, 1)

    output_dir = ROOT / "experiments/balanced_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = args.output_checkpoint or default_checkpoint_path(ROOT, "balanced", cfg, is_legacy)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CAFA-TTA | BALANCED ROBUST TRAINING")
    print(f"Device: {device} | AMP: {use_amp} | Init: {init_path}")
    print(f"Modality: {cfg['modality']} | Size: {size}x{size} | Backbone: {backbone_name}")
    print(f"Batch: {cfg['batch_size']} x Accum {cfg['grad_accum']} | Epochs: {args.epochs} | LR: {args.lr}")
    print("=" * 60, flush=True)

    history = []
    best_score = -1.0
    grad_accum = cfg["grad_accum"]

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"), 1):
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

            running_loss += (loss.item() * grad_accum) * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        original_acc = evaluate(original_loader)
        shifted_acc = evaluate(shifted_loader)
        selection_score = (original_acc + shifted_acc) / 2.0

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "original_val_accuracy": original_acc,
            "shifted_val_accuracy": shifted_acc,
            "selection_score": selection_score,
        })
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Loss: {train_loss:.4f} | "
            f"Orig Val: {original_acc:.4f} | Shifted Val: {shifted_acc:.4f} | Score: {selection_score:.4f}",
            flush=True,
        )

        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_score": best_score,
                    "original_val_acc": original_acc,
                    "shifted_val_acc": shifted_acc,
                    "num_classes": num_classes,
                    "labels": class_names,
                    "architecture": backbone_name,
                    "modality": cfg["modality"],
                    "input_size": size,
                    "normalization": cfg["normalization"],
                    "training_type": "balanced_color_finetuning",
                },
                output_checkpoint,
            )

        scheduler.step()

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
