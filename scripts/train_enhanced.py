
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import PathMNIST
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

SEED = 42
EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Training augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# Validation uses no random augmentation
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

train_dataset = PathMNIST(
    split="train",
    root=str(ROOT / "data/raw/medmnist"),
    transform=train_transform,
    download=False,
    size=28,
)

val_dataset = PathMNIST(
    split="val",
    root=str(ROOT / "data/raw/medmnist"),
    transform=val_transform,
    download=False,
    size=28,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)

# Build model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

source_checkpoint = torch.load(
    ROOT / "checkpoints/resnet18_pathmnist_best.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    source_checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)

output_dir = ROOT / "experiments/enhanced_training"
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = (
    ROOT / "checkpoints/resnet18_pathmnist_enhanced_best.pt"
)

history = []
best_val_acc = 0.0

print("=" * 60)
print("CAFA-TTA | ENHANCED RESNET-18 TRAINING")
print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Starting LR:", LR)
print("AMP:", use_amp)
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:
        images = images.to(
            DEVICE, non_blocking=True
        )
        labels = labels.view(-1).long().to(
            DEVICE, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        train_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        train_total += images.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(
                DEVICE, non_blocking=True
            )
            labels = labels.view(-1).long().to(
                DEVICE, non_blocking=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):
                logits = model(images)

            val_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            val_total += images.size(0)

    val_acc = val_correct / val_total

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_val_acc,
                "num_classes": 9,
                "training_type": "enhanced_finetuning",
            },
            checkpoint_path,
        )

        print(
            f"New best checkpoint saved: "
            f"{best_val_acc:.4f}"
        )

    scheduler.step()

with open(
    output_dir / "history.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(history, f, indent=2)

print("\n" + "=" * 60)
print("ENHANCED TRAINING COMPLETED")
print("Best Validation Accuracy:", best_val_acc)
print("Checkpoint:", checkpoint_path)
print("=" * 60)
