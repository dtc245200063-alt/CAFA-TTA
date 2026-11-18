
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import PathMNIST

# ==========================================
# CAFA-TTA | COMPUTER VISION BASELINE
# ==========================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("CAFA-TTA | RESNET-18 TRAINING")
print("Device:", device)
print("=" * 55)

# 1. Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

# 2. Datasets
data_dir = "data/raw/medmnist"

train_data = PathMNIST(
    split="train",
    root=data_dir,
    transform=transform,
    download=False,
    size=28
)

val_data = PathMNIST(
    split="val",
    root=data_dir,
    transform=transform,
    download=False,
    size=28
)

# 3. DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# 4. ResNet-18
model = models.resnet18(weights=None)

# Thay đổi classifier thành 9 lớp
model.fc = nn.Linear(model.fc.in_features, 9)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# 5. Training loop
best_val_acc = 0.0

Path("checkpoints").mkdir(exist_ok=True)

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(
            device, non_blocking=True
        )
        labels = labels.view(-1).long().to(
            device, non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (
            outputs.argmax(1) == labels
        ).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # 6. Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            outputs = model(images)

            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc:.4f} "
        f"| Val Acc: {val_acc:.4f}",
        flush=True
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_val_acc,
                "num_classes": 9,
            },
            "checkpoints/resnet18_pathmnist_best.pt"
        )

print("=" * 55)
print("TRAINING COMPLETED")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print("Checkpoint: checkpoints/resnet18_pathmnist_best.pt")

if torch.cuda.is_available():
    print(
        "Peak GPU VRAM:",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2
        ),
        "MB"
    )
