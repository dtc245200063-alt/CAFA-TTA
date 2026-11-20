
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

SEED = 42

EPOCHS = 6
BATCH_SIZE = 64
LR = 3e-5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Fixed domain shift used in validation experiments
with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    shift = json.load(f)["shift"]

normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3
)

# Randomized color and geometry augmentation

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),

    # Apply color augmentation to 60% of samples.
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.02,
        )
    ], p=0.6),

    transforms.ToTensor(),
    normalize,
])

original_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

def shifted_transform(image):
    image = TF.adjust_brightness(
        image, shift["brightness"]
    )
    image = TF.adjust_contrast(
        image, shift["contrast"]
    )
    image = TF.adjust_saturation(
        image, shift["saturation"]
    )
    return normalize(TF.to_tensor(image))

data_root = str(ROOT / "data/raw/medmnist")

def make_dataset(split, transform):
    return PathMNIST(
        split=split,
        root=data_root,
        transform=transform,
        download=False,
        size=28,
    )

def make_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

train_loader = make_loader(
    make_dataset("train", train_transform),
    shuffle=True,
)

original_loader = make_loader(
    make_dataset("val", original_transform)
)

shifted_loader = make_loader(
    make_dataset("val", shifted_transform)
)

# Initialize using enhanced checkpoint
model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features, 9
)

source_path = (
    ROOT / "checkpoints/"
    "resnet18_pathmnist_enhanced_best.pt"
)

checkpoint = torch.load(
    source_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda", enabled=use_amp
)

@torch.inference_mode()
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.view(-1).long().to(DEVICE)

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):
            logits = model(images)

        correct += (
            logits.argmax(1) == labels
        ).sum().item()

        total += len(labels)

    return correct / total

output_dir = ROOT / "experiments/balanced_training"
output_dir.mkdir(parents=True, exist_ok=True)

output_checkpoint = (
    ROOT / "checkpoints/"
   "resnet18_pathmnist_balanced_best.pt"
)

# Track original and shifted validation.
# Selection favors their average.
history = []
best_score = -1.0

print("=" * 60)
print("CAFA-TTA | BALANCED ROBUST TRAINING")
print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Starting LR:", LR)
print("Shift:", shift)
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    ):
        images = images.to(DEVICE)
        labels = labels.view(-1).long().to(DEVICE)

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

        running_loss += loss.item() * len(labels)
        correct += (
            logits.argmax(1) == labels
        ).sum().item()
        total += len(labels)

    train_loss = running_loss / total
    train_acc = correct / total

    original_acc = evaluate(original_loader)
    shifted_acc = evaluate(shifted_loader)

    selection_score = (
        original_acc + shifted_acc
    ) / 2.0

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "original_val_accuracy": original_acc,
        "shifted_val_accuracy": shifted_acc,
        "selection_score": selection_score,
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Original Val: {original_acc:.4f} | "
        f"Shifted Val: {shifted_acc:.4f} | "
        f"Score: {selection_score:.4f}",
        flush=True,
    )

 
    if selection_score > best_score:
        best_score = selection_score

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "best_val_score": best_score,
            "original_val_accuracy": original_acc,
            "shifted_val_accuracy": shifted_acc,
            "num_classes": 9,
            "training_type": "balanced_color_finetuning",
        }, output_checkpoint)

        print("New best balanced checkpoint saved")

    scheduler.step()

with open(
    output_dir / "history.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(history, f, indent=2)

print("\nBALANCED TRAINING COMPLETED")
print("Best combined validation score:", best_score)
print("Checkpoint:", output_checkpoint)

