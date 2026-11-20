
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import PathMNIST

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import numpy as np
import matplotlib.pyplot as plt

# =======================================
# CAFA-TTA | BASELINE EVALUATION
# =======================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("CAFA-TTA | EVALUATION")
print("Device:", device)

# 1. Load test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

test_dataset = PathMNIST(
    split="test",
    root="data/raw/medmnist",
    transform=transform,
    download=False,
    size=28,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

# 2. Rebuild model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

checkpoint_path = Path(
    "checkpoints/resnet18_pathmnist_best.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

# 3. Inference
y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        y_true.extend(
            labels.view(-1).cpu().numpy().tolist()
        )

        y_pred.extend(
            predictions.cpu().numpy().tolist()
        )

# 4. Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, average="macro"
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(9)),
    zero_division=0,
)

print("\nRESULTS")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print("\nClassification Report:")
print(report)

# 5. Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(9))
)

output_dir = Path("experiments/baseline")
output_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, cmap="Blues")

ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("PathMNIST - ResNet18 Confusion Matrix")
ax.set_xticks(range(9))
ax.set_yticks(range(9))

fig.colorbar(im, ax=ax)
fig.tight_layout()

fig.savefig(
    output_dir / "confusion_matrix.png",
    dpi=150,
)
plt.close(fig)

# 6. Save metrics
with open(
    output_dir / "metrics.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(f"Accuracy: {accuracy:.6f}\n")
    file.write(f"Macro-F1: {macro_f1:.6f}\n\n")
    file.write(report)

print("\nEvaluation saved at:", output_dir)
print("EVALUATION COMPLETED")
