import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score


DATA_ROOT = Path("/project/data/raw/medmnist")
CHECKPOINT = Path(
    "/outputs/baseline_full_cpu_01/best.pt"
)
RESULT_PATH = Path(
    "/outputs/baseline_full_cpu_01/test_metrics.json"
)

torch.set_num_threads(1)

if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)

if RESULT_PATH.exists():
    raise FileExistsError(
        f"Result already exists: {RESULT_PATH}"
    )

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

test_data = PathMNIST(
    split="test",
    root=str(DATA_ROOT),
    transform=transform,
    download=False,
    size=28,
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features, 9
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.eval()

predictions = []
labels_all = []

print("CAFA-TTA | DOCKER BASELINE TEST", flush=True)
print("Device: CPU", flush=True)
print("Test samples:", len(test_data), flush=True)
print("Checkpoint:", CHECKPOINT, flush=True)

with torch.inference_mode():
    for images, labels in test_loader:
        outputs = model(images)

        predictions.extend(
            outputs.argmax(1).tolist()
        )
        labels_all.extend(
            labels.view(-1).tolist()
        )

accuracy = accuracy_score(
    labels_all, predictions
)

macro_f1 = f1_score(
    labels_all,
    predictions,
    labels=list(range(9)),
    average="macro",
    zero_division=0,
)

metrics = {
    "model": "ResNet18 Baseline",
    "device": "cpu",
    "dataset": "PathMNIST",
    "split": "test",
    "samples": len(test_data),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "checkpoint": CHECKPOINT.name,
}

RESULT_PATH.write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)

print("=" * 55)
print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
print(f"MACRO-F1: {macro_f1:.4f}")
print("RESULT:", RESULT_PATH)
print("FINAL STATUS: PASS")
