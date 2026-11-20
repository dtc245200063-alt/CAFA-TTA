
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# CAFA-TTA | DOMAIN SHIFT EXPERIMENT
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_DIR = "data/raw/medmnist"
CHECKPOINT = "checkpoints/resnet18_pathmnist_best.pt"

BATCH_SIZE = 64

# Chỉ dùng validation để phát triển thuật toán.
# Test được giữ lại cho đánh giá cuối cùng.
SPLIT = "val"

print("=" * 55)
print("CAFA-TTA | DOMAIN SHIFT EVALUATION")
print("Device:", DEVICE)
print("Split:", SPLIT)
print("=" * 55)

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

original_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

# Biến đổi cố định, có thể tái lập.
# Đây là shift mô phỏng, không phải miền bệnh viện thực.
def shifted_transform(image):
    image = TF.adjust_brightness(image, 0.95)
    image = TF.adjust_contrast(image, 1.05)
    image = TF.adjust_saturation(image, 0.95)
    return normalize(TF.to_tensor(image))

def create_loader(transform):
    dataset = PathMNIST(
        split=SPLIT,
        root=DATA_DIR,
        transform=transform,
        download=False,
        size=28
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda"
    )

# Load checkpoint
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

def evaluate(loader):
    true_labels = []
    predictions = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)

            logits = model(images)
            pred = logits.argmax(dim=1)

            predictions.extend(
                pred.cpu().tolist()
            )
            true_labels.extend(
                labels.view(-1).tolist()
            )

    return {
        "accuracy": accuracy_score(
            true_labels, predictions
        ),
        "macro_f1": f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }

print("\n[1/2] Evaluating original images...")
original_results = evaluate(
    create_loader(original_transform)
)

print("[2/2] Evaluating shifted images...")
shifted_results = evaluate(
    create_loader(shifted_transform)
)

print("\n" + "=" * 55)
print("RESULTS")
print("=" * 55)

print(
    f"Original Accuracy: "
    f"{original_results['accuracy']:.4f}"
)
print(
    f"Shifted Accuracy: "
    f"{shifted_results['accuracy']:.4f}"
)
print(
    f"Original Macro-F1: "
    f"{original_results['macro_f1']:.4f}"
)
print(
    f"Shifted Macro-F1: "
    f"{shifted_results['macro_f1']:.4f}"
)

accuracy_change = (
    shifted_results["accuracy"]
    - original_results["accuracy"]
)

print(
    f"Accuracy Change: "
    f"{accuracy_change * 100:+.2f} percentage points"
)

output_dir = Path("experiments/domain_shift")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    "split": SPLIT,
    "shift": {
        "brightness": 0.75,
        "contrast": 1.35,
        "saturation": 0.70
    },
    "original": original_results,
    "shifted": shifted_results,
    "accuracy_change": accuracy_change
}

with open(
    output_dir / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

print("\nResults saved.")
print("DOMAIN SHIFT EVALUATION COMPLETED")
