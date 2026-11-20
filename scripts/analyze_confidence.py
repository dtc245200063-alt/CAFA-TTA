
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernel.confidence import confidence_score

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Use the same validation-domain shift
with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    config = json.load(f)

assert config["split"] == "val"
shift = config["shift"]

normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3
)

def transform(image):
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

dataset = PathMNIST(
    split="val",
    root=str(ROOT / "data/raw/medmnist"),
    transform=transform,
    download=False,
    size=28
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# Load trained source model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

checkpoint = torch.load(
    ROOT / "checkpoints/resnet18_pathmnist_best.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)

# BN-only mode: no gradient, target batch statistics
model.train()
model.requires_grad_(False)

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None

all_scores = []
all_correct = []

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.view(-1).long().to(DEVICE)

        logits = model(images)
        predicted = logits.argmax(dim=1)

        scores = confidence_score(logits)
        correct = predicted.eq(labels)

        all_scores.append(scores.cpu())
        all_correct.append(correct.cpu())

scores = torch.cat(all_scores)
correct = torch.cat(all_correct)

print("=" * 55)
print("CAFA-TTA | CONFIDENCE ANALYSIS")
print("Device:", DEVICE)
print("Shift:", shift)
print("Total images:", len(scores))
print("Overall Accuracy:", round(correct.float().mean().item(), 4))
print("=" * 55)

results = {
    "split": "val",
    "shift": shift,
    "total_samples": len(scores),
    "overall_accuracy": correct.float().mean().item(),
    "thresholds": {}
}

for threshold in [0.3, 0.5, 0.7, 0.9]:
    mask = scores >= threshold
    count = mask.sum().item()

    coverage = count / len(scores)

    selected_accuracy = (
        correct[mask].float().mean().item()
        if count > 0 else None
    )

    rejected_count = (~mask).sum().item()
    rejected_accuracy = (
        correct[~mask].float().mean().item()
        if rejected_count > 0 else None
    )

    print(f"\nThreshold: {threshold}")
    print(f"Selected: {count}")
    print(f"Coverage: {coverage:.2%}")
    print(
        "Selected Accuracy:",
        f"{selected_accuracy:.2%}"
        if selected_accuracy is not None else "N/A"
    )
    print(
        "Rejected Accuracy:",
        f"{rejected_accuracy:.2%}"
        if rejected_accuracy is not None else "N/A"
    )

    results["thresholds"][str(threshold)] = {
        "selected": count,
        "coverage": coverage,
        "selected_accuracy": selected_accuracy,
        "rejected_accuracy": rejected_accuracy
    }

print("\nMean confidence:")
if correct.any():
    print(
        "Correct predictions:",
        round(scores[correct].mean().item(), 4)
    )

if (~correct).any():
    print(
        "Wrong predictions:",
        round(scores[~correct].mean().item(), 4)
    )

output = ROOT / "experiments/confidence"
output.mkdir(parents=True, exist_ok=True)

with open(output / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nCONFIDENCE ANALYSIS COMPLETED")
