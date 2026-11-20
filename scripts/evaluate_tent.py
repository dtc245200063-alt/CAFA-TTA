
import sys
import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernel.tent import TentAdapter

# =========================================
# CAFA-TTA | TENT EVALUATION
# =========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
LEARNING_RATE = 1e-4

checkpoint_path = ROOT / (
    "checkpoints/resnet18_pathmnist_best.pt"
)

config_path = ROOT / (
    "experiments/domain_shift/metrics.json"
)

# Read the saved domain shift configuration
with open(config_path, "r", encoding="utf-8") as f:
    shift_config = json.load(f)

assert shift_config["split"] == "val", (
    "Use validation data for development"
)

shift = shift_config["shift"]

print("=" * 55)
print("CAFA-TTA | TENT DOMAIN ADAPTATION")
print("Device:", DEVICE)
print("Shift config:", shift)
print("=" * 55)

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

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

# Load target domain validation dataset
dataset = PathMNIST(
    split="val",
    root=str(ROOT / "data/raw/medmnist"),
    transform=shifted_transform,
    download=False,
    size=28,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)

# Rebuild source model
source_model = models.resnet18(weights=None)
source_model.fc = nn.Linear(
    source_model.fc.in_features, 9
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

source_model.load_state_dict(
    checkpoint["model_state_dict"]
)

source_model = source_model.to(DEVICE)
source_model.eval()

# TENT receives a separate copy.
# Original source model is never modified.
tent_model = copy.deepcopy(source_model)

adapter = TentAdapter(
    tent_model,
    learning_rate=LEARNING_RATE
)

source_predictions = []
tent_predictions = []
ground_truth = []
adaptation_losses = []

print("\nStarting evaluation...")

# Prequential protocol:
# Predict each batch BEFORE learning from it.
for batch_idx, (images, labels) in enumerate(loader):
    images = images.to(DEVICE)

    # Frozen source-only baseline
    with torch.no_grad():
        source_logits = source_model(images)
        source_pred = source_logits.argmax(dim=1)

    # Adapted model's current predictions
    # No current-batch gradient is applied yet.
    tent_logits = adapter.predict(images)
    tent_pred = tent_logits.argmax(dim=1)

    source_predictions.extend(
        source_pred.cpu().tolist()
    )
    tent_predictions.extend(
        tent_pred.cpu().tolist()
    )
    ground_truth.extend(
        labels.view(-1).tolist()
    )

    # Now adapt using the current batch,
    # without using ground-truth labels.
    loss = adapter.adapt(images)
    adaptation_losses.append(loss)

    if (batch_idx + 1) % 25 == 0:
        print(
            f"Batch {batch_idx + 1}/{len(loader)} "
            f"| TENT Loss: {loss:.4f}",
            flush=True
        )

def compute_metrics(predictions):
    return {
        "accuracy": float(
            accuracy_score(ground_truth, predictions)
        ),
        "macro_f1": float(
            f1_score(
                ground_truth,
                predictions,
                average="macro",
                zero_division=0
            )
        )
    }

source_metrics = compute_metrics(
    source_predictions
)

tent_metrics = compute_metrics(
    tent_predictions
)

print("\n" + "=" * 55)
print("EVALUATION RESULTS")
print("=" * 55)

print(
    f"Source Accuracy: "
    f"{source_metrics['accuracy']:.4f}"
)
print(
    f"TENT Accuracy: "
    f"{tent_metrics['accuracy']:.4f}"
)
print(
    f"Source Macro-F1: "
    f"{source_metrics['macro_f1']:.4f}"
)
print(
    f"TENT Macro-F1: "
    f"{tent_metrics['macro_f1']:.4f}"
)

accuracy_change = (
    tent_metrics["accuracy"]
    - source_metrics["accuracy"]
)

print(
    f"Accuracy Change: "
    f"{accuracy_change * 100:+.2f} "
    "percentage points"
)

print(
    f"Mean TENT Loss: "
    f"{sum(adaptation_losses) / len(adaptation_losses):.4f}"
)

# Save research results
output_dir = ROOT / "experiments/tent"
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    "protocol": "online_prequential",
    "split": "val",
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "shift": shift,
    "source": source_metrics,
    "tent": tent_metrics,
    "accuracy_change": accuracy_change,
    "mean_tent_loss": (
        sum(adaptation_losses) / len(adaptation_losses)
    )
}

with open(
    output_dir / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

print("\nTENT EVALUATION COMPLETED")
print("Saved:", output_dir / "metrics.json")
