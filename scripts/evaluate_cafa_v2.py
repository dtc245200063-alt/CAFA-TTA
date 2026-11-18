
import sys
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
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.feature_extractor import ResNetFeatureExtractor

from core.cafa_agreement_adapter import CAFAAgreementAdapter


# =============================================
# CAFA-TTA | REAL MEDICAL IMAGE EVALUATION
# =============================================

SEED = 42
BATCH_SIZE = 64
LR = 1e-4
THRESHOLD = 0.7
LAMBDA_ALIGN = 0.05

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 55)
print("CAFA-TTA | DOMAIN ADAPTATION EVALUATION")
print("Device:", DEVICE)
print("=" * 55)

# 1. Read domain shift configuration
with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    shift_config = json.load(f)

assert shift_config["split"] == "val"
shift = shift_config["shift"]

print("Shift:", shift)

# 2. Data preprocessing
normalize = transforms.Normalize(
    mean=[0.5] * 3,
    std=[0.5] * 3
)

def target_transform(image):
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
    transform=target_transform,
    download=False,
    size=28
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda"
)

print("Target samples:", len(dataset))

# 3. Load trained ResNet-18
backbone = models.resnet18(weights=None)
backbone.fc = nn.Linear(
    backbone.fc.in_features, 9
)

checkpoint = torch.load(
    ROOT / "checkpoints/resnet18_pathmnist_best.pt",
    map_location="cpu",
    weights_only=True
)

backbone.load_state_dict(
    checkpoint["model_state_dict"]
)

model = ResNetFeatureExtractor(
    backbone
).to(DEVICE)

# 4. Load source prototypes
prototype_data = torch.load(
    ROOT / "checkpoints/source_prototypes.pt",
    map_location="cpu",
    weights_only=True
)

prototypes = prototype_data["prototypes"]

assert prototypes.shape == (9, 512)

# 5. Initialize CAFA-TTA
adapter = CAFAAgreementAdapter(
    model=model,
    prototypes=prototypes,
    lr=LR,
    threshold=THRESHOLD,
    lambda_align=LAMBDA_ALIGN,
    min_samples=2
)

all_labels = []
all_predictions = []

entropy_losses = []
alignment_losses = []
total_losses = []

total_selected = 0
total_updates = 0
total_seen = 0

print("\nStarting CAFA-TTA adaptation...")

# 6. Online prequential evaluation
for batch_idx, (images, labels) in enumerate(loader):

    images = images.to(
        DEVICE,
        non_blocking=True
    )

    # Predict before adapting on this batch.
    with torch.no_grad():
        logits = adapter.predict(images)
        predictions = logits.argmax(dim=1)

    all_predictions.extend(
        predictions.cpu().tolist()
    )

    all_labels.extend(
        labels.view(-1).tolist()
    )

    # Adapt using images only.
    # Ground-truth labels are not passed.
    result = adapter.adapt(images)

    total_seen += images.size(0)
    total_selected += result["selected"]

    if result["updated"]:
        total_updates += 1

        entropy_losses.append(
            result["entropy_loss"]
        )

        alignment_losses.append(
            result["alignment_loss"]
        )

        total_losses.append(
            result["total_loss"]
        )

    if (batch_idx + 1) % 25 == 0:
        print(
            f"Batch {batch_idx + 1}/{len(loader)} "
            f"| Selected: {result['selected']}/{len(images)} "
            f"| Updated: {result['updated']}",
            flush=True
        )

# 7. Evaluate predictions
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

selection_rate = total_selected / total_seen

def mean_or_zero(values):
    return (
        sum(values) / len(values)
        if values else 0.0
    )

print("\n" + "=" * 55)
print("CAFA-TTA RESULTS")
print("=" * 55)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(f"Selection Rate: {selection_rate:.2%}")
print(f"Updated Batches: {total_updates}/{len(loader)}")
print(f"Mean Entropy Loss: {mean_or_zero(entropy_losses):.4f}")
print(f"Mean Alignment Loss: {mean_or_zero(alignment_losses):.4f}")
print(f"Mean Total Loss: {mean_or_zero(total_losses):.4f}")

if DEVICE.type == "cuda":
    print(
        "Peak GPU VRAM:",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2
        ),
        "MB"
    )

# 8. Save results

output_dir = ROOT / "experiments/cafa_v2"
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    
"method": "CAFA-TTA-v2-Agreement",
    "protocol": "online_prequential",
    "split": "val",
    "shift": shift,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "confidence_threshold": THRESHOLD,
    "lambda_align": LAMBDA_ALIGN,
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "selection_rate": float(selection_rate),
    "updated_batches": total_updates,
    "mean_entropy_loss": mean_or_zero(entropy_losses),
    "mean_alignment_loss": mean_or_zero(alignment_losses),
    "mean_total_loss": mean_or_zero(total_losses)
}

with open(
    output_dir / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

print("\nSaved:", output_dir / "metrics.json")
print("CAFA-TTA EVALUATION COMPLETED")
