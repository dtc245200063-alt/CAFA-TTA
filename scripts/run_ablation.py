
import sys
import json
import random
import argparse
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
from core.cafa_adapter import CAFAAdapter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    required=True,
    choices=[
        "entropy_only",
        "confidence_only",
        "alignment_only",
        "full_cafa",
    ]
)
args = parser.parse_args()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
LR = 1e-4

# Choose ablation configuration
configs = {
    "entropy_only": {
        "threshold": 0.0,
        "lambda_align": 0.0,
    },
    "confidence_only": {
        "threshold": 0.7,
        "lambda_align": 0.0,
    },
    "alignment_only": {
        "threshold": 0.0,
        "lambda_align": 0.05,
    },
    "full_cafa": {
        "threshold": 0.7,
        "lambda_align": 0.05,
    },
}

config = configs[args.mode]

print("=" * 55)
print("CAFA-TTA | ABLATION STUDY")
print("Mode:", args.mode)
print("Device:", DEVICE)
print("Configuration:", config)
print("=" * 55)

# Load the same shift
with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    shift_data = json.load(f)

assert shift_data["split"] == "val"
shift = shift_data["shift"]

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
    num_workers=0
)

# Reload original checkpoint every run
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

prototype_data = torch.load(
    ROOT / "checkpoints/source_prototypes.pt",
    map_location="cpu",
    weights_only=True
)

adapter = CAFAAdapter(
    model=model,
    prototypes=prototype_data["prototypes"],
    lr=LR,
    threshold=config["threshold"],
    lambda_align=config["lambda_align"],
    min_samples=2
)

predictions = []
labels_all = []
updated_batches = 0
selected_samples = 0

# Online prequential protocol:
# predict first, then adapt.
for batch_idx, (images, labels) in enumerate(loader):
    images = images.to(DEVICE)

    with torch.no_grad():
        logits = adapter.predict(images)
        predicted = logits.argmax(dim=1)

    predictions.extend(predicted.cpu().tolist())
    labels_all.extend(labels.view(-1).tolist())

    result = adapter.adapt(images)

    selected_samples += result["selected"]
    updated_batches += int(result["updated"])

    if (batch_idx + 1) % 50 == 0:
        print(
            f"Processed {batch_idx + 1}/{len(loader)} batches",
            flush=True
        )

accuracy = accuracy_score(
    labels_all,
    predictions
)

macro_f1 = f1_score(
    labels_all,
    predictions,
    average="macro",
    zero_division=0
)

results = {
    "mode": args.mode,
    "split": "val",
    "protocol": "online_prequential",
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "shift": shift,
    "threshold": config["threshold"],
    "lambda_align": config["lambda_align"],
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "selection_rate": selected_samples / len(dataset),
    "updated_batches": updated_batches,
}

output_dir = ROOT / "experiments/ablation"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{args.mode}.json"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 55)
print("ABLATION RESULTS")
print("Mode:", args.mode)
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")
print(
    f"Selection Rate: "
    f"{results['selection_rate']:.2%}"
)
print(f"Updated Batches: {updated_batches}/{len(loader)}")
print("Saved:", output_path)
print("=" * 55)
