
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64

checkpoint_path = (
    ROOT / "checkpoints/resnet18_pathmnist_enhanced_best.pt"
)

with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    shift_config = json.load(f)

assert shift_config["split"] == "val"
shift = shift_config["shift"]

normalize = transforms.Normalize(
    mean=[0.5] * 3,
    std=[0.5] * 3,
)

def original_transform(image):
    return normalize(TF.to_tensor(image))

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

def make_loader(transform):
    dataset = PathMNIST(
        split="val",
        root=str(ROOT / "data/raw/medmnist"),
        transform=transform,
        download=False,
        size=28,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

@torch.inference_mode()
def evaluate(loader):
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(DEVICE)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        y_true.extend(labels.view(-1).tolist())
        y_pred.extend(predictions.cpu().tolist())

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

print("=" * 55)
print("CAFA-TTA | ENHANCED SOURCE EVALUATION")
print("Device:", DEVICE)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Best train-time val acc:", checkpoint["best_val_acc"])
print("Shift:", shift)
print("=" * 55)

original = evaluate(
    make_loader(original_transform)
)

shifted = evaluate(
    make_loader(shifted_transform)
)

print("\nORIGINAL VALIDATION:")
print("Accuracy:", round(original["accuracy"], 4))
print("Macro-F1:", round(original["macro_f1"], 4))

print("\nSHIFTED VALIDATION:")
print("Accuracy:", round(shifted["accuracy"], 4))
print("Macro-F1:", round(shifted["macro_f1"], 4))

print(
    "\nAccuracy drop:",
    round(
        (original["accuracy"] - shifted["accuracy"]) * 100,
        2
    ),
    "percentage points"
)

output_dir = ROOT / "experiments/enhanced_training"
output_dir.mkdir(parents=True, exist_ok=True)

with open(
    output_dir / "validation_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump({
        "checkpoint": checkpoint_path.name,
        "checkpoint_epoch": checkpoint["epoch"],
        "split": "val",
        "shift": shift,
        "original": original,
        "shifted": shifted,
    }, f, indent=2)

print("\nENHANCED EVALUATION COMPLETED")
