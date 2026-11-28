
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

SHIFTS = {
    "original": (1.0, 1.0, 1.0),
    "light": (0.95, 1.05, 0.95),
    "medium": (0.85, 1.15, 0.85),
    "strong": (0.75, 1.35, 0.70),
}


CHECKPOINTS = {
    "enhanced": "resnet18_pathmnist_enhanced_best.pt",
    "robust": "resnet18_pathmnist_robust_best.pt",
    "balanced": "resnet18_pathmnist_balanced_best.pt",
}

normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3
)


def make_transform(brightness, contrast, saturation):
    def transform(image):
        image = TF.adjust_brightness(
            image, brightness
        )
        image = TF.adjust_contrast(
            image, contrast
        )
        image = TF.adjust_saturation(
            image, saturation
        )
        return normalize(TF.to_tensor(image))

    return transform


def load_model(checkpoint_name):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features, 9
    )

    checkpoint = torch.load(
        ROOT / "checkpoints" / checkpoint_name,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(DEVICE)
    model.eval()

    return model


@torch.inference_mode()
def evaluate(model, loader):
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)

        y_pred.extend(
            logits.argmax(1).cpu().tolist()
        )
        y_true.extend(labels.view(-1).tolist())

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


results = {}

print("=" * 60)
print("CAFA-TTA | DOMAIN SHIFT SEVERITY ANALYSIS")
print("Device:", DEVICE)
print("Split: val")
print("=" * 60)

for model_name, checkpoint_name in CHECKPOINTS.items():
    model = load_model(checkpoint_name)
    results[model_name] = {}

    for shift_name, factors in SHIFTS.items():
        dataset = PathMNIST(
            split="val",
            root=str(ROOT / "data/raw/medmnist"),
            transform=make_transform(*factors),
            download=False,
            size=28,
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        metrics = evaluate(model, loader)
        results[model_name][shift_name] = metrics

        print(
            f"{model_name:9s} "
            f"{shift_name:8s} "
            f"Acc={metrics['accuracy']:.4f} "
            f"F1={metrics['macro_f1']:.4f}",
            flush=True
        )

output = ROOT / "experiments/shift_severity"
output.mkdir(parents=True, exist_ok=True)

with open(
    output / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "split": "val",
            "batch_size": BATCH_SIZE,
            "shifts": SHIFTS,
            "results": results,
        },
        f,
        indent=2
    )

print("\nSHIFT SEVERITY ANALYSIS COMPLETED")
