
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.feature_extractor import ResNetFeatureExtractor
from core.cafa_agreement_adapter import CAFAAgreementAdapter
from kernel.tent import TentAdapter

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEEDS = [42, 123, 2026]
BATCH_SIZE = 64

with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    shift_data = json.load(f)

assert shift_data["split"] == "val"
shift = shift_data["shift"]

normalize = transforms.Normalize(
    [0.5] * 3, [0.5] * 3
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
    split="test",
    root=str(ROOT / "data/raw/medmnist"),
    transform=transform,
    download=False,
    size=28,
)

checkpoint = torch.load(
    ROOT / "checkpoints/resnet18_pathmnist_best.pt",
    map_location="cpu",
    weights_only=True
)

prototype_data = torch.load(
    ROOT / "checkpoints/source_prototypes.pt",
    map_location="cpu",
    weights_only=True
)

prototypes = prototype_data["prototypes"]

def create_model():
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(
        backbone.fc.in_features, 9
    )
    backbone.load_state_dict(
        checkpoint["model_state_dict"]
    )
    return backbone.to(DEVICE)

def evaluate(method, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    backbone = create_model()

    if method == "tent":
        adapter = TentAdapter(
            backbone,
            learning_rate=1e-4
        )
    else:
        feature_model = ResNetFeatureExtractor(
            backbone
        )

        adapter = CAFAAgreementAdapter(
            model=feature_model,
            prototypes=prototypes,
            lr=1e-4,
            threshold=0.7,
            lambda_align=0.05,
            min_samples=2,
        )

    predictions = []
    labels_all = []

    for images, labels in loader:
        images = images.to(DEVICE)

        # Predict BEFORE current-batch adaptation.
        with torch.no_grad():
            logits = adapter.predict(images)
            pred = logits.argmax(dim=1)

        predictions.extend(pred.cpu().tolist())
        labels_all.extend(labels.view(-1).tolist())

        # No target labels are used here.
        adapter.adapt(images)

    return {
        "accuracy": accuracy_score(
            labels_all, predictions
        ),
        "macro_f1": f1_score(
            labels_all,
            predictions,
            average="macro",
            zero_division=0
        ),
    }

results = []

print("=" * 55)
print("CAFA-TTA | ROBUSTNESS EVALUATION")
print("Device:", DEVICE)
print("Samples:", len(dataset))
print("Shift:", shift)
print("=" * 55)

for seed in SEEDS:
    for method in ["tent", "cafa_v2"]:
        metrics = evaluate(method, seed)

        entry = {
            "method": method,
            "seed": seed,
            **metrics,
        }
        results.append(entry)

        print(
            f"Seed={seed:4d} "
            f"Method={method:8s} "
            f"Acc={metrics['accuracy']:.4f} "
            f"F1={metrics['macro_f1']:.4f}",
            flush=True,
        )

print("\nSUMMARY")

for method in ["tent", "cafa_v2"]:
    rows = [
        r for r in results
        if r["method"] == method
    ]

    accuracies = np.array([
        r["accuracy"] for r in rows
    ])

    f1_scores = np.array([
        r["macro_f1"] for r in rows
    ])

    print(
        f"{method:8s} "
        f"Accuracy={accuracies.mean():.4f} "
        f"+/- {accuracies.std(ddof=1):.4f} "
        f"F1={f1_scores.mean():.4f} "
        f"+/- {f1_scores.std(ddof=1):.4f}"
    )

tent_by_seed = {
    r["seed"]: r["accuracy"]
    for r in results
    if r["method"] == "tent"
}

cafa_by_seed = {
    r["seed"]: r["accuracy"]
    for r in results
    if r["method"] == "cafa_v2"
}

differences = np.array([
    cafa_by_seed[s] - tent_by_seed[s]
    for s in SEEDS
])

print(
    "Paired Accuracy Difference (CAFA - TENT):",
    f"{differences.mean() * 100:+.3f}",
    "percentage points"
)

output_dir = ROOT / "experiments/final_test"
output_dir.mkdir(parents=True, exist_ok=True)

with open(
    output_dir / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "protocol": "shuffled_online_prequential",
            "split": "test",
            "shift": shift,
            "batch_size": BATCH_SIZE,
            "seeds": SEEDS,
            "results": results,
            "paired_accuracy_differences": (
                differences.tolist()
            ),
        },
        f,
        indent=2
    )

print("\nROBUSTNESS EVALUATION COMPLETED")
