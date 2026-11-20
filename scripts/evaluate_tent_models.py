
import sys
import json
import copy
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

from kernel.tent import TentAdapter

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

CHECKPOINTS = {
    "enhanced": "resnet18_pathmnist_enhanced_best.pt",
    "robust": "resnet18_pathmnist_robust_best.pt",
    "balanced": "resnet18_pathmnist_balanced_best.pt",
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    shift_config = json.load(f)

assert shift_config["split"] == "val"
shift = shift_config["shift"]

normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3
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


def load_source_model(checkpoint_name):
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        9
    )

    checkpoint = torch.load(
        ROOT / "checkpoints" / checkpoint_name,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model.to(DEVICE).eval()


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def run_experiment(model_name, checkpoint_name):
    print("\n" + "=" * 60)
    print("MODEL:", model_name.upper())
    print("Checkpoint:", checkpoint_name)
    print("=" * 60)

    source_model = load_source_model(
        checkpoint_name
    )

    # TENT must receive independent weights
    tent_model = copy.deepcopy(source_model)

    adapter = TentAdapter(
        tent_model,
        learning_rate=LEARNING_RATE,
    )

    source_predictions = []
    tent_predictions = []
    ground_truth = []
    adaptation_losses = []

    # Online prequential evaluation:
    # predict first, adapt afterward.
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)

        with torch.inference_mode():
            source_logits = source_model(images)
            source_pred = source_logits.argmax(1)

        tent_logits = adapter.predict(images)
        tent_pred = tent_logits.argmax(1)

        source_predictions.extend(
            source_pred.cpu().tolist()
        )

        tent_predictions.extend(
            tent_pred.cpu().tolist()
        )

        ground_truth.extend(
            labels.view(-1).tolist()
        )

        loss = adapter.adapt(images)
        adaptation_losses.append(float(loss))

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(loader)} "
                f"| Loss: {float(loss):.4f}",
                flush=True,
            )

    source_metrics = compute_metrics(
        ground_truth,
        source_predictions,
    )

    tent_metrics = compute_metrics(
        ground_truth,
        tent_predictions,
    )

    accuracy_change = (
        tent_metrics["accuracy"]
        - source_metrics["accuracy"]
    )

    mean_loss = (
        sum(adaptation_losses)
        / len(adaptation_losses)
    )

    results = {
        "model": model_name,
        "checkpoint": checkpoint_name,
        "protocol": "online_prequential",
        "split": "val",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "shuffle": False,
        "shift": shift,
        "source": source_metrics,
        "tent": tent_metrics,
        "accuracy_change": accuracy_change,
        "mean_tent_loss": mean_loss,
    }

    output_dir = (
        ROOT / "experiments"
        / "tent_models"
        / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"Source Acc: "
        f"{source_metrics['accuracy']:.4f}"
    )

    print(
        f"TENT Acc: "
        f"{tent_metrics['accuracy']:.4f}"
    )

    print(
        f"Source F1: "
        f"{source_metrics['macro_f1']:.4f}"
    )

    print(
        f"TENT F1: "
        f"{tent_metrics['macro_f1']:.4f}"
    )

    print(
        f"Accuracy change: "
        f"{accuracy_change * 100:+.2f} pp"
    )

    print("Saved:", output_dir / "metrics.json")

    del adapter
    del tent_model
    del source_model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return results


print("=" * 60)
print("CAFA-TTA | TENT MULTI-MODEL EVALUATION")
print("Device:", DEVICE)
print("Dataset: PathMNIST Validation")
print("Shift:", shift)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("=" * 60)

all_results = {}

for model_name, checkpoint_name in CHECKPOINTS.items():
    result = run_experiment(
        model_name,
        checkpoint_name,
    )

    all_results[model_name] = result

summary_dir = ROOT / "experiments/tent_models"
summary_dir.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    summary_dir / "summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        all_results,
        f,
        indent=2,
    )

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

for name, result in all_results.items():
    print(
        f"{name:9s} | "
        f"Source: {result['source']['accuracy']:.4f} | "
        f"TENT: {result['tent']['accuracy']:.4f} | "
        f"Change: {result['accuracy_change'] * 100:+.2f} pp"
    )

print("\nTENT MULTI-MODEL EVALUATION COMPLETED")
