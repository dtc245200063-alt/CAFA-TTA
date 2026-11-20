
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
from core.cafa_adapter import CAFAAdapter

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
BATCH_SIZE = 64
LR = 1e-4

MODELS = {
    "enhanced": {
        "checkpoint": "resnet18_pathmnist_enhanced_best.pt",
        "prototypes": "source_prototypes_enhanced.pt",
    },
    "robust": {
        "checkpoint": "resnet18_pathmnist_robust_best.pt",
        "prototypes": "source_prototypes_robust.pt",
    },
    "balanced": {
        "checkpoint": "resnet18_pathmnist_balanced_best.pt",
        "prototypes": "source_prototypes_balanced.pt",
    },
}


CONFIGS = {
    "threshold_0_0": {
        "threshold": 0.0,
        "lambda_align": 0.05,
    },
    "threshold_0_5": {
        "threshold": 0.5,
        "lambda_align": 0.05,
    },
    "threshold_0_6": {
        "threshold": 0.6,
        "lambda_align": 0.05,
    },
    "threshold_0_7": {
        "threshold": 0.7,
        "lambda_align": 0.05,
    },
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8",
) as f:
    shift_data = json.load(f)

assert shift_data["split"] == "val"
shift = shift_data["shift"]

normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3,
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
    size=28,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=DEVICE.type == "cuda",
)


def load_model(checkpoint_name):
    backbone = models.resnet18(weights=None)

    backbone.fc = nn.Linear(
        backbone.fc.in_features,
        9,
    )

    checkpoint = torch.load(
        ROOT / "checkpoints" / checkpoint_name,
        map_location="cpu",
        weights_only=True,
    )

    backbone.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return ResNetFeatureExtractor(
        backbone
    ).to(DEVICE)


def load_prototypes(prototype_name, checkpoint_name):
    data = torch.load(
        ROOT / "checkpoints" / prototype_name,
        map_location="cpu",
        weights_only=True,
    )

    assert data["source_checkpoint"] == checkpoint_name

    prototypes = data["prototypes"]

    assert prototypes.shape == (9, 512)
    assert torch.isfinite(prototypes).all()

    return prototypes


def evaluate(model_name, model_config, mode, config):
    set_seed(SEED)

    model = load_model(
        model_config["checkpoint"]
    )

    prototypes = load_prototypes(
        model_config["prototypes"],
        model_config["checkpoint"],
    )

    adapter = CAFAAdapter(
        model=model,
        prototypes=prototypes,
        lr=LR,
        threshold=config["threshold"],
        lambda_align=config["lambda_align"],
        min_samples=2,
    )

    predictions = []
    labels_all = []

    selected_samples = 0
    updated_batches = 0

    for images, labels in loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # Predict BEFORE adapting on current batch.
        with torch.no_grad():
            logits = adapter.predict(images)
            predicted = logits.argmax(dim=1)

        predictions.extend(
            predicted.cpu().tolist()
        )

        labels_all.extend(
            labels.view(-1).tolist()
        )

        # Adapt without ground-truth labels.
        result = adapter.adapt(images)

        selected_samples += int(result["selected"])
        updated_batches += int(result["updated"])

    accuracy = float(
        accuracy_score(
            labels_all,
            predictions,
        )
    )

    macro_f1 = float(
        f1_score(
            labels_all,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    result = {
        "model": model_name,
        "mode": mode,
        "method": "CAFA-v1-ablation",
        "checkpoint": model_config["checkpoint"],
        "prototype_file": model_config["prototypes"],
        "split": "val",
        "protocol": "online_prequential",
        "seed": SEED,
        "shuffle": False,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "shift": shift,
        "threshold": config["threshold"],
        "lambda_align": config["lambda_align"],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "selection_rate": (
            selected_samples / len(dataset)
        ),
        "updated_batches": updated_batches,
        "total_batches": len(loader),
    }

    del adapter
    del model
    del prototypes

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result


print("=" * 65)
print("CAFA-TTA | MULTI-MODEL ABLATION STUDY")
print("Device:", DEVICE)
print("Validation samples:", len(dataset))
print("Shift:", shift)
print("Seed:", SEED)
print("=" * 65)

results = []

for model_name, model_config in MODELS.items():
    print("\n" + "=" * 65)
    print("MODEL:", model_name.upper())
    print("=" * 65)

    for mode, config in CONFIGS.items():
        metrics = evaluate(
            model_name,
            model_config,
            mode,
            config,
        )

        results.append(metrics)

        print(
            f"{mode:18s} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['macro_f1']:.4f} | "
            f"Selected={metrics['selection_rate']:.2%} | "
            f"Updated={metrics['updated_batches']}/"
            f"{metrics['total_batches']}",
            flush=True,
        )

# Save all ablation results without overwriting
# the original baseline experiments.

output_dir = ROOT / "experiments/threshold_sweep"
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_path = output_dir / "metrics.json"

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "split": "val",
            "seed": SEED,
            "shuffle": False,
            "batch_size": BATCH_SIZE,
            "shift": shift,
            "configs": CONFIGS,
            "results": results,
        },
        f,
        indent=2,
    )

print("\n" + "=" * 65)
print("FINAL SUMMARY | ABLATION VS TENT VS CAFA v2")
print("=" * 65)

for model_name in MODELS:
    print("\nMODEL:", model_name.upper())

    for row in results:
        if row["model"] == model_name:
            print(
                f"{row['mode']:18s} "
                f"Acc={row['accuracy']:.4f} "
                f"F1={row['macro_f1']:.4f}"
            )

    tent_path = (
        ROOT / "experiments/tent_models"
        / model_name / "metrics.json"
    )

    cafa_v2_path = (
        ROOT / "experiments/cafa_models"
        / model_name / "metrics.json"
    )

    if tent_path.exists():
        with open(
            tent_path,
            encoding="utf-8",
        ) as f:
            tent_result = json.load(f)

        print(
            "TENT               "
            f"Acc={tent_result['tent']['accuracy']:.4f} "
            f"F1={tent_result['tent']['macro_f1']:.4f}"
        )

    if cafa_v2_path.exists():
        with open(
            cafa_v2_path,
            encoding="utf-8",
        ) as f:
            cafa_result = json.load(f)

        print(
            "CAFA v2 agreement  "
            f"Acc={cafa_result['accuracy']:.4f} "
            f"F1={cafa_result['macro_f1']:.4f}"
        )

print("\nABLATION STUDY COMPLETED")
print("Saved:", output_path)
