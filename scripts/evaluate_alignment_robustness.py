
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
from kernel.tent import TentAdapter
from core.cafa_adapter import CAFAAdapter



# ==========================================
# CAFA-TTA | MULTI-MODEL ROBUSTNESS EVALUATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEEDS = [42, 123, 2026]
BATCH_SIZE = 64
LR = 1e-4
THRESHOLD = 0.7
LAMBDA_ALIGN = 0.05

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

METHODS = ["tent", "alignment_v1"]

# Read the same validation shift configuration
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


def create_backbone(checkpoint_name):
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

    return backbone.to(DEVICE)


def load_prototypes(prototype_name, checkpoint_name):
    data = torch.load(
        ROOT / "checkpoints" / prototype_name,
        map_location="cpu",
        weights_only=True,
    )

    assert data["source_checkpoint"] == checkpoint_name, (
        "Prototype/checkpoint mismatch"
    )

    prototypes = data["prototypes"]

    assert prototypes.shape == (9, 512)
    assert torch.isfinite(prototypes).all()

    return prototypes


def evaluate(model_name, config, method, seed):
    # Reset state for every independent run
    set_seed(seed)

    loader = make_loader(seed)

    backbone = create_backbone(
        config["checkpoint"]
    )

    
    if method == "tent":
        adapter = TentAdapter(
            backbone,
            learning_rate=LR,
        )

    elif method == "alignment_v1":
        feature_model = ResNetFeatureExtractor(
            backbone
        )

        prototypes = load_prototypes(
            config["prototypes"],
            config["checkpoint"],
        )

        adapter = CAFAAdapter(
            model=feature_model,
            prototypes=prototypes,
            lr=LR,
            threshold=THRESHOLD,
            lambda_align=LAMBDA_ALIGN,
            min_samples=2,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    predictions = []
    labels_all = []

    selected_total = 0
    updated_batches = 0
    total_seen = 0

    # Predict first, adapt afterward
    for images, labels in loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.no_grad():
            logits = adapter.predict(images)
            pred = logits.argmax(dim=1)

        predictions.extend(
            pred.cpu().tolist()
        )
        labels_all.extend(
            labels.view(-1).tolist()
        )

        result = adapter.adapt(images)

        total_seen += images.size(0)

        if method == "alignment_v1":
            selected_total += int(
                result["selected"]
            )

            if result["updated"]:
                updated_batches += 1

    metrics = {
        "model": model_name,
        "checkpoint": config["checkpoint"],
        "method": method,
        "seed": seed,
        "accuracy": float(
            accuracy_score(
                labels_all,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels_all,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    if method == "":
        metrics["selection_rate"] = float(
            selected_total / total_seen
        )
        metrics["updated_batches"] = (
            updated_batches
        )

    # Release references before next run
    del adapter
    del backbone

    if method == "":
        del feature_model
        del prototypes

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


print("=" * 65)
print("CAFA-TTA | MULTI-MODEL ROBUSTNESS")
print("Device:", DEVICE)
print("Validation samples:", len(dataset))
print("Seeds:", SEEDS)
print("Shift:", shift)
print("Batch size:", BATCH_SIZE)
print("=" * 65)

results = []

for model_name, config in MODELS.items():
    print("\n" + "=" * 65)
    print("MODEL:", model_name.upper())
    print("=" * 65)

    for seed in SEEDS:
        for method in METHODS:
            metrics = evaluate(
                model_name,
                config,
                method,
                seed,
            )

            results.append(metrics)

            line = (
                f"Seed={seed:4d} "
                f"Method={method:8s} "
                f"Acc={metrics['accuracy']:.4f} "
                f"F1={metrics['macro_f1']:.4f}"
            )

            if method == "":
                line += (
                    f" Selection="
                    f"{metrics['selection_rate']:.2%}"
                )

            print(line, flush=True)


# ==========================================
# Aggregate results
# ==========================================

summary = {}

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

for model_name in MODELS:
    summary[model_name] = {}

    print("\nMODEL:", model_name.upper())

    for method in METHODS:
        rows = [
            r for r in results
            if r["model"] == model_name
            and r["method"] == method
        ]

        accuracies = np.array([
            r["accuracy"] for r in rows
        ])

        f1_values = np.array([
            r["macro_f1"] for r in rows
        ])

        method_summary = {
            "accuracy_mean": float(
                accuracies.mean()
            ),
            "accuracy_std": float(
                accuracies.std(ddof=1)
            ),
            "macro_f1_mean": float(
                f1_values.mean()
            ),
            "macro_f1_std": float(
                f1_values.std(ddof=1)
            ),
        }

        if method == "":
            selections = np.array([
                r["selection_rate"] for r in rows
            ])

            method_summary["selection_mean"] = (
                float(selections.mean())
            )

        summary[model_name][method] = (
            method_summary
        )

        print(
            f"{method:8s} | "
            f"Acc={accuracies.mean():.4f} "
            f"+/- {accuracies.std(ddof=1):.4f} | "
            f"F1={f1_values.mean():.4f} "
            f"+/- {f1_values.std(ddof=1):.4f}"
        )

    # Paired differences using the same seeds
    tent_by_seed = {
        r["seed"]: r["accuracy"]
        for r in results
        if r["model"] == model_name
        and r["method"] == "tent"
    }

   
    alignment_by_seed = {
        r["seed"]: r["accuracy"]
        for r in results
        if r["model"] == model_name
        and r["method"] == "alignment_v1"
    }

    differences = np.array([
        alignment_by_seed[seed]
        - tent_by_seed[seed]
        for seed in SEEDS
    ])

    summary[model_name]["paired_difference"] = {
        "per_seed": differences.tolist(),
        "mean_pp": float(
            differences.mean() * 100
        ),
        "std_pp": float(
            differences.std(ddof=1) * 100
        ),
    }

    print(
        "Paired Alignment v1 - TENT:",
        f"{differences.mean() * 100:+.3f}",
        "percentage points"
    )



# ==========================================
# Save results separately from old runs
# ==========================================

output_dir = (
    ROOT / "experiments"
    / "alignment_robustness"
)

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
            "protocol": "shuffled_online_prequential",
            "split": "val",
            "shift": shift,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "confidence_threshold": THRESHOLD,
            "lambda_align": LAMBDA_ALIGN,
            "seeds": SEEDS,
            "models": MODELS,
            "results": results,
            "summary": summary,
        },
        f,
        indent=2,
    )

print("\nROBUSTNESS EVALUATION COMPLETED")
print("Saved:", output_path)
