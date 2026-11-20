
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
from core.cafa_soft_adapter import CAFASoftAdapter

# ==========================================
# CAFA-TTA v3 | ROBUSTNESS EVALUATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEEDS = [42, 123, 2026]
BATCH_SIZE = 64
LR = 1e-4
ENTROPY_FLOOR = 0.2
LAMBDA_ALIGN = 0.05
MIN_SAMPLES = 2

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

METHODS = ["tent", "cafa_v3"]

# ==========================================
# Dataset and domain shift
# ==========================================

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


# ==========================================
# Reproducibility
# ==========================================

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


# ==========================================
# Model loading
# ==========================================

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


# ==========================================
# Online prequential evaluation
# ==========================================

def evaluate(model_name, config, method, seed):
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

    elif method == "cafa_v3":
        feature_model = ResNetFeatureExtractor(
            backbone
        )

        prototypes = load_prototypes(
            config["prototypes"],
            config["checkpoint"],
        )

        adapter = CAFASoftAdapter(
            model=feature_model,
            prototypes=prototypes,
            lr=LR,
            lambda_align=LAMBDA_ALIGN,
            entropy_floor=ENTROPY_FLOOR,
            min_samples=MIN_SAMPLES,
        )

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    predictions = []
    labels_all = []

    agreement_total = 0
    updated_batches = 0
    total_seen = 0

    # Predict first, adapt afterward.
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

        # Ground-truth labels are not used here.
        result = adapter.adapt(images)

        total_seen += images.size(0)

        if method == "cafa_v3":
            agreement_total += int(
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

    if method == "cafa_v3":
        metrics["agreement_rate"] = float(
            agreement_total / total_seen
        )

        metrics["updated_batches"] = updated_batches
        metrics["total_batches"] = len(loader)

    # Release GPU memory.
    del adapter
    del backbone

    if method == "cafa_v3":
        del feature_model
        del prototypes

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


# ==========================================
# Run experiments
# ==========================================

print("=" * 65)
print("CAFA-TTA v3 | MULTI-MODEL ROBUSTNESS")
print("Device:", DEVICE)
print("Validation samples:", len(dataset))
print("Seeds:", SEEDS)
print("Shift:", shift)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)
print("Entropy floor:", ENTROPY_FLOOR)
print("Alignment coefficient:", LAMBDA_ALIGN)
print("=" * 65)

output_dir = (
    ROOT / "experiments" / "cafa_v3_robustness"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

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

            if method == "cafa_v3":
                line += (
                    f" Agreement="
                    f"{metrics['agreement_rate']:.2%}"
                    f" Updated="
                    f"{metrics['updated_batches']}"
                    f"/{metrics['total_batches']}"
                )

            print(line, flush=True)

            # Save progress after every completed run.
            with open(
                output_dir / "partial_results.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "seeds": SEEDS,
                        "learning_rate": LR,
                        "entropy_floor": ENTROPY_FLOOR,
                        "lambda_align": LAMBDA_ALIGN,
                        "batch_size": BATCH_SIZE,
                        "shift": shift,
                        "results": results,
                    },
                    f,
                    indent=2,
                )


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

        assert len(rows) == len(SEEDS)

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

        if method == "cafa_v3":

            agreements = np.array([
                r["agreement_rate"] for r in rows
            ])

            method_summary["agreement_mean"] = float(
                agreements.mean()
            )

            method_summary["agreement_std"] = float(
                agreements.std(ddof=1)
            )

        summary[model_name][method] = method_summary

        print(
            f"{method:8s} | "
            f"Acc={accuracies.mean():.4f} "
            f"+/- {accuracies.std(ddof=1):.4f} | "
            f"F1={f1_values.mean():.4f} "
            f"+/- {f1_values.std(ddof=1):.4f}"
        )

        if method == "cafa_v3":
            print(
                "Mean Agreement Rate:",
                f"{agreements.mean():.2%}"
            )

    # Compare methods using identical seeds.

    tent_by_seed = {
        r["seed"]: r
        for r in results
        if r["model"] == model_name
        and r["method"] == "tent"
    }

    v3_by_seed = {
        r["seed"]: r
        for r in results
        if r["model"] == model_name
        and r["method"] == "cafa_v3"
    }

    acc_differences = np.array([
        v3_by_seed[seed]["accuracy"]
        - tent_by_seed[seed]["accuracy"]
        for seed in SEEDS
    ])

    f1_differences = np.array([
        v3_by_seed[seed]["macro_f1"]
        - tent_by_seed[seed]["macro_f1"]
        for seed in SEEDS
    ])

    wins = int((acc_differences > 0).sum())
    ties = int((acc_differences == 0).sum())
    losses = int((acc_differences < 0).sum())

    summary[model_name]["paired_difference"] = {
        "seeds": SEEDS,
        "accuracy_per_seed_pp": (
            acc_differences * 100
        ).tolist(),
        "accuracy_mean_pp": float(
            acc_differences.mean() * 100
        ),
        "accuracy_std_pp": float(
            acc_differences.std(ddof=1) * 100
        ),
        "macro_f1_per_seed_pp": (
            f1_differences * 100
        ).tolist(),
        "macro_f1_mean_pp": float(
            f1_differences.mean() * 100
        ),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }

    print(
        "Paired CAFA v3 - TENT Accuracy:",
        f"{acc_differences.mean() * 100:+.3f} pp"
    )

    print(
        "Paired CAFA v3 - TENT Macro-F1:",
        f"{f1_differences.mean() * 100:+.3f} pp"
    )

    print(
        f"Wins/Ties/Losses: {wins}/{ties}/{losses}"
    )


# ==========================================
# Save final results
# ==========================================

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
            "entropy_floor": ENTROPY_FLOOR,
            "lambda_align": LAMBDA_ALIGN,
            "min_samples": MIN_SAMPLES,
            "seeds": SEEDS,
            "models": MODELS,
            "methods": METHODS,
            "results": results,
            "summary": summary,
        },
        f,
        indent=2,
    )

print("\n" + "=" * 65)
print("CAFA-TTA v3 ROBUSTNESS EVALUATION COMPLETED")
print("Saved:", output_path)
print("=" * 65)
