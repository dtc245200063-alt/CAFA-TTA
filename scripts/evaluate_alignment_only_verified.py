
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

# ==========================================
# CAFA-TTA | VERIFIED ALIGNMENT-ONLY
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEEDS = [42, 123, 2026]
BATCH_SIZE = 64
LR = 1e-4

THRESHOLD = 0.0
LAMBDA_ALIGN = 0.10
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

# ==========================================
# Verify experimental configuration
# ==========================================

with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8",
) as f:
    shift_data = json.load(f)

assert shift_data["split"] == "val"

shift = shift_data["shift"]

assert THRESHOLD == 0.0
assert LAMBDA_ALIGN == 0.10

assert shift == {
    "brightness": 0.75,
    "contrast": 1.35,
    "saturation": 0.70,
}

# ==========================================
# Reuse previously verified TENT results
# ==========================================

reference_path = (
    ROOT
    / "experiments"
    / "cafa_v3_robustness"
    / "metrics.json"
)

with open(
    reference_path,
    encoding="utf-8",
) as f:
    reference = json.load(f)

assert reference["split"] == "val"
assert reference["shift"] == shift
assert reference["batch_size"] == BATCH_SIZE
assert reference["learning_rate"] == LR
assert reference["seeds"] == SEEDS
assert reference["models"] == MODELS

assert reference["protocol"] == (
    "shuffled_online_prequential"
)

tent_results = [
    row for row in reference["results"]
    if row["method"] == "tent"
]

tent_lookup = {
    (row["model"], row["seed"]): row
    for row in tent_results
}

expected_keys = {
    (model_name, seed)
    for model_name in MODELS
    for seed in SEEDS
}

assert len(tent_results) == 9
assert set(tent_lookup) == expected_keys

for model_name, seed in expected_keys:

    assert (
        tent_lookup[(model_name, seed)]["checkpoint"]
        == MODELS[model_name]["checkpoint"]
    )

# ==========================================
# Dataset
# ==========================================

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

    return normalize(
        TF.to_tensor(image)
    )


dataset = PathMNIST(
    split="val",
    root=str(ROOT / "data/raw/medmnist"),
    transform=target_transform,
    download=False,
    size=28,
)

assert len(dataset) == 10004


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
# Load checkpoints and source prototypes
# ==========================================

def load_model(config):

    backbone = models.resnet18(
        weights=None
    )

    backbone.fc = nn.Linear(
        backbone.fc.in_features,
        9,
    )

    checkpoint = torch.load(
        ROOT / "checkpoints" / config["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )

    backbone.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return ResNetFeatureExtractor(
        backbone
    ).to(DEVICE)


def load_prototypes(config):

    data = torch.load(
        ROOT / "checkpoints" / config["prototypes"],
        map_location="cpu",
        weights_only=True,
    )

    assert (
        data["source_checkpoint"]
        == config["checkpoint"]
    ), "Prototype/checkpoint mismatch"

    prototypes = data["prototypes"]

    assert prototypes.shape == (9, 512)
    assert torch.isfinite(prototypes).all()

    return prototypes


# ==========================================
# Evaluate Alignment-only
# ==========================================

def evaluate(model_name, config, seed):

    set_seed(seed)

    loader = make_loader(seed)

    model = load_model(config)
    prototypes = load_prototypes(config)

    adapter = CAFAAdapter(
        model=model,
        prototypes=prototypes,
        lr=LR,
        threshold=THRESHOLD,
        lambda_align=LAMBDA_ALIGN,
        min_samples=MIN_SAMPLES,
    )

    predictions = []
    labels_all = []

    selected_total = 0
    total_seen = 0
    updated_batches = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # Predict before adaptation.
        with torch.no_grad():

            logits = adapter.predict(images)

            pred = logits.argmax(dim=1)

        predictions.extend(
            pred.cpu().tolist()
        )

        labels_all.extend(
            labels.view(-1).tolist()
        )

        # No target labels are used.
        result = adapter.adapt(images)

        total_seen += images.size(0)

        selected_total += int(
            result["selected"]
        )

        if result["updated"]:
            updated_batches += 1

    selection_rate = (
        selected_total / total_seen
    )

    metrics = {
        "model": model_name,
        "checkpoint": config["checkpoint"],
        "method": "alignment_only_verified",
        "seed": seed,
        "confidence_threshold": THRESHOLD,
        "lambda_align": LAMBDA_ALIGN,
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
        "selection_rate": selection_rate,
        "updated_batches": updated_batches,
        "total_batches": len(loader),
        "samples": total_seen,
    }

    assert total_seen == len(dataset)

    # With threshold=0, every valid sample
    # should pass confidence selection.
    if selection_rate != 1.0:

        print(
            "WARNING: Expected 100% selection, got",
            f"{selection_rate:.4%}"
        )

    if updated_batches != len(loader):

        print(
            "WARNING: Some batches were not updated."
        )

    del adapter
    del model
    del prototypes

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


# ==========================================
# Output configuration
# ==========================================

output_dir = (
    ROOT
    / "experiments"
    / "alignment_only_verified"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

metadata = {
    "protocol": "shuffled_online_prequential",
    "split": "val",
    "shift": shift,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "confidence_threshold": THRESHOLD,
    "lambda_align": LAMBDA_ALIGN,
    "min_samples": MIN_SAMPLES,
    "seeds": SEEDS,
    "models": MODELS,
    "reference_metrics": (
        "experiments/cafa_v3_robustness/metrics.json"
    ),
    "tent_results_reused": True,
}

TOTAL_RUNS = (
    len(MODELS) * len(SEEDS)
)


def save_json(path, data):

    temp_path = path.with_suffix(".tmp")

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

    temp_path.replace(path)


# ==========================================
# Run only missing experiments
# ==========================================

print("=" * 75)
print("CAFA-TTA | VERIFIED ALIGNMENT-ONLY")
print("Device:", DEVICE)
print("Dataset: PathMNIST Validation")
print("Samples:", len(dataset))
print("Seeds:", SEEDS)
print("Shift:", shift)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)
print("Confidence threshold:", THRESHOLD)
print("Alignment coefficient:", LAMBDA_ALIGN)
print("Reused TENT results:", len(tent_results))
print("New runs:", TOTAL_RUNS)
print("=" * 75)

results = []

for model_name, config in MODELS.items():

    print("\n" + "=" * 75)
    print("MODEL:", model_name.upper())
    print("=" * 75)

    for seed in SEEDS:

        metrics = evaluate(
            model_name,
            config,
            seed,
        )

        tent = tent_lookup[
            (model_name, seed)
        ]

        acc_delta = (
            metrics["accuracy"]
            - tent["accuracy"]
        ) * 100

        f1_delta = (
            metrics["macro_f1"]
            - tent["macro_f1"]
        ) * 100

        metrics["cafa_vs_tent_accuracy_pp"] = (
            acc_delta
        )

        metrics["cafa_vs_tent_macro_f1_pp"] = (
            f1_delta
        )

        results.append(metrics)

        print(
            f"[{len(results)}/{TOTAL_RUNS}] "
            f"Seed={seed:4d} | "
            f"TENT={tent['accuracy']:.4f} | "
            f"Alignment={metrics['accuracy']:.4f} | "
            f"F1={metrics['macro_f1']:.4f} | "
            f"Delta={acc_delta:+.3f} pp | "
            f"Selection={metrics['selection_rate']:.2%} | "
            f"Updated={metrics['updated_batches']}"
            f"/{metrics['total_batches']}",
            flush=True,
        )

        save_json(
            output_dir / "partial_results.json",
            {
                **metadata,
                "completed_runs": len(results),
                "total_runs": TOTAL_RUNS,
                "results": results,
                "tent_reference_results": tent_results,
            },
        )


# ==========================================
# Final summary
# ==========================================

assert len(results) == TOTAL_RUNS

summary = {}

print("\n" + "=" * 75)
print("FINAL VERIFIED ALIGNMENT SUMMARY")
print("=" * 75)

all_acc_deltas = []
all_f1_deltas = []

for model_name in MODELS:

    rows = [
        row for row in results
        if row["model"] == model_name
    ]

    assert len(rows) == len(SEEDS)

    tent_rows = [
        tent_lookup[(model_name, seed)]
        for seed in SEEDS
    ]

    tent_acc = np.array([
        row["accuracy"]
        for row in tent_rows
    ])

    tent_f1 = np.array([
        row["macro_f1"]
        for row in tent_rows
    ])

    alignment_acc = np.array([
        row["accuracy"]
        for row in rows
    ])

    alignment_f1 = np.array([
        row["macro_f1"]
        for row in rows
    ])

    acc_deltas = np.array([
        row["cafa_vs_tent_accuracy_pp"]
        for row in rows
    ])

    f1_deltas = np.array([
        row["cafa_vs_tent_macro_f1_pp"]
        for row in rows
    ])

    all_acc_deltas.extend(acc_deltas)
    all_f1_deltas.extend(f1_deltas)

    wins = int(
        (acc_deltas > 0).sum()
    )

    ties = int(
        (acc_deltas == 0).sum()
    )

    losses = int(
        (acc_deltas < 0).sum()
    )

    summary[model_name] = {
        "tent": {
            "accuracy_mean": float(
                tent_acc.mean()
            ),
            "accuracy_std": float(
                tent_acc.std(ddof=1)
            ),
            "macro_f1_mean": float(
                tent_f1.mean()
            ),
            "macro_f1_std": float(
                tent_f1.std(ddof=1)
            ),
        },
        "alignment_only_verified": {
            "accuracy_mean": float(
                alignment_acc.mean()
            ),
            "accuracy_std": float(
                alignment_acc.std(ddof=1)
            ),
            "macro_f1_mean": float(
                alignment_f1.mean()
            ),
            "macro_f1_std": float(
                alignment_f1.std(ddof=1)
            ),
            "selection_mean": float(
                np.mean([
                    row["selection_rate"]
                    for row in rows
                ])
            ),
        },
        "paired_difference": {
            "accuracy_mean_pp": float(
                acc_deltas.mean()
            ),
            "accuracy_std_pp": float(
                acc_deltas.std(ddof=1)
            ),
            "macro_f1_mean_pp": float(
                f1_deltas.mean()
            ),
            "accuracy_per_seed_pp": (
                acc_deltas.tolist()
            ),
            "macro_f1_per_seed_pp": (
                f1_deltas.tolist()
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
    }

    print("\nMODEL:", model_name.upper())

    print(
        f"TENT (reused) | "
        f"Acc={tent_acc.mean():.4f} "
        f"+/- {tent_acc.std(ddof=1):.4f} | "
        f"F1={tent_f1.mean():.4f}"
    )

    print(
        f"Alignment     | "
        f"Acc={alignment_acc.mean():.4f} "
        f"+/- {alignment_acc.std(ddof=1):.4f} | "
        f"F1={alignment_f1.mean():.4f}"
    )

    print(
        "Paired Accuracy Delta:",
        f"{acc_deltas.mean():+.3f} pp"
    )

    print(
        "Paired Macro-F1 Delta:",
        f"{f1_deltas.mean():+.3f} pp"
    )

    print(
        f"Wins/Ties/Losses: "
        f"{wins}/{ties}/{losses}"
    )


all_acc_deltas = np.array(
    all_acc_deltas
)

all_f1_deltas = np.array(
    all_f1_deltas
)

overall = {
    "mean_accuracy_delta_pp": float(
        all_acc_deltas.mean()
    ),
    "mean_macro_f1_delta_pp": float(
        all_f1_deltas.mean()
    ),
    "wins": int(
        (all_acc_deltas > 0).sum()
    ),
    "ties": int(
        (all_acc_deltas == 0).sum()
    ),
    "losses": int(
        (all_acc_deltas < 0).sum()
    ),
}

print("\nOVERALL COMPARISON")

print(
    "Mean Accuracy Delta:",
    f"{overall['mean_accuracy_delta_pp']:+.3f} pp"
)

print(
    "Mean Macro-F1 Delta:",
    f"{overall['mean_macro_f1_delta_pp']:+.3f} pp"
)

print(
    "Wins/Ties/Losses:",
    f"{overall['wins']}/"
    f"{overall['ties']}/"
    f"{overall['losses']}"
)

# ==========================================
# Save verified results
# ==========================================

output_path = output_dir / "metrics.json"

save_json(
    output_path,
    {
        **metadata,
        "completed_runs": len(results),
        "total_runs": TOTAL_RUNS,
        "results": results,
        "tent_reference_results": tent_results,
        "summary": summary,
        "overall": overall,
    },
)

print("\n" + "=" * 75)
print("VERIFIED ALIGNMENT EVALUATION COMPLETED")
print("New runs:", len(results), "/", TOTAL_RUNS)
print("Reused TENT runs:", len(tent_results))
print("Saved:", output_path)
print("=" * 75)
