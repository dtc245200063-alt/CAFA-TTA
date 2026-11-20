
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
from core.cafa_soft_adapter import CAFASoftAdapter

# ==========================================
# CAFA-TTA v3 | ABLATION STUDY
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEEDS = [42, 123, 2026]
BATCH_SIZE = 64
LR = 1e-4
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

VARIANTS = {
    "entropy_baseline": {
        "entropy_floor": 1.0,
        "lambda_align": 0.0,
    },
    "soft_entropy_only": {
        "entropy_floor": 0.2,
        "lambda_align": 0.0,
    },
    "agreement_align_only": {
        "entropy_floor": 1.0,
        "lambda_align": 0.05,
    },
    "full_cafa_v3": {
        "entropy_floor": 0.2,
        "lambda_align": 0.05,
    },
}

# ==========================================
# Dataset and target shift
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
# Model and prototype loading
# ==========================================

def load_model(config):
    backbone = models.resnet18(weights=None)

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
# Controlled ablation evaluation
# ==========================================

def evaluate(model_name, config, variant, seed):
    set_seed(seed)

    loader = make_loader(seed)

    model = load_model(config)
    prototypes = load_prototypes(config)

    params = VARIANTS[variant]

    adapter = CAFASoftAdapter(
        model=model,
        prototypes=prototypes,
        lr=LR,
        lambda_align=params["lambda_align"],
        entropy_floor=params["entropy_floor"],
        min_samples=MIN_SAMPLES,
    )

    predictions = []
    labels_all = []

    agreement_total = 0
    updated_batches = 0
    total_seen = 0

    entropy_losses = []
    alignment_losses = []

    for images, labels in loader:
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # Predict first, before adaptation.
        with torch.no_grad():
            logits = adapter.predict(images)
            pred = logits.argmax(dim=1)

        predictions.extend(
            pred.cpu().tolist()
        )

        labels_all.extend(
            labels.view(-1).tolist()
        )

        # Adapt without target labels.
        result = adapter.adapt(images)

        total_seen += images.size(0)

        agreement_total += int(
            result["selected"]
        )

        if result["updated"]:
            updated_batches += 1

            entropy_losses.append(
                result["entropy_loss"]
            )

            alignment_losses.append(
                result["alignment_loss"]
            )

    raw_alignment_loss = (
        float(np.mean(alignment_losses))
        if alignment_losses else None
    )

    metrics = {
        "model": model_name,
        "checkpoint": config["checkpoint"],
        "variant": variant,
        "seed": seed,
        "entropy_floor": params["entropy_floor"],
        "lambda_align": params["lambda_align"],
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
        "agreement_rate": float(
            agreement_total / total_seen
        ),
        "updated_batches": updated_batches,
        "total_batches": len(loader),
        "mean_entropy_loss": (
            float(np.mean(entropy_losses))
            if entropy_losses else None
        ),
        "mean_raw_alignment_loss": (
            raw_alignment_loss
        ),
        "mean_effective_alignment_loss": (
            params["lambda_align"] * raw_alignment_loss
            if raw_alignment_loss is not None
            else None
        ),
    }

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
    ROOT / "experiments" / "cafa_v3_ablation"
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
    "min_samples": MIN_SAMPLES,
    "seeds": SEEDS,
    "models": MODELS,
    "variants": VARIANTS,
}

TOTAL_RUNS = (
    len(MODELS)
    * len(SEEDS)
    * len(VARIANTS)
)

# ==========================================
# Run all ablation experiments
# ==========================================

print("=" * 70)
print("CAFA-TTA v3 | ABLATION STUDY")
print("Device:", DEVICE)
print("Validation samples:", len(dataset))
print("Seeds:", SEEDS)
print("Shift:", shift)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)
print("Total runs:", TOTAL_RUNS)
print("=" * 70)

results = []

for model_name, config in MODELS.items():

    print("\n" + "=" * 70)
    print("MODEL:", model_name.upper())
    print("=" * 70)

    for seed in SEEDS:
        for variant in VARIANTS:

            metrics = evaluate(
                model_name,
                config,
                variant,
                seed,
            )

            results.append(metrics)

            print(
                f"[{len(results):2d}/{TOTAL_RUNS}] "
                f"Seed={seed:4d} | "
                f"{variant:21s} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['macro_f1']:.4f} | "
                f"Agreement={metrics['agreement_rate']:.2%} | "
                f"Updated={metrics['updated_batches']}"
                f"/{metrics['total_batches']}",
                flush=True,
            )

            # Save progress after every run.
            with open(
                output_dir / "partial_results.json",
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    {
                        **metadata,
                        "completed_runs": len(results),
                        "total_runs": TOTAL_RUNS,
                        "results": results,
                    },
                    f,
                    indent=2,
                )


# ==========================================
# Aggregate ablation results
# ==========================================

summary = {}

print("\n" + "=" * 70)
print("FINAL ABLATION SUMMARY")
print("=" * 70)

for model_name in MODELS:

    print("\nMODEL:", model_name.upper())

    summary[model_name] = {}

    model_rows = [
        row for row in results
        if row["model"] == model_name
    ]

    baseline_by_seed = {
        row["seed"]: row
        for row in model_rows
        if row["variant"] == "entropy_baseline"
    }

    for variant in VARIANTS:

        rows = [
            row for row in model_rows
            if row["variant"] == variant
        ]

        assert len(rows) == len(SEEDS)

        accuracies = np.array([
            row["accuracy"] for row in rows
        ])

        f1_values = np.array([
            row["macro_f1"] for row in rows
        ])

        acc_differences = np.array([
            row["accuracy"]
            - baseline_by_seed[row["seed"]]["accuracy"]
            for row in rows
        ]) * 100

        f1_differences = np.array([
            row["macro_f1"]
            - baseline_by_seed[row["seed"]]["macro_f1"]
            for row in rows
        ]) * 100

        summary[model_name][variant] = {
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
            "accuracy_difference_pp": float(
                acc_differences.mean()
            ),
            "macro_f1_difference_pp": float(
                f1_differences.mean()
            ),
            "accuracy_per_seed_pp": (
                acc_differences.tolist()
            ),
        }

        print(
            f"{variant:21s} | "
            f"Acc={accuracies.mean():.4f} "
            f"+/- {accuracies.std(ddof=1):.4f} | "
            f"F1={f1_values.mean():.4f} "
            f"+/- {f1_values.std(ddof=1):.4f} | "
            f"Delta={acc_differences.mean():+.3f} pp"
        )

    # ======================================
    # Analyze individual component effects
    # ======================================

    model_summary = summary[model_name]

    base = model_summary[
        "entropy_baseline"
    ]["accuracy_mean"]

    soft = model_summary[
        "soft_entropy_only"
    ]["accuracy_mean"]

    align = model_summary[
        "agreement_align_only"
    ]["accuracy_mean"]

    full = model_summary[
        "full_cafa_v3"
    ]["accuracy_mean"]

    component_effects = {
        "soft_entropy_alone_pp": (
            (soft - base) * 100
        ),
        "agreement_alignment_alone_pp": (
            (align - base) * 100
        ),
        "alignment_given_soft_entropy_pp": (
            (full - soft) * 100
        ),
        "soft_entropy_given_alignment_pp": (
            (full - align) * 100
        ),
        "full_improvement_pp": (
            (full - base) * 100
        ),
        "interaction_pp": (
            (full - soft - align + base) * 100
        ),
    }

    summary[model_name][
        "component_effects"
    ] = component_effects

    print("\nCOMPONENT EFFECTS:")

    for key, value in component_effects.items():
        print(f"  {key}: {value:+.3f} pp")


# ==========================================
# Save final metrics
# ==========================================

output_path = output_dir / "metrics.json"

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            **metadata,
            "completed_runs": len(results),
            "results": results,
            "summary": summary,
        },
        f,
        indent=2,
    )

print("\n" + "=" * 70)
print("CAFA-TTA v3 ABLATION STUDY COMPLETED")
print("Completed runs:", len(results), "/", TOTAL_RUNS)
print("Saved:", output_path)
print("=" * 70)
