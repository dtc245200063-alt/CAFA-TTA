
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
# CAFA-TTA v3 | DOMAIN SHIFT SEVERITY
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
BATCH_SIZE = 64
LR = 1e-4
ENTROPY_FLOOR = 0.2
LAMBDA_ALIGN = 0.05
MIN_SAMPLES = 2

SHIFTS = {
    "original": (1.0, 1.0, 1.0),
    "light": (0.95, 1.05, 0.95),
    "medium": (0.85, 1.15, 0.85),
    "strong": (0.75, 1.35, 0.70),
}

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

METHODS = [
    "source_only",
    "bn_only",
    "tent",
    "cafa_v3",
]

# ==========================================
# Reproducibility
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# Dataset and domain shift
# ==========================================

normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3,
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

        return normalize(
            TF.to_tensor(image)
        )

    return transform


def make_loader(shift_name):

    factors = SHIFTS[shift_name]

    dataset = PathMNIST(
        split="val",
        root=str(ROOT / "data/raw/medmnist"),
        transform=make_transform(*factors),
        download=False,
        size=28,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


# ==========================================
# Load source checkpoint
# ==========================================

def create_backbone(checkpoint_name):

    backbone = models.resnet18(
        weights=None
    )

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
# Configure BN-only baseline
# ==========================================

def configure_target_bn(model):

    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():

        if isinstance(module, nn.BatchNorm2d):

            module.train()

            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    return model


# ==========================================
# Evaluate one independent run
# ==========================================

def evaluate(model_name, config, shift_name, method):

    set_seed(SEED)

    loader = make_loader(shift_name)

    backbone = create_backbone(
        config["checkpoint"]
    )

    adapter = None
    feature_model = None
    prototypes = None

    if method == "source_only":

        backbone.eval()

        for param in backbone.parameters():
            param.requires_grad_(False)

    elif method == "bn_only":

        backbone = configure_target_bn(
            backbone
        )

    elif method == "tent":

        adapter = TentAdapter(
            backbone,
            learning_rate=LR,
        )

    elif method == "cafa_v3":

        feature_model = ResNetFeatureExtractor(
            backbone
        )

        prototypes = load_prototypes(
            config
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

    total_seen = 0
    agreement_total = 0
    updated_batches = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        # Predict before any parameter update.
        with torch.no_grad():

            if adapter is None:
                logits = backbone(images)
            else:
                logits = adapter.predict(images)

            pred = logits.argmax(dim=1)

        predictions.extend(
            pred.cpu().tolist()
        )

        labels_all.extend(
            labels.view(-1).tolist()
        )

        total_seen += images.size(0)

        # Adapt without ground-truth labels.
        if adapter is not None:

            result = adapter.adapt(images)

            if method == "cafa_v3":

                agreement_total += int(
                    result["selected"]
                )

                if result["updated"]:
                    updated_batches += 1

    metrics = {
        "model": model_name,
        "checkpoint": config["checkpoint"],
        "shift": shift_name,
        "method": method,
        "samples": total_seen,
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

        metrics["updated_batches"] = (
            updated_batches
        )

        metrics["total_batches"] = len(loader)

    del adapter
    del feature_model
    del prototypes
    del backbone
    del loader

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


# ==========================================
# Output configuration
# ==========================================

output_dir = (
    ROOT
    / "experiments"
    / "cafa_v3_severity"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

TOTAL_RUNS = (
    len(MODELS)
    * len(SHIFTS)
    * len(METHODS)
)

metadata = {
    "protocol": "ordered_online_prequential",
    "split": "val",
    "seed": SEED,
    "shuffle": False,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "entropy_floor": ENTROPY_FLOOR,
    "lambda_align": LAMBDA_ALIGN,
    "min_samples": MIN_SAMPLES,
    "models": MODELS,
    "shifts": SHIFTS,
    "shift_factor_order": [
        "brightness",
        "contrast",
        "saturation",
    ],
    "methods": METHODS,
}

# ==========================================
# Run experiments
# ==========================================

print("=" * 75)
print("CAFA-TTA v3 | DOMAIN SHIFT SEVERITY")
print("Device:", DEVICE)
print("Dataset: PathMNIST Validation")
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)
print("Entropy floor:", ENTROPY_FLOOR)
print("Alignment coefficient:", LAMBDA_ALIGN)
print("Shuffle: False")
print("Total runs:", TOTAL_RUNS)
print("=" * 75)

results = []

for model_name, config in MODELS.items():

    print("\n" + "=" * 75)
    print("MODEL:", model_name.upper())
    print("=" * 75)

    for shift_name in SHIFTS:

        print(
            "\nSHIFT:",
            shift_name.upper(),
            SHIFTS[shift_name],
        )

        for method in METHODS:

            metrics = evaluate(
                model_name,
                config,
                shift_name,
                method,
            )

            results.append(metrics)

            line = (
                f"[{len(results):2d}/{TOTAL_RUNS}] "
                f"{method:12s} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"F1={metrics['macro_f1']:.4f}"
            )

            if method == "cafa_v3":

                line += (
                    f" | Agreement="
                    f"{metrics['agreement_rate']:.2%}"
                    f" | Updated="
                    f"{metrics['updated_batches']}"
                    f"/{metrics['total_batches']}"
                )

            print(line, flush=True)

            # Save progress after each run.
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
# Aggregate results
# ==========================================

summary = {}

print("\n" + "=" * 75)
print("FINAL SEVERITY SUMMARY | ACCURACY")
print("=" * 75)

for model_name in MODELS:

    summary[model_name] = {}

    print("\nMODEL:", model_name.upper())

    print(
        f"{'Shift':10s} | "
        f"{'Source':>8s} | "
        f"{'BN-only':>8s} | "
        f"{'TENT':>8s} | "
        f"{'CAFA v3':>8s} | "
        f"{'Delta pp':>9s}"
    )

    print("-" * 75)

    for shift_name in SHIFTS:

        rows = [
            row for row in results
            if row["model"] == model_name
            and row["shift"] == shift_name
        ]

        by_method = {
            row["method"]: row
            for row in rows
        }

        assert set(by_method) == set(METHODS)

        source = by_method[
            "source_only"
        ]["accuracy"]

        bn = by_method[
            "bn_only"
        ]["accuracy"]

        tent = by_method[
            "tent"
        ]["accuracy"]

        cafa = by_method[
            "cafa_v3"
        ]["accuracy"]

        delta_tent = (
            cafa - tent
        ) * 100

        delta_bn = (
            cafa - bn
        ) * 100

        delta_source = (
            cafa - source
        ) * 100

        f1_delta_tent = (
            by_method["cafa_v3"]["macro_f1"]
            - by_method["tent"]["macro_f1"]
        ) * 100

        summary[model_name][shift_name] = {
            "methods": by_method,
            "cafa_vs_tent_pp": float(
                delta_tent
            ),
            "cafa_vs_bn_only_pp": float(
                delta_bn
            ),
            "cafa_vs_source_only_pp": float(
                delta_source
            ),
            "cafa_vs_tent_macro_f1_pp": float(
                f1_delta_tent
            ),
        }

        print(
            f"{shift_name:10s} | "
            f"{source:8.4f} | "
            f"{bn:8.4f} | "
            f"{tent:8.4f} | "
            f"{cafa:8.4f} | "
            f"{delta_tent:+8.3f}"
        )


# ==========================================
# Macro-F1 summary
# ==========================================

print("\n" + "=" * 75)
print("FINAL SEVERITY SUMMARY | MACRO-F1")
print("=" * 75)

for model_name in MODELS:

    print("\nMODEL:", model_name.upper())

    print(
        f"{'Shift':10s} | "
        f"{'Source':>8s} | "
        f"{'BN-only':>8s} | "
        f"{'TENT':>8s} | "
        f"{'CAFA v3':>8s} | "
        f"{'Delta pp':>9s}"
    )

    print("-" * 75)

    for shift_name in SHIFTS:

        row = summary[
            model_name
        ][shift_name]

        methods = row["methods"]

        source = methods[
            "source_only"
        ]["macro_f1"]

        bn = methods[
            "bn_only"
        ]["macro_f1"]

        tent = methods[
            "tent"
        ]["macro_f1"]

        cafa = methods[
            "cafa_v3"
        ]["macro_f1"]

        delta = (
            cafa - tent
        ) * 100

        print(
            f"{shift_name:10s} | "
            f"{source:8.4f} | "
            f"{bn:8.4f} | "
            f"{tent:8.4f} | "
            f"{cafa:8.4f} | "
            f"{delta:+8.3f}"
        )


# ==========================================
# Overall comparison
# ==========================================

acc_deltas = []
f1_deltas = []

shift_summary = {}

print("\n" + "=" * 75)
print("CAFA v3 VS TENT | BY SHIFT")
print("=" * 75)

for shift_name in SHIFTS:

    shift_acc = [
        summary[name][shift_name][
            "cafa_vs_tent_pp"
        ]
        for name in MODELS
    ]

    shift_f1 = [
        summary[name][shift_name][
            "cafa_vs_tent_macro_f1_pp"
        ]
        for name in MODELS
    ]

    acc_deltas.extend(shift_acc)
    f1_deltas.extend(shift_f1)

    shift_summary[shift_name] = {
        "mean_accuracy_delta_pp": float(
            np.mean(shift_acc)
        ),
        "mean_macro_f1_delta_pp": float(
            np.mean(shift_f1)
        ),
    }

    print(
        f"{shift_name:10s} | "
        f"Mean Acc Delta="
        f"{np.mean(shift_acc):+.3f} pp | "
        f"Mean F1 Delta="
        f"{np.mean(shift_f1):+.3f} pp"
    )

acc_deltas = np.array(acc_deltas)
f1_deltas = np.array(f1_deltas)

wins = int(
    (acc_deltas > 0).sum()
)

ties = int(
    (acc_deltas == 0).sum()
)

losses = int(
    (acc_deltas < 0).sum()
)

overall = {
    "mean_accuracy_delta_pp": float(
        acc_deltas.mean()
    ),
    "mean_macro_f1_delta_pp": float(
        f1_deltas.mean()
    ),
    "accuracy_wins": wins,
    "accuracy_ties": ties,
    "accuracy_losses": losses,
    "total_comparisons": len(acc_deltas),
}

print("\nOVERALL COMPARISON")

print(
    "Mean Accuracy Delta:",
    f"{acc_deltas.mean():+.3f} pp"
)

print(
    "Mean Macro-F1 Delta:",
    f"{f1_deltas.mean():+.3f} pp"
)

print(
    f"Wins/Ties/Losses: "
    f"{wins}/{ties}/{losses}"
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
            **metadata,
            "completed_runs": len(results),
            "total_runs": TOTAL_RUNS,
            "results": results,
            "summary": summary,
            "shift_summary": shift_summary,
            "overall": overall,
        },
        f,
        indent=2,
    )

print("\n" + "=" * 75)
print("CAFA-TTA v3 SEVERITY EVALUATION COMPLETED")
print("Completed runs:", len(results), "/", TOTAL_RUNS)
print("Saved:", output_path)
print("=" * 75)
