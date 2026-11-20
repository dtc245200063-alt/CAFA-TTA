
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
# CAFA-TTA v3 | MULTI-TYPE CORRUPTION STUDY
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
NOISE_SEED = 2026

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

CORRUPTIONS = {
    "gaussian_noise": {
        "light": 0.02,
        "medium": 0.05,
        "strong": 0.10,
    },
    "gaussian_blur": {
        "light": 0.5,
        "medium": 1.0,
        "strong": 2.0,
    },
    "hue_shift": {
        "light": 0.03,
        "medium": 0.07,
        "strong": 0.15,
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


normalize = transforms.Normalize(
    [0.5] * 3,
    [0.5] * 3,
)

# ==========================================
# Corrupted target dataset
# ==========================================

class CorruptedPathMNIST(Dataset):

    def __init__(
        self,
        corruption,
        severity,
    ):
        self.corruption = corruption
        self.severity = severity

        self.strength = CORRUPTIONS[
            corruption
        ][severity]

        self.dataset = PathMNIST(
            split="val",
            root=str(
                ROOT / "data/raw/medmnist"
            ),
            transform=None,
            download=False,
            size=28,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, label = self.dataset[index]

        if self.corruption == "gaussian_noise":

            image = TF.to_tensor(image)

            # Deterministic noise for every image.
            generator = torch.Generator()
            generator.manual_seed(
                NOISE_SEED + index
            )

            noise = torch.randn(
                image.shape,
                generator=generator,
                dtype=image.dtype,
            )

            image = (
                image + self.strength * noise
            ).clamp(0.0, 1.0)

        elif self.corruption == "gaussian_blur":

            image = TF.to_tensor(image)

            image = TF.gaussian_blur(
                image,
                kernel_size=[13, 13],
                sigma=[
                    self.strength,
                    self.strength,
                ],
            )

        elif self.corruption == "hue_shift":

            image = TF.adjust_hue(
                image,
                self.strength,
            )

            image = TF.to_tensor(image)

        else:

            raise ValueError(
                f"Unknown corruption: "
                f"{self.corruption}"
            )

        image = normalize(image)

        return image, label


def make_loader(corruption, severity):

    dataset = CorruptedPathMNIST(
        corruption,
        severity,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


# ==========================================
# Source model and prototypes
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
# BN-only baseline
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
# Evaluate one independent experiment
# ==========================================

def evaluate(
    model_name,
    config,
    corruption,
    severity,
    method,
):

    set_seed(SEED)

    loader = make_loader(
        corruption,
        severity,
    )

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

        # Predict first.
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

        # Adapt afterward, without target labels.
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
        "corruption": corruption,
        "severity": severity,
        "strength": CORRUPTIONS[
            corruption
        ][severity],
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
    / "cafa_v3_corruptions"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

partial_path = (
    output_dir / "partial_results.json"
)

output_path = (
    output_dir / "metrics.json"
)

TOTAL_RUNS = (
    len(MODELS)
    * sum(
        len(levels)
        for levels in CORRUPTIONS.values()
    )
    * len(METHODS)
)

metadata = {
    "protocol": "ordered_online_prequential",
    "split": "val",
    "seed": SEED,
    "noise_seed": NOISE_SEED,
    "shuffle": False,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "entropy_floor": ENTROPY_FLOOR,
    "lambda_align": LAMBDA_ALIGN,
    "min_samples": MIN_SAMPLES,
    "models": MODELS,
    "corruptions": CORRUPTIONS,
    "gaussian_blur_kernel_size": 13,
    "methods": METHODS,
}


def save_json(path, data):

    # Atomic replacement to protect saved results.
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


def run_key(row):

    return (
        row["model"],
        row["corruption"],
        row["severity"],
        row["method"],
    )


# ==========================================
# Resume unfinished experiments
# ==========================================

results = []
completed = set()

if partial_path.exists():

    with open(
        partial_path,
        encoding="utf-8",
    ) as f:

        previous = json.load(f)

    if previous["metadata"] != metadata:

        raise RuntimeError(
            "Saved configuration differs. "
            "Do not mix experiment results."
        )

    results = previous["results"]

    completed = {
        run_key(row)
        for row in results
    }

    assert len(results) == len(completed)


# ==========================================
# Run corruption experiments
# ==========================================

print("=" * 75)
print("CAFA-TTA v3 | MULTI-TYPE CORRUPTIONS")
print("Device:", DEVICE)
print("Dataset: PathMNIST Validation")
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)
print("Entropy floor:", ENTROPY_FLOOR)
print("Alignment coefficient:", LAMBDA_ALIGN)
print("Corruptions:", CORRUPTIONS)
print("Total runs:", TOTAL_RUNS)
print("Already completed:", len(completed))
print("=" * 75)

for model_name, config in MODELS.items():

    print("\n" + "=" * 75)
    print("MODEL:", model_name.upper())
    print("=" * 75)

    for corruption, levels in CORRUPTIONS.items():

        for severity, strength in levels.items():

            print(
                "\nCORRUPTION:",
                corruption.upper(),
                "|",
                severity.upper(),
                "| Strength:",
                strength,
            )

            for method in METHODS:

                key = (
                    model_name,
                    corruption,
                    severity,
                    method,
                )

                if key in completed:

                    print(
                        f"SKIP {method}: "
                        "already completed",
                        flush=True,
                    )

                    continue

                metrics = evaluate(
                    model_name,
                    config,
                    corruption,
                    severity,
                    method,
                )

                results.append(metrics)
                completed.add(key)

                line = (
                    f"[{len(results):3d}/{TOTAL_RUNS}] "
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

                # Save completed runs for resuming.
                save_json(
                    partial_path,
                    {
                        "metadata": metadata,
                        "completed_runs": len(results),
                        "total_runs": TOTAL_RUNS,
                        "results": results,
                    },
                )


# ==========================================
# Aggregate results
# ==========================================

assert len(results) == TOTAL_RUNS

lookup = {
    run_key(row): row
    for row in results
}

summary = {}
all_acc_deltas = []
all_f1_deltas = []

print("\n" + "=" * 75)
print("FINAL CORRUPTION SUMMARY | ACCURACY")
print("=" * 75)

for model_name in MODELS:

    summary[model_name] = {}

    print("\nMODEL:", model_name.upper())

    for corruption, levels in CORRUPTIONS.items():

        summary[model_name][corruption] = {}

        print("\nCORRUPTION:", corruption.upper())

        print(
            f"{'Severity':10s} | "
            f"{'Source':>8s} | "
            f"{'BN-only':>8s} | "
            f"{'TENT':>8s} | "
            f"{'CAFA v3':>8s} | "
            f"{'Delta pp':>9s}"
        )

        print("-" * 75)

        for severity in levels:

            by_method = {
                method: lookup[
                    (
                        model_name,
                        corruption,
                        severity,
                        method,
                    )
                ]
                for method in METHODS
            }

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

            acc_delta = (
                cafa - tent
            ) * 100

            f1_delta = (
                by_method["cafa_v3"]["macro_f1"]
                - by_method["tent"]["macro_f1"]
            ) * 100

            all_acc_deltas.append(acc_delta)
            all_f1_deltas.append(f1_delta)

            summary[model_name][corruption][
                severity
            ] = {
                "methods": by_method,
                "cafa_vs_tent_pp": float(
                    acc_delta
                ),
                "cafa_vs_tent_macro_f1_pp": float(
                    f1_delta
                ),
                "cafa_vs_bn_only_pp": float(
                    (cafa - bn) * 100
                ),
                "cafa_vs_source_only_pp": float(
                    (cafa - source) * 100
                ),
            }

            print(
                f"{severity:10s} | "
                f"{source:8.4f} | "
                f"{bn:8.4f} | "
                f"{tent:8.4f} | "
                f"{cafa:8.4f} | "
                f"{acc_delta:+8.3f}"
            )


# ==========================================
# Macro-F1 summary
# ==========================================

print("\n" + "=" * 75)
print("FINAL CORRUPTION SUMMARY | MACRO-F1")
print("=" * 75)

for model_name in MODELS:

    print("\nMODEL:", model_name.upper())

    for corruption, levels in CORRUPTIONS.items():

        print("\nCORRUPTION:", corruption.upper())

        print(
            f"{'Severity':10s} | "
            f"{'Source':>8s} | "
            f"{'BN-only':>8s} | "
            f"{'TENT':>8s} | "
            f"{'CAFA v3':>8s} | "
            f"{'Delta pp':>9s}"
        )

        print("-" * 75)

        for severity in levels:

            row = summary[
                model_name
            ][corruption][severity]

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

            print(
                f"{severity:10s} | "
                f"{source:8.4f} | "
                f"{bn:8.4f} | "
                f"{tent:8.4f} | "
                f"{cafa:8.4f} | "
                f"{(cafa-tent)*100:+8.3f}"
            )


# ==========================================
# Results by corruption type
# ==========================================

corruption_summary = {}

print("\n" + "=" * 75)
print("CAFA v3 VS TENT | BY CORRUPTION")
print("=" * 75)

for corruption, levels in CORRUPTIONS.items():

    acc_deltas = np.array([
        summary[m][corruption][s][
            "cafa_vs_tent_pp"
        ]
        for m in MODELS
        for s in levels
    ])

    f1_deltas = np.array([
        summary[m][corruption][s][
            "cafa_vs_tent_macro_f1_pp"
        ]
        for m in MODELS
        for s in levels
    ])

    wins = int(
        (acc_deltas > 0).sum()
    )

    ties = int(
        (acc_deltas == 0).sum()
    )

    losses = int(
        (acc_deltas < 0).sum()
    )

    corruption_summary[corruption] = {
        "mean_accuracy_delta_pp": float(
            acc_deltas.mean()
        ),
        "mean_macro_f1_delta_pp": float(
            f1_deltas.mean()
        ),
        "accuracy_wins": wins,
        "accuracy_ties": ties,
        "accuracy_losses": losses,
    }

    print(
        f"{corruption:16s} | "
        f"Acc Delta={acc_deltas.mean():+.3f} pp | "
        f"F1 Delta={f1_deltas.mean():+.3f} pp | "
        f"W/T/L={wins}/{ties}/{losses}"
    )


# ==========================================
# Overall results
# ==========================================

all_acc_deltas = np.array(
    all_acc_deltas
)

all_f1_deltas = np.array(
    all_f1_deltas
)

wins = int(
    (all_acc_deltas > 0).sum()
)

ties = int(
    (all_acc_deltas == 0).sum()
)

losses = int(
    (all_acc_deltas < 0).sum()
)

overall = {
    "mean_accuracy_delta_pp": float(
        all_acc_deltas.mean()
    ),
    "mean_macro_f1_delta_pp": float(
        all_f1_deltas.mean()
    ),
    "accuracy_wins": wins,
    "accuracy_ties": ties,
    "accuracy_losses": losses,
    "total_comparisons": len(
        all_acc_deltas
    ),
}

print("\nOVERALL COMPARISON")

print(
    "Mean Accuracy Delta:",
    f"{all_acc_deltas.mean():+.3f} pp"
)

print(
    "Mean Macro-F1 Delta:",
    f"{all_f1_deltas.mean():+.3f} pp"
)

print(
    f"Wins/Ties/Losses: "
    f"{wins}/{ties}/{losses}"
)

# ==========================================
# Save final results
# ==========================================

save_json(
    output_path,
    {
        **metadata,
        "completed_runs": len(results),
        "total_runs": TOTAL_RUNS,
        "results": results,
        "summary": summary,
        "corruption_summary": corruption_summary,
        "overall": overall,
    },
)

print("\n" + "=" * 75)
print("CAFA-TTA v3 CORRUPTION EVALUATION COMPLETED")
print(
    "Completed runs:",
    len(results),
    "/",
    TOTAL_RUNS,
)
print("Saved:", output_path)
print("=" * 75)
