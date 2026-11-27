
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


# ============================================
# CAFA-TTA v2 | MULTI-MODEL EVALUATION
# ============================================

SEED = 42
BATCH_SIZE = 64
LR = 1e-4
THRESHOLD = 0.7
LAMBDA_ALIGN = 0.05
MIN_SAMPLES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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


# 1. Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# 2. Read domain shift configuration
with open(
    ROOT / "experiments/domain_shift/metrics.json",
    "r",
    encoding="utf-8",
) as f:
    shift_config = json.load(f)

assert shift_config["split"] == "val"

shift = shift_config["shift"]

normalize = transforms.Normalize(
    mean=[0.5] * 3,
    std=[0.5] * 3,
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


# 3. Load validation data once
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

    checkpoint_path = (
        ROOT / "checkpoints" / checkpoint_name
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    backbone.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = ResNetFeatureExtractor(
        backbone
    ).to(DEVICE)

    return model


def load_prototypes(prototype_name, checkpoint_name):
    prototype_path = (
        ROOT / "checkpoints" / prototype_name
    )

    prototype_data = torch.load(
        prototype_path,
        map_location="cpu",
        weights_only=True,
    )

    prototypes = prototype_data["prototypes"]

    assert prototypes.shape == (9, 512)
    assert torch.isfinite(prototypes).all()

    # Prevent checkpoint/prototype mismatches.
    assert (
        prototype_data["source_checkpoint"]
        == checkpoint_name
    ), (
        "Prototype does not match model: "
        f"{prototype_name}"
    )

    return prototypes


def mean_or_zero(values):
    if not values:
        return 0.0

    return float(sum(values) / len(values))


def run_experiment(model_name, config):
    print("\n" + "=" * 60)
    print("MODEL:", model_name.upper())
    print("Checkpoint:", config["checkpoint"])
    print("Prototypes:", config["prototypes"])
    print("=" * 60)

    # Reset random state between models.
    set_seed(SEED)

    model = load_model(
        config["checkpoint"]
    )

    prototypes = load_prototypes(
        config["prototypes"],
        config["checkpoint"],
    )

    adapter = CAFAAgreementAdapter(
        model=model,
        prototypes=prototypes,
        lr=LR,
        threshold=THRESHOLD,
        lambda_align=LAMBDA_ALIGN,
        min_samples=MIN_SAMPLES,
    )

    all_labels = []
    all_predictions = []

    entropy_losses = []
    alignment_losses = []
    total_losses = []

    total_selected = 0
    total_updates = 0
    total_seen = 0

    # Online prequential protocol:
    # predict before adapting on each batch.
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        with torch.no_grad():
            logits = adapter.predict(images)
            predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.view(-1).tolist()
        )

        # Adapt using unlabeled images.
        result = adapter.adapt(images)

        total_seen += images.size(0)
        total_selected += int(result["selected"])

        if result["updated"]:
            total_updates += 1

            entropy_losses.append(
                float(result["entropy_loss"])
            )

            alignment_losses.append(
                float(result["alignment_loss"])
            )

            total_losses.append(
                float(result["total_loss"])
            )

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Batch {batch_idx + 1}/{len(loader)} "
                f"| Selected: {result['selected']}/{len(images)} "
                f"| Updated: {result['updated']}",
                flush=True,
            )

    accuracy = float(
        accuracy_score(
            all_labels,
            all_predictions,
        )
    )

    macro_f1 = float(
        f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    selection_rate = (
        total_selected / total_seen
    )

    results = {
        "method": "CAFA-TTA-v2-Agreement",
        "model": model_name,
        "checkpoint": config["checkpoint"],
        "prototype_file": config["prototypes"],
        "prototype_mode": "source_eval",
        "protocol": "online_prequential",
        "split": "val",
        "seed": SEED,
        "shuffle": False,
        "shift": shift,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "confidence_threshold": THRESHOLD,
        "lambda_align": LAMBDA_ALIGN,
        "min_samples": MIN_SAMPLES,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "selection_rate": selection_rate,
        "selected_samples": total_selected,
        "total_samples": total_seen,
        "updated_batches": total_updates,
        "total_batches": len(loader),
        "mean_entropy_loss": mean_or_zero(
            entropy_losses
        ),
        "mean_alignment_loss": mean_or_zero(
            alignment_losses
        ),
        "mean_total_loss": mean_or_zero(
            total_losses
        ),
    }

    output_dir = (
        ROOT / "experiments"
        / "cafa_models"
        / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\nRESULTS:", model_name.upper())
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Selection Rate: {selection_rate:.2%}")
    print(
        f"Updated Batches: "
        f"{total_updates}/{len(loader)}"
    )
    print(
        f"Mean Entropy Loss: "
        f"{results['mean_entropy_loss']:.4f}"
    )
    print(
        f"Mean Alignment Loss: "
        f"{results['mean_alignment_loss']:.4f}"
    )
    print("Saved:", output_dir / "metrics.json")

    # Release GPU memory before next model.
    del adapter
    del model
    del prototypes

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return results


print("=" * 60)
print("CAFA-TTA v2 | MULTI-MODEL EVALUATION")
print("Device:", DEVICE)
print("Dataset: PathMNIST Validation")
print("Target samples:", len(dataset))
print("Shift:", shift)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LR)
print("Confidence threshold:", THRESHOLD)
print("Alignment coefficient:", LAMBDA_ALIGN)
print("=" * 60)

all_results = {}

for model_name, config in MODELS.items():
    result = run_experiment(
        model_name,
        config,
    )

    all_results[model_name] = result


# 4. Save summary
summary_dir = ROOT / "experiments/cafa_models"
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


# 5. Compare with existing TENT results
print("\n" + "=" * 70)
print("FINAL SUMMARY | TENT VS CAFA-TTA v2")
print("=" * 70)

for name, cafa_result in all_results.items():
    tent_path = (
        ROOT / "experiments"
        / "tent_models"
        / name
        / "metrics.json"
    )

    if tent_path.exists():
        with open(
            tent_path,
            "r",
            encoding="utf-8",
        ) as f:
            tent_result = json.load(f)

        # Ensure comparison uses the same protocol.
        assert tent_result["checkpoint"] == cafa_result["checkpoint"]
        assert tent_result["split"] == cafa_result["split"]
        assert tent_result["batch_size"] == cafa_result["batch_size"]
        assert tent_result["shift"] == cafa_result["shift"]
        assert tent_result["shuffle"] == cafa_result["shuffle"]

        tent_acc = tent_result["tent"]["accuracy"]
        cafa_acc = cafa_result["accuracy"]

        difference_pp = (
            cafa_acc - tent_acc
        ) * 100

        print(
            f"{name:9s} | "
            f"TENT: {tent_acc:.4f} | "
            f"CAFA: {cafa_acc:.4f} | "
            f"Difference: {difference_pp:+.2f} pp | "
            f"Selection: {cafa_result['selection_rate']:.2%}"
        )

    else:
        print(
            f"{name:9s} | "
            f"CAFA: {cafa_result['accuracy']:.4f} | "
            "TENT result unavailable"
        )

print("\nCAFA-TTA MULTI-MODEL EVALUATION COMPLETED")
