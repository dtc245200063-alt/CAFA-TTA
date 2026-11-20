
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from medmnist import PathMNIST

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.feature_extractor import ResNetFeatureExtractor
from kernel.confidence import confidence_score

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
THRESHOLD = 0.7

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


def load_model(config):
    backbone = models.resnet18(weights=None)

    backbone.fc = nn.Linear(
        backbone.fc.in_features,
        9
    )

    checkpoint = torch.load(
        ROOT / "checkpoints" / config["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )

    backbone.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = ResNetFeatureExtractor(
        backbone
    ).to(DEVICE)

    # Match CAFA's target-batch BN behavior.
    model.train()

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    return model


def load_prototypes(config):
    data = torch.load(
        ROOT / "checkpoints" / config["prototypes"],
        map_location="cpu",
        weights_only=True,
    )

    assert data["source_checkpoint"] == config["checkpoint"]

    prototypes = data["prototypes"].to(DEVICE)

    assert prototypes.shape == (9, 512)
    assert torch.isfinite(prototypes).all()

    return F.normalize(
        prototypes.float(),
        p=2,
        dim=1
    )


def group_stats(mask, correct):
    count = int(mask.sum().item())
    total = len(mask)

    correct_count = int(
        (mask & correct).sum().item()
    )

    return {
        "count": count,
        "coverage": count / total,
        "correct": correct_count,
        "accuracy": (
            correct_count / count
            if count > 0 else None
        ),
    }


@torch.inference_mode()
def evaluate(config):
    model = load_model(config)
    prototypes = load_prototypes(config)

    labels_list = []
    classifier_list = []
    prototype_list = []
    confidence_list = []

    for images, labels in loader:
        images = images.to(
            DEVICE,
            non_blocking=True
        )

        features, logits = model(images)

        classifier_pred = logits.argmax(dim=1)

        features = F.normalize(
            features.float(),
            p=2,
            dim=1
        )

        similarities = features @ prototypes.T
        prototype_pred = similarities.argmax(dim=1)

        confidence = confidence_score(logits)

        labels_list.append(labels.view(-1).long().cpu())
        classifier_list.append(classifier_pred.cpu())
        prototype_list.append(prototype_pred.cpu())
        confidence_list.append(confidence.cpu())

    labels = torch.cat(labels_list)
    classifier = torch.cat(classifier_list)
    prototype = torch.cat(prototype_list)
    confidence = torch.cat(confidence_list)

    correct = classifier == labels

    agreement = classifier == prototype
    confident = confidence >= THRESHOLD
    joint = agreement & confident

    groups = {
        "all": torch.ones_like(correct, dtype=torch.bool),
        "agreement": agreement,
        "disagreement": ~agreement,
        "confidence_only": confident,
        "joint_selected": joint,
        "joint_rejected": ~joint,
    }

    stats = {
        name: group_stats(mask, correct)
        for name, mask in groups.items()
    }

    # How many correct predictions get discarded?
    correct_rejected = int(
        (correct & ~joint).sum().item()
    )

    total_correct = int(correct.sum().item())

    stats["correct_rejected_fraction"] = (
        correct_rejected / total_correct
        if total_correct else None
    )

    stats["prototype_accuracy"] = float(
        (prototype == labels).float().mean().item()
    )

    stats["mean_confidence"] = float(
        confidence.mean().item()
    )

    del model
    del prototypes

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return stats


def format_accuracy(value):
    return (
        f"{value:.2%}"
        if value is not None else "N/A"
    )


print("=" * 65)
print("CAFA-TTA | PSEUDO-LABEL QUALITY ANALYSIS")
print("Device:", DEVICE)
print("Samples:", len(dataset))
print("Threshold:", THRESHOLD)
print("Shift:", shift)
print("=" * 65)

results = {}

for name, config in MODELS.items():
    stats = evaluate(config)
    results[name] = stats

    print("\n" + "=" * 65)
    print("MODEL:", name.upper())
    print("=" * 65)

    for group in [
        "all",
        "agreement",
        "disagreement",
        "confidence_only",
        "joint_selected",
        "joint_rejected",
    ]:
        row = stats[group]

        print(
            f"{group:17s} | "
            f"Count={row['count']:5d} | "
            f"Coverage={row['coverage']:.2%} | "
            f"Accuracy={format_accuracy(row['accuracy'])}"
        )

    print(
        "Prototype Accuracy:",
        f"{stats['prototype_accuracy']:.2%}"
    )

    print(
        "Correct predictions rejected:",
        format_accuracy(
            stats["correct_rejected_fraction"]
        )
    )


output_dir = (
    ROOT / "experiments/pseudo_label_analysis"
)
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "metrics.json"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "split": "val",
            "batch_size": BATCH_SIZE,
            "threshold": THRESHOLD,
            "shift": shift,
            "results": results,
        },
        f,
        indent=2,
    )

print("\nPSEUDO-LABEL ANALYSIS COMPLETED")
print("Saved:", output_path)
