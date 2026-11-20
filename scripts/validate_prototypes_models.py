
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64

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
    shift_data = json.load(f)

assert shift_data["split"] == "val"
shift = shift_data["shift"]

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


def load_prototypes(config):
    data = torch.load(
        ROOT / "checkpoints" / config["prototypes"],
        map_location="cpu",
        weights_only=True,
    )

    assert (
        data["source_checkpoint"]
        == config["checkpoint"]
    ), "Prototype and checkpoint do not match"

    prototypes = data["prototypes"]

    assert prototypes.shape == (9, 512)
    assert torch.isfinite(prototypes).all()

    return F.normalize(
        prototypes.to(DEVICE).float(),
        p=2,
        dim=1
    )


def build_model(config, use_target_bn):
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

    if use_target_bn:
        model.train()

        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    else:
        model.eval()

    return model


@torch.inference_mode()
def evaluate(model, prototypes):
    total = 0
    classifier_correct = 0
    prototype_correct = 0

    true_cosine_sum = 0.0
    nearest_cosine_sum = 0.0
    prediction_agreement = 0

    for images, labels in loader:
        images = images.to(
            DEVICE,
            non_blocking=True
        )
        labels = labels.view(-1).long().to(DEVICE)

        features, logits = model(images)

        classifier_pred = logits.argmax(dim=1)

        features = F.normalize(
            features.float(),
            p=2,
            dim=1
        )

        similarities = features @ prototypes.T

        prototype_pred = similarities.argmax(dim=1)

        batch_indices = torch.arange(
            len(labels),
            device=DEVICE
        )

        classifier_correct += (
            classifier_pred == labels
        ).sum().item()

        prototype_correct += (
            prototype_pred == labels
        ).sum().item()

        prediction_agreement += (
            classifier_pred == prototype_pred
        ).sum().item()

        true_cosine_sum += similarities[
            batch_indices,
            labels
        ].sum().item()

        nearest_cosine_sum += (
            similarities.max(dim=1).values.sum().item()
        )

        total += len(labels)

    return {
        "classifier_accuracy": classifier_correct / total,
        "prototype_accuracy": prototype_correct / total,
        "classifier_prototype_agreement": (
            prediction_agreement / total
        ),
        "mean_true_class_cosine": (
            true_cosine_sum / total
        ),
        "mean_nearest_cosine": (
            nearest_cosine_sum / total
        ),
    }


print("=" * 65)
print("CAFA-TTA | MULTI-MODEL PROTOTYPE VALIDATION")
print("Device:", DEVICE)
print("Validation samples:", len(dataset))
print("Shift:", shift)
print("Batch size:", BATCH_SIZE)
print("=" * 65)

results = {}

for name, config in MODELS.items():
    print("\n" + "=" * 65)
    print("MODEL:", name.upper())
    print("=" * 65)

    prototypes = load_prototypes(config)
    results[name] = {}

    for mode_name, use_target_bn in [
        ("source_bn", False),
        ("target_batch_bn", True),
    ]:
        model = build_model(
            config,
            use_target_bn=use_target_bn
        )

        metrics = evaluate(model, prototypes)
        results[name][mode_name] = metrics

        print("\nBN Mode:", mode_name)
        print(
            "Classifier Accuracy:",
            f"{metrics['classifier_accuracy']:.4f}"
        )
        print(
            "Prototype Accuracy:",
            f"{metrics['prototype_accuracy']:.4f}"
        )
        print(
            "Classifier-Prototype Agreement:",
            f"{metrics['classifier_prototype_agreement']:.4f}"
        )
        print(
            "Mean True-Class Cosine:",
            f"{metrics['mean_true_class_cosine']:.4f}"
        )
        print(
            "Mean Nearest Cosine:",
            f"{metrics['mean_nearest_cosine']:.4f}"
        )

        del model

    del prototypes

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


output_dir = (
    ROOT / "experiments/prototype_validation_models"
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
            "shift": shift,
            "batch_size": BATCH_SIZE,
            "models": MODELS,
            "results": results,
        },
        f,
        indent=2
    )

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

for name, model_results in results.items():
    print("\nMODEL:", name.upper())

    for mode_name, metrics in model_results.items():
        print(
            f"{mode_name:16s} | "
            f"Classifier={metrics['classifier_accuracy']:.4f} | "
            f"Prototype={metrics['prototype_accuracy']:.4f} | "
            f"Agreement={metrics['classifier_prototype_agreement']:.4f} | "
            f"TrueCosine={metrics['mean_true_class_cosine']:.4f}"
        )

print("\nPROTOTYPE VALIDATION COMPLETED")
print("Saved:", output_path)
