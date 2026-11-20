
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

print("=" * 55)
print("CAFA-TTA | PROTOTYPE VALIDATION")
print("Device:", DEVICE)
print("=" * 55)

# Load source prototypes
proto_data = torch.load(
    ROOT / "checkpoints/source_prototypes.pt",
    map_location="cpu",
    weights_only=True
)

prototypes = proto_data["prototypes"].to(DEVICE)
prototypes = F.normalize(prototypes, dim=1)

assert prototypes.shape == (9, 512)

# Load the exact simulated shift settings
with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    shift = json.load(f)["shift"]

normalize = transforms.Normalize(
    mean=[0.5] * 3,
    std=[0.5] * 3
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
    size=28
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

def build_model(use_target_bn=False):
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(
        backbone.fc.in_features, 9
    )

    checkpoint = torch.load(
        ROOT / "checkpoints/resnet18_pathmnist_best.pt",
        map_location="cpu",
        weights_only=True
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

@torch.no_grad()
def evaluate_features(model):
    total = 0
    correct = 0
    sum_true_similarity = 0.0
    sum_pred_similarity = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.view(-1).long().to(DEVICE)

        features, logits = model(images)

        features = F.normalize(features, dim=1)
        similarity = features @ prototypes.T

        prototype_predictions = similarity.argmax(dim=1)

        correct += (
            prototype_predictions == labels
        ).sum().item()

        batch_indices = torch.arange(
            len(labels), device=DEVICE
        )

        sum_true_similarity += similarity[
            batch_indices, labels
        ].sum().item()

        sum_pred_similarity += similarity.max(
            dim=1
        ).values.sum().item()

        total += len(labels)

    return {
        "prototype_accuracy": correct / total,
        "mean_true_class_cosine": (
            sum_true_similarity / total
        ),
        "mean_nearest_cosine": (
            sum_pred_similarity / total
        )
    }

source_bn_model = build_model(
    use_target_bn=False
)

target_bn_model = build_model(
    use_target_bn=True
)

print("\nEvaluating source BN features...")
source_results = evaluate_features(source_bn_model)

print("Evaluating target batch BN features...")
target_results = evaluate_features(target_bn_model)

print("\n" + "=" * 55)
print("RESULTS")
print("=" * 55)

for name, result in [
    ("Source BN", source_results),
    ("Target Batch BN", target_results)
]:
    print(f"\n{name}:")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")

output = ROOT / "experiments/prototype_validation"
output.mkdir(parents=True, exist_ok=True)

with open(output / "metrics.json", "w") as f:
    json.dump({
        "split": "val",
        "shift": shift,
        "source_bn": source_results,
        "target_batch_bn": target_results
    }, f, indent=2)

print("\nPROTOTYPE VALIDATION COMPLETED")
