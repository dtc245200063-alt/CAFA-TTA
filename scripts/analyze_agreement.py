
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

THRESHOLD = 0.7
BATCH_SIZE = 64

# Load the same domain shift as previous experiments.
with open(
    ROOT / "experiments/domain_shift/metrics.json",
    encoding="utf-8"
) as f:
    shift_config = json.load(f)

assert shift_config["split"] == "val"
shift = shift_config["shift"]

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
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load trained classifier.
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

# Use target batch statistics, no adaptation.
model.train()
model.requires_grad_(False)

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None

# Load source prototypes.
data = torch.load(
    ROOT / "checkpoints/source_prototypes.pt",
    map_location="cpu",
    weights_only=True
)

prototypes = F.normalize(
    data["prototypes"].to(DEVICE),
    dim=1
)

classifier_correct = []
prototype_correct = []
agreement_all = []
confidence_all = []

with torch.inference_mode():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.view(-1).long().to(DEVICE)

        features, logits = model(images)

        features = F.normalize(features, dim=1)
        similarity = features @ prototypes.T

        cls_pred = logits.argmax(dim=1)
        proto_pred = similarity.argmax(dim=1)

        confidence = confidence_score(logits)

        classifier_correct.append(
            cls_pred.eq(labels).cpu()
        )
        prototype_correct.append(
            proto_pred.eq(labels).cpu()
        )
        agreement_all.append(
            cls_pred.eq(proto_pred).cpu()
        )
        confidence_all.append(
            confidence.cpu()
        )

cls_correct = torch.cat(classifier_correct)
proto_correct = torch.cat(prototype_correct)
agreement = torch.cat(agreement_all)
confidence = torch.cat(confidence_all)

confident = confidence >= THRESHOLD
joint_mask = agreement & confident

def selection_metrics(mask):
    selected = int(mask.sum().item())
    coverage = selected / len(mask)

    selected_accuracy = (
        cls_correct[mask].float().mean().item()
        if selected > 0 else None
    )

    return {
        "selected": selected,
        "coverage": coverage,
        "pseudo_label_accuracy": selected_accuracy
    }

results = {
    "split": "val",
    "shift": shift,
    "batch_size": BATCH_SIZE,
    "confidence_threshold": THRESHOLD,
    "classifier_accuracy": (
        cls_correct.float().mean().item()
    ),
    "prototype_accuracy": (
        proto_correct.float().mean().item()
    ),
    "confidence_only": selection_metrics(confident),
    "agreement_only": selection_metrics(agreement),
    "agreement_and_confidence": selection_metrics(
        joint_mask
    )
}

print("=" * 55)
print("CAFA-TTA | CLASSIFIER-PROTOTYPE AGREEMENT")
print("Device:", DEVICE)
print("Total Images:", len(dataset))
print("Shift:", shift)
print("=" * 55)

print(
    "Classifier Accuracy:",
    f"{results['classifier_accuracy']:.2%}"
)

print(
    "Prototype Accuracy:",
    f"{results['prototype_accuracy']:.2%}"
)

for name in [
    "confidence_only",
    "agreement_only",
    "agreement_and_confidence"
]:
    metric = results[name]

    print("\n" + name)
    print("Selected:", metric["selected"])
    print("Coverage:", f"{metric['coverage']:.2%}")

    accuracy = metric["pseudo_label_accuracy"]
    print(
        "Pseudo-label Accuracy:",
        f"{accuracy:.2%}"
        if accuracy is not None else "N/A"
    )

output_dir = ROOT / "experiments/agreement"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "metrics.json"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

print("\nSaved:", output_path)
print("AGREEMENT ANALYSIS COMPLETED")
